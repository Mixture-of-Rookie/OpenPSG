#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import copy
import pickle
import warnings
import argparse

import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet.core import BitmapMasks, build_assigner
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from openpsg.datasets import build_dataset
from openpsg.models.relation_heads.approaches import Result
from openpsg.models.relation_heads.approaches import RelationSampler

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--pan-file', help='path to the panoptic results.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--use-tar-label',
        action='store_true',
        help='Whether to use target label (i.e., label by assignment).')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--gpu-collect',
                        action='store_true',
                        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument(
        '--submit',
        action='store_true',
        help=
        'save output to a json file and save the panoptic mask as a png image into a folder for grading purpose'
    )

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.submit, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if cfg.model.get('pretrained'):
        cfg.model.pretrained = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = os.path.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = copy.deepcopy(cfg.data.test)
    pipeline = []
    for p in cfg.data.train.pipeline:
        if p['type'] == 'RandomFlip':
            pipeline.append({'type': 'RandomFlip', 'flip_ratio': 0.0})
        if p['type'] not in ['Resize', 'Pad']:
            pipeline.append(p)
    dataset.pipeline = pipeline
    dataset = build_dataset(dataset)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False)

    # build relaiton sampler for evaluation
    relation_sampler = RelationSampler(
        type='Motif',
        pos_iou_thr=0.5,
        require_overlap=False,
        num_sample_per_gt_rel=4,
        num_rel_per_image=1024,
        pos_fraction=0.25,
        test_overlap=False,
        use_gt_box=False)
    sample_function = relation_sampler.segm_relsample

    # load det results
    # with open(args.pan_file, 'rb') as f:
    #     pan_results = pickle.load(f)

    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        gt_rels = data['gt_rels']._data[0]
        gt_masks = data['gt_masks']._data[0]
        gt_bboxes = data['gt_bboxes']._data[0]
        gt_labels = data['gt_labels']._data[0]
        gt_relmaps = data['gt_relmaps']._data[0]

        # preapre gt_result
        gt_labels = [gt_label + 1 for gt_label in gt_labels]

        gt_result = Result(
            rels=gt_rels,
            masks=gt_masks,
            bboxes=gt_bboxes,
            labels=gt_labels,
            relmaps=gt_relmaps,
            rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels]
            if gt_rels is not None else None,
            rel_labels=[rel[:, -1].clone() for rel in gt_rels]
            if gt_rels is not None else None,
        )

        # fetch pre-extract panoptic results
        img_metas = data['img_metas']._data[0]
        filename = img_metas[0]['filename'].split(
            '/', 3)[-1].split('/')[-1].split('.')[0]
        # pan_result = pan_results[filename]
        out_dir = '/mnt/disk6T/mask2former_swin_1x_psg_val_instance_80/'
        pan_result = np.load(os.path.join(out_dir, filename + '.npz'))
        segms = pan_result['segms']
        labels = pan_result['labels']

        if args.use_tar_label:
            mask_assigner = build_assigner({'type': 'MaskIoUAssigner',
                                            'pos_iou_thr': 0.3,
                                            'neg_iou_thr': 0.3,
                                            'min_pos_iou': 0.3,
                                            'match_low_quality': True,
                                            'ignore_iof_thr': -1})
            assign_result = mask_assigner.assign(
                masks=segms.astype('uint8'),
                gt_masks=gt_masks[0].masks,
                gt_labels=gt_labels[0] - 1,
            )
            labels = (assign_result.labels + 1).numpy()

        # convert to BitmapMasks
        height, width = segms.shape[1:]
        masks = BitmapMasks(segms, height, width)
        bboxes = masks.get_bboxes()
        det_result = Result(
            masks=[masks],
            bboxes=bboxes,
            labels=[torch.tensor(labels).to(gt_labels[0])],
        )
        rel_labels, rel_idx_pairs, _ = sample_function(det_result, gt_result)

        # filter background rels
        rel_idx = torch.nonzero(rel_labels[0]).flatten()

        if rel_idx.shape[0] == 0 and bboxes.shape[0] != 0:
            rel_labels = torch.tensor([1])
            rel_idx_pairs = torch.tensor([[0, 0]])
        else:
            rel_labels = rel_labels[0][rel_idx]
            rel_idx_pairs = rel_idx_pairs[0][rel_idx]

        # prepare det_result as pseudo output from the model
        det_result.masks = masks
        det_result.labels = labels
        obj_score = np.ones((labels.shape[0], 1), dtype='float32')
        det_result.refine_bboxes = np.concatenate([bboxes, obj_score], axis=1)
        det_result.rel_pair_idxes = rel_idx_pairs.numpy()
        det_result.rel_labels = rel_labels.numpy()
        det_result.rel_dists = np.eye(57)[rel_labels.numpy()]

        results.extend(det_result)

        batch_size = len(img_metas)
        for _ in range(batch_size):
            prog_bar.update()

    # Instead of forwarding by model,
    # we directly use the output of realtion sampler to
    # compute the upper bound of the performance
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            for result in results:
                result.masks = None
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
            import ipdb; ipdb.set_trace()
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(results, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

if __name__ == '__main__':
    main()
