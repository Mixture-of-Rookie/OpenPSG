#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import pickle
import numpy as np
from tqdm import tqdm
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet import __version__
from mmdet.core import BitmapMasks, mask
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.utils import collect_env, get_root_logger
from mmdet.datasets import build_dataloader
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from openpsg.datasets import build_dataset
from openpsg.models.relation_heads.approaches import Result
from openpsg.models.relation_heads.approaches import RelationSampler



def parse_args():# {{{
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--det-file', help='Path to the detection results.')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training',
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)',
    )
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)',
    )
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args# }}}


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False): torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # load det results
    with open(args.det_file, 'rb') as f:
        det_results = pickle.load(f)

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

    # build dataset & dataloader
    split = 'val'

    if split == 'train':
        dataset = copy.deepcopy(cfg.data.train)
    elif split == 'val':
        dataset = copy.deepcopy(cfg.data.val)
    else:
        raise ValueError

    pipeline = []
    for p in cfg.data.train.pipeline:
        if p['type'] == 'RandomFlip':
            pipeline.append({'type': 'RandomFlip', 'flip_ratio': 0.0})
        if p['type'] not in ['Resize', 'Pad']:
            pipeline.append(p)
    dataset.pipeline = pipeline
    dataset = build_dataset(dataset)

    cfg.data.samples_per_gpu = 1
    dataloader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        shuffle=False,
    )

    for i, data_batch in enumerate(tqdm(dataloader)):
        img_metas = data_batch['img_metas']._data[0]
        gt_bboxes = data_batch['gt_bboxes']._data[0]
        gt_labels = data_batch['gt_labels']._data[0]
        gt_rels = data_batch['gt_rels']._data[0]
        gt_relmaps = data_batch['gt_relmaps']._data[0]
        gt_masks = data_batch['gt_masks']._data[0]

        gt_labels = [gt_label + 1 for gt_label in gt_labels]

        gt_result = Result(
            bboxes=gt_bboxes,
            labels=gt_labels,
            masks=gt_masks,
            rels=gt_rels,
            relmaps=gt_relmaps,
            rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels]
            if gt_rels is not None else None,
            rel_labels=[rel[:, -1].clone() for rel in gt_rels]
            if gt_rels is not None else None,
        )

        assert img_metas[0]['flip'] is False

        filename = img_metas[0]['filename'].split('/', 3)[-1]
        pan_result = det_results[filename]

        # convert panoptic segments to bboxes
        ids = np.unique(pan_result)[::-1]
        legal_indices = ids != 133  # for VOID label
        ids = ids[legal_indices]  # exclude VOID label

        # Extract class labels, (N), 1-index?
        labels = np.array([id % INSTANCE_OFFSET for id in ids],
                           dtype=np.int64) + 1
        segms = pan_result[None] == ids[:, None, None]

        # convert to BitmapMasks
        height, width = segms.shape[1:]
        masks = BitmapMasks(segms, height, width)
        det_result = Result(
            labels=[torch.tensor(labels).to(gt_bboxes[0])],
            masks=[masks]
        )
        rel_labels, rel_idx_pairs, _ = sample_function(det_result, gt_result)

    print(relation_sampler.num_hit, relation_sampler.num_total)

if __name__ == '__main__':
    main()
