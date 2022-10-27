#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core import BitmapMasks, build_assigner
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS, Mask2Former
from mmdet.models.builder import build_head
from mmdet.models.detectors.single_stage import SingleStageDetector

from openpsg.models.relation_heads.approaches import Result

@DETECTORS.register_module()
class SceneGraphMask2Former(Mask2Former):
    def __init__(
        self,
        backbone,
        neck=None,
        panoptic_head=None,
        panoptic_fusion_head=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        # for scene graph
        relation_head=None,
    ):
        super(SceneGraphMask2Former, self).__init__(
            backbone,
            neck=neck,
            panoptic_head=panoptic_head,
            panoptic_fusion_head=panoptic_fusion_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

        # Init relation head if relation_head is not None:
        if relation_head is not None:
            self.relation_head = build_head(relation_head)


    @property
    def with_relation(self):
        return hasattr(self,
            "relation_head") and self.relation_head is not None

    def forward_train(
        self,
        imgs,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_masks=None,
        gt_rels=None,
        gt_relmaps=None,
        rescale=False,
        **kwargs,
    ):

        # add batch_input_shape in img_metas
        super(SingleStageDetector, self).forward_train(imgs, img_metas)

        # Change gt to 1-index here
        gt_labels = [label + 1 for label in gt_labels]

        (
            bboxes,
            labels,
            target_labels,
            dists,
            pan_masks,
            pan_results,
            points,
            query_feats,
        ) = self.detector_simple_test(
            imgs,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_masks,
            rescale=rescale,
        )

        # Filter out empty predictions
        idxes_to_filter = [i for i, b in enumerate(bboxes) if len(b) == 0]

        param_need_filter = [
            bboxes, labels, dists, target_labels, gt_bboxes,
            gt_labels, gt_rels, img_metas, points, pan_results,
            gt_masks, gt_relmaps, pan_masks, query_feats
        ]

        for idx, param in enumerate(param_need_filter):
            if param_need_filter[idx]:
                param_need_filter[idx] = [
                    x for i, x in enumerate(param)
                    if i not in idxes_to_filter
                ]

        (bboxes, labels, dists, target_labels, gt_bboxes,
         gt_labels, gt_rels, img_metas, points, pan_results,
         gt_masks, gt_relmaps, pan_masks, query_feats) = param_need_filter


        gt_result = Result(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            relmaps=gt_relmaps,
            masks=gt_masks,
            rel_pair_idxes=[rel[:, :2].clone() for rel in gt_rels]
            if gt_rels is not None else None,
            rel_labels=[rel[:, -1].clone() for rel in gt_rels]
            if gt_rels is not None else None,
            img_shape=[meta['img_shape'] for meta in img_metas],
        )

        det_result = Result(
            bboxes=bboxes,
            labels=labels,
            dists=dists,
            masks=pan_masks,
            pan_results=pan_results,
            points=points,
            target_labels=target_labels,
            query_feats=query_feats,
            img_shape=[meta['img_shape'] for meta in img_metas]
        )

        det_result = self.relation_head(img_metas, det_result, gt_result)
        return self.relation_head.loss(det_result)


    def forward_test(self, imgs, img_metas, **kwargs):# {{{
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        key_first = kwargs.pop('key_first', False)


        assert num_augs == 1
        return self.relation_simple_test(imgs[0],
                                         img_metas[0],
                                         key_first=key_first,
                                         **kwargs)# }}}


    def detector_simple_test(# {{{
        self,
        imgs,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_masks,
        rescale=False,
        is_testing=False,
        **kwargs
    ):
        pan_seg_masks = None
        if not is_testing:
            det_results = self.simple_test_sg_bboxes(imgs,
                                                     img_metas,
                                                     rescale=rescale,
                                                     **kwargs)
            det_bboxes = [r['bboxes'] for r in det_results]
            det_labels = [r['labels'] for r in det_results]
            query_feats = [r['query_feats'] for r in det_results]
            pan_results = None

            target_labels = []
            # MaxIOUAssigner
            bbox_assigner = build_assigner({'type': 'MaxIoUAssigner',
                                            'pos_iou_thr': 0.5,
                                            'neg_iou_thr': 0.5,
                                            'min_pos_iou': 0.5,
                                            'match_low_quality': True,
                                            'ignore_iof_thr': -1})
            for i in range(len(img_metas)):
                assign_result = bbox_assigner.assign(
                    det_bboxes[i],
                    gt_bboxes[i],
                    gt_labels=gt_labels[i] - 1,
                )
                target_labels.append(assign_result.labels + 1)
        else:
            det_results = self.simple_test_sg_bboxes(imgs,
                                                     img_metas,
                                                     rescale=rescale,
                                                     **kwargs)
            det_bboxes = [r['bboxes'] for r in det_results]
            det_labels = [r['labels'] for r in det_results]
            query_feats = [r['query_feats'] for r in det_results]
            pan_seg_masks = [r['masks'] for r in det_results]

            # to reshape pan_seg_masks
            mask_size = (img_metas[0]['ori_shape'][0],
                         img_metas[0]['ori_shape'][1])
            pan_seg_masks = F.interpolate(
                torch.Tensor(pan_seg_masks[0]).unsqueeze(1),
                size=mask_size).squeeze(1).bool()
            pan_seg_masks = [pan_seg_masks.numpy()]


            det_results_for_pan = self.simple_test_sg_bboxes(imgs,
                                                             img_metas,
                                                             rescale=True,
                                                             **kwargs)
            pan_results = [r['pan_results'] for r in det_results_for_pan]

            target_labels = None

        det_dists = [
            F.one_hot(det_label,
                      num_classes=self.num_classes + 1).to(det_bboxes[0])
            for det_label in det_labels
        ]

        det_bboxes = [
            torch.cat([b, b.new_ones(len(b), 1)], dim=-1)
            for b in det_bboxes
        ]

        return det_bboxes, det_labels, target_labels, \
            det_dists, pan_seg_masks, pan_results, None, query_feats# }}}


    def simple_test_sg_bboxes(self, imgs, img_metas, rescale=False, **kwargs):# {{{
        """Test without Augmentation. Convert panoptic segments to bounding boxes."""
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results, query_feats = self.panoptic_head.simple_test(
            feats, img_metas, return_query_feats=True, **kwargs)
        results = self.panoptic_fusion_head.simple_test(
            mask_cls_results, mask_pred_results, img_metas, rescale=rescale, query_feats=query_feats, **kwargs)

        for i in range(len(results)):
            assert 'pan_results' in results[i], 'panoptic results not found.'
            results[i]['pan_results'] = results[i]['pan_results'].detach().cpu().numpy()
            query_feats = results[i]['query_feats']
            query_feats_indicator = results[i]['query_feats_indicator']
            
            # Convert panoptic results to bboxes
            pan_results = results[i]['pan_results']
            ids = np.unique(pan_results)[::-1]
            legal_indices = ids != self.num_classes
            ids = ids[legal_indices]

            # fetch query_feats
            query_idx = np.array([query_feats_indicator[id] for id in ids],
                                 dtype=np.int32)
            query_feats = query_feats[query_idx]

            # Extract class labels (1-index)
            labels = np.array([id % INSTANCE_OFFSET for id in ids],
                              dtype=np.int64) + 1
            segms = pan_results[None] == ids[:, None, None]

            # Convert to bboxes
            height, width = segms.shape[1:]
            bboxes = BitmapMasks(segms, height, width).get_bboxes()

            # Convert to tensor
            bboxes = torch.tensor(bboxes).to(feats[0].device)
            labels = torch.tensor(labels).to(feats[0].device)

            results[i]['masks'] = segms
            results[i]['bboxes'] = bboxes
            results[i]['labels'] = labels
            results[i]['query_feats'] = query_feats

        return results# }}}


    def relation_simple_test(# {{{
        self,
        img,
        img_meta,
        gt_bboxes=None,
        gt_labels=None,
        gt_masks=None,
        rescale=False,
        key_first=False,
    ):
        # Rescale should be forbidden here since the bboxes and masks
        # will be used in relation module.
        bboxes, labels, target_labels, dists, pan_masks, pan_results, points, query_feats \
            = self.detector_simple_test(
            img,
            img_meta,
            gt_bboxes,
            gt_labels,
            gt_masks,
            rescale=False,
            is_testing=True,
        )

        det_result = Result(
            bboxes=bboxes,
            labels=labels,
            dists=dists,
            masks=pan_masks,
            pan_results=pan_results,
            points=points,
            target_labels=target_labels,
            query_feats=query_feats,
            img_shape=[meta['img_shape'] for meta in img_meta]
        )

        # If empty prediction
        if len(bboxes[0]) == 0:
            return det_result

        det_result = self.relation_head(img_meta, det_result, is_testing=True)
        scale_factor = img_meta[0]['scale_factor']
        det_result = self.relation_head.get_result(det_result,
                                                   scale_factor,
                                                   rescale=rescale,
                                                   key_first=key_first)
        if pan_masks is not None:
            det_result.masks = np.array(pan_masks[0])

        return det_result# }}}
