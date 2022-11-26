#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import normal_init, xavier_init

from .approaches import TransformerContext
from .relation_head import RelationHead

@HEADS.register_module()
class TransformerHead(RelationHead):
    def __init__(self, **kwargs):
        super(TransformerHead, self).__init__(**kwargs)
        self.context_layer = TransformerContext(self.head_config,
                                                self.obj_classes,
                                                self.rel_classes)

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.context_pooling_dim)
        self.rel_compress = nn.Linear(self.context_pooling_dim, self.num_predicates, bias=True)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_predicates, bias=True)

        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim, self.context_pooling_dim)
        else:
            self.union_single_not_match = False

    def init_weights(self):
        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

        normal_init(self.post_emb, mean=0, std=10.0 * (1.0 / self.hidden_dim) ** 0.5)
        xavier_init(self.post_cat)
        xavier_init(self.rel_compress)
        xavier_init(self.ctx_compress)

        if self.union_single_not_match:
            xavier_init(self.up_dim)


    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
        """
        Obtain the relation prediction results based on detection results.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_result: (Result): Result containing bbox, label, mask, point, rels,
                etc. According to different mode, all the contents have been
                set correctly. Feel free to  use it.
            gt_result : (Result): The ground truth information.
            is_testing:
        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        refine_obj_scores, obj_preds, edge_ctx = self.context_layer(roi_feats, det_result)

        if is_testing and ignore_classes is not None:
            refine_obj_scores = self.process_ignore_objects(refine_obj_scores, ignore_classes)
            obj_preds = refine_obj_scores[:, 1:].max(1)[1] + 1

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
                det_result.rel_pair_idxes, head_reps, tail_reps, obj_preds):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]),
                          dim=-1))
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                            dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        # adapted from the Scene-Graph-Benchmark by K. Tang
        ctx_gate = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_feats)
            else:
                visual_rep = ctx_gate * union_feats

        rel_scores = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        return det_result


@HEADS.register_module()
class Mask2FormerTransformerHead(RelationHead):
    def __init__(self, **kwargs):
        super(Mask2FormerTransformerHead, self).__init__(**kwargs)

        self.bbox_head = nn.Sequential(*[
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        ])

        self.union_head = nn.Sequential(*[
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        ])
        self.context_layer = TransformerContext(self.head_config,
                                                self.obj_classes,
                                                self.rel_classes)

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2,
                                  self.context_pooling_dim)
        self.rel_compress = nn.Linear(self.context_pooling_dim,
                                      self.num_predicates,
                                      bias=True)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2,
                                      self.num_predicates,
                                      bias=True)

        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim,
                                    self.context_pooling_dim)
        else:
            self.union_single_not_match = False

    def init_weights(self):
        self.context_layer.init_weights()

        normal_init(self.post_emb, mean=0, std=10.0 * (1.0 / self.hidden_dim) ** 0.5)
        xavier_init(self.post_cat)
        xavier_init(self.rel_compress)
        xavier_init(self.ctx_compress)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

    def forward(self,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
        """Obtain the relation prediction results based on detection results.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_result: (Result): Result containing bbox, label, mask, point, rels,
                etc. According to different mode, all the contents have been
                set correctly. Feel free to  use it.
            gt_result : (Result): The ground truth information.
            is_testing:
        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        roi_feats, union_feats, det_result = self.frontend_features(
            img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        refine_obj_scores, obj_preds, edge_ctx = self.context_layer(
            roi_feats, det_result)

        if is_testing and ignore_classes is not None:
            refine_obj_scores = self.process_ignore_objects(refine_obj_scores, ignore_classes)
            obj_preds = refine_obj_scores[:, 1:].max(1)[1] + 1

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
                det_result.rel_pair_idxes, head_reps, tail_reps, obj_preds):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]),
                          dim=-1))
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                            dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        # adapted from the Scene-Graph-Benchmark by K. Tang
        ctx_gate = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_feats)
            else:
                visual_rep = ctx_gate * union_feats

        rel_scores = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(
                pair_pred.long())

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores
        return det_result

    def frontend_features(self, img_meta, det_result, gt_result, assigner='mask'):
        query_feats = det_result.query_feats

        if gt_result is not None and gt_result.rels is not None:
            assert self.mode == 'sgdet'
            if assigner == 'bbox':
                sample_function = self.relation_sampler.detect_relsample
            elif assigner == 'mask':
                sample_function = self.relation_sampler.segm_relsample

            sample_res = sample_function(det_result, gt_result)

            if len(sample_res) == 4:
                rel_labels, rel_pair_idxes, rel_matrix, \
                    key_rel_labels = sample_res
            else:
                rel_labels, rel_pair_idxes, rel_matrix = sample_res
                key_rel_labels = None
        else:
            rel_labels, rel_matrix, key_rel_labels = None, None, None
            rel_pair_idxes = self.relation_sampler.prepare_test_pairs(det_result)

        det_result.rel_pair_idxes = rel_pair_idxes
        det_result.relmaps = rel_matrix
        det_result.target_rel_labels = rel_labels
        det_result.target_key_rel_labels = key_rel_labels

        roi_feats = self.get_bbox_feat(query_feats)
        union_feats = self.get_relation_feat(query_feats, rel_pair_idxes)

        return roi_feats + union_feats + (det_result, )# }}}

    def get_bbox_feat(self, query_feats):
        all_query_feats = torch.cat(query_feats, dim=0)
        return (self.bbox_head(all_query_feats), )

    def get_relation_feat(self, query_feats, rel_pair_idxes):
        union_feats = []
        for query_feat, rel_pair_idx in zip(query_feats, rel_pair_idxes):
            head_feat = query_feat[rel_pair_idx[:, 0]]
            tail_feat = query_feat[rel_pair_idx[:, 1]]
            union_feat = torch.cat([head_feat, tail_feat], dim=-1)
            union_feat = self.union_head(union_feat)
            union_feats.append(union_feat)
        return (torch.cat(union_feats, dim=0), )
