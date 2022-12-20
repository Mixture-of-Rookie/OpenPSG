#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn

from mmdet.core import bbox2roi
from mmdet.models import HEADS
from mmcv.cnn import normal_init, xavier_init

from .relation_head import RelationHead
from .approaches import TransformerEncoder
from .approaches.motif_util import (
    encode_box_info,
    psg_obj_edge_vectors,
)

@HEADS.register_module()
class D2STRHead(RelationHead):
    def __init__(self, **kwargs):
        super(D2STRHead, self).__init__(**kwargs)
        # inputs of object encoder
        self.embed_dim = self.head_config.embed_dim
        self.hidden_dim = self.head_config.hidden_dim
        self.in_channels = self.head_config.roi_dim
        self.label_embed1 = nn.Embedding(self.num_classes, self.embed_dim)

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128,
                                 self.hidden_dim)

        # object encoder
        self.obj_encoder = TransformerEncoder(
            n_layer=self.head_config.obj_layer,
            n_dim=self.head_config.obj_dim,
            n_head=self.head_config.num_head,
            drop=self.head_config.drop,
            attn_drop=self.head_config.attn_drop,
            drop_path=self.head_config.drop_path,
        )

        # object decoder
        self.obj_decoder = nn.Linear(self.hidden_dim, self.num_classes)

        # inputs of relation encoder
        self.rel_dim = self.hidden_dim * 2
        self.lin_rel = nn.Linear(self.hidden_dim, self.rel_dim)
        self.lin_up_rel = nn.Linear(self.rel_dim, self.in_channels)
        self.lin_mix = nn.Linear(self.in_channels * 2, self.rel_dim)
        self.label_embed2 = nn.Embedding(self.num_classes, self.embed_dim)

        # relation encoder
        self.rel_encoder = TransformerEncoder(
            n_layer=self.head_config.rel_layer,
            n_dim=self.head_config.rel_dim,
            n_head=self.head_config.num_head,
            mlp_ratio=self.head_config.rel_mlp_ratio,
            drop=self.head_config.drop,
            attn_drop=self.head_config.attn_drop,
            drop_path=self.head_config.drop_path,
        )

        # relation decoder
        self.rel_decoder = nn.Linear(self.rel_dim, self.num_predicates)
        self.use_vision = self.head_config.use_vision
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.rel2cxt = nn.Linear(self.rel_dim, self.context_pooling_dim)
        self.vis_decoder = nn.Linear(self.context_pooling_dim, self.num_predicates)

        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim,
                                    self.context_pooling_dim)
        else:
            self.union_single_not_match = False

    def init_weights(self):
        # initialize embedding with glove vector
        glove_vec = psg_obj_edge_vectors(wv_dir=self.head_config.glove_dir,
                                         wv_dim=self.embed_dim)
        with torch.no_grad():
            self.label_embed1.weight.copy_(glove_vec, non_blocking=True)
            self.label_embed2.weight.copy_(glove_vec, non_blocking=True)

        normal_init(self.lin_rel,
                    mean=0,
                    std=10.0 * (1.0 / self.hidden_dim) ** 0.5)

        xavier_init(self.obj_decoder)
        xavier_init(self.rel_decoder)
        xavier_init(self.vis_decoder)
        xavier_init(self.rel2cxt)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

    def forward(self, img, img_meta, det_result, gt_result=None,
                is_testing=False, ignore_classes=None):
        # 0. extract feats & sample object-pairs
        roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        # 1. prepare inputs of object encoder
        obj_inputs = self.prepare_obj_inputs(roi_feats, det_result)

        # 2. forward object encoder
        num_objs = [len(b) for b in det_result.bboxes]
        obj_feats, attn_list = self.obj_encoder(obj_inputs, num_objs)

        if is_testing:
            # from dense to sparse
            rel_pair_idxes = self.relation_sampler.prepare_test_pairs(
                det_result, attn_weights=attn_list[-1], threshold=0.0,
            )
            det_result.rel_pair_idxes = rel_pair_idxes

            rois = bbox2roi(det_result.bboxes)
            union_feats = self.relation_roi_extractor(img,
                                                      img_meta,
                                                      rois,
                                                      rel_pair_idx=rel_pair_idxes)[0]

        # 3. forward object decoder
        obj_dists = self.obj_decoder(obj_feats)
        obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        # 4. prepare inputs of relation encoder
        rel_inputs = self.prepare_rel_inputs(obj_feats, union_feats,
                                             det_result, num_objs)

        # 5. forward relation encoder
        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        rel_feats, _ = self.rel_encoder(rel_inputs, num_rels)

        # 6. forward relation decoder
        rel_dists = self.rel_decoder(rel_feats)

        if self.use_vision:
            ctx_gate = self.rel2cxt(rel_feats)
            if self.union_single_not_match:
                vis_feats = ctx_gate * self.up_dim(union_feats)
            else:
                vis_feats = ctx_gate * union_feats
            rel_dists = rel_dists + self.vis_decoder(vis_feats)

        # 7. result results
        if self.training:
            det_result.target_labels = torch.cat(
                det_result.target_labels, dim=-1
            )
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1
            )
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)

        det_result.attn_list = attn_list
        det_result.refine_scores = obj_dists
        det_result.rel_scores = rel_dists
        return det_result


    def prepare_obj_inputs(self, roi_feats, det_result):
        # the inputs of object encoder consists of:
        # 1): roi_feats; 2): bbox embedding; 3): label embeding
        bbox_embeds = self.bbox_embed(encode_box_info(det_result))

        obj_dists = torch.cat(det_result.dists, dim=0).detach()
        label_embeds = obj_dists @ self.label_embed1.weight

        obj_inputs = torch.cat((roi_feats, label_embeds, bbox_embeds), -1)
        obj_inputs = self.lin_obj(obj_inputs)
        return obj_inputs


    def prepare_rel_inputs(self, obj_feats, union_feats, det_result, num_objs):
        rel_feats = self.lin_rel(obj_feats).view(-1, 2, self.hidden_dim)

        head_feats = rel_feats[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_feats = rel_feats[:, 1].contiguous().view(-1, self.hidden_dim)
        head_feats = head_feats.split(num_objs, dim=0)
        tail_feats = tail_feats.split(num_objs, dim=0)

        pair_feats = []
        rel_pair_idxes = det_result.rel_pair_idxes
        for pair_idx, head_feat, tail_feat in zip(
            rel_pair_idxes, head_feats, tail_feats):
            pair_feats.append(
                torch.cat((head_feat[pair_idx[:, 0]], tail_feat[pair_idx[:, 1]]),
                          dim=-1))
        pair_feats = torch.cat(pair_feats, dim=0)

        up_pair_feats = self.lin_up_rel(pair_feats)
        rel_inputs = self.lin_mix(
            torch.cat((up_pair_feats, union_feats), dim=1))
        return rel_inputs


@HEADS.register_module()
class Mask2FormerD2STRHead(RelationHead):
    def __init__(self, **kwargs):
        super(Mask2FormerD2STRHead, self).__init__(**kwargs)

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

        # inputs of object encoder
        self.embed_dim = self.head_config.embed_dim
        self.hidden_dim = self.head_config.hidden_dim
        self.in_channels = self.head_config.roi_dim
        self.label_embed1 = nn.Embedding(self.num_classes, self.embed_dim)

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128,
                                 self.hidden_dim)

        # object encoder
        self.obj_encoder = TransformerEncoder(
            n_layer=self.head_config.obj_layer,
            n_dim=self.head_config.obj_dim,
            n_head=self.head_config.num_head,
            drop=self.head_config.drop,
            attn_drop=self.head_config.attn_drop,
            drop_path=self.head_config.drop_path,
        )

        # object decoder
        self.obj_decoder = nn.Linear(self.hidden_dim, self.num_classes)

        # inputs of relation encoder
        self.rel_dim = self.hidden_dim * 2
        self.lin_rel = nn.Linear(self.hidden_dim, self.rel_dim)
        self.lin_up_rel = nn.Linear(self.rel_dim, self.in_channels)
        self.lin_mix = nn.Linear(self.in_channels * 2, self.rel_dim)
        self.label_embed2 = nn.Embedding(self.num_classes, self.embed_dim)

        # relation encoder
        self.rel_encoder = TransformerEncoder(
            n_layer=self.head_config.rel_layer,
            n_dim=self.head_config.rel_dim,
            n_head=self.head_config.num_head,
            mlp_ratio=self.head_config.rel_mlp_ratio,
            drop=self.head_config.drop,
            attn_drop=self.head_config.attn_drop,
            drop_path=self.head_config.drop_path,
        )

        # relation decoder
        self.rel_decoder = nn.Linear(self.rel_dim, self.num_predicates)
        self.use_vision = self.head_config.use_vision
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.rel2cxt = nn.Linear(self.rel_dim, self.context_pooling_dim)
        self.vis_decoder = nn.Linear(self.context_pooling_dim, self.num_predicates)

        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim,
                                    self.context_pooling_dim)
        else:
            self.union_single_not_match = False

        self.use_spatial = self.head_config.use_spatial
        self.spatial_size = self.head_config.spatial_size
        if self.use_spatial:
            self.spatial_lin = nn.Linear(2 * self.spatial_size *
                                             self.spatial_size,
                                         self.hidden_dim)


    def init_weights(self):
        # initialize embedding with glove vector
        glove_vec = psg_obj_edge_vectors(wv_dir=self.head_config.glove_dir,
                                         wv_dim=self.embed_dim)
        with torch.no_grad():
            self.label_embed1.weight.copy_(glove_vec, non_blocking=True)
            self.label_embed2.weight.copy_(glove_vec, non_blocking=True)

        normal_init(self.lin_rel,
                    mean=0,
                    std=10.0 * (1.0 / self.hidden_dim) ** 0.5)

        xavier_init(self.bbox_head)
        xavier_init(self.union_head)
        xavier_init(self.obj_decoder)
        xavier_init(self.rel_decoder)
        xavier_init(self.vis_decoder)
        xavier_init(self.rel2cxt)

        if self.union_single_not_match:
            xavier_init(self.up_dim)


    def frontend_features(self, img_meta, det_result, gt_result, assigner='mask'):
        bboxes = det_result.bboxes
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
        union_feats = self.get_relation_feat(query_feats, rel_pair_idxes,
                                             use_spatial=self.use_spatial,
                                             bboxes=bboxes,
                                             img_meta=img_meta,
                                             spatial_size=self.spatial_size)

        return roi_feats + union_feats + (det_result, )


    def get_bbox_feat(self, query_feats):
        all_query_feats = torch.cat(query_feats, dim=0)
        return (self.bbox_head(all_query_feats), )


    def get_relation_feat(self, query_feats, rel_pair_idxes, use_spatial=False,
                          bboxes=None, img_meta=None, spatial_size=None):
        union_feats = []
        for i, (query_feat, rel_pair_idx) in enumerate(zip(query_feats,
                                                           rel_pair_idxes)):
            head_feat = query_feat[rel_pair_idx[:, 0]]
            tail_feat = query_feat[rel_pair_idx[:, 1]]
            union_feat = torch.cat([head_feat, tail_feat], dim=-1)
            if use_spatial:
                bbox = bboxes[i]
                num_rel = len(rel_pair_idx)
                dummy_x_range = (torch.arange(spatial_size).to(
                    rel_pair_idx.device).view(1, 1,
                                                -1).expand(num_rel, spatial_size,
                                                           spatial_size))
                dummy_y_range = (torch.arange(spatial_size).to(
                    rel_pair_idx.device).view(1, -1,
                                                1).expand(num_rel, spatial_size,
                                                          spatial_size))
                img_shape = torch.tensor(img_meta[i]['img_shape'][:2]).tile(num_rel, 1)
                img_shape = img_shape.to(bbox)

                # resize bbox to the spatial_size
                head_bbox = bbox[rel_pair_idx[:, 0]][:, :4]
                head_bbox[:, 0::2] *= spatial_size / img_shape[:, 1:2]
                head_bbox[:, 1::2] *= spatial_size / img_shape[:, 0:1]
                tail_bbox = bbox[rel_pair_idx[:, 1]][:, :4]
                tail_bbox[:, 0::2] *= spatial_size / img_shape[:, 1:2]
                tail_bbox[:, 1::2] *= spatial_size / img_shape[:, 0:1]

                head_rect = ((dummy_x_range >= head_bbox[:, 0].floor().view(
                    -1, 1, 1).long())
                             & (dummy_x_range <= head_bbox[:, 2].ceil().view(
                                 -1, 1, 1).long())
                             & (dummy_y_range >= head_bbox[:, 1].floor().view(
                                 -1, 1, 1).long())
                             & (dummy_y_range <= head_bbox[:, 3].ceil().view(
                                 -1, 1, 1).long())).float()
                tail_rect = ((dummy_x_range >= tail_bbox[:, 0].floor().view(
                    -1, 1, 1).long())
                             & (dummy_x_range <= tail_bbox[:, 2].ceil().view(
                                 -1, 1, 1).long())
                             & (dummy_y_range >= tail_bbox[:, 1].floor().view(
                                 -1, 1, 1).long())
                             & (dummy_y_range <= tail_bbox[:, 3].ceil().view(
                                 -1, 1, 1).long())).float()

                rect_input = torch.stack((head_rect, tail_rect),
                                         dim=1)  # (num_rel, 2, rect_size, rect_size)
                rect_feat = self.spatial_lin(rect_input.view(rect_input.size(0),
                                                             -1))
                union_feat = union_feat + rect_feat

            union_feat = self.union_head(union_feat)
            union_feats.append(union_feat)
        return (torch.cat(union_feats, dim=0), )


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

        # 1. prepare inputs of object encoder
        obj_inputs = self.prepare_obj_inputs(roi_feats, det_result)

        # 2. forward object encoder
        num_objs = [len(b) for b in det_result.bboxes]
        obj_feats, attn_list = self.obj_encoder(obj_inputs, num_objs)

        # 3. forward object decoder
        obj_dists = self.obj_decoder(obj_feats)
        obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        # 4. prepare inputs of relation encoder
        rel_inputs = self.prepare_rel_inputs(obj_feats, union_feats,
                                             det_result, num_objs)

        # 5. forward relation encoder
        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        rel_feats, _ = self.rel_encoder(rel_inputs, num_rels)

        # 6. forward relation decoder
        rel_dists = self.rel_decoder(rel_feats)

        if self.use_vision:
            ctx_gate = self.rel2cxt(rel_feats)
            if self.union_single_not_match:
                vis_feats = ctx_gate * self.up_dim(union_feats)
            else:
                vis_feats = ctx_gate * union_feats
            rel_dists = rel_dists + self.vis_decoder(vis_feats)

        # 7. result results
        if self.training:
            det_result.target_labels = torch.cat(
                det_result.target_labels, dim=-1
            )
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1
            )
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)

        det_result.attn_list = attn_list
        det_result.refine_scores = obj_dists
        det_result.rel_scores = rel_dists
        return det_result


    def prepare_obj_inputs(self, roi_feats, det_result):
        # the inputs of object encoder consists of:
        # 1): roi_feats; 2): bbox embedding; 3): label embeding
        bbox_embeds = self.bbox_embed(encode_box_info(det_result))

        obj_dists = torch.cat(det_result.dists, dim=0).detach()
        label_embeds = obj_dists @ self.label_embed1.weight

        obj_inputs = torch.cat((roi_feats, label_embeds, bbox_embeds), -1)
        obj_inputs = self.lin_obj(obj_inputs)
        return obj_inputs


    def prepare_rel_inputs(self, obj_feats, union_feats, det_result, num_objs):
        rel_feats = self.lin_rel(obj_feats).view(-1, 2, self.hidden_dim)

        head_feats = rel_feats[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_feats = rel_feats[:, 1].contiguous().view(-1, self.hidden_dim)
        head_feats = head_feats.split(num_objs, dim=0)
        tail_feats = tail_feats.split(num_objs, dim=0)

        pair_feats = []
        rel_pair_idxes = det_result.rel_pair_idxes
        for pair_idx, head_feat, tail_feat in zip(
            rel_pair_idxes, head_feats, tail_feats):
            pair_feats.append(
                torch.cat((head_feat[pair_idx[:, 0]], tail_feat[pair_idx[:, 1]]),
                          dim=-1))
        pair_feats = torch.cat(pair_feats, dim=0)

        up_pair_feats = self.lin_up_rel(pair_feats)
        rel_inputs = self.lin_mix(
            torch.cat((up_pair_feats, union_feats), dim=1))
        return rel_inputs
