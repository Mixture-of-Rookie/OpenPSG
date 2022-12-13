#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
