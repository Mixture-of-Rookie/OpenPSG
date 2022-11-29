#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import normal_init, xavier_init

from .approaches import GTransformerContext, BaseTransformerEncoder
from .relation_head import RelationHead

@HEADS.register_module()
class RTPBHead(RelationHead):
    def __init__(self, **kwargs):
        super(RTPBHead, self).__init__(**kwargs)

        self.context_layer = GTransformerContext(self.head_config,
                                                 self.obj_classes,
                                                 self.rel_classes)
        
        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.in_channels = self.head_config.roi_dim
        self.edge_repr_dim = self.hidden_dim * 2
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_obj_edge_repr = nn.Linear(self.hidden_dim, self.edge_repr_dim)

        self.epsilon = 0.001
        self.use_rel_graph = self.head_config.use_rel_graph
        self.use_graph_encode = self.head_config.use_graph_encode
        self.graph_enc_strategy = self.head_config.graph_encode_strategy

        if self.use_graph_encode:
            if self.graph_enc_strategy == 'trans':
                # encode relationship with trans
                self.pred_up_dim = nn.Linear(self.edge_repr_dim, self.in_channels)
                self.mix_ctx = nn.Linear(self.in_channels + self.in_channels, self.edge_repr_dim)
                n_layer = self.head_config.rel_layer
                num_head = self.head_config.num_head
                k_dim = self.head_config.key_dim
                v_dim = self.head_config.val_dim
                dropout_rate = self.head_config.dropout_rate
                self.graph_encoder = nn.ModuleList([
                    BaseTransformerEncoder(input_dim=self.edge_repr_dim,
                                           out_dim=self.edge_repr_dim,
                                           n_layer=n_layer,
                                           num_head=num_head,
                                           k_dim=k_dim,
                                           v_dim=v_dim,
                                           dropout_rate=dropout_rate)
                ])
            elif self.graph_enc_strategy == 'cross_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'all_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'mix':
                raise NotImplementedError

        # final classification
        self.rel_visual_clf = nn.Linear(self.context_pooling_dim,
                                        self.num_predicates)
        self.rel_clf = nn.Linear(self.edge_repr_dim, self.num_predicates)
        self.post_rel2ctx = nn.Linear(self.edge_repr_dim, self.context_pooling_dim)

        # about visual feature of union boxes
        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim, self.context_pooling_dim)
        else:
            self.union_single_not_match = False

        # bias module
        self.bias_module = None

    def init_weights(self):
        normal_init(self.post_obj_edge_repr,
                    mean=0,
                    std=10.0 * (1.0 / self.hidden_dim) ** 0.5)
        
        xavier_init(self.rel_visual_clf)
        xavier_init(self.rel_clf)
        xavier_init(self.post_rel2ctx)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

        self.bbox_roi_extractor.init_weights()
        self.relation_roi_extractor.init_weights()
        self.context_layer.init_weights()

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
        roi_feats, union_feats, det_result = self.frontend_features(
            img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        refine_obj_scores, obj_preds, obj_feats, _ = self.context_layer(roi_feats, det_result)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # rel encoding
        obj_repr_for_edge = self.post_obj_edge_repr(
            obj_feats).view(-1, 2, self.hidden_dim)
        edge_rep, obj_pair_labels = self.composeEdgeRepr(obj_repr_for_edge,
                                                         obj_preds,
                                                         det_result.rel_pair_idxes,
                                                         num_objs)

        rel_positive_prob = torch.ones_like(edge_rep[:, 0])
        if self.use_graph_encode:
            if self.use_rel_graph:
                rel_adj_list = self.build_rel_graph(rel_positive_prob, num_rels,
                                                    det_result.rel_pair_idxes,
                                                    num_objs)
            else:
                rel_adj_list = [None] * len(num_rels)

            # union_features
            if self.graph_enc_strategy == 'cat_gcn':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'trans':
                edge_rep = self.pred_up_dim(edge_rep)
                edge_rep = torch.cat((edge_rep, union_feats), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
                if not self.use_rel_graph:
                    rel_adj_list = None
                for encoder in self.graph_encoder:
                    edge_rep = encoder(edge_rep, num_rels, rel_adj_list)
            elif self.graph_enc_strategy == 'cross_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'all_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'mix':
                raise NotImplementedError
            else:
                raise NotImplementedError

        # rel classification
        rel_scores = self.rel_classification(edge_rep, union_feats)

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        return det_result
        
    def composeEdgeRepr(self, obj_repr_for_edge, obj_preds, rel_pair_idxs,
                        num_objs):
        # from object level feature to pairwise relation level feature
        pred_reps = []
        pair_preds = []

        head_rep = obj_repr_for_edge[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = obj_repr_for_edge[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        for pair_idx, head_rep, tail_rep, obj_pred in zip(
                rel_pair_idxs, head_reps, tail_reps, obj_preds):
            pred_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]),
                          dim=-1))
            pair_preds.append(
                torch.stack(
                    (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                    dim=1))

        pair_rel_rep = torch.cat(pred_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)
        return pair_rel_rep, pair_pred

    def build_rel_graph(self, rel_positive_prob, num_rels, rel_pair_idxs,
                        num_objs):
        positive_rel_split = torch.split(rel_positive_prob, num_rels)
        rel_graph = []
        for rel_cls, rel_pair_idx, num_obj in zip(
                positive_rel_split, rel_pair_idxs, num_objs):
            num_rel = rel_pair_idx.size(0)
            rel_obj_matrix = torch.zeros((num_rel, num_obj), device=rel_cls.device)
            idx = torch.arange(num_rel)
            valid_score = rel_cls.float()

            rel_obj_matrix[idx, rel_pair_idx[:, 0]] += valid_score
            rel_obj_matrix[idx, rel_pair_idx[:, 1]] += valid_score

            adj = torch.matmul(rel_obj_matrix, rel_obj_matrix.T)
            adj[idx, idx] = 1
            adj = adj + self.epsilon
            rel_graph.append(adj)

        return rel_graph

    def rel_classification(self, edge_rep, union_feats):
        # rel cls
        rel_dists = self.rel_clf(edge_rep)
        # remove bias
        if not self.training and self.head_config.remove_bias:
            rel_dists = rel_dists - self.rel_clf.bias

        # use union box and mask convolution
        if self.use_vision:
            ctx_gate = self.post_rel2ctx(edge_rep)
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_feats)
            else:
                visual_rep = ctx_gate * union_feats
            rel_dists = rel_dists + self.rel_visual_clf(visual_rep)

        # use bias module
        #TODO support bias module
        bias = None

        if bias is not None:
            rel_dists = rel_dists + bias

        return rel_dists


@HEADS.register_module()
class Mask2FormerRTPBHead(RelationHead):
    def __init__(self, **kwargs):
        super(Mask2FormerRTPBHead, self).__init__(**kwargs)

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
        self.context_layer = GTransformerContext(self.head_config,
                                                 self.obj_classes,
                                                 self.rel_classes)

        # post decoding
        self.use_vision = self.head_config.use_vision
        self.hidden_dim = self.head_config.hidden_dim
        self.in_channels = self.head_config.roi_dim
        self.edge_repr_dim = self.hidden_dim * 2
        self.context_pooling_dim = self.head_config.context_pooling_dim
        self.post_obj_edge_repr = nn.Linear(self.hidden_dim, self.edge_repr_dim)

        self.epsilon = 0.001
        self.use_rel_graph = self.head_config.use_rel_graph
        self.use_graph_encode = self.head_config.use_graph_encode
        self.graph_enc_strategy = self.head_config.graph_encode_strategy

        if self.use_graph_encode:
            if self.graph_enc_strategy == 'trans':
                # encode relationship with trans
                self.pred_up_dim = nn.Linear(self.edge_repr_dim, self.in_channels)
                self.mix_ctx = nn.Linear(self.in_channels + self.in_channels, self.edge_repr_dim)
                n_layer = self.head_config.rel_layer
                num_head = self.head_config.num_head
                k_dim = self.head_config.key_dim
                v_dim = self.head_config.val_dim
                dropout_rate = self.head_config.dropout_rate
                self.graph_encoder = nn.ModuleList([
                    BaseTransformerEncoder(input_dim=self.edge_repr_dim,
                                           out_dim=self.edge_repr_dim,
                                           n_layer=n_layer,
                                           num_head=num_head,
                                           k_dim=k_dim,
                                           v_dim=v_dim,
                                           dropout_rate=dropout_rate)
                ])
            elif self.graph_enc_strategy == 'cross_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'all_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'mix':
                raise NotImplementedError

        # final classification
        self.rel_visual_clf = nn.Linear(self.context_pooling_dim,
                                        self.num_predicates)
        self.rel_clf = nn.Linear(self.edge_repr_dim, self.num_predicates)
        self.post_rel2ctx = nn.Linear(self.edge_repr_dim, self.context_pooling_dim)

        # about visual feature of union boxes
        if self.context_pooling_dim != self.head_config.roi_dim:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(self.head_config.roi_dim, self.context_pooling_dim)
        else:
            self.union_single_not_match = False

        # bias module
        self.bias_module = None

    def init_weights(self):
        normal_init(self.post_obj_edge_repr,
                    mean=0,
                    std=10.0 * (1.0 / self.hidden_dim) ** 0.5)
        
        xavier_init(self.rel_visual_clf)
        xavier_init(self.rel_clf)
        xavier_init(self.post_rel2ctx)

        if self.union_single_not_match:
            xavier_init(self.up_dim)

        self.context_layer.init_weights()

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

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        refine_obj_scores, obj_preds, obj_feats, _ = self.context_layer(roi_feats, det_result)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # rel encoding
        obj_repr_for_edge = self.post_obj_edge_repr(
            obj_feats).view(-1, 2, self.hidden_dim)
        edge_rep, obj_pair_labels = self.composeEdgeRepr(obj_repr_for_edge,
                                                         obj_preds,
                                                         det_result.rel_pair_idxes,
                                                         num_objs)

        rel_positive_prob = torch.ones_like(edge_rep[:, 0])
        if self.use_graph_encode:
            if self.use_rel_graph:
                rel_adj_list = self.build_rel_graph(rel_positive_prob, num_rels,
                                                    det_result.rel_pair_idxes,
                                                    num_objs)
            else:
                rel_adj_list = [None] * len(num_rels)

            # union_features
            if self.graph_enc_strategy == 'cat_gcn':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'trans':
                edge_rep = self.pred_up_dim(edge_rep)
                edge_rep = torch.cat((edge_rep, union_feats), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
                if not self.use_rel_graph:
                    rel_adj_list = None
                for encoder in self.graph_encoder:
                    edge_rep = encoder(edge_rep, num_rels, rel_adj_list)
            elif self.graph_enc_strategy == 'cross_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'all_trans':
                raise NotImplementedError
            elif self.graph_enc_strategy == 'mix':
                raise NotImplementedError
            else:
                raise NotImplementedError

        # rel classification
        rel_scores = self.rel_classification(edge_rep, union_feats)

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels,
                                                 dim=-1)
            det_result.target_rel_labels = torch.cat(
                det_result.target_rel_labels, dim=-1)
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        return det_result

    def composeEdgeRepr(self, obj_repr_for_edge, obj_preds, rel_pair_idxs,
                        num_objs):
        # from object level feature to pairwise relation level feature
        pred_reps = []
        pair_preds = []

        head_rep = obj_repr_for_edge[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = obj_repr_for_edge[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        for pair_idx, head_rep, tail_rep, obj_pred in zip(
                rel_pair_idxs, head_reps, tail_reps, obj_preds):
            pred_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]),
                          dim=-1))
            pair_preds.append(
                torch.stack(
                    (obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]),
                    dim=1))

        pair_rel_rep = torch.cat(pred_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)
        return pair_rel_rep, pair_pred

    def build_rel_graph(self, rel_positive_prob, num_rels, rel_pair_idxs,
                        num_objs):
        positive_rel_split = torch.split(rel_positive_prob, num_rels)
        rel_graph = []
        for rel_cls, rel_pair_idx, num_obj in zip(
                positive_rel_split, rel_pair_idxs, num_objs):
            num_rel = rel_pair_idx.size(0)
            rel_obj_matrix = torch.zeros((num_rel, num_obj), device=rel_cls.device)
            idx = torch.arange(num_rel)
            valid_score = rel_cls.float()

            rel_obj_matrix[idx, rel_pair_idx[:, 0]] += valid_score
            rel_obj_matrix[idx, rel_pair_idx[:, 1]] += valid_score

            adj = torch.matmul(rel_obj_matrix, rel_obj_matrix.T)
            adj[idx, idx] = 1
            adj = adj + self.epsilon
            rel_graph.append(adj)

        return rel_graph

    def rel_classification(self, edge_rep, union_feats):
        # rel cls
        rel_dists = self.rel_clf(edge_rep)
        # remove bias
        if not self.training and self.head_config.remove_bias:
            rel_dists = rel_dists - self.rel_clf.bias

        # use union box and mask convolution
        if self.use_vision:
            ctx_gate = self.post_rel2ctx(edge_rep)
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_feats)
            else:
                visual_rep = ctx_gate * union_feats
            rel_dists = rel_dists + self.rel_visual_clf(visual_rep)

        # use bias module
        #TODO support bias module
        bias = None

        if bias is not None:
            rel_dists = rel_dists + bias

        return rel_dists

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
