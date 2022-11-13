#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init

from .motif_util import (encode_box_info, get_dropout_mask, obj_edge_vectors,
                         to_onehot)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, graph_mask_type='add'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.graph_mask_type = graph_mask_type

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            if mask.dtype is not torch.bool:
                # positive value is for rel; -1 for padding;
                if self.graph_mask_type == 'add':
                    attn = attn + mask
                elif self.graph_mask_type == 'mul':
                    attn = attn * mask
                # elif self.graph_mask_type == 'none':
                #     pass
                else:
                    pass
                # fill padding node with -inf
                fill_mask = mask.lt(0)
                attn = attn.masked_fill(fill_mask, -np.inf)
            else:
                attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, d_model)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()  # len_k==len_v

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = residual + self.layer_norm(output)

        return output, attn


class PositionWiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Merge adjacent information. Equal to linear layer if kernel size is 1
        Args:
            x (bsz, len, dim)
        Returns:
            output (bsz, len, dim)
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = residual + self.layer_norm(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 dropout=0.1, graph_matrix=None):
        super().__init__()
        self.graph_matrix = graph_matrix
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input_feats, count_split, graph_mask=None):
        """
        Args:
            input_feats [Tensor] (#total_box, d_model) : bounding box features of a batch
            num_objs [list of int] (bsz, ) : number of bounding box of each image

        Returns:
            enc_output [Tensor] (#total_box, d_model)
        """
        input_feats = input_feats.split(count_split, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(count_split)
        device = input_feats.device
        pad_len = max(count_split)
        num_objs_ = torch.LongTensor(count_split).to(device).unsqueeze(1).expand(-1, pad_len)

        if graph_mask is not None:
            if graph_mask.dtype is not torch.bool:
                slf_attn_mask = graph_mask
            else:
                slf_attn_mask = graph_mask.logical_not()
        else:
            slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(
                1).expand(-1, pad_len, -1)  # (bsz, pad_len, pad_len) False->include  True -> not

        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(
            num_objs_).unsqueeze(-1)  # (bsz, pad_len, 1) True -> include; False -> not

        # -- Forward
        enc_output = input_feats
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output


class GTransformerContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes):
        super(GTransformerContext, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.in_channels = self.cfg.roi_dim
        self.obj_dim = self.in_channels
        self.embed_dim = self.cfg.embed_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.nms_thresh = self.cfg.test_nms_thres

        self.dropout_rate = self.cfg.dropout_rate
        self.obj_layer = self.cfg.obj_layer
        self.num_head = self.cfg.num_head
        self.inner_dim = self.cfg.inner_dim
        self.k_dim = self.cfg.key_dim
        self.v_dim = self.cfg.val_dim

        self.use_gt_box = self.cfg.use_gt_box
        self.use_gt_label = self.cfg.use_gt_label

        # mode
        if self.cfg.use_gt_box:
            if self.cfg.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        embed_vecs = obj_edge_vectors(self.obj_classes,
                                      wv_dir=self.cfg.glove_dir,
                                      wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1)])

        # for other embed operation
        self.lin_obj = nn.Linear(self.in_channels + self.embed_dim + 128,
                                 self.hidden_dim)
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_classes)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head,
                                              self.k_dim, self.v_dim,
                                              self.hidden_dim, self.inner_dim,
                                              self.dropout_rate)

    def init_weights(self):
        xavier_init(self.lin_obj)
        xavier_init(self.out_obj)

    def forward(self, x, det_result, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.use_gt_label:
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.cfg.use_gt_label:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        num_objs = [len(b) for b in det_result.bboxes]
        pos_embed = self.pos_embed(encode_box_info(det_result))

        # encode objects with transformer
        obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)
        obj_pre_rep = self.lin_obj(obj_pre_rep)

        obj_feats = self.context_obj(obj_pre_rep, num_objs, graph_mask=None)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_feats)
            # use_decoder_nms = self.mode == 'sgdet' and not self.training
            use_decoder_nms = False
            if use_decoder_nms:
                raise NotImplementedError
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        return obj_dists, obj_preds, obj_feats, None


class BaseTransformerEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, n_layer, num_head, k_dim, v_dim,
                 dropout_rate=0.1):
        super(BaseTransformerEncoder, self).__init__()

        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_head = num_head
        self.dropout_rate = dropout_rate

        self.graph_encoder = TransformerEncoder(n_layer, self.num_head,
                                                self.k_dim, self.v_dim,
                                                input_dim, out_dim, self.dropout_rate)

    def forward(self, x, counts, adj_matrices=None):
        if adj_matrices is not None:
            adj_matrices = self.build_padding_adj(adj_matrices, counts)
        x = self.graph_encoder(x, counts, adj_matrices)
        return x

    @staticmethod
    def build_padding_adj(adj_matrices, counts):
        """
        expand the adj matrix to the same size, and stack them into one Tensor

        """
        padding_size = max(counts)
        index = torch.arange(padding_size).long()

        res = []
        for adj in adj_matrices:
            expand_mat = torch.zeros(size=(padding_size, padding_size)) - 1
            expand_mat[index, index] = 1
            expand_mat = expand_mat.to(adj)
            adj_count = adj.size(0)
            expand_mat[:adj_count, :adj_count] = adj
            res.append(expand_mat)
        return torch.stack(res)

