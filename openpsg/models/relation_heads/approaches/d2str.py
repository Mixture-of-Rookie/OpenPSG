# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_dim, n_head=8, mlp_ratio=4., drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(n_dim)
        self.attn = nn.MultiheadAttention(n_dim, n_head, dropout=attn_drop,
                                          batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(n_dim)
        mlp_hidden_dim = int(n_dim * mlp_ratio)
        self.mlp = MLP(in_features=n_dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def attention(self, x, padding_mask):
        return self.attn(x, x, x, key_padding_mask=padding_mask)

    def forward(self, x, padding_mask):
        x2, attn = self.attention(self.norm1(x), padding_mask)
        x = x + self.drop_path(x2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_dim, n_head, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(n_dim, n_head, mlp_ratio)
            for _ in range(n_layer)
        ])

    def forward(self, x, token_list):
        x = x.split(token_list, dim=0)
        x = nn.utils.rnn.pad_sequence(x, batch_first=True)

        bs = len(token_list)
        max_len = max(token_list)
        
        num_objs = torch.LongTensor(token_list).to(x.device).unsqueeze(1).expand(-1, max_len)
        padding_mask = torch.arange(
            max_len, device=x.device).long().unsqueeze(0).expand(
            bs, max_len).ge(num_objs)

        attn_list = []
        for block in self.blocks:
            x, attn = block(x, padding_mask)
            attn_list.append(attn)

        # flatten x and discard pad tokens
        unpad_mask = torch.arange(
            max_len, device=x.device).unsqueeze(0).expand(
            bs, max_len).lt(num_objs)
        return x[unpad_mask], attn_list
