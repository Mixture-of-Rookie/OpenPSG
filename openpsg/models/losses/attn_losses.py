
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class AttnMarginLoss(nn.Module):
    def __init__(self):
        super(AttnMarginLoss, self).__init__()
        self.margin_ranking_loss = nn.MarginRankingLoss()

    def forward(self, inputs1, inputs2, targets, margin=0., reduction='mean'):
        num_objs = inputs1.shape[0]

        loss = 0.
        for input1, input2, target in zip(inputs1, inputs2, targets):
            loss += F.margin_ranking_loss(input1, input2, target, margin=margin, reduction=reduction)
        return loss
