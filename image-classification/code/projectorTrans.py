# Projector module to fuse transformer output with the feature map.

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, CT, C, head=16, groups=16):
        super(Projector , self).__init__()
        self.proj_value_conv = nn.Conv1d(CT, C, kernel_size=1)
        self.proj_key_conv = nn.Conv1d(CT, C, kernel_size=1)
        self.proj_query_conv = nn.Conv2d(C, CT, kernel_size=1,groups=groups)
        self.head = head

    def forward(self, feature, token):
        N, L, CT = token.shape
        token = token.view(N, CT, L)
        h = self.head
        proj_v = self.proj_value_conv(token).view(N, h, -1, L)
        proj_k = self.proj_key_conv(token).view(N, h, -1, L)
        proj_q = self.proj_query_conv(feature)
        N, C, H, W = proj_q.shape
        proj_q = proj_q.view(N, h, C // h, H * W).permute(0, 1, 3, 2)
        proj_coef = F.softmax(torch.Tensor.matmul(proj_q, proj_k) / np.sqrt(C / h), dim=3)
        proj = torch.Tensor.matmul(proj_v, proj_coef.permute(0, 1, 3, 2))
        _, _, H, W = feature.shape
        return feature + proj.view(N, -1, H, W), token
