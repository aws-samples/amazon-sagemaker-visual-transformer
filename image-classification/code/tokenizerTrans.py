# Tokenizer module to convert feature maps into visual tokens.

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tokenizer(nn.Module):
    def __init__(self, L, CT, C, head=16, groups=16, dynamic=False, input_channels=256):
        super(Tokenizer , self).__init__()
         # Code for adjusting the channel sizes in case C is not equal to CT
        self.feature = nn.Conv2d(input_channels, C, kernel_size=1)
        if not dynamic :
            # use static weights to compute token coefficients.
            self.conv_token_coef = nn.Conv2d(C, L, kernel_size=1) 
        else:
            # use previous tokens to compute a query weight, which is
            # then used to compute token coefficients. 
            self.conv_query = nn.Conv1d(CT, C, kernel_size=1) 
            self.conv_key = nn.Conv2d(C, C, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(C, C,kernel_size=1, groups=groups)
        self.head = head
        self.dynamic = dynamic
        self.C = C

    def forward(self, feature, tokens=0):
        N, C, H, W = feature.shape
        if C != self.C:
            feature = self.feature(feature)
        # compute token coefficients 
        #feature: N, C, H, W, token: N, CT, L
        if not self.dynamic : 
            token_coef = self.conv_token_coef(feature)
            N, L, H, W = token_coef.shape
            token_coef = token_coef.view(N, 1, L, H * W)
            token_coef = token_coef.permute(0, 1, 3, 2) # N, 1, HW, L 
            token_coef = token_coef / np.sqrt(feature.shape[1])
        else: 
            L = tokens.shape[2]
            # Split input tokens
            T_a, T_b = tokens[:, :, :L // 2], tokens[:, :, L // 2:] 
            query = self.conv_query(T_a)
            N, C, L_a = query.shape
            query = query.view(N, self.head, C // self.head, L_a) 
            N, C, H, W = feature.shape
            key = self.conv_key(feature).view(N, self.head, C // self.head, H * W) # N, h, C//h, HW 
            # Compute token coefficients.
            # N, h, HW, L_a
            token_coef = torch.Tensor.matmul(key.permute(0, 1, 3, 2), query) 
            token_coef = token_coef / np.sqrt(C / self.head)
        N, C, H, W = feature.shape
        token_coef = F.softmax(token_coef , dim=2)
        value = self.conv_value(feature).view(N, self.head, C // self.head, H * W) # N, h, C//h, HW
        # extract tokens from the feature map
        # static tokens: N, C, L. dynamic tokens: N, C, L_a
        tokens = torch.Tensor.matmul(value, token_coef).view(N, C, -1)
        tokens = tokens.view(N, L, C)
        return feature, tokens
