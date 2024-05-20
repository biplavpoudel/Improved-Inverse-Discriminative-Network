# This is a crude implementation of Enhanced Spatial Attention Module

import torch
import torch.nn as nn
import numpy as np


class ESA(nn.Module):
    def __init_(self):
        super(ESA, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv_stride = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_group = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=0, padding=1, groups=4)
        self.upsampler = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        original_input = input

        reduced_feature = self.conv1x1_1(input) # Apply first 1x1 convolution
        strided_feature = self.conv_stride(reduced_feature) # Apply stride convolution
        pooled_feature = self.pool(strided_feature) # Apply pooling
        grouped_feature = self.conv_group(pooled_feature) # Apply grouped convolution
        upsampled_feature = self.upsampler(grouped_feature) # Upsample to original size
        reduced2_feature = self.conv1x1_2(upsampled_feature) # Apply second 1x1 convolution
        attention_mask = self.sigmoid(reduced2_feature) # Generate attention mask

        # Apply attention mask using element-wise Hadamard multiplication
        out = original_input * attention_mask
        return out
