# This is a crude implementation of Enhanced Spatial Attention Module

import torch
import torch.nn as nn
import numpy as np


class ESA(nn.Module):
    def __init__(self, in_channels=32, reduction_factor=4):
        super(ESA, self).__init__()
        reduced_channels = int(in_channels//reduction_factor)
        self.conv1x1_1 = nn.Conv2d(in_channels=in_channels, out_channels=reduced_channels, kernel_size=1, stride=1)
        self.conv_stride = nn.Conv2d(in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=7, stride=3)
        # Grouped Convolution with number of convolutions equal to number of input channels
        self.conv_group = nn.Conv2d(in_channels=reduced_channels, out_channels=reduced_channels, kernel_size=3, stride=1, padding=1, groups=reduced_channels)
        # self.upsampler = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=False)
        self.conv1x1_2 = nn.Conv2d(in_channels=reduced_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        reduced_feature = self.conv1x1_1(input) # First 1x1 convolution to reduce channels
        strided_feature = self.conv_stride(reduced_feature)  # Stride convolution
        pooled_feature = self.pool(strided_feature)     # Pooling
        grouped_feature = self.conv_group(pooled_feature)    # Grouped convolution
        # upsampled_feature = self.upsampler(grouped_feature)     # Upsample to original size
        upsampled_feature = nn.functional.interpolate(grouped_feature, size=input.size()[2:], mode='bilinear', align_corners=False)
        reduced2_feature = self.conv1x1_2(upsampled_feature)     # Second 1x1 convolution to restore channel size
        attention = self.sigmoid(reduced2_feature)     # Generate attention mask

        # Scaled connection
        masked_feature_map = input * attention
        return masked_feature_map


if __name__ == '__main__':
    model = ESA()
    # print(model)
    input = torch.randn(1, 32, 115, 220)
    print("Input Shape: ", input.shape)
    out = model(input)
    print("Output Shape: ", out.shape)
