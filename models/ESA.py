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
        # print("Shape after first 1x1 conv: ", reduced_feature.shape)
        strided_feature = self.conv_stride(reduced_feature)  # Stride convolution
        # print("Shape after stride conv: ", strided_feature.shape)
        pooled_feature = self.pool(strided_feature)     # Pooling
        # print("Shape after pooling: ", pooled_feature.shape)
        grouped_feature = self.conv_group(pooled_feature)    # Grouped convolution
        # print("Shape after grouping: ", grouped_feature.shape)
        # upsampled_feature = self.upsampler(grouped_feature)     # Upsample to original size
        upsampled_feature = nn.functional.interpolate(grouped_feature, size=input.size()[2:], mode='bilinear', align_corners=False)
        # print("Shape after upsampling: ", upsampled_feature.shape)
        reduced2_feature = self.conv1x1_2(upsampled_feature)     # Second 1x1 convolution to restore channel size
        # print("Shape after conv1x1_2: ", reduced2_feature.shape)
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
