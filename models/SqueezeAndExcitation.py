# Squeeze and Excitation block

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        pooled = self.avg_pool(x).squeeze(-1).squeeze(-1)
        print("The size of squeezed pooled feature is: ", pooled.shape)
        excite = self.excite(pooled)
        scaled = x * excite.unsqueeze(-1).unsqueeze(-1)
        return scaled


if __name__ == '__main__':
    input = torch.randn(1, 32, 224, 224)
    model = SEBlock(in_channels=32, reduction_ratio=4)
    output = model(input)
    print(output.size())
