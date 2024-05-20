import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, inputs):
        half = inputs.size()[1] // 2
        reference = inputs[:, :half, :, :]
        reference_inverse = 255 - reference
        test = inputs[:, half:, :, :]
        del inputs
        test_inverse = 255 - test

        feature_reference = self.module(reference)
        feature_reference_inverse = self.module(reference_inverse)
        feature_test = self.module(test)
        feature_test_inverse = self.module(test_inverse)

        return feature_reference, feature_reference_inverse, feature_test, feature_test_inverse


if __name__ == '__main__':
    model = ConvModule()
    print(model)

    input = torch.randn(1, 2, 11, 28)
    output = model(input)

    for items in output:
        print(items.shape)
