import torch
import torch.nn as nn
from models.stream import stream
from models.ConvModule import ConvModule
from torchsummary import summary
import torchvision


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.stream = stream()
        self.module = ConvModule()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        

    def forward(self, inputs):

        reference, reference_inverse, test, test_inverse = self.module(inputs)
        reference, reference_inverse = self.stream(reference, reference_inverse)
        test, test_inverse = self.stream(test, test_inverse)

        cat_1 = torch.cat((test, reference_inverse), dim=1)
        cat_2 = torch.cat((reference, test), dim=1)
        cat_3 = torch.cat((reference, test_inverse), dim=1)

        del reference, reference_inverse, test, test_inverse

        cat_1 = self.sub_forward(cat_1)
        cat_2 = self.sub_forward(cat_2)
        cat_3 = self.sub_forward(cat_3)

        return cat_1, cat_2, cat_3
    
    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, inputs.size()[1])
        out = self.classifier(out)

        return out


if __name__ == '__main__':
    net = net().cuda()
    input = torch.randn(1, 2, 115, 220)
    print("The model summary is: ")
    summary(model=net, input_size=(2, 115, 220), batch_size=1, device="cuda")
    # out_1, out_2, out_3 = net(input)
    # print(out_1, out_2, out_3)
