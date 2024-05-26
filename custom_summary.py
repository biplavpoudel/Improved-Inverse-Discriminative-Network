import torch
import torch.nn as nn
from torchsummary import summary


class CustomSummary:
    def __init__(self, model, input_size, device="cuda"):
        self.model = model.to(device)
        self.input_size = input_size
        self.device = device

    def print_summary(self):
        # Print the standard summary
        summary(self.model, self.input_size, batch_size=1, device=self.device)

    def print_detailed_summary(self):
        # Print detailed layer information
        self._print_layer_information()

        # Print parameters and buffers
        self._print_params_and_buffers()

    def _print_layer_information(self):
        print("\nDetailed Layer Information:\n")
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Sequential):
                continue
            print(f"{name}: {layer}")

    def _print_params_and_buffers(self):
        print("\nParameters and Buffers:\n")
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.shape}")
        for name, buffer in self.model.named_buffers():
            print(f"{name}: {buffer.shape}")


if __name__ == '__main__':
    # Import the model (ensure the correct import path)
    from models.SqueezeAndExcitation import SEBlock
    from models.ESA import ESA
    from models.net import net
    from models.ConvModule import ConvModule
    from models.stream import stream

    # model = SEBlock(in_channels=32).cuda()
    # model = ESA().cuda()
    # model = net().cuda()
    # model = ConvModule().cuda()
    model = stream().cuda()
    net_input_size = (2, 115, 220)
    esa_size = (32, 115, 220)
    stream_size = [(32, 115, 200), (32, 115, 200)]
    custom_summary = CustomSummary(model, stream_size)
    custom_summary.print_summary()
