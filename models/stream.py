# This is a rough implementation of four SE blocks for each input stream
import torch
import torch.nn as nn
from torchsummary import summary
from models.ESA import ESA
from models.SqueezeAndExcitation import SEBlock
# import torchvision.models as models


class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()

		self.stream = nn.Sequential(
			nn.Conv2d(32, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=1),

			nn.Conv2d(32, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(64, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(96, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),
			)

		self.squeeze_excitation = nn.ModuleList([
			SEBlock(in_channels=32),
			SEBlock(in_channels=64),
			SEBlock(in_channels=96),
			SEBlock(in_channels=128)
		])

		self.spatial_attention = nn.ModuleList([
			ESA(in_channels=32),
			ESA(in_channels=64),
			ESA(in_channels=96),
			ESA(in_channels=128)
		])

	def forward(self, reference, inverse):
		for i in range(4):
			for j in range(5):
				reference = self.stream[j + i * 5](reference)
				inverse = self.stream[j + i * 5](inverse)

			reference, inverse = self.attention(i, inverse, reference)
			# print("Sizes of each feature maps are:", reference.shape, inverse.shape)

		return reference, inverse

	def attention(self, i, inverse, discriminative):
		# print(inverse.size(), discriminative.size())
		excited_inverse = self.squeeze_excitation[i](inverse)

		g = self.spatial_attention[i](excited_inverse)
		g = g + excited_inverse 	# SEResNet Block for inverse stream
		# print("Size of g:", g.shape)
		aggregated_features = g * discriminative
		# print("Size of aggregated features:", aggregated_features.shape)
		channel_attention = self.squeeze_excitation[i](aggregated_features)

		temp = aggregated_features * channel_attention
		out = temp + discriminative		# SEResNet Block for inverse stream
		#
		return out, excited_inverse


if __name__ == '__main__':
	model = stream().cuda()
	# r, i = model(torch.ones(1, 32, 115, 200), torch.ones(1, 32, 115, 200))
	# print(r.size(), i.size())
	summary(model, input_size=[(32, 115, 200), (32, 115, 200)], batch_size=1, device='cuda')