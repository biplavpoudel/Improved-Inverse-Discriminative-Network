# This is a rough implementation of four SE blocks for each input stream
import torch
import torch.nn as nn
from models.ESA import ESA
from models.SqueezeAndExcitation import SEBlock
# import torchvision.models as models
# May implement ResNetSE block in the future but not now


class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()

		self.stream = nn.Sequential(
			nn.Conv2d(32, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

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
			nn.MaxPool2d(2, stride=2)
			)

		self.squeeze_excitation1 = SEBlock(in_channels=32)
		self.squeeze_excitation2 = SEBlock(in_channels=64)
		self.squeeze_excitation3 = SEBlock(in_channels=96)
		self.squeeze_excitation4 = SEBlock(in_channels=128)

		self.spatial_attention1 = ESA(in_channels=32)
		self.spatial_attention2 = ESA(in_channels=64)
		self.spatial_attention3 = ESA(in_channels=96)
		self.spatial_attention4 = ESA(in_channels=128)

	def forward(self, reference, inverse):
		for i in range(4):
			inverse = self.stream[0 + i * 5](inverse)
			inverse = self.stream[1 + i * 5](inverse)
			inverse = self.stream[2 + i * 5](inverse)
			inverse = self.stream[3 + i * 5](inverse)
			inverse = self.stream[4 + i * 5](inverse)
			excited_feature = getattr(self, 'squeeze_excitation' + str(i+1))(inverse)
			
			attention = getattr(self, 'spatial_attention' + str(i+1))(excited_feature)

			reference = self.stream[0 + i * 5](reference)
			reference = self.stream[1 + i * 5](reference)
			reference = self.stream[2 + i * 5](reference) * attention
			reference = self.stream[3 + i * 5](reference)
			reference = self.stream[4 + i * 5](reference)

		return reference, inverse


if __name__ == '__main__':
	model = stream()
	r, i = model(torch.ones(1, 32, 115, 220), torch.ones(1, 32, 115, 220))
	# print(r.size(), i.size())
