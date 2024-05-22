import torch
import torch.nn as nn
from models.ESA import ESA


class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()
		self.spatial_attention = ESA()

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

		self.Conv_1x1 = nn.Conv2d(128, 256, 1, stride=1, padding=0)
		self.Conv_32 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
		self.Conv_64 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.Conv_96 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
		self.Conv_128 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

		self.fc_32 = nn.Linear(32, 32)
		self.fc_64 = nn.Linear(64, 64)
		self.fc_96 = nn.Linear(96, 96)
		self.fc_128 = nn.Linear(128, 128)

		self.max_pool = nn.MaxPool2d(2, stride=2)

	def forward(self, reference, inverse):
		for i in range(4):
			reference = self.stream[0 + i * 5](reference)
			reference = self.stream[1 + i * 5](reference)
			inverse = self.stream[0 + i * 5](inverse)
			inverse = self.stream[1 + i * 5](inverse)
			inverse = self.stream[2 + i * 5](inverse)
			inverse = self.stream[3 + i * 5](inverse)
			inverse = self.stream[4 + i * 5](inverse)
			reference = self.spatial_attention(inverse)
			reference = self.stream[2 + i * 5](reference)
			reference = self.stream[3 + i * 5](reference)
			reference = self.stream[4 + i * 5](reference)

		return reference, inverse


if __name__ == '__main__':
	model = stream()
	r, i = model(torch.ones(1, 32, 115, 220), torch.ones(1, 32, 115, 220))
	# print(r.size(), i.size())
