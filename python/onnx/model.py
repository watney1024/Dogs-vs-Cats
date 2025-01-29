import torch
from torch import nn

class Bilinear(nn.Module):
    def __init__(self, input_shape):
        super(Bilinear, self).__init__()
        self.input_shape = input_shape
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second Convolutional Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third Convolutional Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth Convolutional Block
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Average Pooling
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifiers = nn.Sequential(
            nn.Linear(128 ** 2, 1),
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x) # 在需要开启量化的层前后加入 量化层 和 反量化层
        x = self.features(x)
        x = self.dequant(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 128, 4 ** 2)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 4 ** 2).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.quant(x)
        x = torch.sigmoid(self.classifiers(x))
        x = self.dequant(x)
        return x.squeeze(1)
