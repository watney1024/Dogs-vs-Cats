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


import torch.nn.functional as F
from torch.autograd import Function


class myQuant(Function):
    @staticmethod
    def forward(ctx, x):
        min_val, max_val = x.min(), x.max()  # TODO 莫名报错
        # min_val, max_val = torch.tensor(0), torch.tensor(1)  # 计算最小值和最大值
        scale = (max_val - min_val) / 255    # 计算 scale
        zero_point = -min_val / scale        # 计算 zero_point
        zero_point = torch.round(zero_point).clamp(0, 255)  # 限制范围
        # 量化
        x = torch.round(x / scale + zero_point)  # 转换为 int8
        x = x.float()
        return [x, scale, zero_point]

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.clone()  # 近似梯度 = 直接传递梯度
        return grad_x, None  # 第二个返回值为 None，表示 func 无梯度
    
    @staticmethod
    def symbolic(g: torch.Graph, x: torch.Tensor):
        # x, scale, zero_point = g.op("MyQuant", x)
        return g.op("MyQuant", x)

class myDequant(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point):
        x = (x.float() - zero_point.float()) * scale  # 反量化
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output.clone()  # 近似梯度 = 直接传递梯度
        return grad_x, None  # 第二个返回值为 None，表示 func 无梯度
    
    @staticmethod
    def symbolic(g: torch.Graph, x: torch.Tensor):
        return g.op("MyDequant", x)

# 使用自定义量化算子的模型
class Bilinear2(nn.Module):
    def __init__(self, input_shape):
        super(Bilinear2, self).__init__()
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
        self.my_quant = myQuant()
        self.my_dequant = myDequant()

    def forward(self, x):
        x, scale, zero_point = self.my_quant.apply(x)
        x = self.features(x)
        x = self.my_dequant.apply(x, scale, zero_point)
        batch_size = x.size(0)
        x = x.view(batch_size, 128, 4 ** 2)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 4 ** 2).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x, scale, zero_point = self.my_quant.apply(x)
        x = torch.sigmoid(self.classifiers(x))
        x = self.my_dequant.apply(x, scale, zero_point)
        return x.squeeze(1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化
        # 全连接层：根据最后的输出特征尺寸来确定输入维度
        # 假设经过3个卷积层和池化层后，特征图的尺寸为 (64, 18, 18)（这里的计算需要根据输入尺寸来推算）
        self.fc1 = nn.Linear(64 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 10)  # 假设10类输出（可以根据实际情况调整）

    def forward(self, x):
        x, scale, zero_point = myQuant.apply(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = myDequant.apply(x, scale, zero_point)
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x