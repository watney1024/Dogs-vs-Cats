import os
import time
import psutil

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 训练集目录
TRAIN_DIRS = ['./dataset_torch/train1', './dataset_torch/train2', './dataset_torch/train3', './dataset_torch/train4',
              './dataset_torch/train5']
# 验证集目录
VAL_DIRS = ['./dataset_torch/val1', './dataset_torch/val2', './dataset_torch/val3', './dataset_torch/val4',
            './dataset_torch/val5']

# # 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 定义归一化转换，将像素值归一化到 [-1, 1] 之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
input_shape = (3, 150, 150)
train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    normalize  # 应用归一化
])

val_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    normalize  # 应用归一化
])

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

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device).float()  # Ensure y is float
        output = model(image)
        cur_loss = loss_fn(output, y)
        cur_acc = torch.sum((y == output.round()).int()).float() / output.shape[0]
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    train_loss = loss / n
    train_acc = current / n
    print('train_loss: {:.4f}     train_acc: {:.4f}'.format(train_loss, train_acc))
    return train_loss, train_acc

def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    start = time.time()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device).float()  # Ensure y is float
            output = model(image)
            cur_loss = loss_fn(output, y)
            cur_acc = torch.sum((y == output.round()).int()).float() / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
    end = time.time()
    print(f'time cost = {end - start}')
    val_loss = loss / n
    val_acc = current / n
    print('val_loss: {:.4f}     val_acc: {:.4f}'.format(val_loss, val_acc))
    return val_loss, val_acc

# 定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(bottom=0)  # 设置y轴的最小值为0
    plt.xlim(left=0)  # 设置x轴的最小值为0，如果epoch从1开始，可以去掉这行
    plt.title("loss")
    plt.show()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.ylim(bottom=0)  # 设置y轴的最小值为0
    plt.xlim(left=0)  # 设置x轴的最小值为0，如果epoch从1开始，可以去掉这行
    plt.title("acc ")
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(3407)
    if device == 'cuda':
        torch.cuda.manual_seed_all(3407)

    # 开始训练
    activation_sizes = [] # 存储中间激活值大小
    # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for i in range(1):
        model = Bilinear(input_shape).to(device)
        
        # 定义钩子函数并注册到 model 所有层
        def hook_fn(module, input, output):
            size_in_bytes = output.element_size() * output.numel() # 元素大小 * 元素总数
            activation_sizes.append(size_in_bytes)
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU)): # 选择需要监控的层
                layer.register_forward_hook(hook_fn)
        model.qconfig = get_default_qat_qconfig('fbgemm') # 配置 QAT
        prepare_qat(model, inplace=True)

        # 定义一个损失函数
        loss_fn = nn.BCELoss()
        # 定义一个优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # 学习率每隔10轮变为原来的0.5
        # lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        ROOT_TRAIN = TRAIN_DIRS[i]
        ROOT_TEST = VAL_DIRS[i]
        train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
        val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=6)
        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []
        epoch = 1
        min_acc = 0
        best_epoch = 0
        for t in range(epoch):
            print(f"epoch{t + 1}\n-----------")
            print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")
            start = time.time()
            train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
            val_loss, val_acc = val(val_dataloader, model, loss_fn)

            loss_train.append(train_loss)
            acc_train.append(train_acc)
            loss_val.append(val_loss)
            acc_val.append(val_acc)
            # 保存最好的模型权重
            if val_acc > min_acc:
                folder = 'save_model'
                if not os.path.exists(folder):
                    os.mkdir('save_model')
                min_acc = val_acc
                print(f"save best model, 第{t + 1}轮")
                best_epoch = t
                torch.save(model.state_dict(), 'save_model/best_bilinear.pth')
            # 保存最后一轮的权重文件
            if t == epoch - 1:
                torch.save(model.state_dict(), 'save_model/last_bilinear.pth')
            # lr_scheduler.step()
            end = time.time()
            print(end - start)
        # matplot_loss(loss_train, loss_val)
        # matplot_acc(acc_train, acc_val)
    print('finish training')

    print('start validation')
    # 测试不开启量化 model 最大中间激活值内存
    activation_sizes= []
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
    print(f"model activation sizes: {max(activation_sizes) / (1024 ** 2):.4f} MB")
    # 测试开启量化后 model 最大中间激活值内存
    torch.backends.quantized.engine = 'fbgemm'
    quantized_model = torch.quantization.convert(model, inplace=False)
    activation_sizes= []
    val_loss, val_acc = val(val_dataloader, quantized_model, loss_fn)
    print(f"quantized_model activation sizes: {max(activation_sizes) / (1024 ** 2):.4f} MB")