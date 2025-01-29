import time
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


# 训练集目录
TRAIN_DIRS = ['../dataset_torch/train1', '../dataset_torch/train2', '../dataset_torch/train3', '../dataset_torch/train4',
              '../dataset_torch/train5']
# 验证集目录
VAL_DIRS = ['../dataset_torch/val1', '../dataset_torch/val2', '../dataset_torch/val3', '../dataset_torch/val4',
            '../dataset_torch/val5']

# 定义归一化转换，将像素值归一化到 [-1, 1] 之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, device):
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

def val(dataloader, model, loss_fn, device):
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
