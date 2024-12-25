# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from Net import Cnn_250, AlexNet_250, Dnn_250, Bilinear_250, Cnn_150, AlexNet_150, Dnn_150, Bilinear_150, Bilinear_150_bnrelu

# 配置 logging
logging.basicConfig(filename='training.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')

# 设置参数
# 解析命令行参数
parser = argparse.ArgumentParser(description='Training script with variable batch size.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
parser.add_argument('--input_shape', type=int, default=150, help='Size of picture')
parser.add_argument('--model', type=str, default='Cnn', help='Type of model')
args = parser.parse_args()

# 将命令行参数记录到日志
logging.info("Command line arguments:")
for arg_name in vars(args):
    logging.info(f"{arg_name}: {getattr(args, arg_name)}")

model_dict = {
    'Cnn_250': Cnn_250,
    'AlexNet_250': AlexNet_250,
    'Dnn_250': Dnn_250,
    'Bilinear_250': Bilinear_250,
    'Cnn_150': Cnn_150,
    'AlexNet_150': AlexNet_150,
    'Dnn_150': Dnn_150,
    'Bilinear_150': Bilinear_150,
    'Bilinear_150_bnrelu': Bilinear_150_bnrelu
}
model_name = args.model
batch_size = args.batch_size
epochs = args.epochs
input_shape = args.input_shape
lr = args.lr

# 训练集目录
TRAIN_DIRS = ['./dataset_torch/train1', './dataset_torch/train2', './dataset_torch/train3', './dataset_torch/train4',
              './dataset_torch/train5']
# 验证集目录
VAL_DIRS = ['./dataset_torch/val1', './dataset_torch/val2', './dataset_torch/val3', './dataset_torch/val4',
            './dataset_torch/val5']

# 定义归一化转换，将像素值归一化到 [-1, 1] 之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((input_shape, input_shape)),
    transforms.ToTensor(),
    normalize  # 应用归一化
])

val_transform = transforms.Compose([
    transforms.Resize((input_shape, input_shape)),
    transforms.ToTensor(),
    normalize  # 应用归一化
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(3407)
if device == 'cuda':
    torch.cuda.manual_seed_all(3407)


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
        n += 1

    train_loss = loss / n
    train_acc = current / n
    logging.info('train_loss: {:.4f}     train_acc: {:.4f}'.format(train_loss, train_acc))
    return train_loss, train_acc


def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device).float()  # Ensure y is float
            output = model(image)
            cur_loss = loss_fn(output, y)
            cur_acc = torch.sum((y == output.round()).int()).float() / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    val_loss = loss / n
    val_acc = current / n
    logging.info('val_loss: {:.4f}     val_acc: {:.4f}'.format(val_loss, val_acc))
    return val_loss, val_acc


# 定义画图函数
def matplot_loss(train_loss, val_loss, fold_number):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(bottom=0)  # 设置y轴的最小值为0
    plt.xlim(left=0)  # 设置x轴的最小值为0，如果epoch从1开始，可以去掉这行
    plt.title("loss")
    file_name = f"{model_name}_fold_{fold_number + 1}_batch_{batch_size}_epochs_{epochs}_input_{input_shape}_loss.png"
    plt.savefig(file_name)
    plt.show()


def matplot_acc(train_acc, val_acc, fold_number):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.ylim(bottom=0)  # 设置y轴的最小值为0
    plt.xlim(left=0)  # 设置x轴的最小值为0，如果epoch从1开始，可以去掉这行
    plt.title("acc ")
    file_name = f"{model_name}_fold_{fold_number + 1}_batch_{batch_size}_epochs_{epochs}_input_{input_shape}_acc.png"
    plt.savefig(file_name)
    plt.show()


# 开始训练
all_loss_train = [0.0, 0.0, 0.0, 0.0, 0.0]
all_acc_train = [0.0, 0.0, 0.0, 0.0, 0.0]
all_loss_val = [0.0, 0.0, 0.0, 0.0, 0.0]
all_acc_val = [0.0, 0.0, 0.0, 0.0, 0.0]

avg_loss_train = 0
avg_acc_train = 0
avg_loss_val = 0
avg_acc_val = 0

for i in range(5):
    logging.info(f"Fold {i + 1}\n")
    model = model_dict[model_name]().to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ROOT_TRAIN = TRAIN_DIRS[i]
    ROOT_TEST = VAL_DIRS[i]
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    epoch = epochs
    min_acc = 0
    best_epoch = 0
    for t in range(epoch):
        start = time.time()
        logging.info(f"epoch{t + 1}\n-----------")
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
            all_acc_val[i] = val_acc
            all_loss_val[i] = val_loss
            all_acc_train[i] = train_acc
            all_loss_train[i] = train_loss

            min_acc = val_acc
            best_epoch = t
            file_name = f"{model_name}_batch_{args.batch_size}_epochs_{args.epochs}_input_{args.input_shape}best.pth"
            torch.save(model.state_dict(), os.path.join(folder, file_name))
        # 保存最后一轮的权重文件
        # if t == epoch - 1:
        #     torch.save(model.state_dict(), 'save_model/last_bilinear.pth')
        end = time.time()
        logging.info(f"Epoch time: {end - start}\n")

        # all_time = end-start
        # logging.info("all_time",all_time)
    logging.info(f"\nbest epoch {best_epoch}\n")
    avg_acc_val += all_acc_val[i]
    avg_loss_val += all_loss_val[i]
    avg_acc_train += all_acc_train[i]
    avg_loss_train += all_loss_train[i]

    matplot_loss(loss_train, loss_val, i)
    matplot_acc(acc_train, acc_val, i)

logging.info("")
logging.info('avg_train_loss: {:.4f}     avg_train_acc: {:.4f}'.format(avg_loss_train / 5, avg_acc_train / 5))
logging.info('avg_val_loss: {:.4f}     avg_val_acc: {:.4f}'.format(avg_loss_val / 5, avg_acc_val / 5))
logging.info('Done!\n')
