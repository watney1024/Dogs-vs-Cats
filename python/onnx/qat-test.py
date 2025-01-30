import os
import time

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.quantization import get_default_qat_qconfig, prepare_qat
from torchvision.datasets import ImageFolder

from model import Bilinear
from train import TRAIN_DIRS, VAL_DIRS, train_transform, val_transform, train, val


if __name__ == "__main__":
    input_shape = (3, 150, 150)
    device = 'cpu'

    # 开始训练
    activation_sizes = [] # 存储中间激活值大小
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
            train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
            val_loss, val_acc = val(val_dataloader, model, loss_fn, device)

            loss_train.append(train_loss)
            acc_train.append(train_acc)
            loss_val.append(val_loss)
            acc_val.append(val_acc)
            end = time.time()
            print(end - start)
    print('finish training')

    print('start validation')
    # 测试不开启量化 model 最大中间激活值内存
    activation_sizes= []
    val_loss, val_acc = val(val_dataloader, model, loss_fn, device)
    print(f"model activation sizes: {max(activation_sizes) / (1024 ** 2):.4f} MB")
    # 测试开启量化后 model 最大中间激活值内存
    torch.backends.quantized.engine = 'fbgemm'
    quantized_model = torch.quantization.convert(model, inplace=False)
    activation_sizes= []
    val_loss, val_acc = val(val_dataloader, quantized_model, loss_fn, device)
    print(f"quantized_model activation sizes: {max(activation_sizes) / (1024 ** 2):.4f} MB")