import onnx
import torch
import torch.onnx
from torch.utils.data import DataLoader
from torch.quantization import get_default_qat_qconfig, prepare_qat
from torchvision.datasets import ImageFolder

from model import Bilinear2, SimpleCNN
from train import VAL_DIRS, val_transform


if __name__ == "__main__":
    input_shape = (3, 150, 150)
    model = SimpleCNN()

    # model.qconfig = get_default_qat_qconfig('fbgemm') # 配置 QAT
    # prepare_qat(model, inplace=True)

    # torch.backends.quantized.engine = 'fbgemm'
    # quantized_model = torch.quantization.convert(model, inplace=False)
    quantized_model = model

    dataset = ImageFolder(VAL_DIRS[0], transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)
    input, label = next(iter(dataloader))

    # 4. 导出 ONNX
    torch.onnx.export(
        quantized_model, 
        input, 
        "Bilinear.onnx",   # 输出文件名
        export_params=True,   # 存储模型参数
        opset_version=16,     # ONNX 版本
        do_constant_folding=True,  # 进行常量折叠优化
        input_names=["input"],     # 输入名称
        output_names=["output"],   # 输出名称
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # 动态批量大小
    )

    print("Successfully converted to ONNX format!")

    # 加载 ONNX 模型
    onnx_model = onnx.load("Bilinear.onnx")

    # 检查模型是否有效
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型加载成功！")
    
    print(onnx.helper.printable_graph(onnx_model.graph))
