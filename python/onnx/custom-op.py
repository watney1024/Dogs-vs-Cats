import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.onnx import OperatorExportTypes


class CustomRot90AndScale(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        x = torch.rot90(x, k=1, dims=(3, 2))  # clockwise 90
        x *= 1.2
        return x

    @staticmethod
    def symbolic(g: torch.Graph, x: torch.Tensor):
        return g.op("Rot90AndScale", x, k_i=1, scale_f=1.2, clockwise_s="yes")


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return CustomRot90AndScale.apply(x)


if __name__ == '__main__':
    with torch.inference_mode():
        custum_model = MyModel()
        x = torch.randn(1, 3, 224, 224)

        torch.onnx.export(model=custum_model,
                          args=(x,),
                          f="custom_rot90.onnx",
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {2: "h0", 3: "w0"},
                                        "output": {2: "w0", 3: "h0"}},
                          opset_version=16,
                          operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH)

    # 加载 ONNX 模型
    onnx_model = onnx.load("custom_rot90.onnx")
    # 检查模型是否有效
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型加载成功！")
    print(onnx.helper.printable_graph(onnx_model.graph))

