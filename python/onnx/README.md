## 使用 pytorch 提供的 api 实现基本量化操作
```
在 cpu 上运行量化测试（暂不支持 gpu）
要求数据集 dataset_torch 在 python 文件夹下
$ python qat-test.py
```

## 学习 onnx
### onnx 是什么
    ONNX（Open Neural Network Exchange）是一个由微软提出的一种开放的神经网络交换格式，用于在不同的深度学习框架之间进行模型的转换和交流。
    使用ONNX，你可以使用不同的深度学习框架（如PyTorch、TensorFlow等）进行模型的训练和定义，然后将模型导出为ONNX格式。导出后的ONNX模型包含了模型的结构和权重参数等信息。
### onnx 模型部署
    导出模型参见 model2onnx.py
### onnx 模型组成
    ONNX模型主要由三部分组成：Graph、Node、Tensor
    Graph:
        nodes:
        inputs:
        outputs:
        name:
        doc_string:
        opset_version:
        ... TODO
        opset 算子集合：ONNX模型其实本质上就是将源模型通过各种基础算子来搭建出来，所以算子集合中包括了各种基础算子，当模型中使用到该算子时，则直接调用即可，但是随着模型复杂度的提升，模型可能会使用到算子集合中没有的算子，所以算子集合也是需要不断更新的，因此opset可以理解为version，当将模型转成ONNX失败时，可以尝试提高opset的版本。
    Node:
        op：该节点执行的操作，表示该节点是什么算子，如：加算子、卷积算子等
        name：节点名字
        attrs：将属性名称映射到属性值的字典。可以理解为节点的参数，如：加算子没有参数，但是对于FC算子，他有output_number的参数。
        inputs：节点的输入。需要注意的是：如FC算子，他是有权重weight和偏置bias的，这部分权值ONNX是将其当成节点的输入的，以Tensor的形式进行保存。
    Tensor:
        name:
        dtype:
        shape: