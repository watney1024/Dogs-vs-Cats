#!/bin/bash
#激活虚拟环境

# 定义模型名称
MODEL_NAME="Bilinear_150"

# 定义批处理大小
BATCH_SIZE=16

# 定义学习率
LEARNING_RATE=0.0001

# 定义训练周期
EPOCHS=100

# 定义输入图片尺寸 (通道数, 高度, 宽度)
INPUT_SHAPE=150

# 运行Python训练脚本
python auto_training.py \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --input_shape "${INPUT_SHAPE}" \
    --model ${MODEL_NAME}