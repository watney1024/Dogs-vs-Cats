#!/bin/bash
#激活虚拟环境
#Cnn_250, AlexNet_250, Dnn_250, Bilinear_250, Cnn_150, AlexNet_150, Dnn_150, Bilinear_150, Bilinear_150_bnrelu

# 定义模型名称
MODEL_NAME1="Cnn_150"
# 定义模型名称
MODEL_NAME2="AlexNet_150"

# 定义批处理大小
BATCH_SIZE=16

# 定义学习率
LEARNING_RATE=0.0001

# 定义训练周期
EPOCHS=100

# 定义输入图片尺寸 (通道数, 高度, 宽度)
INPUT_SHAPE=150


python auto_training.py \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --input_shape "${INPUT_SHAPE}" \
    --model ${MODEL_NAME1}

python auto_training.py \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --input_shape "${INPUT_SHAPE}" \
    --model ${MODEL_NAME2}

