# Goal

依照04_training_framework的结构，学习构建一个简单的ResNet来处理MNIST

# Construction

```
05_resnet
├── checkpoints
├── configs
├── datasets
├── models
├── plots
├── README.md
├── results
├── test.py
├── train.py
└── utils
```

# Config

epoch=10

learning rate=1e-3

train batch size=64

test batch size=1000

device=mps

seed=42

# Train

```bash
/opt/miniconda3/envs/pytorch/bin/python train.py
```

# Results

num_params=160938

accuracy=0.9935



num_params=86826

accuracy=0.9895

# Conclusion

参数量缩小一半，并没有显著降低acc，但是目前参数量仍大于CNN。

# Q&A

## Question

1. 几个 stage?每个 stage 堆几个 Residual? 

2. stem 怎么设计? 

3. 通道方案用 32→64→… 还是 16→32→…?

4. 分类头为什么用 GlobalAvgPool 而不是 Flatten + Linear(784→10)?


## Answer

1. 因为MNIST数据集没有很大，且分辨率是28x28，因此选择2个stage，一个堆一个Res块就行了
2. stem我打算就设计一个简单的projection，直接变成32通道就行，不降低分辨率
3. 32-64
4. GlobalAvgPool是对每个通道的整张特征图取一个平均值。[B, C, H, W] → [B, C, 1, 1],再 Flatten 成 [B, C]。这样既减少了参数量，也抹平了位置差异