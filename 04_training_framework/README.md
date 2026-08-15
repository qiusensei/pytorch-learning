# Goal

将 MNIST CNN 训练拆分为配置、数据、模型、工具等多个模块。

# Construction

```
04_training_framework
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

seed=42

accuracy=0.9887



seed=1000

accuracy=0.9907

# Conclusion

相同seed一致，不同seed会有波动。

# Output

- `results/train_log.csv`
- `plots/loss_curve_cnn.png`
- `plots/accuracy_curve_cnn.png`
- `checkpoints/mnist_cnn.pth`

# Q&A

## Question

1. 为什么 `mnist.py` 不直接导入 `default.py`？

2. 为什么训练日志比只记录最终 accuracy 更有价值？

## Answer

1. 因为要把各个py文件里面写成单独的函数，负责接收参数传出参数，不需要它自己找参数，这样每个py模块就是独立的。
2. 最终accuracy只能反映最后一个时刻的结果。而训练日志更加全面，包含训练细节，它能够保存每个epoch的变化:
   - loss 是否持续下降；
   - accuracy 是否收敛；
   - 是否出现过拟合；
   - 不同模型或不同 seed 的收敛速度差异。