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

| 模型                  | 参数量 | 最终acc |
| --------------------- | ------ | ------- |
| 03_mnist_cnn/baseline | 34,826 | 0.9899  |
| 05_resnet             | 86,826 | 0.9895  |

1. 参数量:ResNet 比 CNN 大 2.5 倍，ResNet 每一层都带 2 个 BN,且 3×3 卷积通道更大；CNN的通道方案更小。
2. acc 几乎持平(98.99% vs 98.95%)，因为这个MNIST任务太简单,深度优势体现不出来。
3. 收敛速度：ResNet的收敛速度更快。
4. 不同 seed 波动就有 ±0.2%,所以这点差距在噪声内,不能断言"ResNet 更差"。

# Q&A

## Question

1. 几个 stage?每个 stage 堆几个 Residual? 
2. stem 怎么设计? 
3. 分类头为什么用 GlobalAvgPool 而不是 Flatten + Linear(784→10)?

## Answer

1. 因为MNIST数据集没有很大，且分辨率是28x28，因此选择2个stage，一个堆一个Res块就行了。
2. stem我打算就设计一个简单的projection，直接变成32通道就行，不降低分辨率。
3. GlobalAvgPool是对每个通道的整张特征图取一个平均值。[B, C, H, W] → [B, C, 1, 1],再 Flatten 成 [B, C]。这样既减少了参数量，也抹平了位置差异。使用AdaptiveAvgPool2d((1,1))实现。

# 批量实验重构

为了让"一次跑 3 个 seed × 3 个 lr"不再手动改配置，做了如下重构：

## 改动

- `models/model.py`：新增 `build_net(device)`，把搭网络的代码从 train/test 中抽出，统一复用。
- `utils/train.py`：拆成两层
  - `set_seed(seed, device)`：按设备设置随机种子（CPU / MPS / CUDA）。
  - `train_loop(...)`：纯训练循环，返回每个 epoch 的 loss/acc。
  - `run_experiment(seed, lr, cfg)`：编排一次完整实验（设种子→建模型→建数据→训练→保存→画图→写日志）。
- `train.py`：瘦身为纯编排层，双层循环遍历 `cfg.seeds × cfg.lrs`，跑完打印汇总表。
- `utils/checkpoint.py`：`save_model`/`load_model` 增加 `seed`、`lr` 参数，checkpoint 按 `mnist_resnet_seed{seed}_lr{lr}.pth` 命名，避免互相覆盖。
- `configs/default.py`：`seed`/`lr` 改为列表 `seeds`/`lrs`；`device` 可在 `"mps"`/`"cuda"`/`"cpu"` 间切换。

## 关键顺序

每次实验必须按 `set_seed → build_net → get_loaders` 的顺序执行，否则随机初始化与数据 shuffle 无法按 seed 复现。

## device 切换

`configs/default.py` 里的 `device = pick_device()` 会自动检测设备（cuda > mps > cpu），换机器无需改配置。想强制指定可用 `pick_device("cuda")` / `pick_device("mps")` / `pick_device("cpu")`，设备不可用时会明确报错。

- `"mps"`：Apple Silicon GPU
- `"cuda"`：NVIDIA GPU
- `"cpu"`：纯 CPU

所有 `.to(device)` 已贯穿模型、数据与评估。
