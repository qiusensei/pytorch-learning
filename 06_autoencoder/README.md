# Goal

依照 `04_training_framework` 的模块化结构，从零实现一个基于 MLP 的最简 AutoEncoder（自编码器），完成从${28\times28}$图像到 $64$ 维潜在空间（Latent Space）的压缩与重建，探索无监督表征学习（Representation Learning）的本质。

# Construction

```txt
06_autoencoder
├── checkpoints/
├── configs/
├── datasets/
├── models/
│   └── autoencoder.py
├── plots/
├── results/
├── utils/
│   ├── checkpoint.py
│   ├── evaluate.py
│   ├── plot.py
│   ├── print.py
│   └── train.py
├── test.py
├── train.py
└── README.md
```

# Model Architecture

* **Input**: `(B, 1, 28, 28)` 归一化到 $[0.0, 1.0]$
* **Encoder**:
  * `nn.Flatten()` $\to$ `(B, 784)`
  * `nn.Linear(784, 64)` + `nn.ReLU()` $\to$ `(B, 64)` （**Bottleneck 潜在向量 $z$**）
* **Decoder**:
  * `nn.Linear(64, 784)` + `nn.Sigmoid()` $\to$ `(B, 784)`
  * `nn.Unflatten(1, (1, 28, 28))` $\to$ `(B, 1, 28, 28)` 重建图 $\hat{x}$
* **Loss Function**: `nn.MSELoss()` （度量重建图像 $\hat{x}$ 与原图 $x$ 的像素级均方误差）

# Config

* `num_epochs`: 10
* `lrs`: `[5e-4]`
* `seeds`: `[42, 919, 1000]`
* `train_batch_size`: 256
* `test_batch_size`: 1000
* `device`: 自动检测（cuda > mps > cpu）

# How to Run

```bash
# 训练与批量实验 (多 seed sweep + 自动画图 + CSV 记录)
python train.py

# 评估与加载 checkpoint 测试
python test.py
```

# Results

* **网络总参数量**: 101,200 （$(784\times 64 + 64) + (64\times 784 + 784)$）
* **实验指标记录**（请填入终端最终输出的各 seed 表现）：

| Seed | Learning Rate | Last Train Loss | Last Test Loss |
| :--: | :-----------: | :-------------: | :------------: |
|  42  |     5e-4      |      填写       |      填写      |
| 919  |     5e-4      |      填写       |      填写      |
| 1000 |     5e-4      |      填写       |      填写      |

# Key Takeaways & Reflections

### 1. 为什么重构任务不能用 Accuracy，必须用 MSE Loss？
* 分类任务目标是 $X \to Y$（预测离散类别标签），评估可以用分类准确率；
* AutoEncoder 是无监督/自监督重建任务 $X \to Z \to \hat{X}$（目标是 $\hat{x} \approx x$），输出是连续像素矩阵，必须用度量连续距离的 `MSELoss`。

### 2. 为什么 Decoder 输出层必须配合 `nn.Sigmoid()`？
* MNIST 原图经过 `ToTensor()` 归一化后像素严格在 $[0, 1]$ 区间；
* `nn.Linear` 输出的是无界实数 $(-\infty, +\infty)$，加上 `Sigmoid()` 强制将重建像素压缩至 $(0, 1)$，使输出分布与输入分布严格一致，保证训练稳定。

### 3. 信息瓶颈（Bottleneck）在表征学习中扮演什么角色？
* 如果没有狭窄的 Bottleneck（例如隐层也是 784 维），网络会偷懒学成恒等映射（直接抄像素，不学任何特征）；
* 正是因为只有 64 维，网络**被迫**抛弃冗余背景和噪声，在低维流形上自动提炼数字的粗细、倾斜、结构等关键语义特征。

### 4. 标准 AutoEncoder 与 DAE（去噪自编码器）的区别？
* 标准 AE 输入是干净图像 $x$，目标是重建 $x$，只做忠实压缩与还原；
* DAE 输入故意添加噪声 $\tilde{x}$，但训练目标依然是还原干净图像 $x$。只有在目标不一致时，网络才会主动学习流形投影以消除噪声。

### 5. 与MMD-AAE
* 训练完成后，Encoder 提取出的 $64$ 维向量 $z$ 就是高质量的无监督图像特征表示；
* 后续论文利用此特征，通过 **MMD（最大均值差异）** 强迫不同 Domain（领域）提取出的 $z$ 分布对齐，从而实现跨域泛化。