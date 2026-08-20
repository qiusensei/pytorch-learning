# Goal

实现高斯径向基核（RBF Kernel）与无偏 MMD² 计算函数，通过对比实验验证 MMD 作为非参数化分布距离度量在均值漂移、方差差异、采样规模、核带宽σ及高维特征下的数值行为，为 Domain Generalization 及 MMD-AAE 论文复现建立数学与代码直觉。

# Construction

```txt
07_mmd
├── README.md
├── __pycache__
├── experiments.py
├── mmd.py
└── result.md
```

# MMD Architecture

* gaussian_kernel(x, y, sigma)
* mmd_square(x, y, sigma)

前者是高斯核计算，后者是mmd函数计算

# Config

* n_samples=500
* sigma=1.0
* dim=10
* seed=42

后续有更改n_samples, sigma, dim等进行运行。

# How to Run

```bash
python experiments.py
```

# Results

见同目录下面的result.md。