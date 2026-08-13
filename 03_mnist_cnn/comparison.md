# Experiment Setup

数据集：MNIST

优化器：Aadm

lr = 1e-3

epochs = 10

batch_size = 64

单次运行

# Results

|   Model    | E1 loss | Acc_max(epoch) | Final acc |
| :--------: | :-----: | :------------: | :-------: |
|  baseline  | 0.1900  |  0.9909(E10)   |  0.9909   |
| batchnorm  | 0.1215  |   0.9907(E8)   |  0.9899   |
|  dropout   | 0.2451  |  0.9910(E10)   |  0.9910   |
| bn_dropout | 0.1699  |  0.9929(E10)   |  0.9929   |

# Question

1. BN是否加快收敛？
2. Dropout是否降低过拟合
3. BN+Dropout是否一定更好

# Answer

1. 很明显BN的E1 loss比baseline要低，收敛速度比baseline要快。
2. Dropout的提升近乎看不见。
3. BN+Dropout也并没有看到更好的效果。