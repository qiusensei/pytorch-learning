# 基础验证实验

| 实验 | 样本 A | 样本 B | 预期 MMD |
|---|---|---|---|
| 同分布 | $\mathcal{N}(0, 1)$ | $\mathcal{N}(0, 1)$ | ≈ 0（小） |
| 均值不同 | $\mathcal{N}(0, 1)$ | $\mathcal{N}(3, 1)$ | 大 |
| 方差不同 | $\mathcal{N}(0, 1)$ | $\mathcal{N}(0, 4)$ | 中等 |
| 均值逐渐远离 | $\mathcal{N}(0, 1)$ | $\mathcal{N}(\mu, 1)$, $\mu=0,1,2,...,5$ | 递增 |

以下实验都在seed=42, sigma=1.0下面进行。

## 同分布

n_samples=100 MMD²的期望值: 0.000458

## 均值不同

n_samples=100 MMD²的期望值: 0.007980

## 方差不同

n_samples=100 MMD²的期望值: 0.004220

## 均值逐渐远离

$\mu=0$：n_samples=100 MMD²的期望值: 0.000458

$\mu=1$：n_samples=100 MMD²的期望值: 0.006260

$\mu=2$：n_samples=100 MMD²的期望值: 0.007970

$\mu=5$：n_samples=100 MMD²的期望值: 0.007980

# 进阶观察

## **sample size 的影响**

固定分布，改变 $n$（100, 200, 500, 1000），观察 MMD 估计值的稳定性

以下都是同分布的计算结果

n_samples=100 MMD²的期望值: 0.000458

n_samples=200 MMD²的期望值: 0.000342

n_samples=500 MMD²的期望值: 0.000166

n_samples=1000 MMD²的期望值: 0.000050

可以看出，随着采样数数量的增加，同分布的MMD²结果一直在下降，说明更加确定它们是同分布。

## **sigma 的影响**：

固定分布和样本量，改变 kernel bandwidth $\sigma$（0.1, 1.0, 10.0），观察 MMD 的灵敏度

下面是同分布，固定样本量为500

sigma=0.1：n_samples=500 MMD²的期望值: -0.000000000000000000000000000006552

sigma=1.0：n_samples=500 MMD²的期望值: 0.000166

sigma=10.0：n_samples=500 MMD²的期望值: 0.000293

可以看出，sigma是在放大MMD²的计算结果，sigma大的时候计算结果更明显。

## **多维数据**

从 1D 扩展到 2D/10D，观察 MMD 是否仍然能区分不同分布

下面都是sigma=1.0，分布分别为$\mathcal{N}(0, 1)$和$\mathcal{N}(0, 4)$的情况

1D：n_samples=500 MMD²的期望值: 0.079106

2D：n_samples=500 MMD²的期望值: 0.118385

5D：n_samples=500 MMD²的期望值: 0.048195

10D：n_samples=500 MMD²的期望值: 0.003869