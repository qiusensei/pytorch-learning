# 模型变化

在baseline的基础上，加入了dropout，模型有下面的不同：

```python
self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(p=0.5),				#添加了一个nn.Dropout
            nn.Linear(1600, 10)
        )
```

也就是每次以0.5的概率在学习的时候随机删除神经元，用来降低模型的过拟合程度。

# 训练时和测试时行为不同?

> train.py 里 net.train() / net.eval() 在切换什么?训练时按 p=0.5 随机丢弃,测试时全部保留——为什么测试不能随机丢?

net.eval()会把模型调整为测试模式，这样nn.Drouout就不会生效。

训练的时候是要使用Dropout，相当于训练了很多个子网络，来抗衡过拟合，最后测试的时候则不用Dropout是为了集成这些子网络的平均性能。

# Dropout可能降低train accuracy

因为Dropout删除神经元的时候，会让模型在学习的过程中某些特征没有被学习到，因此学习速度上比较慢，那么一开始的acc也就比较低了。当然这里的acc低是为了让整体模型能有更强的泛化能力。