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