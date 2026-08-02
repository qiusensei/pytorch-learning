# MNIST
在学习这部分之前我就一直很好奇MNIST是什么，查询之后知道它是一个手写数字的数据集。

专业的解释是下面这样的：
> MNIST数据库（Modified National Institute of Standards and Technology database）是一个大型数据库的手写数字是通常用于训练各种图像处理系统。

MNIST有6万张训练图+1万张测试图，每张是28×28灰度图，标签是0-9(代表数字0-9)。
## MNIST的数据类型
MNIST里面存放的图片像素可以认为是用28x28的矩阵储存的，而矩阵中的每个数字都在0-255之间，代表了这个像素的灰度值。
# MLP
MLP的全称则是Multi-Layer Perceptron，也就是多层感知机的意思。  
> 多层感知机 = 全连接神经网络。
> 
MLP只有全连接层（Linear）的神经网络，没有卷积。
# iter() and next()
iter是把loader打包成一个迭代器。  
> 准确的说。iter()是拿到了一个指向loader的“指针”。

next就是返回值并将这个指针往后面移动一格。  

**StopIteration**就是代表迭代器到头了取不出来了。
>for循环就是靠着捕捉StopIteration来自动结束的

# nn.Sequential()
和它的名字一样，顺序，也就是把括号里的参数按照顺序进行正向计算。它放在model.py里面。  
原来的初始化神经网络的时候，需要一层一层的写self.layer与self.relu，在forward的时候又要写一次，很复杂。
```python
self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            ...
        )
```
使用nn.Sequential()去定义神经网络里的隐藏层与激活层，在forward的时候只要写一个return self.net(x)就可以了。  
## nn.Flatten()
这是一个自动展开的函数，比如一张图像(B, 1, 28, 28)，在使用nn.Flatten()之后会自动展开成(B, 784)。  
只需要把nn.Flatten()放在nn.Sequential()的第一层，net(images)的时候放入的图像就会自动拉直。
# 训练过程
## 超参数设置
在main函数之外设置训练循环次数epoch与学习率lr。
## 模型初始化损失函数
进入主函数之后，调用model里面的类创建自己的神经网络net，并使用nn.CrossEntropyLoss()作为损失函数。
```python
net = model.MyMLP()
criterion = nn.CrossEntropyLoss()
```
criterion用来保存着损失函数，后面会调用它来计算误差值。
## 优化器设置
```python
optimizer = optim.Adam(net.parameters(), lr=lr)
```
调用optim里的Adam，这是一个动态调整参数的优化器，以后应该还会学习到别的。  
Adam 给每个参数动态调整学习率：
- 梯度大的参数 → 自动缩小更新量（防震荡）
- 梯度小的参数 → 自动放大更新量（防龟速）
## 训练循环
net.train()将网络设置为训练模式。  
> 为什么要有这两个？因为 Dropout、BatchNorm 这类层在训练和评估时的行为不同。MLP 目前没影响，但以后 CNN/ResNet 忘了切换，评估结果会直接崩。  

用for循环进入训练循环，利用交叉熵与反向传播更新网络里面的权重与偏置。
```python
for epoch in range(num_epochs):   # ← 外层:整个训练集过一遍 = 1 个 epoch
    running_loss = 0.0            # ← 每轮归零!
    for images, labels in dataset.train_loader:
        outputs = net(images)               # 1. 前向
        loss = criterion(outputs, labels)   # 2. 算损失
        optimizer.zero_grad()               # 3. 清梯度,防累加
        loss.backward()                     # 4. 算梯度
        optimizer.step()                    # 5. 用梯度更新参数
        running_loss += loss.item()         # 6. 把一个epoch里面的误差全部在一块
```
也就是最基本的一个结构，一次次算损失然后反向传播算梯度然后更新参数。
### loss.item()
把带计算图的张量变成普通Python浮点数，这样方便了running_loss的累加求和。
# 评估过程
评估之前，用net.eval()调成评估模式，然后使用torch.no_grad()关掉梯度计算，这样可以节省内存。
```python
outputs = net(images)                      # 用之前训练好的net对测试集进行预测看看对不对
preds = outputs.argmax(dim=1)              # 沿类别维取最大下标 → 预测类别
total_correct += (preds == labels).sum().item()
accuracy = total_correct / total_samples
```
# 模型保存
```python
torch.save(net.state_dict(), "checkpoints/mnist_mlp.pth")
```
最后使用torch.save方法保存模型，保存的格式是state_dict() = {参数名: 张量} 字典，是一种标准格式，日后如果再使用可以直接加载。  
state_dict():它返回一个 OrderedDict,键是参数名(如fc1.weight、fc1.bias),值是张量。简单的说它只保存权重,不保存模型结构。
