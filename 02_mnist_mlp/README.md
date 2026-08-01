# MNIST
在学习这部分之前我就一直很好奇MNIST是什么，查询之后知道它是一个手写数字的数据集。

专业的解释是下面这样的：
> MNIST数据库（Modified National Institute of Standards and Technology database）是一个大型数据库的手写数字是通常用于训练各种图像处理系统。

MNIST有6万张训练图+1万张测试图，每张是28×28灰度图，标签是0-9(代表数字0-9)。
## 数据类型
MNIST里面存放的图片像素可以认为是用28x28的矩阵储存的，而矩阵中的每个数字都在0-255之间，代表了这个像素的灰度值。
# MLP
MLP的全称则是Multi-Layer Perceptron，也就是多层感知机的意思。  
> 多层感知机 = 全连接神经网络。
> 
MLP只有全连接层（Linear）的神经网络，没有卷积。  


# iter() and next()
iter是把loader打包成一个迭代器。  
> 准确的说。iter()是拿到了一个指向loader的“指针”。
> 
next就是返回值并将这个指针往后面移动一格。  

**StopIteration**就是代表迭代器到头了取不出来了。
>for循环就是靠着捕捉StopIteration来自动结束的
> 
