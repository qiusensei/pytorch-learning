# 我在这里面学习了基本的pytorch操作
## tensor_demo
在这里面我学习了与之前学习的numpy差不多的操作，创建随机数组、全1数组、全0数组。

这里创建的应该是tensor类型的数组，虽然看上去与numpy差不多，但是应该会更便于后面的pytorch计算

## autograd_demo
这里学习了如何使用pytorch进行快速的反向传播计算
x初始化成一个tensor变量之后，再令y是x的函数

### y.backward()
使用y.backward()就能从y开始进行反向传播的计算
梯度值会自动储存到x.grad里，使用pytorch非常方便

### y.grad_fn
.grad_fn是用来记录"这个张量是通过什么运算算出来的"。

比如：
b = a+1
c = a*b
d = c**2

那么：
b.grad_fn = <AddBackward0>（由加法算出来的）
c.grad_fn = <MulBackward0>（由乘法算出来的）
d.grad_fn = <PowBackward0>（由平方算出来的）
y.grad_fn在我的例子里打印出来的是<SumBackward0>说明y是由sum()算出来的

## my_net
在这里我学习了如何使用nn.Module作为父类创建了一个简单网络MyNet
### super.__init__()
MyNet继承的是nn.Module，nn.Module自己有一个__init__过程，用于维护参数注册表、层的记录等
这里使用这个super是为了让nn.Module自己初始化一边，让后面自己的模型能够获得这些参数表与层。
### nn.Linear() and nn.ReLU()
这个可以方便快捷的创建Affine层。
比如我想创建一个权重W（形状 (8，4)）和 偏置b（形状 (4,)）的Affine层

nn.Linear()规则如下：
nn.Linear(**输入维度**, 输出维度)
  → weight 形状 = (输出维度, **输入维度**)
  → bias   形状 = (输出维度,)

所以只需要让self.layer1 = nn.Linear(4,8)即可，自动创建，非常方便
同时可以使用nn.ReLU()来快捷的创建激活函数。
### 调用
在主程序里创建model = MyNet()之后，我们需要初始化一个能够放入网络的张量x
因为nn.Linear(4, 8)的**输入维度**是4，所以x的最后一维必须是4，这里创建的是(5,4)
直接使用model(x)就可以把x放到网络里进行计算
### model.named_parameters()
model.parameters()会返回model里每一个参数的列表的生成器（generator），但是用它打印出来的就是一串数字。
如果想更加清晰的知道每个打印出来的列表是什么，就要用到model.named_parameters()，他会返回名称与参数
用name，param接收返回的名字与参数后，用for循环打印，结果如下：
layer1.weight: shape=torch.Size([8, 4])
layer1.bias: shape=torch.Size([8])
layer2.weight: shape=torch.Size([2, 8])
layer2.bias: shape=torch.Size([2])

# 总结
至此，我在tensor_demo里学习类似numpy的torch操作
然后用autograd_demo学习了如何使用torch进行快速的反向传播计算
最后在my_net里学习使用了nn.Module里创建了自己第一个网络模型model并用它打印输出了网络中的权重与偏置