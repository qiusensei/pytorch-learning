import torch

#下面初步学习了如何使用pytorch进行快速的反向传播计算并提取梯度

# x = 2.0

x = torch.tensor(2.0,requires_grad=True)
y = x**2 + 2*x +1
y.backward()
print("x=2时y的梯度",x.grad)

# x = [1.0,2.0,3.0]

x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
y = (x**2).sum()
y.backward()
print("x=[1.0,2.0,3.0]时y在各分量上的梯度",x.grad)

print("y.grad_fn",y.grad_fn)


