import torch

# 1.创建 4 种不同的 Tensor（tensor、randn、zeros、ones），每个都用 print() 打印出来

x_tensor = torch.tensor([[1,2,3],[4,5,6]])

x_randn = torch.randn(2,3)      #What the difference of randn and rand?

x_zeros = torch.zeros(2,3)

x_ones = torch.ones(2,3)
print(x_tensor,x_randn,x_zeros,x_ones,sep="\n")

# 2.创建一个形状为 (2, 3, 4) 的随机 Tensor，然后对它依次做以下操作：
# 用 .shape 打印原始形状•用 .reshape() 变成 (6, 4)，打印形状
# 用 .view() 变成 (6, 4)，打印形状•用 .permute(2, 0, 1) 交换维度
# 打印形状•用 .transpose(0, 2) 交换两维，打印形状

x_q2 = torch.randn(2,3,4)
print(x_q2.shape)

# 3.创建两个矩阵：m1 形状 (3, 4)，m2 形状 (4, 5)•用 torch.matmul() 做乘法，打印结果的形状•用 @ 运算符做乘法，打印结果的形状
#
# 4.在文件末尾加一行注释： # ✅ Tensor 基础操作完成

