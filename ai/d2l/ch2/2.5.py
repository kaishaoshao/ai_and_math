# 自动微分
# 例子
import torch
x = torch.arange(4.0)
print(x)
# 一个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。
x.requires_grad_(True) # 等价于 x = torch.arange(4.0, requires_grad=True)
print(x.grad) # 默认值是None
# 计算y
y  = 2 * torch.dot(x, x)
print(y)
# 通过调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
x.grad
print(x.grad)
print(x.grad == 4 * x)
# 在默认情况下, PyTorch会累积梯度
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)
# 非标量变量的反向传播
x.grad.zero_()
y = x * x
# 等价于 y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

# python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)