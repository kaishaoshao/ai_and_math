## 线性代数
# 标量 
# 标量由一个元素的张量表示
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y, "\t", x * y, "\t", x / y,"\t",x ** y)

# 向量
# 向量可以被视为标量值组成的列表
x = torch.arange(4)
print("\n", x)
# 我们可以使用下标来引用向量的任一个元素
print("\n", x[3])

