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
print("\n", len(x))
# 当用张量表示一个向量时，张量只有一个轴，使用shape属性来获取张量形状
# 为一个元素组
print("\n", x.shape)

# 矩阵
# 向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶
# 当矩阵具有相同数量的行和列时，其形状将变为正方形；
# 因此，它被称为方阵（square matrix）
A = torch.arange(20).reshape(5,4)
print("\n", A)
# 访问矩阵的转置
print("\n", A.T)
# 对称矩阵转置与原始矩阵相同
B = torch.tensor([[1,2,3], [2,0,4], [3,4,5]])
print("\n", B)
if torch.all(B == B.T):
    print("\n", B.T)
print("\n", B == B.T)

# 张量
# 张量是描述具有任意数量轴的n维数组的通用方法
X = torch.arange(24).reshape(2,3,4)
print("\n", X)
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A.clone() # 通过分配新内存，将A的一个副本分配到B
print("\n", A, "\n", A + B)
print("\n", A * B)
# 将张量乘以或加上一个标量不会改变张量的形状，
# 其中张量的每个元素都将与标量相加或相乘
a = 2
X = torch.arange(24).reshape(2,3,4)
print("\n", a+X, "\n", (a * X).shape)
# 降维
# 我们可以对任意张量进行的一个有用的操作是计算其
# 所有元素的总和
