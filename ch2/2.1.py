import torch

# 入门

# 张量 不表示一个由数值组成的数组，数组可能有多个维度
# 一个轴的张量为向量 两个轴为矩阵 两个以上轴的没有特殊名称
x = torch.arange(12)
print(x)

# 通过shape属性来访问张量的形状
shape = x.shape
print(shape)

# 如果只想知道张量中元素的总数，将及即形状的所有元素乘积，可以检查大小
size = x.numel()
print(size)

# 如果想改变一个张量的形状而不改变元素数量和元素之值，可以使用reshape
X = x.reshape(3,4)
print(X)
# 我们不需要通过手动指定每个维度的形状，仅仅需要制定高度或者宽度其中之一
# 将然后另外的变量设置为-1
X2 = x.reshape(3,-1)
print(X2)

# 初始化全0,1的张量
zeros = torch.zeros((2,3,4))
print("\n",zeros)
ones = torch.ones((2,3,4))
print("\n",ones)

# 随机初始化参数，从均值为0,标准差为1的高斯分布中随机采样(正态分布)
randn = torch.randn(3,4)
print("\n",randn)

# 还可以提供包含数值的python列表，来为所需的张量中的每个元素赋予
# 确定值，最外层的列表对应于轴0,内层的列表对应于轴1
y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print("\n",y)

# 运算符
# 在数学表示法中，符号d |:R->R 来表示一元标量运算符，这意味着该函数从任何
# 实数（R）映射到另外一个实数，同样，通过符号f:R,R->R表示二元标量运算符
# 这表示该函数接收两个输入，并产生一个输出，给定同一形状的任意两个向量u和v
# 和二元运算符,我们可以得到向量c = F(u,v)。  具体计算方法是Ci<-f(Ui,Vi)
# 其中Ci、Ui和Vi分别是向量c、u、v中的元素，我们通过将标量函数升级为按元素向量
# 运算来生成向量值F：Rx,Rx->Rx
# 对于任意具有相同形状的张量，常见的标准算术运算符(+、-、*、/和**)都可以被升级
# 为按元素运算，我们可以在同一形状的任意两个张量上调用按元素操作
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("\n", x + y, "\n", x - y, "\n", x * y, "\n", x / y, "\n", x ** y)

# 还可以使用现成的函数
exp = torch.exp(x)
print("\n", exp)

# 我们也可以把多个张量连接在一起，把它们端到端地堆叠起来形成一个更大的张量
# 只需要提供张量列表，并给出哪个轴连结
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3] ,[1, 2, 3, 4], [4, 3, 2, 1]])
XY0 = torch.cat((X, Y), dim=0)
XY1 = torch.cat((X, Y), dim=1)
print("\n", XY0, "\n",XY1)

# 通过逻辑运算符构建二元张量
print(X == Y)

# 对所有元素进行求和
Xsum = X.sum()
print("\n", Xsum)


# 广播机制
# 我们一般是将形装相同的两个张量执行元素操作，某些情况，即使形状不同，我们
# 仍然可以通过调用广播机制来执行元素操作
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("\n", a, "\n", b, "\n", a+b)

# 索引和切片
# 第一个元素索引0，最后一个是-1
print("\n", X[-1], "\n", X[1:3])
# 如果我们想为多个元素赋值相同的值，我们只需要索引所有元素，然后为他们赋值
# 例如：[0:2, :]访问第一行和第二行，其中":"代表沿轴1（列）的所有元素
X[0:2, :] = 12
print("\n", X)

# 节省内存
# 运行一些操作可能会导致为新结果分配内存。 例如，如果我们用Y = X + Y，
# 我们将取消引用Y指向的张量，而是指向新分配的内存处的张量。
before = id(Y)
Y = Y + X
print("\n", id(Y) == before)
# 可以使用切片的方法将操作的结果分配给先前的数组，例如Y[:] = <experssion>
# 新建矩阵Z，其形状与Y相同，使用zeros_like来分配一个全0的快
Z = torch.zeros_like(Y)
print("\nid(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))
# 如果后续的计算没有使用X，我们也可以使用X[:] = X + Y 或 X += Y来减少操作的内存开销
before = id(X)
X += Y
print("\n", id(X) == before)

# 转换为其他Python对象
# 将深度学习框架定义的张量转换为NumPy张量（ndarray）很容易，反之也同样容易。
# torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改
# 另一个张量。
A = X.numpy()
B = torch.tensor(A)
print("\n", type(A),type(B))
# 将大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数
a = torch.tensor([3.5])
print("\n", a, a.item(), float(a), int(a))

# 练习
# X > Y | X < Y
X_new = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y_new = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("\n", X > Y, "\n", X < Y)

# 三维张量 && 广播机制
a1 = torch.arange(27).reshape((3,3,3))
b1 = torch.arange(4).reshape(2,2)
print("\n", a1, "\n", b, "\n", a+b)
