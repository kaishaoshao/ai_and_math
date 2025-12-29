import math
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def f(x):
    return 3*x**2 - 4*x + 5

f(3.0)
print(f(3.0))

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)

h = 0.0001
x = -3.0
f(x+h)
print((f(x+h) - f(x)) / h)

a = 2.0
b = -3.0
c = 10.0
d = a * b + c
d1 = a * b + c
a += h 
d2 = a * b + c
print('d', d)
print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1) / h) 


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # 梯度
        self._prev = set(_children)
        self._op = _op
        self.label = label
  
    def __repr__(self):
        return f"Value(data={self.data}))"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = Value(4.0, label='c')
print(a)
print(a + b)
print(a * b)

# 在 Python 中，几乎所有的运算符都对应一个以双下划线开头和结尾的方法。当你写下 a + b 时，Python 解释器实际上在后台执行了如下逻辑：
# 检查左操作数：检查变量 a 是否有一个名为 __add__ 的方法。
# 调用方法：如果 a 有这个方法，Python 就会调用 a.__add__(b)。
# 传入参数：此时，a 作为 self 传入，b 作为 other 传入。
# 底层等价关系： a + b 等价于 a.__add__(b) a * b 等价于 a.__mul__(b)

# python 独有的鸭子类型
# 统一接口：不论是数字相加、字符串拼接，还是列表合并，在底层都是调用各自类的 __add__。
# 自定义行为：你可以定义任何类的加法逻辑。例如在你的代码中，你定义了 Value 对象的加法是将其内部的 data 属性相加，并返回一个新的 Val

d = a * b + c
print(d._prev)
print(d._op)

e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
print(d)

f
