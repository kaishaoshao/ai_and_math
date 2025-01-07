# Coding a Neural
import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias

    # 前向传播    
    def feedforward(self,inputs):
        # 
        total = np.dot(self.weights,inputs) + self.bias
        return sigmoid(total)
    
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n =  Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3 
print(n.feedforward(x))


