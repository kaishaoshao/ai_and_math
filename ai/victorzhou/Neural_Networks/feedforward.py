import numpy as np
from neurons import sigmoid, Neuron

'''
A neural network with:
    - 2 inputs
   	- a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
'''
class OurNeuralNetwork:
    
    def __init__(self):
        weights = np.array([0, 1]) 
        bias = 0
        
        # The Neuron class here is from the previous section  
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1
   
network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.72106