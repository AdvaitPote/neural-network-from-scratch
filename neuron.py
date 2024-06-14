import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron():
    weights = np.array([])
    bias = 0

    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)