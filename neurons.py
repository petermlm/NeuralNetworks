"""
This was just an implementation of perceptrons and sigmoids. They are not used
in the NN implementation.
"""


import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed(self, input_vals):
        val = np.dot(self.weights, input_vals) + self.bias
        return 1 if val > 0 else 0


class Perceptron(Neuron):
    pass


class Sigmoid(Neuron):
    def feed(self, input_vals):
        val = super().feed(input_vals)
        return 1 / (1 + np.e ** -val)


if __name__ == "__main__":
    # Make a perception and a sigmoid
    # The weights are -2 for both inputs, the bias is 3
    per = Perceptron(np.array([-2, -2]), 3)
    sig = Sigmoid(np.array([-2, -2]), 3)

    # Output of neurons with given input
    print(0, 0, per.feed(np.array([0, 0])))
    print(0, 1, per.feed(np.array([0, 1])))
    print(1, 0, per.feed(np.array([1, 0])))
    print(1, 1, per.feed(np.array([1, 1])))

    print(0, 0, sig.feed(np.array([0, 0])))
    print(0, 1, sig.feed(np.array([0, 1])))
    print(1, 0, sig.feed(np.array([1, 0])))
    print(1, 1, sig.feed(np.array([1, 1])))
