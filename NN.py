import random


import numpy as np
import numpy.random


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


"""
Neural Network
"""

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

        self.weights = [[]]
        for i, iv in enumerate(self.layers[1:]):
            matrix = np.random.random((iv, self.layers[i]))
            self.weights.append(matrix)

        self.bias = []
        for i, iv in enumerate(self.layers):
            matrix = np.random.random((iv, 1))
            self.bias.append(matrix)

        self.activ = []
        for i, iv in enumerate(self.layers):
            matrix = np.array([0.0 for i in range(iv)])
            self.activ.append(matrix)

        self.errors = [[]] * len(self.layers)

    def activate(self, z_val):
        return 1 / (1 + np.e ** -z_val)

    def feedForward(self, input_vals):
        # Activation for the first layer is just the values of input
        self.activ[0] = input_vals[:]

        # For the following layers, use the normal feed function
        for k, layer in enumerate(self.layers[1:]):
            l = k+1

            for j in range(layer):
                w_vec = np.array(self.weights[l][j])
                x_vec = np.array(self.activ[l-1])

                z_val = np.dot(w_vec, x_vec) + self.bias[l][j]
                sig_val = self.activate(z_val)
                self.activ[l][j] = sig_val

    def costDeriv(self, intended):
        self.step1(intended)
        self.step2()
        self.step3()
        self.step4()

    def step1(self, intended):
        L = len(self.layers) - 1

        # Calculate derivatives
        deriv_vec = self.activ[L] - intended

        # Calculate sigma vector
        sig_vec = []
        for j in range(self.layers[L]):
            z = np.dot(self.weights[L][j], self.activ[L-1])
            z += self.bias[L][j]

            sig_vec.append(self.activate(z))

        self.errors[L] = sig_vec * deriv_vec

    def step2(self):
        L = len(self.layers) - 1

        for l in range(L-1, 0, -1):
            # Bring error back
            error_back = self.weights[l+1].T * self.errors[l+1]

            # Calculate sigma vector
            sig_vec = []
            for j in range(self.layers[l]):
                z = np.dot(self.weights[l][j], self.activ[l-1])
                z += self.bias[l][j]

                sig_vec.append(self.activate(z))

            self.errors[l] = error_back * sig_vec

    def step3(self):
        self.cost_b = self.errors

    def step4(self):
        self.cost_w = [[]]

        for l in range(1, len(self.layers)):
            rows = self.layers[l]
            cols = self.layers[l-1]
            self.cost_w.append(np.zeros((rows, cols)))

            for j in range(rows):
                for k in range(cols):
                    self.cost_w[l][j][k] = self.activ[l-1][k] * self.errors[l][j]
