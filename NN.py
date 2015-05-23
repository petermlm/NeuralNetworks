import random


import numpy as np
import numpy.random


"""
Neural Network
"""

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

        self.weights = [np.random.random((iv, self.layers[i-1])) for i, iv in enumerate(self.layers)]
        self.bias = [np.random.random((iv, 1)) for i, iv in enumerate(self.layers)]
        self.activ = [np.zeros(iv) for i, iv in enumerate(self.layers)]
        self.errors = [[]] * len(self.layers)

    def activate(self, z_val):
        """
        The activation function
        """

        return 1 / (1 + np.e ** -z_val)

    def feedForward(self, input_vals):
        """
        Calculates the activation values for a given input in this network
        """

        # Activation for the first layer is just the values of input
        self.activ[0] = input_vals[:]

        # For the following layers, use the normal feed function
        for k, layer in enumerate(self.layers[1:]):
            l = k+1

            for j in range(layer):
                w_vec = np.array(self.weights[l][j])
                x_vec = np.array(self.activ[l-1])

                # Dot product between the values of the previous layer with the
                # weights, plus the bias
                z_val = np.dot(w_vec, x_vec) + self.bias[l][j]
                sig_val = self.activate(z_val)
                self.activ[l][j] = sig_val

    def backPropagation(self, input_vals, output_vals):
        """
        Does the Back Propagation algorithm for one sample
        """

        # Feed this input forward
        self.feedForward(input_vals)

        # Calculate the derivatives using the four equations
        self.step1(output_vals)
        self.step2()
        self.step3()
        self.step4()

    def step1(self, target):
        """
        Error for last layer
        """

        L = len(self.layers) - 1

        # Calculate derivatives
        deriv_vec = self.activ[L] - target

        # Calculate sigma vector
        sig_vec = []
        for j in range(self.layers[L]):
            z = np.dot(self.weights[L][j], self.activ[L-1])
            z += self.bias[L][j]

            sig_vec.append(self.activate(z))

        self.errors[L] = sig_vec * deriv_vec

    def step2(self):
        """
        Error for layers L-1 to 2
        """

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
        """
        Derivative of cost function in order of the biases
        """

        self.cost_b = self.errors

    def step4(self):
        """
        Derivative of cost function in order of the weights
        """

        self.cost_w = [[]]

        for l in range(1, len(self.layers)):
            rows = self.layers[l]
            cols = self.layers[l-1]
            self.cost_w.append(np.zeros((rows, cols)))

            for j in range(rows):
                for k in range(cols):
                    self.cost_w[l][j][k] = self.activ[l-1][k] * self.errors[l][j]

    def train(self, training_set, its=300, step=0.5):
        """
        Given a training set, train the weights and biases of the network using
        gradient descent. Back Propagation is used to calculate the derivatives
        of the quadratic cost function. The training set is of the form:

        [(inputs, outputs),
         (inputs, outputs),
         ...
         (inputs, outputs)]

         The inputs and outputs are lists
        """

        n = len(training_set)

        for it in range(its):
            if it % 10 == 0:
                print("Iterations:", it)

            sum_cost_w = [np.zeros((iv, self.layers[i-1])) for i, iv in enumerate(self.layers)]
            sum_cost_b = [np.zeros((iv, 1)) for i, iv in enumerate(self.layers)]

            # Get the weights and bias
            for i in training_set:
                self.backPropagation(i[0], i[1])

                for l, vl in enumerate(self.cost_w[1:], 1):
                    sum_cost_w[l] += self.cost_w[l]

                for l, vl in enumerate(self.cost_b[1:], 1):
                    sum_cost_b[l] += self.cost_b[l]

            for i, iv in enumerate(sum_cost_w[1:], 1):
                self.weights[i] -= step/n * iv

            for i, iv in enumerate(sum_cost_b):
                self.bias[i] -= step/n * iv
