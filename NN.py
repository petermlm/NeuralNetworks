import random


import numpy as np
import numpy.random


"""
Neural Network
"""


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

        self.weights = [2*np.random.random((iv, self.layers[i-1])) - 1
                        for i, iv in enumerate(self.layers)]
        self.bias = [2*np.random.random((iv, 1)) - 1
                     for i, iv in enumerate(self.layers)]
        self.activ = [np.zeros((iv, 1))
                      for i, iv in enumerate(self.layers)]
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
        self.activ[0] = np.array(input_vals[:])

        # For the following layers, use the normal feed function
        for l in range(1, len((self.layers))):
            z_vec = np.dot(self.weights[l], self.activ[l-1]) + self.bias[l]
            self.activ[l] = self.activate(z_vec)

        return self.activ[-1]

    def backPropagation(self, input_vals, output_vals):
        """
        Does the Back Propagation algorithm for one sample
        """

        # Feed this input forward
        self.feedForward(input_vals)

        # Calculate the derivatives using the four equations
        self.step1(output_vals)  # Output error
        self.step2()  # Back propagate error
        self.step3()  # Calculate the gradient of cost with respect to bias
        self.step4()  # Calculate the gradient of cost with respect to weights

    def step1(self, target):
        """
        Output error, or error for last layer
        """

        L = len(self.layers) - 1

        # Calculate derivatives
        deriv_vec = self.activ[L] - target

        # Calculate sigma vector
        z_vec = np.dot(self.weights[L], self.activ[L-1]) + self.bias[L]
        sig_vec = self.activate(z_vec)

        self.errors[L] = deriv_vec * sig_vec

    def step2(self):
        """
        Error for layers L-1 to 2, back propagation
        """

        L = len(self.layers) - 1

        for l in range(L-1, 0, -1):
            # Bring error back
            error_back = np.dot(self.weights[l+1].T, self.errors[l+1])

            # Calculate sigma vector
            z_vec = np.dot(self.weights[l], self.activ[l-1]) + self.bias[l]
            sig_vec = self.activate(z_vec)

            self.errors[l] = error_back * sig_vec

    def step3(self):
        """
        Gradient of cost function in order of the biases
        """

        self.cost_b = self.errors

    def step4(self):
        """
        Gradient of cost function in order of the weights
        """

        self.cost_w = [[]]

        for l in range(1, len(self.layers)):
            deriv = np.dot(self.errors[l], self.activ[l-1].T)
            self.cost_w.append(deriv)

    def train(self, training_set, its=300, step=0.5, verbose=False):
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
            if verbose and it % 10 == 0:
                print("Iterations:", it)

            # Will have the sums of gradients for all examples
            sum_cost_w = [np.zeros((iv, self.layers[i-1]))
                          for i, iv in enumerate(self.layers)]
            sum_cost_b = [np.zeros((iv, 1))
                          for i, iv in enumerate(self.layers)]

            # Run every training example and sum its gradient
            for i in training_set:
                self.backPropagation(i[0], i[1])

                for l, vl in enumerate(self.cost_w[1:], 1):
                    sum_cost_w[l] += self.cost_w[l]

                for l, vl in enumerate(self.cost_b[1:], 1):
                    sum_cost_b[l] += self.cost_b[l]

            # Update weights and biases using the rules
            for l, lv in enumerate(sum_cost_w[1:], 1):
                self.weights[l] -= step/n * lv

            for l, lv in enumerate(sum_cost_b):
                self.bias[l] -= step/n * lv
