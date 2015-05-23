import random
import numpy as np

from NN import NeuralNetwork


if __name__ == "__main__":
    # A very simple neural network, only 1 input, 1 output, no middle layers
    net = NeuralNetwork([1, 1])

    # Make a big (and predictable) training set that states that for an input
    # 0, the output should be 1, and vice versa
    training = []

    for i in range(100):
        t = ([0], [1])
        training.append(t)

    for i in range(100):
        t = ([1], [0])
        training.append(t)

    # Train the network
    print("Starting training")
    net.train(training, its=300, step=0.5)

    # After training, feed a few test values into the network to see the output
    # Input 1, should see something close to 0
    net.feedForward(np.array([1]))
    print(net.activ[1])

    # Input 0, should see something close to 1
    net.feedForward(np.array([0]))
    print(net.activ[1])

    # Input 0.2, should see something close to 1, but not as close as the
    # previous example
    net.feedForward(np.array([0.2]))
    print(net.activ[1])
