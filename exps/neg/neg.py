"""
This experiment is a very elementary one. Its objective is to observe the
behaviour of a simple network in this implementation in terms of training,
feeding, and serialization.

The network only has 1 input layer and 1 output layer. No middle layer.

To begin, a training set is created. That training set consists on 200 tuples
which have the input and the output of the network. 100 of those tuples have
the input as 0 and the output as 1, the other 100 have the contrary.

So the network will output a values closer to 0 as the input is closer to 1 and
vice versa.

The network is trained with the following parameters:

    iteration = 300
    step      = 0.5

After training, the behaviour of the network is seen with the following inputs:

    Input: 1.0  - Expected output: ~0
    Input: 0.0  - Expected output: ~1
    Input: 0.2  - Expected output: ~1*

The output of 0.2* should be close to 1, but not as close as 0.0.

Finally the network is dumped into a serialization file and loaded again. A new
network is also created from that serialization. The same tests are run again
the loaded network and the newly created network. This tests if the network
serialization is being done right.
"""

import sys
sys.path.append("../..")

import random
import numpy as np

from NN import NeuralNetwork


if __name__ == "__main__":
    """
    Dataset
    """

    # Make a big (and predictable) training set that states that for an input
    # 0, the output should be 1, and vice versa
    training = []

    for i in range(100):
        t = ([0], [1])
        training.append(t)

    for i in range(100):
        t = ([1], [0])
        training.append(t)

    """
    Create, train, and test the Neural Network
    """

    # A very simple neural network, only 1 input, 1 output, no middle layers
    net = NeuralNetwork([1, 1])

    # Train the network
    print("Starting training")
    net.train(training, its=300, step=1.0, mini_batch_size=100, verbose=True, cross_entropy=True)

    print("")
    print("Results:")

    # After training, feed a few test values into the network to see the output
    # Input 1, should see something close to 0
    print(net.feedForward(np.array([1])))

    # Input 0, should see something close to 1
    print(net.feedForward(np.array([0])))

    # Input 0.2, should see something close to 1, but not as close as the
    # previous example
    print(net.feedForward(np.array([0.2])))

    """
    Serialize the network and try it
    """

    # Dump the network into a file, load it and test it again
    with open("neg_network_file", "wb") as f:
        net.dump(f)

    with open("neg_network_file", "rb") as f:
        net.load(f)

    print("")
    print("After serialization:")

    print(net.feedForward(np.array([1])))
    print(net.feedForward(np.array([0])))
    print(net.feedForward(np.array([0.2])))

    """
    Create a network from serialization
    """

    # New network
    print("")
    print("Creating new network from serialization:")

    new_net = NeuralNetwork()
    with open("neg_network_file", "rb") as f:
        new_net.load(f)

    print(new_net.feedForward(np.array([1])))
    print(new_net.feedForward(np.array([0])))
    print(new_net.feedForward(np.array([0.2])))

    """
    Create new network and train using cross entropy cost function
    """

    # A very simple neural network, only 1 input, 1 output, no middle layers
    net = NeuralNetwork([1, 1])

    # Train the network
    print("Starting training")
    net.train(training, its=300, step=1.0, mini_batch_size=100, verbose=True, cross_entropy=True)

    print("")
    print("Results with Cross Entropy cost function:")

    # After training, feed a few test values into the network to see the output
    # Input 1, should see something close to 0
    print(net.feedForward(np.array([1])))

    # Input 0, should see something close to 1
    print(net.feedForward(np.array([0])))

    # Input 0.2, should see something close to 1, but not as close as the
    # previous example
    print(net.feedForward(np.array([0.2])))
