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
    net.train(training, its=300, step=0.5, verbose=True)

    # After training, feed a few test values into the network to see the output
    # Input 1, should see something close to 0
    print(net.feedForward(np.array([1])))

    # Input 0, should see something close to 1
    print(net.feedForward(np.array([0])))

    # Input 0.2, should see something close to 1, but not as close as the
    # previous example
    print(net.feedForward(np.array([0.2])))

    # Dump the network into a file, load it and test it again
    print("Serialization:")

    with open("exp1_network_file", "wb") as f:
        net.dump(f)

    with open("exp1_network_file", "rb") as f:
        net.load(f)

    print(net.feedForward(np.array([1])))
    print(net.feedForward(np.array([0])))
    print(net.feedForward(np.array([0.2])))

    # Ne2 network
    print("New network, which takes from previous network")

    new_net = NeuralNetwork()
    with open("exp1_network_file", "rb") as f:
        new_net.load(f)

    print(new_net.feedForward(np.array([1])))
    print(new_net.feedForward(np.array([0])))
    print(new_net.feedForward(np.array([0.2])))
