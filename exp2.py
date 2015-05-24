import random
import numpy as np

from NN import NeuralNetwork


def printResults(input_vec, results):
    arrow = ["", "", ""]
    if results[0] > results[1] and results[0] > results[2]:
        arrow[0] = " <---"
    elif results[1] > results[0] and results[1] > results[2]:
        arrow[1] = " <---"
    elif results[2] > results[1] and results[2] > results[0]:
        arrow[2] = " <---"

    print("Input: %s, %s" % (input_vec[0], input_vec[1]))
    print("Output:")
    print("\tGroup 1: %s%s" % (results[0], arrow[0]))
    print("\tGroup 2: %s%s" % (results[1], arrow[1]))
    print("\tGroup 3: %s%s" % (results[2], arrow[2]))
    print("")


if __name__ == "__main__":
    # Neural network with 2 inputs, 3 middle layer neurons, a 1 output
    net = NeuralNetwork([2, 2, 3])

    # Define a training example consisting of a point with two coordinates. If
    # that point is inside a few areas, then the network should output 1,
    # otherwise, it should output 0
    training = []

    # Point inside area 1
    g1 = g2 = g3 = 0
    for i in range(1000):
        px = random.random()
        py = random.random()

        if 0.0 <= px < 0.5 and 0.0 <= py < 0.5:
            out_g1 = 1.0
            out_g2 = 0.0
            out_g3 = 0.0
            g1 += 1
        elif 0.5 <= px <= 1.0 and 0.5 <= py <= 1.0:
            out_g1 = 0.0
            out_g2 = 1.0
            out_g3 = 0.0
            g2 += 1
        else:
            out_g1 = 0.0
            out_g2 = 0.0
            out_g3 = 1.0
            g3 += 1

        training.append(([[px], [py]], [[out_g1], [out_g2], [out_g3]]))

    print("Samples:")
    print("\tGroup 1: %s" % (g1))
    print("\tGroup 2: %s" % (g2))
    print("\tGroup 3: %s" % (g3))
    print("")

    print("Starting training")
    net.train(training, its=300, step=1)

    net.feedForward(np.array([[0.0], [0.0]]))
    printResults(np.array([[0.0], [0.0]]), net.activ[-1])
    net.feedForward(np.array([[1.0], [1.0]]))
    printResults(np.array([[1.0], [1.0]]), net.activ[-1])
    net.feedForward(np.array([[0.2], [0.2]]))
    printResults(np.array([[0.2], [0.2]]), net.activ[-1])
    net.feedForward(np.array([[0.3], [0.7]]))
    printResults(np.array([[0.3], [0.7]]), net.activ[-1])
    net.feedForward(np.array([[0.8], [0.1]]))
    printResults(np.array([[0.8], [0.1]]), net.activ[-1])
    net.feedForward(np.array([[0.9], [0.9]]))
    printResults(np.array([[0.9], [0.9]]), net.activ[-1])
