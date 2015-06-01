import random
import numpy as np

from NN import NeuralNetwork


def printResults(input_vec, results):
    """
    Prints results, which are a list
    """

    m_index = np.argmax(results)

    arrow = ["", "", "", ""]
    arrow[m_index] = " <---"

    print("Input: %s, %s" % (input_vec[0], input_vec[1]))

    print("Output:")
    for i, v in enumerate(results):
        print("\tGroup %s: %s%s" % (i+1, results[i], arrow[i]))

    print("")


def makeRandomData(n):
    """
    Generates random data for some experiments. The data will consist of a
    point with two coordinates. They will belong to one of four groups, which
    are:

    Group 1: 0.0 <= px < 0.5 and 0.0 <= py < 0.5
    Group 2: 0.0 <= px < 0.5 and 0.5 <= py <= 1.0
    Group 3: 0.5 <= px <= 1.0 and 0.0 <= py < 0.5
    Group 4: 0.5 <= px <= 1.0 and 0.5 <= py <= 1.0
    """

    training = []
    counts = [0] * 4

    for i in range(n):
        px = random.random()
        py = random.random()

        if 0.0 <= px < 0.5 and 0.0 <= py < 0.5:
            out_g1 = 1.0
            out_g2 = 0.0
            out_g3 = 0.0
            out_g4 = 0.0
            counts[0] += 1
        elif 0.0 <= px < 0.5 and 0.5 <= py <= 1.0:
            out_g1 = 0.0
            out_g2 = 1.0
            out_g3 = 0.0
            out_g4 = 0.0
            counts[1] += 1
        elif 0.5 <= px <= 1.0 and 0.0 <= py < 0.5:
            out_g1 = 0.0
            out_g2 = 0.0
            out_g3 = 1.0
            out_g4 = 0.0
            counts[2] += 1
        else:
            out_g1 = 0.0
            out_g2 = 0.0
            out_g3 = 0.0
            out_g4 = 1.0
            counts[3] += 1

        training.append(([[px], [py]], [[out_g1], [out_g2], [out_g3], [out_g4]]))

    # Write some output
    for i, v in enumerate(counts):
        print("\tGroup %s: %s" % (i+1, v))

    return training


def makeRandomTests(net, test_data):
    """
    Uses test dataset to calculate a confusion matrix and a hit rate
    """

    correct = incorrect = 0
    confusion = [[0 for i in range(4)] for j in range(4)]

    for i in test_data:
        input_val = np.array(i[0])
        exp_output = np.array(i[1])

        res = net.feedForward(input_val)

        if np.argmax(res) == np.argmax(exp_output):
            correct += 1
        else:
            incorrect += 1

        confusion[np.argmax(exp_output)][np.argmax(res)] += 1

    print("Correct: %s; Incorrect: %s" % (correct, incorrect))
    print("Hit rate: %s" % (correct / len(test_data)))
    print("Confusion matrix:")
    for i in confusion:
        print(i)


def makeFixedTests(net):
    tests = [np.array([[0.1], [0.3]]),
             np.array([[0.1], [0.8]]),
             np.array([[0.9], [0.2]]),
             np.array([[0.8], [0.9]])]

    for i in tests:
        res = net.feedForward(i)
        printResults(i, res)


if __name__ == "__main__":
    # Neural network with 2 inputs and 4 outputs
    net = NeuralNetwork([2, 4])

    # Make training and test data
    print("=== Training Samples ===")
    train_data = makeRandomData(5000)

    print("")
    print("=== Random Testing Samples ===")
    test_data = makeRandomData(1000)

    print("")
    print("=== Starting training ===")
    net.train(train_data, its=500, step=0.5, verbose=True)

    print("")
    print("=== Starting random tests ===")
    makeRandomTests(net, test_data)

    print("")
    print("=== Trying hand fixed tests ===")
    makeFixedTests(net)
