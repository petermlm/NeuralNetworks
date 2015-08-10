import sys
sys.path.append("../..")

import numpy as np

from NN import NeuralNetwork

imgs_test = "mnist/train-images-idx3-ubyte"
labels_test = "mnist/train-labels-idx1-ubyte"

def testNetwork(net):
    # Open images and labels files
    imgs = open(imgs_test, "rb")
    labels = open(labels_test, "rb")

    # Get information of images
    imgs.read(4)  # Magic number
    imgs_num = int.from_bytes(imgs.read(4), "big")  # Number of images
    rows_num = int.from_bytes(imgs.read(4), "big")  # Rows
    cols_num = int.from_bytes(imgs.read(4), "big")  # Cols

    # Get information of labels
    labels.read(4)  # Magic number
    labels_num = int.from_bytes(labels.read(4), "big")  # Number of labels

    # Where information is stored
    correct = incorrect = 0
    confusion = [[0 for i in range(10)] for j in range(10)]

    # Read some images and labels
    for i in range(imgs_num):
        net_in = [[k/255] for k in imgs.read(28*28)]
        net_out = [[0] for k in range(10)]
        net_out[ord(labels.read(1))][0] = 1.0

        res = net.feedForward(np.array(net_in))

        if np.argmax(res) == np.argmax(net_out):
            correct += 1
        else:
            incorrect += 1

        confusion[np.argmax(net_out)][np.argmax(res)] += 1

    # Close the files
    imgs.close()
    labels.close()

    # Write some output
    print("Correct: %s; Incorrect: %s" % (correct, incorrect))
    print("Hit rate: %s" % (correct / imgs_num))
    print("Confusion matrix:")
    for i in confusion:
        print(i)


if __name__ == "__main__":
    net = NeuralNetwork([28*28, 500, 10])

    with open("network_files/mnist_500_network_file", "rb") as f:
        net.load(f)

    testNetwork(net)
