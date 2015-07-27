import sys
sys.path.append("../..")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot

from NN import NeuralNetwork


# Need to be decompressed from mnist.tar file
imgs_test = "mnist_exp/mnist/t10k-images-idx3-ubyte"
labels_test = "mnist_exp/mnist/t10k-labels-idx1-ubyte"
imgs_train = "mnist_exp/mnist/train-images-idx3-ubyte"
labels_train = "mnist_exp/mnist/train-labels-idx1-ubyte"

# Default values to be used by default experiment
default_batch_size = 100  # Size of default batch for training
default_train_its = 300  # Default number of iterations
default_train_step = 0.5  # Default training step

# Values to be used taken from best of experiment 2
exp_batch_size = 100  # Size of default batch for training
exp_train_its = 300  # Default number of iterations
exp_train_step = 0.5  # Default training step


def trainNetwork(net, batch_size, train_its, train_step, verbose=True, cross_entropy=True):
    # Open images and labels files
    imgs = open(imgs_train, "rb")
    labels = open(labels_train, "rb")

    # Get information of images
    imgs.read(4)  # Magic number
    imgs_num = int.from_bytes(imgs.read(4), "big")  # Number of images
    rows_num = int.from_bytes(imgs.read(4), "big")  # Rows
    cols_num = int.from_bytes(imgs.read(4), "big")  # Cols

    # Get information of labels
    labels.read(4)  # Magic number
    labels_num = int.from_bytes(labels.read(4), "big")  # Number of labels

    # Read some images and labels
    i = 0
    while i < imgs_num:
        training = []

        for j in range(min(batch_size, imgs_num-i)):
            net_in = [[k/255] for k in imgs.read(28*28)]
            net_out = [[0] for k in range(10)]
            net_out[ord(labels.read(1))][0] = 1.0

            training.append((net_in, net_out))

            if verbose and i % 1000 == 0:
                print("Examples used: ", i)

            i += 1

        net.train(training, its=train_its, step=train_step, cross_entropy=cross_entropy)

    # Close the files
    imgs.close()
    labels.close()


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


def calcHitRate(net):
    """
    Calculates hit rate given a few tests
    """

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
    correct = 0

    # Read some images and labels
    for i in range(imgs_num):
        net_in = [[k/255] for k in imgs.read(28*28)]
        net_out = [[0] for k in range(10)]
        net_out[ord(labels.read(1))][0] = 1.0

        res = net.feedForward(np.array(net_in))

        if np.argmax(res) == np.argmax(net_out):
            correct += 1

    # Close the files
    imgs.close()
    labels.close()

    # Write some output
    return correct / imgs_num


def default():
    net = NeuralNetwork([28*28, 15, 10])
    trainNetwork(net, default_batch_size, default_train_its,
                 default_train_step)
    testNetwork(net)


def exp_no_inner_layer():
    net = NeuralNetwork([28*28, 10])
    trainNetwork(net, default_batch_size, default_train_its,
                 default_train_step)
    testNetwork(net)


def exp_var_inner_layer(res_file_name):
    # Clear output file
    open(res_file_name, "w").close()

    # Make every iteration
    for i in range(10, 21):
        # print("Layer: %s" % (i))

        net = NeuralNetwork([28*28, i, 10])
        trainNetwork(net, exp_batch_size, exp_train_its, exp_train_step,
                     verbose=False)

        with open(res_file_name, "a") as out_file:
            out_file.write(str(i) + ";" + str(calcHitRate(net)) + "\n")


def exp_500():
    net = NeuralNetwork([28*28, 500, 10])
    trainNetwork(net, default_batch_size, default_train_its,
                 default_train_step, cross_entropy=True)
    testNetwork(net)


if __name__ == "__main__":
    # If there are no arguments, just run the default experiment
    if len(sys.argv) <= 1:
        default()

    elif sys.argv[1] == "no_inner_layer":
        exp_no_inner_layer()

    elif sys.argv[1] == "var_inner_layer":
        exp_var_inner_layer("mnist_exp_results/mnist_var_inner_layer_res")

    elif sys.argv[1] == "500":
        exp_500()
