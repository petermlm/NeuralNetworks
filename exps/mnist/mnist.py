import sys
sys.path.append("../..")

import numpy as np

from NN import NeuralNetwork


# Need to be decompressed from mnist.tar file
imgs_test = "mnist/t10k-images-idx3-ubyte"
labels_test = "mnist/t10k-labels-idx1-ubyte"
imgs_train = "mnist/train-images-idx3-ubyte"
labels_train = "mnist/train-labels-idx1-ubyte"

# Default values to be used by default experiment
default_its = 400  # Default number of iterations
default_step = 0.5  # Default training step
default_mini_batch_size = 10  # Default mini batch size


def trainNetwork(net, verbose=True, cross_entropy=True):
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

    if verbose:
        print("Reading images and labels for training")

    # Read some images and labels
    training = []

    for i in range(imgs_num):
        if verbose and i % 10000 == 0:
            print("Read: %s" %(i))

        net_in = [[k/255] for k in imgs.read(28*28)]
        net_out = [[0] for k in range(10)]
        net_out[ord(labels.read(1))][0] = 1.0

        training.append((net_in, net_out))

    # Train the network
    net.train(training, its=default_its, step=default_step, mini_batch_size=default_mini_batch_size, verbose=verbose, cross_entropy=cross_entropy)

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
    trainNetwork(net, verbose=True)
    testNetwork(net)


def exp_no_inner_layer():
    net = NeuralNetwork([28*28, 10])
    trainNetwork(net, verbose=True)
    testNetwork(net)


def exp_var_inner_layer(res_file_name):
    # Clear output file
    open(res_file_name, "w").close()

    # Make every iteration
    for i in range(10, 500, 10):
        print("Inner Layer: %s" % (i))

        net = NeuralNetwork([28*28, i, 10])
        trainNetwork(net, verbose=False)

        with open(res_file_name, "a") as out_file:
            out_file.write(str(i) + ";" + str(calcHitRate(net)) + "\n")


def exp_500():
    net = NeuralNetwork([28*28, 500, 10])
    trainNetwork(net, cross_entropy=True)

    testNetwork(net)

    # Dump the network into a file, load it and test it again
    with open("network_files/mnist_500_network_file", "wb") as f:
        net.dump(f)


if __name__ == "__main__":
    # If there are no arguments, just run the default experiment
    if len(sys.argv) <= 1:
        default()

    elif sys.argv[1] == "no_inner_layer":
        exp_no_inner_layer()

    elif sys.argv[1] == "var_inner_layer":
        exp_var_inner_layer("mnist_results/mnist_var_inner_layer_res")

    elif sys.argv[1] == "500":
        exp_500()
