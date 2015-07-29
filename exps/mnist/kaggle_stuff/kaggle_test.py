import sys
sys.path.append("../../..")

import csv

import numpy as np

from NN import NeuralNetwork


# Need to be decompressed from kaggle_test_dataset.tar file
imgs_test = "test.csv"


if __name__ == "__main__":
    # Create network from serialization file
    net = NeuralNetwork()

    with open("../network_files/mnist_500_network_file", "rb") as f:
        net.load(f)

    # Open images files
    imgs = open(imgs_test, "r")
    imgs_reader = csv.reader(imgs, delimiter=',')

    # Make the output
    print("Result")

    first = True
    for i in imgs_reader:
        if first:
            first = False
            continue

        net_in = [[int(j)/255] for j in i]
        res = np.argmax(net.feedForward(np.array(net_in)))
        print(res)

    # Close the file
    imgs.close()
