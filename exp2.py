import numpy as np


from NN import Perceptron, Sigmoid, NeuralNetwork

per = Perceptron(np.array([-2, -2]), 3)
sig = Sigmoid(np.array([-2, -2]), 3)
net = NeuralNetwork([2, 3, 1])

# print(per.feed(np.array([0, 0])))
# print(per.feed(np.array([0, 1])))
# print(per.feed(np.array([1, 0])))
# print(per.feed(np.array([1, 1])))

# print(sig.feed(np.array([0, 0])))
# print(sig.feed(np.array([0, 1])))
# print(sig.feed(np.array([1, 0])))
# print(sig.feed(np.array([1, 1])))

net.feedForward(np.array([2.0, 4.0]))

# for i in net.activ:
#     print(i)

net.costDeriv(np.array([10]))
