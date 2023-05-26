from mnist_tutorial import get_data, init_network, predict
import sys, os
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

x, _ = get_data()

network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W2.shape)
print(W3.shape)