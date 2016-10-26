"""
This looks simpler!
"""

import numpy as np
import mnist_loader

from miniflow import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

network = Linear(inputs, weights, bias)
network = Sigmoid(network)
network = MSE(network)

train_SGD(network, training_data, 1000)