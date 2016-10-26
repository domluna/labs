import numpy as np
from miniflowlayer import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)
g = Sigmoid(f)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

ideal_output = np.array(
    [[1.23394576e-04, 9.82013790e-01],
     [1.23394576e-04, 9.82013790e-01]])

gradiented_layers = forward_and_backward(feed_dict, MSE, ideal_output)

"""
what's the output?
"""
print(cost)