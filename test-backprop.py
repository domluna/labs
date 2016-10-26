import numpy as np
from miniflowlayer import *

inputs, weights, bias = Input(), Input(), Input()

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])
ideal_output = np.array(
    [[1.23394576e-04, 9.82013790e-01],
     [1.23394576e-04, 9.82013790e-01]])

f = Linear(inputs, weights, bias)
g = Sigmoid(f)
cost = MSE(g)

feed_dict = {inputs: x, weights: w, bias: b}
gradiented_layers = forward_and_backward(feed_dict, ideal_output)

"""
what's the output?
"""
# print(gradiented_layers)
for l in gradiented_layers:
    print(l.gradients)