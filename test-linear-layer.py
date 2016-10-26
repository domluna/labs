import numpy as np
from miniflowlayer import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

"""
Output should be:
[[-9., 4.],
[-9., 4.]]
"""
print(output)