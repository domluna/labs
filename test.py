from miniflow import *

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 4, y: 5}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output)