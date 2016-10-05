import numpy as np

# NODE DEFINITIONS
#
# ALL NODES ARE SUBLCASSES OF THE Node CLASS.
# 
# In the exercises you will implement the `forward`
# and `backward` methods of the Add, Mul, Linear, and CrossEntropyWithSoftmax
# nodes. .
#
# The Input node is already implemented for you. All the `dvalues` 
# have been initialized for each node class as well.
#
# Look for the TODOs!

class Node(object):
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        # store here the values computed in the forward pass 
        # that will be used in the backward pass
        self.cache = {}
        # set this value on the forward pass
        self.value = None
        # set these dvalues on the backward pass
        # the key should be an input node and the
        # value the gradient for that node
        self.dvalues = {}
        self.typname = type(self).__name__

        for n in self.input_nodes:
            n.output_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # an Input node has no incoming nodes
        # so we pass an empty list
        Node.__init__(self, [])

    # NOTE: Input node is the only node where the value
    # is passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous nodes from self.input_nodes
    #
    # Example:
    # val0 = self.input_nodes[0].value
    def forward(self, value):
        self.value = value

    def backward(self):
        # An Input node has no inputs so we refer to ourself
        # for the dvalue
        self.dvalues = {self: 0}
        for n in self.output_nodes:
            self.dvalues[self] += n.dvalues[self]

class Mul(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        # HINT: We've done this one for you to get your started
        # Multiply the input nodes and store the result in `self.value`.
        # The Add Node is pretty similar ... :-)
        self.cache[0] = self.input_nodes[0].value
        self.cache[1] = self.input_nodes[1].value
        self.value = self.cache[0] * self.cache[1]

    def backward(self):
        # TODO: implement
        # Look back to the case study example in the notebook.
        self.dvalues = {n: 0 for n in self.input_nodes}


class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        # Look back to the case study example in the notebook.
        self.dvalues = {n: 0 for n in self.input_nodes}


class Linear(Node):
    def __init__(self, x, w, b):
        Node.__init__(self, [x, w, b])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}


class Sigmoid(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    def _sigmoid(self, x):
        # TODO: implement sigmoid function
        pass

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}


# NOTE: assume y is a vector with values 0-9
# easier to work with than a one-hot encoding
class CrossEntropyWithSoftmax(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def _predict(self):
        probs = self._softmax(self.input_nodes[0].value)
        return np.argmax(probs, axis=1)

    def _accuracy(self):
        preds = self._predict()
        return np.mean(preds == self.input_nodes[1].value)

    def _softmax(self, x):
        # TODO: implement softmax function
        pass

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        assert len(self.output_nodes) == 0
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}


#
# YOU DON'T HAVE TO IMPLEMENT ANYTHING IN THESE FUNCTIONS.
#


def value_and_grad(node, feed_dict, wrt=[]):
    """
    Performs a forward and backward pass. The value of `node` after the forward pass will be returned along with the gradients of all nodes in `wrt`.

    Arguments:

        `node`: A node in the graph, should be the output node (have no outgoing edges.
        `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

        `wrt`: A list of nodes. The gradient for each node will be returned.
    """
    assert node.output_nodes == []
    input_nodes = [n for n in feed_dict.keys()]
    # maybe refactor so we don't call this everytime? the graph is small
    # so it's probably not an issue
    nodes = topological_sort(input_nodes)

    # forward pass
    for n in nodes:
        if n.typname == 'Input':
            v = feed_dict[n]
            n.forward(v)
        else:
            n.forward()

    # backward pass
    for n in nodes[::-1]:
        n.backward()

    return node.value, [n.dvalues[n] for n in wrt]


def accuracy(node, feed_dict):
    """
    Computes the accuracy of the model. All the weights and data(features, labels) should be in `feed_dict`.

    Arguments:

        `node`: A node in the graph, should be the output node (have no outgoing edges.
        `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    """
    input_nodes = [n for n in feed_dict.keys()]
    nodes = topological_sort(input_nodes)
    # doesn't make sense is output node isn't Softmax
    assert node.typname == 'CrossEntropyWithSoftmax'
    assert nodes[-1].typname == 'CrossEntropyWithSoftmax'
    

    # forward pass on all nodes except the last
    for n in nodes[:-1]:
        if n.typname == 'Input':
            v = feed_dict[n]
            n.forward(v)
        else:
            n.forward()

    return nodes[-1]._accuracy()


def topological_sort(input_nodes):
    """
    Sort the nodes in topological order.

    All nodes should be reachable through the `input_nodes`.

    Returns a list of sorted nodes.
    """
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.output_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        L.append(n)
        for m in n.output_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
