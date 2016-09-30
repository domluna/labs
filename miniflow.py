import numpy as np

#
# Node definitions
#

class Node(object):
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        # store values computed in the forward pass 
        # that can will be used in the backward pass here
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
    # is passed as an argument to forward.
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

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        pass

class Mul(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        pass

class Linear(Node):
    def __init__(self, x, w, b):
        Node.__init__(self, [x, w, b])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        pass

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
        pass


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
        pass


def topological_sort(input_nodes):
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

def value_and_grad(node, feed_dict, wrt=[]):
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
