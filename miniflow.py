"""
In the exercises you will implement the `forward`
and `backward` methods of the Mul, Linear, and CrossEntropyWithSoftmax
nodes.

Q: What do I do in the forward method?

A:

    1. Perform the computation described in the notebook for that node.
        - You need the input node values to do this. Here's how you would access
        the value of the first node:

            first_node_value = self.inbound_nodes[0].value

    2. Store the final result in `self.value`.

    Here's the forward function for the Add node:

        def forward(self):
            self.value = self.inbound_nodes[0].value + self.inbound_nodes[1].value


Q: What do I do in the backward method?

A:

    1. Compute the derivative of the current node with respect to each input node.
    2. Multiply the above by the derivative of the each output node with respect to
    respect to the current node.
    3. Accumulate and store the results in `self.gradients`.

    Here's the backward function for the Add node:

        def backward(self):
            # Initialize all the derivatives to 0
            self.gradients = {n: 0 for n in self.inbound_nodes}

            # If no output nodes pretend the output is 1.
            # NOTE: for a matrix you could do this with `numpy.ones` or `numpy.ones_like`
            if len(self.outbound_nodes) == 0:
                self.gradients[self.inbound_nodes[0]] += 1 * 1
                self.gradients[self.inbound_nodes[1]] += 1 * 1
                return

            # Accumulate for all output nodes (recall case study)
            for n in self.outbound_nodes:
                # Derivative of output node w.r.t current node
                # we can use the self keyword to refer to the current node.
                grad = n.gradients[self] 

                # The derivative of the Add node w.r.t both input nodes is 1 (recall 
                the notebook).
                self.gradients[self.inbound_nodes[0]] += 1 * grad
                self.gradients[self.inbound_nodes[1]] += 1 * grad

The Input and Add nodes have already been implemented for you. All the `gradients` 
have been initialized for each node class as well.

Look for the TODOs!
"""
import numpy as np

#
# NODE DEFINITIONS
#
# ALL NODES ARE SUBLCASSES OF THE Node CLASS.
#


class Node(object):
    def __init__(self, inbound_nodes):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        # store here the values computed in the forward pass 
        # that will be used in the backward pass
        self.cache = {}
        # set this value on the forward pass
        self.value = None
        # set these gradients on the backward pass
        # the key should be an input node and the
        # value the gradient for that node
        self.gradients = {}
        self.typname = type(self).__name__

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.
        
        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.
        
        Compute the gradient of the current node with respect
        to the input nodes. The gradient of the loss with respect
        to the current node should already be computed in the `gradients`
        attribute of the output nodes.
        """
        raise NotImplemented
        

# NOTE: This node is just here to pass dummy gradients backwards for testing
# purposes.
class DummyGrad(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    def forward(self):
        self.value = self.inbound_nodes[0].value

    def backward(self, grad):
        self.gradients = {n: grad for n in self.inbound_nodes}


class Input(Node):
    def __init__(self):
        # an Input node has no incoming nodes
        # so we pass an empty list
        Node.__init__(self, [])

    # NOTE: Input node is the only node where the value
    # is passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous nodes from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value):
        self.value = value

    def backward(self):
        # An Input node has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]
            

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        self.value = self.inbound_nodes[0].value + self.inbound_nodes[1].value

    def backward(self):
        self.gradients = {n: 0 for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += 1 * grad
            self.gradients[self.inbound_nodes[1]] += 1 * grad


class Mul(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        # Look back to the case study example in the notebook.
        self.gradients = {n: 0 for n in self.inbound_nodes}


class Linear(Node):
    def __init__(self, x, w, b):
        Node.__init__(self, [x, w, b])

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}


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
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}


# NOTE: assume y is a vector with values 0-9
# easier to work with than a one-hot encoding
class CrossEntropyWithSoftmax(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def _predict(self):
        probs = self._softmax(self.inbound_nodes[0].value)
        return np.argmax(probs, axis=1)

    def _accuracy(self):
        preds = self._predict()
        return np.mean(preds == self.inbound_nodes[1].value)

    def _softmax(self, x):
        # TODO: implement softmax function
        pass

    def forward(self):
        # TODO: implement
        pass

    def backward(self):
        # TODO: implement
        assert len(self.outbound_nodes) == 0
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}


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
    assert node.outbound_nodes == []
    input_nodes = [n for n in feed_dict.keys()]
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
        if n.typname == 'DummyGrad':
            g = feed_dict[n]
            n.backward(g)
        else:
            n.backward()

    return node.value, [n.gradients[n] for n in wrt]


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
        for m in n.outbound_nodes:
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
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
