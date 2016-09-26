import numpy as np

class Graph(object):
    pass

# TODO: move all these functions into Graph class

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
    # TODO: refactor so we don't class this everytime
    nodes = topological_sort(input_nodes)

    # forward pass
    for n in nodes:
        if isinstance(n, Input):
            v = feed_dict[n]
            n.forward(v)
        else:
            n.forward()

    # backward pass
    for n in nodes[::-1]:
        n.backward()

    return node.value, [n.dvalues[n] for n in wrt]

# TODO: figure this out
def prediction(node, feed_dict):
    input_nodes = [n for n in feed_dict.keys()]
    nodes = topological_sort(input_nodes)
    assert isinstance(node, CrossEntropyLoss)

    # forward pass
    for n in nodes:
        if isinstance(n, Input):
            v = feed_dict[n]
            n.forward(v)
        else:
            n.forward()

    return np.argmax(node.cache[0], axis=1)


class Node(object):
    """docstring for Node."""
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.cache = {}
        self.value = 0
        self.dvalues = {}

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

class Input(Node):
    def __init__(self):
        self.input_nodes = []
        self.output_nodes = []

    def forward(self, val):
        self.value = val

    def backward(self):
        # An Input node has no inputs so we refer to ourself
        # for the dvalue
        self.dvalues = {self: 0}
        for n in self.output_nodes:
            self.dvalues[self] += 1.0 * n.dvalues[self]

class Add(Node):
    def __init__(self, x, y):
        self.input_nodes = [x, y]
        self.output_nodes = []
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def forward(self):
        self.value = self.input_nodes[0].value + self.input_nodes[1].value

    def backward(self):
        self.dvalues = {n: 0 for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += 1
            self.dvalues[self.input_nodes[1]] += 1
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += 1 * dval
            self.dvalues[self.input_nodes[1]] += 1 * dval


class Sub(Node):
    def __init__(self, x, y):
        self.input_nodes = [x, y]
        self.output_nodes = []
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def forward(self):
        self.value = self.input_nodes[0].value + self.input_nodes[1].value

    def backward(self):
        self.dvalues = {n: 0 for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += 1
            self.dvalues[self.input_nodes[1]] += -1
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += 1 * dval
            self.dvalues[self.input_nodes[1]] += -1 * dval


class Neg(Node):
    def __init__(self, x):
        self.input_nodes = [x]
        self.output_nodes = []
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def forward(self):
        self.value = -self.input_nodes[0].value

    def backward(self):
        self.dvalues = {n: 0 for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += -1
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += -1 * dval

class Mul(Node):
    def __init__(self, x, y):
        self.input_nodes = [x, y]
        self.output_nodes = []
        self.cache = {}
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def forward(self):
        self.cache[0] = self.input_nodes[0].value
        self.cache[1] = self.input_nodes[1].value
        val = self.cache[0] * self.cache[1]
        self.value = val

    def backward(self):
        self.dvalues = {n: 0 for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += self.cache[1]
            self.dvalues[self.input_nodes[1]] += self.cache[0]
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += self.cache[1] * dval
            self.dvalues[self.input_nodes[1]] += self.cache[0] * dval

class Linear(Node):
    # TODO: numpy array assertions
    def __init__(self, x, w, b):
        self.input_nodes = [x, w, b]
        self.output_nodes = []
        self.cache = {}
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def forward(self):
        self.cache[0] = self.input_nodes[0].value
        self.cache[1] = self.input_nodes[1].value
        self.cache[2] = self.input_nodes[2].value
        self.value = np.dot(self.cache[0], self.cache[1]) + self.cache[2]

    def backward(self):
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += np.dot(self.value, self.cache[1].T)
            self.dvalues[self.input_nodes[1]] += np.dot(self.cache[0].T, self.value)
            self.dvalues[self.input_nodes[2]] += 1
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += np.dot(dval, self.cache[1].T)
            self.dvalues[self.input_nodes[1]] += np.dot(self.cache[0].T, dval)
            self.dvalues[self.input_nodes[2]] += np.sum(dval, axis=0, keepdims=True)

class Sigmoid(Node):
    def __init__(self, x):
        self.input_nodes = [x]
        self.output_nodes = []
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.input_nodes[0].value)

    # derivative of sigmoid(x) is (1 - sigmoid(x)) * sigmoid(x)
    def backward(self):
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += (1 - self.value) * self.value
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += (1 - self.value) * self.value * dval


class CrossEntropyLoss(Node):
    def __init__(self, x, y):
        self.input_nodes = [x, y]
        self.output_nodes = []
        self.cache = {}
        for n in self.input_nodes:
            n.output_nodes.append(self)

    def _softmax(self, x):
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return probs

    def forward(self):
        probs = self._softmax(self.input_nodes[0].value)
        y = self.input_nodes[1].value
        self.cache[0] = probs
        self.cache[1] = y
        n = probs.shape[0]
        logprobs = -np.log(probs[range(n), y])
        self.value = np.sum(logprobs) / n

    # we know this is a loss so we can be a bit less generic here
    # should have 0 output nodes
    def backward(self):
        assert len(self.output_nodes) == 0
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}
        dprobs = self.cache[0]
        y = self.cache[1]
        n = dprobs.shape[0]
        dprobs[range(n), y] -= 1
        dprobs /= n
        # leave the gradient for the 2nd node all 0s, we don't care about the gradient
        # for the labels
        self.dvalues[self.input_nodes[0]] = dprobs

### TESTS

# f = (x * y) + (x * z)
def test1():
    x, y, z = Input(), Input(), Input()
    g = Mul(x, y)
    h = Mul(x, z)
    f = Add(g, h)
    feed_dict = {x: 3, y: 4, z: -5}
    loss, grad = value_and_grad(f, feed_dict, (x, y, z))
    print(loss, grad)
    assert loss == -3
    assert grad == [-1, 3, 3]

def main():
    test1()
    print('Tests pass!')

if __name__ == '__main__':
    main()
