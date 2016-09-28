import numpy as np

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

class Node(object):
    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.cache = {}
        self.value = None
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
        Node.__init__(self, [])

    def forward(self, value):
        self.value = value

    def backward(self):
        # An Input node has no inputs so we refer to ourself
        # for the dvalue
        self.dvalues = {self: 0}
        for n in self.output_nodes:
            self.dvalues[self] += 1 * n.dvalues[self]

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x,y])

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

class Mul(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x,y])

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
        Node.__init__(self, [x,w,b])

    def forward(self):
        self.cache[0] = self.input_nodes[0].value
        self.cache[1] = self.input_nodes[1].value
        self.cache[2] = self.input_nodes[2].value
        self.value = np.dot(self.cache[0], self.cache[1]) + self.cache[2]

    def backward(self):
        self.dvalues = {n: np.zeros_like(n.value) for n in self.input_nodes}
        if len(self.output_nodes) == 0:
            self.dvalues[self.input_nodes[0]] += np.dot(np.ones_like(self.value), self.cache[1].T)
            self.dvalues[self.input_nodes[1]] += np.dot(self.cache[0].T, np.ones_like(self.value))
            self.dvalues[self.input_nodes[2]] += self.value.shape[0] # equivalent to summing this amount of 1s
            return
        for n in self.output_nodes:
            dval = n.dvalues[self]
            self.dvalues[self.input_nodes[0]] += np.dot(dval, self.cache[1].T)
            self.dvalues[self.input_nodes[1]] += np.dot(self.cache[0].T, dval)
            self.dvalues[self.input_nodes[2]] += np.sum(dval, axis=0, keepdims=True)

class Sigmoid(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

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


class CrossEntropyWithSoftmax(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x,y])

    def _softmax(self, x):
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return probs

    def _predict(self):
        probs = self._softmax(self.input_nodes[0].value)
        return np.argmax(probs, axis=1)

    def _accuracy(self):
        preds = self._predict()
        return np.mean(preds == self.input_nodes[1].value)

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
        # combined derivative of softmax and cross entropy
        dprobs = np.copy(self.cache[0])
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
    # print(loss, grad)
    assert loss == -3
    assert grad == [-1, 3, 3]

# Linear test
def test2():
    x_in, w_in, b_in = Input(), Input(), Input()
    f = Linear(x_in, w_in, b_in)

    x = np.array([[-1., -2.], [-1, -2]])
    w = np.array([[2., -3], [2., -3]])
    b = np.array([-3., -3]).reshape(1, -1)

    feed_dict = {x_in: x, w_in: w, b_in: b}
    loss, grad = value_and_grad(f, feed_dict, (x_in, w_in, b_in))
    # print(loss, grad)
    assert np.allclose(loss, np.array([[-9.,  6.], [-9.,  6.]]))
    assert np.allclose(grad[0], np.array([[-1.,  -1.], [-1.,  -1.]]))
    assert np.allclose(grad[1], np.array([[-2.,  -2.], [-4., -4.]]))
    assert np.allclose(grad[2], np.array([[2., 2.]]))

# Sigmoid test
def test3():
    x_in = Input()
    f = Sigmoid(x_in)

    x = np.array([-10., 0, 10])
    feed_dict = {x_in: x}
    loss, grad = value_and_grad(f, feed_dict, [x_in])
    # print(loss, grad)
    assert np.allclose(loss, np.array([0., 0.5, 1.]), atol=1.e-4)
    assert np.allclose(grad, np.array([0., 0.25, 0.]), atol=1.e-4)

# CrossEntropyWithSoftmax test
def test4():
    x_in = Input()
    y_in = Input()
    f = CrossEntropyWithSoftmax(x_in, y_in)

    # pretend output of a softmax
    x = np.array([[0.5, 1., 1.5]])
    y = np.array([1])
    feed_dict = {x_in: x, y_in: y}
    loss, grad = value_and_grad(f, feed_dict, wrt=[x_in])
    # print(loss, grad)
    assert np.allclose(loss, 1.1802, atol=1.e-4)
    assert np.allclose(grad, np.array([[0.1863, -0.6928,  0.5064]]), atol=1.e-4)


def main():
    test1()
    test2()
    test3()
    test4()
    print('Tests pass!')

if __name__ == '__main__':
    main()
