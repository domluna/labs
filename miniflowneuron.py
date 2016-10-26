#
# NEURON DEFINITIONS
#
# ALL neurons ARE SUBLCASSES OF THE Neuron CLASS.
#


# class Neuron:
#   def __init__(self, inbound_neurons=[]):
#     # Neurons from which this Neuron receives values.
#     self.inbound_neurons = inbound_neurons
#     # Neurons to which this Neuron passes values.
#     self.outbound_neurons = []
#     # A calculated value
#     self.value = None
#     # Cache stores the values computed in the forward pass
#     # that will be used in the backward pass.
#     self.cache = {}
#     # Set these gradients on the backward pass.
#     # The key should be an input Neuron and the
#     # value the gradient for that Neuron.
#     self.gradients = {}
#     # Hack because isinstance acts weird.
#     self.typename = type(self).__name__
#     # Add this Neuron as an outbound Neuron on its inputs.
#     for n in self.inbound_neurons:
#       n.outbound_neurons.append(self)

class Neuron:
    def __init__(self, inbound_neurons=[]):
        # Neurons from which this Node receives values
        self.inbound_neurons = inbound_neurons
        # Neurons to which this Node passes values
        self.outbound_neurons = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.

        Compute the gradient of the current Neuron with respect
        to the input neurons. The gradient of the loss with respect
        to the current Neuron should already be computed in the `gradients`
        attribute of the output neurons.
        """
        raise NotImplemented


# NOTE: This Neuron is just here to pass dummy gradients backwards for testing
# purposes.
class DummyGrad(Neuron):
  def __init__(self, x):
    Neuron.__init__(self, [x])

  def forward(self):
    self.value = self.inbound_neurons[0].value

  def backward(self, grad):
    self.gradients = {n: grad for n in self.inbound_neurons}


class Input(Neuron):
    def __init__(self):
        # An Input Neuron has no inbound neurons,
        # so no need to pass anything to the Neuron instantiator
        Neuron.__init__(self)

        # NOTE: Input Neuron is the only Neuron where the value
        # may be passed as an argument to forward().
        #
        # All other Neuron implementations should get the value
        # of the previous neurons from self.inbound_neurons
        #
        # Example:
        # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value

    def backward(self):
        # An Input Neuron has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_neurons:
            self.gradients[self] += n.gradients[self]


class Add(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        # x_value = self.inbound_neurons[0].value
        # y_value = self.inbound_neurons[1].value
        # self.value = x_value + y_value
        self.value = 0
        for Neuron in self.inbound_neurons:
            self.value += Neuron.value

    def backward(self):
        self.gradients = {n: 0 for n in self.inbound_neurons}
        for n in self.outbound_neurons:
            # get the computed gradient of this Neuron from the output Neuron
            grad = n.gradients[self]
            # set the gradient from the input neurons to the partial of this
            self.gradients[self.inbound_neurons[0]] += 1 * grad
            self.gradients[self.inbound_neurons[1]] += 1 * grad


class Mul(Neuron):
  def __init__(self, x, y):
    Neuron.__init__(self, [x, y])

  def forward(self):
    # TODO: implement
    pass

  def backward(self):
    # TODO: implement
    # Look back to the case study example in the notebook.
    self.gradients = {n: 0 for n in self.inbound_neurons}


class Linear(Neuron):
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)
        self.weights = weights
        self.bias = bias

    def forward(self):
        weighted_inputs = 0
        for n in range(len(self.inbound_neurons)):
            input_value = self.inbound_neurons[n].value
            weight = self.weights[n].value
            weighted_inputs += input_value * weight
        self.value = weighted_inputs + self.bias.value

    def backward(self):
        # TODO: implement
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_neurons}


class Sigmoid(Neuron):
  def __init__(self, x):
    Neuron.__init__(self, [x])

  def _sigmoid(self, x):
    # TODO: implement sigmoid function
    pass

  def forward(self):
    # TODO: implement
    pass

  def backward(self):
    # TODO: implement
    self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_neurons}


# NOTE: assume y is a vector with values 0-9
# easier to work with than a one-hot encoding
class CrossEntropyWithSoftmax(Neuron):
  def __init__(self, x, y):
    Neuron.__init__(self, [x, y])

  def _predict(self):
    probs = self._softmax(self.inbound_neurons[0].value)
    return np.argmax(probs, axis=1)

  def _accuracy(self):
    preds = self._predict()
    return np.mean(preds == self.inbound_neurons[1].value)

  def _softmax(self, x):
    # TODO: implement softmax function
    pass

  def forward(self):
    # TODO: implement
    pass

  def backward(self):
    # TODO: implement
    assert len(self.outbound_neurons) == 0
    self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_neurons}


#
# YOU DON'T HAVE TO IMPLEMENT ANYTHING IN THESE FUNCTIONS.
#


def value_and_grad(Neuron, feed_dict, wrt=[]):
  """
  Performs a forward and backward pass. The `value` of Neuron after the forward pass will be returned along with the gradients of all neurons in wrt.

  Arguments:

    `Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
    `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.

    `wrt`: 'With Respect To'. A list of neurons. The gradient for each Neuron will be returned.
  """
  assert Neuron.outbound_neurons == []
  input_neurons = [n for n in feed_dict.keys()]
  # Creates a flattened list of neurons in a valid operational order.
  neurons = topological_sort(input_neurons)

  # forward pass
  for n in neurons:
    if n.typename == 'Input':
      v = feed_dict[n]
      n.forward(v)
    else:
      n.forward()

  # backward pass
  for n in neurons[::-1]:
    if n.typename == 'DummyGrad':
      g = feed_dict[n]
      n.backward(g)
    else:
      n.backward()

  return Neuron.value, [n.gradients[n] for n in wrt]


def accuracy(Neuron, feed_dict):
  """
  Computes the accuracy of the model. All the weights and data(features, labels) should be in `feed_dict`.

  Arguments:

    `Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges.
    `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.
  """
  input_neurons = [n for n in feed_dict.keys()]
  neurons = topological_sort(input_neurons)
  # doesn't make sense is output Neuron isn't Softmax
  assert Neuron.typename == 'CrossEntropyWithSoftmax'
  assert neurons[-1].typename == 'CrossEntropyWithSoftmax'


  # forward pass on all neurons except the last
  for n in neurons[:-1]:
    if n.typename == 'Input':
      v = feed_dict[n]
      n.forward(v)
    else:
      n.forward()

  return neurons[-1]._accuracy()


# def topological_sort(input_neurons):
#     """
#     Sort the neurons in topological order using Kahn's Algorithm.

#     All neurons should be reachable through the `input_neurons`.

#     Returns a list of sorted neurons.
#     """

#     G = {}
#     neurons = [n for n in input_neurons]
#     while len(neurons) > 0:
#         n = neurons.pop(0)
#         if n not in G:
#             G[n] = {'in': set(), 'out': set()}
#         for m in n.outbound_neurons:
#             if m not in G:
#                 G[m] = {'in': set(), 'out': set()}
#             G[n]['out'].add(m)
#             G[m]['in'].add(n)
#             neurons.append(m)

#     L = []
#     S = set(input_neurons)
#     while len(S) > 0:
#         n = S.pop()
#         L.append(n)
#         for m in n.outbound_neurons:
#             G[n]['out'].remove(m)
#             G[m]['in'].remove(n)
#             # if no other incoming edges add to S
#             if len(G[m]['in']) == 0:
#                 S.add(m)
#     return L



def topological_sort(feed_dict):
    """
    Sort the neurons in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.

    Returns a list of sorted neurons.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output Neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_Neuron.value


# def value_and_grad(Neuron, feed_dict, wrt=[]):
#     """
#     Performs a forward and backward pass. The `value` of Neuron after the forward pass will be returned along with the gradients of all neurons in wrt.

#     Arguments:

#         `Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
#         `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.

#         `wrt`: 'With Respect To'. A list of neurons. The gradient for each Neuron will be returned.
#     """
#     assert Neuron.outbound_neurons == []
#     input_neurons = [n for n in feed_dict.keys()]
#     # Creates a flattened list of neurons in a valid operational order.
#     neurons = topological_sort(input_neurons)

#     # forward pass
#     for n in neurons:
#         if n.typename == 'Input':
#             v = feed_dict[n]
#             n.forward(v)
#         else:
#             n.forward()

#     # backward pass
#     for n in neurons[::-1]:
#         if n.typename == 'DummyGrad':
#             g = feed_dict[n]
#             n.backward(g)
#         else:
#             n.backward()

#     return Neuron.value, [n.gradients[n] for n in wrt]