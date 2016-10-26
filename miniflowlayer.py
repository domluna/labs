import numpy as np

class Layer:
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        # New property! Keys are the inputs to this layer and
        # their values are the partials of this layer with
        # respect to that input.
        self.gradients = {}
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError


class Input(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input layer has no inputs so the gradient starts
        # at zero.
        self.gradients = {self: 0}
        for n in self.outbound_layers:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        # Initialize a partial for each of the inbound_layers.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through the outputs. The gradient will change depending
        # on each output.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this layer's inputs.
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, self.inbound_layers[1].value.T)
            # Set the partial of the loss with respect to this layer's weights.
            self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, grad)
            # Set the partial of the loss with respect to this layer's bias.
            self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Layer):
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_layers[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        # TODO: implement
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_Layer}


# NOTE: assume y is a vector with values 0-9
# easier to work with than a one-hot encoding
class CrossEntropyWithSoftmax(Layer):
  def __init__(self, x, y):
    Layer.__init__(self, [x, y])

  def _predict(self):
    probs = self._softmax(self.inbound_Layer[0].value)
    return np.argmax(probs, axis=1)

  def accuracy(self):
    preds = self._predict()
    return np.mean(preds == self.inbound_Layer[1].value)

  def _softmax(self, x):
    # TODO: implement softmax function
    pass

  def forward(self):
    # TODO: implement
    pass

  def backward(self):
    # TODO: implement
    assert len(self.outbound_Layers) == 0
    self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_Layer}


def topological_sort(feed_dict):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(feed_dict, cost_function, ideal_output):
    """
    Performs a forward pass and a backward pass through a list of sorted Layers.

    Arguments:

        `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
        `cost_function`: The cost function to use.
        `ideal_output`: The correct output. Used to calculate cost.
    """

    sorted_layers = topological_sort(feed_dict)

    # Forward pass
    for n in sorted_layers:
        n.forward()

    # Calculate the cost against the output layer.
    cost = cost_function(ideal_output, sorted_layers[-1].value, feed_dict)

    # Backward pass
    reversed_layers = layers[::-1] # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in reversed_layers:
        n.backward()

    return sorted_nodes


def MSE(actual_output, ideal_output, feed_dict):
    """
    Calculates the mean squared error.

    Arguments:
        `actual_output`: a numpy array
        `ideal_output`: a numpy array
        `feed_dict`: the same feed_dict that sets the inputs
    """
    first = 1. / (2. * len(feed_dict))
    norm = np.linalg.norm(ideal_output - actual_output)
    return first * np.square(norm)


# NOTE: This layer is just here to pass dummy gradients backwards for testing
# purposes.
# class DummyGrad(Layer):
#   def __init__(self, x):
#     Layer.__init__(self, [x])

#   def forward(self):
#     self.value = self.inbound_Layer[0].value

#   def backward(self, grad):
#     self.gradients = {n: grad for n in self.inbound_Layers}

# def value_and_grad(layer, feed_dict, wrt=[]):
#   """
#   Performs a forward and backward pass. The `value` of layer after the forward pass will be returned along with the gradients of all layers in wrt.

#   Arguments:

#     `layer`: A layer in the graph, should be the output layer (have no outgoing edges).
#     `feed_dict`: A dictionary where the key is a `Input` layer and the value is the respective value feed to that layer.

#     `wrt`: 'With Respect To'. A list of layers. The gradient for each layer will be returned.
#   """
#   assert layer.outbound_layers == []
#   input_layers = [n for n in feed_dict.keys()]
#   # Creates a flattened list of layers in a valid operational order.
#   layers = topological_sort(input_layers)

#   # forward pass
#   for n in layers:
#     if isinstance(n, Input):
#       v = feed_dict[n]
#       n.forward(v)
#     else:
#       n.forward()

#   # backward pass
#   for n in layers[::-1]:
#     if isinstance(n, DummyGrad):
#       g = feed_dict[n]
#       n.backward(g)
#     else:
#       n.backward()

#   return layer.value, [n.gradients[n] for n in wrt]


def accuracy(output_layer, feed_dict):
  """
  Computes the accuracy of the model. All the weights and data(features, labels) should be in `feed_dict`.

  Arguments:

    `output_layer`: A layer in the graph, should be the output layer (have no outgoing edges.
    `feed_dict`: A dictionary where the key is a `Input` layer and the value is the respective value feed to that layer.
  """
  input_layers = [n for n in feed_dict.keys()]
  layers = topological_sort(input_layers)
  # doesn't make sense if output layer isn't Softmax
  # assert output_layer.typename == 'CrossEntropyWithSoftmax'
  # assert layers[-1].typename == 'CrossEntropyWithSoftmax'


  # forward pass on all layers except the last
  for n in layers[:-1]:
    if isinstance(n, Input):
      v = feed_dict[n]
      n.forward(v)
    else:
      n.forward()

  return layers[-1].accuracy()


# def value_and_grad(Layer, feed_dict, wrt=[]):
#     """
#     Performs a forward and backward pass. The `value` of Layer after the forward pass will be returned along with the gradients of all Layers in wrt.

#     Arguments:

#         `Layer`: A Layer in the graph, should be the output Layer (have no outgoing edges).
#         `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.

#         `wrt`: 'With Respect To'. A list of Layers. The gradient for each Layer will be returned.
#     """
#     assert Layer.outbound_Layers == []
#     input_Layers = [n for n in feed_dict.keys()]
#     # Creates a flattened list of Layers in a valid operational order.
#     Layers = topological_sort(input_Layers)

#     # forward pass
#     for n in Layers:
#         if n.typename == 'Input':
#             v = feed_dict[n]
#             n.forward(v)
#         else:
#             n.forward()

#     # backward pass
#     for n in Layers[::-1]:
#         if n.typename == 'DummyGrad':
#             g = feed_dict[n]
#             n.backward(g)
#         else:
#             n.backward()

#     return Layer.value, [n.gradients[n] for n in wrt]