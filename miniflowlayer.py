import numpy as np
import random

class Layer:
    """
    Base class for layers in the network.

    Arguments:

        `inbound_layers`: A list of layers with edges into this layer.
    """
    def __init__(self, inbound_layers=[]):
        """
        Layer's constructor (runs when the object is instantiated). Sets
        properties that all layers need.
        """
        # A list of layers with edges into this layer.
        self.inbound_layers = inbound_layers
        # The eventual value of this layer. Set by running
        # the forward() method.
        self.value = None
        # A list of layers that this layer outputs to.
        self.outbound_layers = []
        # New property! Keys are the inputs to this layer and
        # their values are the partials of this layer with
        # respect to that input.
        self.gradients = {}
        # Sets this layer as an outbound layer for all of
        # this layer's inputs.
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        """
        Every layer that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward():
        """
        Every layer that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Layer):
    """
    A generic input into the network.
    """
    def __init__(self):
        # The base class constructor has to run to set all
        # the properties here.
        #
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input layer has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_layers:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    """
    Represents a layer that performs a linear transform.
    """
    def __init__(self, inbound_layer, weights, bias):
        # The base class (Layer) constructor. Weights and bias
        # are treated like inbound layers.
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_layers.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this layer's inputs.
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, self.inbound_layers[1].value.T)
            # Set the partial of the loss with respect to this layer's weights.
            self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this layer's bias.
            self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Layer):
    """
    Represents a layer that performs the sigmoid activation function.
    """
    def __init__(self, layer):
        # The base class constructor.
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbound_layers[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Sum the gradients over all the outputs.
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_layers[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Layer):
    def __init__(self, inbound_layer):
        """
        The mean squared error cost function.
        Should be used as the last layer for a network.

        Arguments:
            `inbound_layer`: A layer with an activation function.
        """
        # Call the base class' constructor.
        Layer.__init__(self, [inbound_layer])
        """
        These two properties are set during topological_sort()
        """
        # The ideal_output for forward().
        self.ideal_output = None
        # The number of inputs for forward().
        self.n_inputs = None

    def forward(self):
        """
        Calculates the mean squared error.
        """
        actual_output = self.inbound_layers[0].value
        first_term = 1. / (2. * self.n_inputs)
        norm = np.linalg.norm(self.ideal_output - actual_output)
        self.value = first_term * np.square(norm)

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradients[self.inbound_layers[0]] = -2 * self.inbound_layers[0].value


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


def topological_sort(feed_dict, ideal_output):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
    `ideal_output`: The correct output value for the last activation layer.

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
        if isinstance(n, MSE):
            n.ideal_output = ideal_output
            n.n_inputs = len(feed_dict)

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(feed_dict, ideal_output):
    """
    Performs a forward pass and a backward pass through a list of sorted Layers.

    Arguments:

        `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
        `ideal_output`: The correct output value for the last activation layer.
    """

    sorted_layers = topological_sort(feed_dict, ideal_output)

    # Forward pass
    for n in sorted_layers:
        n.forward()

    # Backward pass
    reversed_layers = sorted_layers[::-1] # see: https://docs.python.org/2.3/whatsnew/section-slices.html

    for n in reversed_layers:
        n.backward()

    return sorted_layers


def train_SGD(feed_dict, ideal_output, trainables=[], epochs=1, learning_rate=1e-2):
    """
    Performs a forward pass and a backward pass through a list of sorted Layers.

    Arguments:

        `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
        `ideal_output`: The correct output value for the last activation layer.
        `trainables`: Inputs that need to be modified by SGD.
        `epochs`: The number of times to train against all training inputs.
        `learning_rate`: The step size for changes by each gradient.
    """

    sorted_layers = topological_sort(feed_dict, ideal_output)

    for i in range(epochs):
        # Forward pass
        for n in sorted_layers:
            n.forward()

        # Backward pass
        reversed_layers = sorted_layers[::-1] # see: https://docs.python.org/2.3/whatsnew/section-slices.html

        for n in reversed_layers:
            n.backward()

        # Performs SGD
        gradient_cost = [n.gradients[n] for n in trainables]
        for n in range(len(trainables)):
            trainables[n].value -= learning_rate * gradient_cost[n]
        print('Epoch: ' + str(i) + ', Loss: ' + str(sorted_layers[-1].value))

    return sorted_layers[-1].value


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