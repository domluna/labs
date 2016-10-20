As described in the last lesson, a neural network is a graph of mathematical functions. Nodes in each layer perform mathematical function using inputs from nodes in the previous layer (except for the input layer itself). For example, any node could be represented as  *f(x, y)*, where *x* and *y* are input values from nodes in the previous layer. This means that each node also creates an output value, which may be passed to nodes in the next layer (except for the output layer).

Every layer between the input layer and the output layer is called a **hidden layer**.

By propagating values from first layer, the input layer, through all the mathematical functions represented by each node, the network outputs a value. This is process is called a **forward pass**.

Here's an example of a very simple forward pass.

<video width="50%" controls loop >
  <source src="https://s3.amazonaws.com/content.udacity-data.com/courses/carnd/videos/input-to-output-scaled.mp4" type="video/mp4">
  Your browser does not support the video tag. <a href="https://s3.amazonaws.com/content.udacity-data.com/courses/fend/nytimes-fixed.mp4" target="_blank">Click here to see the animation.</a>
</video>

Notice that the output layer performs a mathematical function, addition, on its inputs. There is no hidden layer.

The nodes and their links, also called *edges*, create a graph structure. Though this example is fairly simple, it isn't hard to imagine that increasingly complex graphs with increasingly complex calculations can calculate... well... *almost anything*.

There are generally two steps to create neural networks:

1. Define the graph of nodes.
2. Propagate values through the graph.

`miniflow` works the same. You'll define the nodes and edges of your network with one method and then run it with another. `miniflow` comes with some starter code to help you out. Let's take a look.

### Nodes and miniflow

We'll use a Python class to represent a generic node in `miniflow`.

```
class Node:
    def __init__(self):
        # Properties will go here!
```

We know that each node may receive multiple inputs from other nodes and creates a single output (which will likely be passed to other nodes). Let's add two lists to store references to the inbound nodes and the outbound nodes.

```
class Node:
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
```

Each node will eventually calculate a value that represents its output. Let's initialize the `value` to `None` to indicate that it exists but hasn't been set yet.

```
class Node:
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
```

Each node will need to be able to pass values forward and perform backpropagation (more on that later). For now, let's add two placeholder methods for forward and backward propagation.

```
class Node:
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None

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

        You'll compute this later.
        """
        raise NotImplemented
```

### Nodes that Calculate

While `Node` defines the base set of properties that every node may hold, only specialized [subclasses](https://docs.python.org/3/tutorial/classes.html#inheritance) of `Node` may end up in the graph. As part of this lab, you'll be building  subclasses of `Node` that can perform calculations and hold values. For example, the `Input` class defines nodes in the input layer.

```
class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous nodes from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value
```

Unlike the other subclasses of `Node`, `Input` node does not actually calculate anything - it just holds a `value` (that you can either set explicitly or with `forward()`). This is the value of the input to the network.

`Add`, another subclass of `Node`, actually can perform calculations (addition).

```
class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        """
        You'll be writing math here in the first quiz!
        """
```

### Defining Graphs and Propagating Values

`miniflow` has two methods to help you define and then run values through your graphs.

In order to define your network, you'll need to define the order of operations for your nodes. Order of operations matters because you will run the node calculations one after another (i.e. in series) due to the fact that Python is single threaded unless you use the `thread` module, which we won't for this exercise. Given that the input to some nodes depends on the outputs of others, you need to flatten the graph in such a way where all the input dependencies for each node are resolved before trying to run its calculation. This is a technique called a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting).

The `topological_sort()` function implements topological sorting using [Kahn's Algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm). The details of this method are not important, rather the *output* is. `topological_sort()` returns a sorted list of nodes in which all of the calculations can run in series. `topological_sort()` takes in a `feed_dict`, which is a dictionary of the input values. Here's an example use case:

```
x, y = Input(), Input()

sorted_nodes = toplogical_sort({x: 10, y: 20})
```

(You can find the source code for `topological_sort()` in miniflow.py in the programming quiz below.)

The other method at your disposal is `forward_pass()`, which actually runs the network and outputs a value.

```
def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

    assert output_node.outbound_nodes == []
    for n in sorted_nodes:
        n.forward()

    return output_node.value
```

### Quiz 1 - Passing Values Forward

I want you to create and run this graph!

[[image of graph]]

I'm giving you nn.py and miniflow.py.

The neural network architecture is already there for you in nn.py. It's your job to finish `miniflow` to make it work.

For this quiz, I want you to:

1. Open `nn.py` below. **You don't need to change anything.** I just want you to see how `miniflow` works.
2. Open miniflow.py. **Finish the `forward` method on the `Add` class.**
3. Test your network by hitting "Test Run!" When the output looks right, hit "Submit!"