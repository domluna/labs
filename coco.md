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

### `Node`s and `miniflow`

We'll use a Python class to represent a generic node in `miniflow`.

```
class Node:
    def __init__(self):
        # An optional description of the node - most useful for outputs.
        self.label = label
```

We know that each node may receive multiple inputs from other nodes and creates a single output (which will likely be passed to other nodes). Let's add two lists to store references to the inbound nodes and the outbound nodes.

```
class Node:
    def __init__(self, inbound_nodes=[]):
        # An optional description of the node - most useful for outputs.
        self.label = label
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
```

This node will eventually calculate a value. Let's initialize the `value` to `None` to indicate that it exists but hasn't been set yet.

```
class Node:
    def __init__(self, inbound_nodes=[]):
        # An optional description of the node - most useful for outputs.
        self.label = label
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
        # An optional description of the node - most useful for outputs.
        self.label = label
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

### Defining Graphs and Propagating Values

`miniflow` has two methods to help you define and then run values through your graphs.

In order to define your network, you'll need to define the order of operations for your nodes. Order of operations matters because you will run the node calculations one after another (i.e. in series) due to the fact that Python is single threaded unless you use the `thread` module, which we won't for this exercise. Given that the input to some nodes depends on the outputs of others, you need to flatten the graph in such a way where all the input dependencies for each node are resolved before trying to run its calculation. This is a technique called a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting).

The `topological_sort()` function implements topological sorting using [Kahn's Algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm). The details of this method are not important, rather the *output* is. `topological_sort()` returns a sorted list of nodes in which all of the calculations can run in series.

```
def topological_sort(input_nodes):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

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
```

The other


### Quiz 1 - Passing Values Forward

I want you to create and run the same graph that you just saw! For this quiz, you'll be given:

1. A generic `Node` class (more on this in a moment).
2. A method for connecting nodes in a graph.
3. A method for running the graph.

It will be your job to:

1. Define an `Add` node, which will be a subclass of `Node`.
2. Define the action to take on a forward pass with `Add`.
3. Test your network!
