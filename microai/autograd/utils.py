from typing import Callable
from graphviz import Digraph

from microai.autograd.core import Variable


def trace(root: Variable, label: Callable[[Variable], str] = lambda x: x.label):
    nodes, edges, open = [], [], [root]

    while open:
        node = open.pop(0)
        nodes.append(label(node))
        for child in node._children:
            edges.append((label(node), label(child)))
            open.append(child)

    nodes = list(reversed(nodes))
    edges = list(reversed([(e[1], e[0]) for e in edges]))

    return nodes, edges


def graph(root: Variable):
    nodes, edges = trace(root, label=lambda n: f"{n.label} = {n.data} (grad {n.grad})" if n.label else f"{n.data}")

    graph = Digraph("G", graph_attr={"rankdir": "LR"})
    for node in nodes: graph.node(node, shape="record")
    for edge in edges: graph.edge(*edge)

    return graph
