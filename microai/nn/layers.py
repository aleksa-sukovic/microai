import random

from typing import List

from microai.autograd.core import Variable
from microai.nn.modules import Module


class Neuron(Module):
    def __init__(self, in_features: int, bias: int = 0):
        self.weights = [Variable(random.uniform(-1, 1)) for _ in range(in_features)]
        self.bias = Variable(bias)

    def __call__(self, x):
        return sum([w * s for w, s in zip(self.weights, x)], self.bias)

    def parameters(self) -> List[Variable]:
        return self.weights + [self.bias]


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.weights = [Neuron(in_features) for _ in range(out_features)]

    def __call__(self, x):
        # assumes x is a (1, in_features) vector, not a batch
        out = [n(x) for n in self.weights]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Variable]:
        return [p for neuron in self.weights for p in neuron.parameters()]


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Variable]:
        return [p for layer in self.layers for p in layer.parameters()]
