from typing import List

from microai.autograd import Variable


class Optimizer:
    def __init__(self, parameters: List[Variable], lr: float = 1e-3):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p._grad = 0

    def step(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            p._value -= self.lr * p.grad
