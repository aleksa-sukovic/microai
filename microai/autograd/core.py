import math

from typing import List, Union


class Variable:
    def __init__(self, value: Union[float, int] = None,  children: List["Variable"] = [], label: str = None):
        self.label = label
        self._children = children
        self._value = value if value is not None else None
        self._grad = 0.0
        self._forward() # ensure .data is always defined

    @property
    def data(self):
        return self._value

    @property
    def grad(self):
        return self._grad

    def backward(self, grad: float = None):
        self._backward(1.0 if grad is None else grad)

    def _forward(self):
        pass

    def _backward(self, grad: float):
        self._grad += grad

    def _to_var(self, other: float | int | None) -> "Variable":
        return Variable(other) if isinstance(other, int) or isinstance(other, float) else other

    def exp(self) -> "Variable":
        return ExpVariable(children=[self])

    def log(self) -> "Variable":
        return LogVariable(children=[self])

    def relu(self) -> "Variable":
        return ReLUVariable(children=[self])

    def __add__(self, other: "Variable") -> "Variable":
        return AddVariable(children=[self, self._to_var(other)])

    def __radd__(self, other: "Variable") -> "Variable":
        return AddVariable(children=[self._to_var(other), self])

    def __sub__(self, other: "Variable") -> "Variable":
       return AddVariable(children=[self, -self._to_var(other)])

    def __rsub__(self, other: "Variable") -> "Variable":
       return AddVariable(children=[self._to_var(other), -self])

    def __mul__(self, other: "Variable") -> "Variable":
        return MulVariable(children=[self, self._to_var(other)])

    def __rmul__(self, other: "Variable") -> "Variable":
        return MulVariable(children=[self._to_var(other), self])

    def __neg__(self) -> "Variable":
        return MulVariable(children=[self, Variable(-1)])

    def __pow__(self, other: int) -> "Variable":
        return PowVariable(children=[self, Variable(other)])

    def __truediv__(self, other: "Variable") -> "Variable":
        return MulVariable(children=[self, self._to_var(other) ** -1])

    def __rtruediv__(self, other: "Variable") -> "Variable":
        return MulVariable(children=[self._to_var(other), self ** -1])

    def __repr__(self) -> str:
        return f"Var(data={self.data:.2f}, grad={self.grad:.2f}, label={self.label})"


class AddVariable(Variable):
    def __init__(self, value: float | int = None, children: List[Variable] = []):
        super().__init__(value, children, label="+")

    def _forward(self):
        self._value = self._children[0].data + self._children[1].data

    def _backward(self, grad: float):
        self._grad += grad
        self._children[0]._backward(grad)
        self._children[1]._backward(grad)


class MulVariable(Variable):
    def __init__(self, value: float | int = None, children: List[Variable] = []):
        super().__init__(value, children, label="*")

    def _forward(self):
        self._value = self._children[0].data * self._children[1].data

    def _backward(self, grad: float):
        self._grad += grad
        self._children[0]._backward(grad * self._children[1].data)
        self._children[1]._backward(grad * self._children[0].data)


class PowVariable(Variable):
    def __init__(self, value: float | int = None, children: List[Variable] = []):
        super().__init__(value, children, label="**")

    def _forward(self):
        self._value = self._children[0].data ** self._children[1].data

    def _backward(self, grad: float):
        self._grad += grad
        base, exp = self._children[0].data, self._children[1].data
        self._children[0]._backward(grad * exp * base ** (exp - 1))


class ExpVariable(Variable):
    def __init__(self, value: float | int = None, children: List[Variable] = []):
        super().__init__(value, children, label="exp")

    def _forward(self):
        self._value = math.exp(self._children[0].data)

    def _backward(self, grad: float):
        self._grad += grad
        self._children[0].backward(self.data * grad)


class LogVariable(Variable):
    def __init__(self, value: float | int = None, children: List[Variable] = []):
        super().__init__(value, children, label="log")

    def _forward(self):
        self._value = math.log(self._children[0].data)

    def _backward(self, grad: float):
        self._grad += grad
        self._children[0].backward((1 / self._children[0].data) * grad)


class ReLUVariable(Variable):
    def __init__(self, value: float | int = None, children: List[Variable] = []):
        super().__init__(value, children, label="relu")

    def _forward(self):
        self._value = max(0, self._children[0].data)

    def _backward(self, grad: float):
        self._grad += grad
        self._children[0].backward(int(self.data > 0) * grad)
