from .layers import Module


class Tanh(Module):
    def __call__(self, x):
        x = x if isinstance(x, list) else [x]
        out = [((2 * act).exp() - 1) / ((2 * act).exp() + 1) for act in x]
        return out[0] if len(out) == 1 else out


class Sigmoid(Module):
    def __call__(self, x):
        x = x if isinstance(x, list) else [x]
        out = [1 / (1 + (-act).exp()) for act in x]
        return out[0] if len(out) == 1 else out


class ReLU(Module):
    def __call__(self, x):
        x = x if isinstance(x, list) else [x]
        out = [act.relu() for act in x]
        return out[0] if len(out) == 1 else out
