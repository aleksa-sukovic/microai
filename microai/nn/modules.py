from typing import List
from microai.autograd.core import Variable


class Module:
    def parameters(self) -> List[Variable]:
        return []

    def __call__(self, x):
        raise NotImplementedError()
