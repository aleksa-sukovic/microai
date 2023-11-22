from microai.nn.optimizers import Optimizer


class Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.epoch = 0
        self.base_lr = optimizer.lr

    def step(self):
        raise NotImplementedError()


class ExponentialLR(Scheduler):
    def __init__(self, optimizer: Optimizer, gamma: float = 0.9):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.base_lr * self.gamma ** self.epoch
