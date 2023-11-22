from typing import Literal


class Loss:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError()


class MSELoss(Loss):
    def __call__(self, y_true, y_pred):
        return sum([(yt - yp)**2 for yt, yp in zip(y_true, y_pred)])


class BCELoss(Loss):
    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        self.reduction = reduction

    def __call__(self, y_true, y_pred):
        loss = []
        for yt, yp in zip(y_true, y_pred):
            loss.append(-yt * yp.log() - (1 - yt) * (1 - yp).log())
        return sum(loss) if self.reduction == "sum" else sum(loss) / len(loss)
