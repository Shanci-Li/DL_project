# This is the helper function file

import math
import torch

# disable the auto_grad
torch.set_grad_enabled(False)


# ------------------ Classes ------------------

class Module(object):
    """
        base class to inherit from
    """

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_wrt_output):
        raise NotImplementedError

    def param(self):
        return []


class LossMSE:
    """
        Class for MSE Loss function
    """

    def __init__(self):
        pass

    @staticmethod
    def calculate_mse(predictions, targets):
        return (targets - predictions).pow(2).mean()

    @staticmethod
    def grad_wrt_loss(predictions, targets):
        return -2 * (targets - predictions).mean()
