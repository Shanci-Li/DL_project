# This is the helper function file

import math
import torch
from torch import FloatTensor, LongTensor, Tensor


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
    def calc_loss(predictions, targets):
        """
            calculate MSE loss
        """
        return (targets - predictions).pow(2).mean()

    @staticmethod
    def grad_wrt_loss(predictions, targets):
        grad = -2 * (targets - predictions)
        return grad


def generate_data(nb_samples):
    inputs = torch.empty(nb_samples, 2).uniform_(0, 1)
    center = Tensor([0.5, 0.5]).view(1, -1)
    distances = torch.norm((inputs - center).abs(), 2, 1)
    labels = (distances < 1 / math.sqrt(2 * math.pi)).type(LongTensor)
    return inputs.t(), labels
