import random

from helpers import Module
import torch
import math


class SGD:
    """
        SGD optimizor class
    """

    def __init__(self, model):
        self.model = model

    def update(self, lr):
        # make sure learning rate > 0
        assert lr > 0
        param_list = self.model.param()
        for module in param_list:
            for (w, dw) in module:
                # make sure that the gradients exist
                if w is None or dw is None:
                    continue
                else:
                    w -= (lr * torch.mean(dw, 1, True))

    def zero_grad(self):
        param_list = self.model.param()
        for module in param_list:
            for (w, dw) in module:
                # make sure that the gradients exist
                if w is None or dw is None:
                    continue
                else:
                    dw.fill_(0.)

