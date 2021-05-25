from helpers import Module
import torch
import math


class Sequential(Module):
    """
        A sequential of Modules to create the network
    """

    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for arg in args:
            self.modules.append(arg)
        self.train_loss = 0
        self.test_loss = 0
        self.size_layer = []

    def forward(self, inputs):
        for module in self.modules:
            inputs = module.forward(inputs)
        return inputs

    def backward(self, grad_wrt_outputs):
        for module in self.modules[::-1]:
            grad_wrt_outputs = module.backward(grad_wrt_outputs)

    def param(self):
        parameters = []
        for module in self.modules:
            parameters.append(module.param())
        return parameters


