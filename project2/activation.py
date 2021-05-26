from helpers import Module
import torch
import math


# implements of activation layer: Relu and Tanh

class Relu(Module):
    """
        Method: forward()
                backward()
    """

    def __init__(self):
        super().__init__()
        self.s = 0
        self.x = 0

    def forward(self, inputs):
        self.s = inputs
        self.x = self.s.clamp(min=0)
        return self.x

    def backward(self, grad_wrt_output):
        # l'(s) = l'(x) * dsigma(s)  point-wise product
        # grad_wrt_output is l'(x), which is dl/dx
        dsigma_s = self.s.sign().clamp(min=0)
        return dsigma_s * grad_wrt_output

    def param(self):
        """"
        Return a list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        This list should be empty for parameterless modules (e.g. ReLU).
        """
        return [(None, None)]


class Tanh(Module):
    """
        Method: forward()
                backward()
    """

    def __init__(self):
        super().__init__()
        self.s = 0
        self.x = 0

    def forward(self, inputs):
        self.s = inputs
        self.x = self.s.tanh()
        return self.x

    def backward(self, grad_wrt_output):
        # l'(s) = l'(x) * dsigma(s)  point-wise product
        # grad_wrt_output is l'(x), which is dl/dx
        # dsigma(s) = (tanh_s)' = 1 - (tanh_s)^2
        dsigma_s = 1 - self.s.tanh()**2
        return grad_wrt_output * dsigma_s

    def param(self):
        return [(None, None)]

