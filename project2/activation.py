from helpers import Module
import torch


# implements of activation layer: Relu and Tanh

class Relu(Module):
    """
        Relu activation layer
        Method: forward()
                backward()
    """

    def __init__(self):
        super().__init__()
        self.s = 0
        self.x = 0

    def forward(self, inputs):
        # x = sigma(s)
        self.s = inputs
        self.x = self.s.clamp(min=0)
        return self.x

    def backward(self, grad_wrt_output):
        # l'(s) = l'(x) * d_sigma(s)  point-wise product
        # grad_wrt_output is l'(x), which is dl/dx
        d_sigma_s = self.s.sign().clamp(min=0)
        return grad_wrt_output * d_sigma_s

    def param(self):
        """"
        Return a list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        This list should be empty for parameterless modules (e.g. ReLU).
        """
        return [(None, None)]

    def moment(self):
        # parameters of the layer when update weights using momentum method
        return [(None, None, None)]

    def adam(self):
        # parameters of the layer when update weights using adam method
        return [(None, None, None, None, None)]


class Tanh(Module):
    """
        Tanh activation layer
        Method: forward()
                backward()
    """

    def __init__(self):
        super().__init__()
        self.s = 0
        self.x = 0

    def forward(self, inputs):
        # x = sigma(s)
        self.s = inputs
        self.x = self.s.tanh()
        return self.x

    def backward(self, grad_wrt_output):
        # l'(s) = l'(x) * d_sigma(s)  point-wise product
        # grad_wrt_output is l'(x), which is dl/dx
        # d_sigma(s) = (tanh_s)' = 1 - (tanh_s)^2
        d_sigma_s = 1 - self.s.tanh() ** 2
        return grad_wrt_output * d_sigma_s

    def param(self):
        return [(None, None)]

    def moment(self):
        return [(None, None, None)]

    def adam(self):
        # parameters of the layer when update weights using adam method
        return [(None, None, None, None, None)]


class Sigmoid(Module):
    """
        Sigmoid activation layer
        Method: forward()
                backward()
    """

    def __init__(self):
        super().__init__()
        self.s = 0
        self.x = 0

    def forward(self, inputs):
        # x = sigma(s)
        self.s = inputs
        self.x = torch.sigmoid(self.s)
        return self.x

    def backward(self, grad_wrt_output):
        # l'(s) = l'(x) * d_sigma(s)  point-wise product
        # grad_wrt_output is l'(x), which is dl/dx
        # d_sigma(s) = sigmoid(s) * (1 - sigmoid(s))
        d_sigma_s = self.x * (1 - self.x)
        return grad_wrt_output * d_sigma_s

    def param(self):
        return [(None, None)]

    def moment(self):
        return [(None, None, None)]

    def adam(self):
        # parameters of the layer when update weights using adam method
        return [(None, None, None, None, None)]
