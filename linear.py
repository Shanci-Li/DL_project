from helpers import Module
import torch


class Linear(Module):
    """
        Linear layer for fully connected network
    """

    def __init__(self, in_features, out_features):
        """
        Initialization of the Linear layer
        :param in_features: number of input units
        :param out_features: number of output units
        """
        torch.manual_seed(0)
        # w: weights b:bias x:input data dw:gradient of w db:gradients of b
        self.w = torch.empty(out_features, in_features).normal_()
        self.b = torch.empty(out_features).normal_()
        self.x = 0
        self.dw = torch.zeros(self.w.size())
        self.db = torch.zeros(self.b.size())

    def forward(self, inputs):
        # s_(l+1) = w_(l+1) @ x_l + b_(l+1)
        self.x = inputs
        assert inputs.size() == self.b.size()
        return self.w.mv(self.x) + self.b

    def backward(self, grad_wrt_output):
        # ds = grad_wrt_output
        # db = ds
        # dw = ds @ x_(l-1).T
        # dx_(l-1) = w.T @ ds
        self.db = grad_wrt_output
        self.dw = grad_wrt_output.mv(self.w.t())
        grad_wrt_inputs = self.w.t().mv(grad_wrt_output)
        return grad_wrt_inputs

    def param(self):
        return [(self.w, self.dw), (self.b, self.db)]
