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
        # torch.manual_seed(0)
        # w: weights b:bias x:input data dw:gradient of w db:gradients of b
        self.w = torch.empty(out_features, in_features).normal_()
        self.b = torch.empty(out_features, 1).normal_()
        self.dw = torch.zeros(self.w.size())
        self.db = torch.zeros(self.b.size())
        self.momentum_dw = torch.zeros(self.w.size())
        self.momentum_db = torch.zeros(self.b.size())
        self.x = 0

    def forward(self, inputs):
        # s_(l+1) = w_(l+1) @ x_l + b_(l+1)
        self.x = inputs
        assert inputs.size(0) == self.w.size(1)
        return self.w @ self.x + self.b

    def backward(self, grad_wrt_output):
        # ds = grad_wrt_output
        # db = ds
        # dw = ds @ x_(l-1).T
        # dx_(l-1) = w.T @ ds
        self.momentum_dw, self.momentum_db = self.dw, self.db
        self.db = grad_wrt_output
        self.dw = grad_wrt_output.mm(self.x.t())
        grad_wrt_inputs = self.w.t().mm(grad_wrt_output)
        return grad_wrt_inputs

    def param(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def moment(self):
        return [(self.w, self.dw, self.momentum_dw), (self.b, self.db, self.momentum_db)]
