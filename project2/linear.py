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
        :self parameters:
            w: weights          b:bias         x:input data
            dw:gradient of w    db:gradients of b
        """
        # initialize all the self parameters
        # sgd parameters
        self.x = 0
        self.w = torch.empty(out_features, in_features).normal_(0, 1)
        self.b = torch.empty(out_features, 1).normal_(0, 1)
        self.dw = torch.zeros(self.w.size())
        self.db = torch.zeros(self.b.size())

        # momentum parameters
        self.momentum_dw = torch.zeros(self.w.size())
        self.momentum_db = torch.zeros(self.b.size())

        # adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_dw_mt = torch.zeros(self.w.size())
        self.adam_db_mt = torch.zeros(self.b.size())
        self.adam_dw_vt = torch.zeros(self.w.size())
        self.adam_db_vt = torch.zeros(self.b.size())
        self.adam_t = torch.zeros(1)

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

        # backward-pass calculation
        self.db += grad_wrt_output.mean()
        self.dw += grad_wrt_output.mm(self.x.t()) / grad_wrt_output.size(1)
        grad_wrt_inputs = self.w.t().mm(grad_wrt_output)

        # buffer of dw and db for momentum method
        self.momentum_dw, self.momentum_db = self.dw, self.db

        # buffers for adam method
        self.adam_dw_mt = self.beta1 * self.adam_dw_mt + (1 - self.beta1) * self.dw
        self.adam_db_mt = self.beta1 * self.adam_db_mt + (1 - self.beta1) * self.db
        self.adam_dw_vt = self.beta2 * self.adam_dw_vt + (1 - self.beta2) * self.dw.pow(2)
        self.adam_db_vt = self.beta2 * self.adam_db_vt + (1 - self.beta2) * self.db.pow(2)
        self.adam_t += 1

        return grad_wrt_inputs

    def param(self):
        # weights and bias in format as list: [(w, dw), (b, db)]
        return [(self.w, self.dw), (self.b, self.db)]

    def moment(self):
        # weights and bias and memory for the epoch before
        # this is the model.param when using sgd_momentum method
        return [(self.w, self.dw, self.momentum_dw), (self.b, self.db, self.momentum_db)]

    def adam(self):
        # parameters of the layer when update weights using adam method
        return [(self.w, self.dw, self.adam_dw_mt, self.adam_dw_vt, self.adam_t),
                (self.b, self.db, self.adam_db_mt, self.adam_db_vt, self.adam_t)]
