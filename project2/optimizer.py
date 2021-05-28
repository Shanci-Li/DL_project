import torch


class SGD:
    """
        SGD optimizor class
    """

    def __init__(self, model, method):
        self.model = model
        self.method = method

    def update(self, lr):
        # make sure learning rate > 0
        if self.method == 'sgd':
            param_list = self.model.param()
            # iterate through layers
            for module in param_list:
                for (w, dw) in module:
                    # make sure that the gradients exist
                    if dw is None:
                        continue
                    else:
                        w -= (lr['eta'] * torch.mean(dw, 1, True))

        elif self.method == 'sgd_momentum':
            momentum_list = self.model.moment()
            for module in momentum_list:
                for (w, dw, mo_dw) in module:
                    # make sure that the gradients exist
                    if dw is None:
                        continue
                    else:
                        w -= (lr['gamma'] * torch.mean(mo_dw, 1, True) + lr['eta'] * torch.mean(dw, 1, True))

    def zero_grad(self):
        param_list = self.model.param()
        # iterate through layers
        for module in param_list:
            for (w, dw) in module:
                # skip the activation layers
                if w is None and dw is None:
                    continue
                else:
                    # set all the gradients 0
                    dw.fill_(0.)

