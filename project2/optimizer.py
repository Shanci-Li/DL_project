import torch


class SGD:
    """
        SGD optimizor class
    """

    def __init__(self, model, method):
        self.model = model
        self.method = method

        # parameter for adam method
        self.eps = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999

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
                        w -= (lr['eta'] * dw)

        elif self.method == 'momentum':
            momentum_list = self.model.moment()
            for module in momentum_list:
                for (w, dw, mo_dw) in module:
                    # make sure that the gradients exist
                    if dw is None:
                        continue
                    else:
                        w -= (lr['gamma'] * mo_dw + lr['eta'] * dw)

        elif self.method == 'adam':
            adam_list = self.model.adam()
            for module in adam_list:
                for (w, dw, mt, vt, t) in module:
                    # make sure that the gradients exist
                    if dw is None:
                        continue
                    else:
                        mt_hat = mt / (1 - self.beta1**t)
                        vt_hat = vt / (1 - self.beta2**t)
                        w -= (lr['eta'] * mt_hat / (vt_hat.sqrt() + self.eps))

    def zero_grad(self):
        if self.method == 'adam':
            adam_list = self.model.adam()
            for module in adam_list:
                for (w, dw, mt, vt, t) in module:
                    # make sure that the gradients exist
                    if w is None and dw is None:
                        continue
                    else:
                        dw.fill_(0.)
                        mt.fill_(0.)
                        vt.fill_(0.)
                        t.fill_(0.)

        else:
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
