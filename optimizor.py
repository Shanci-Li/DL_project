import random

from helpers import Module
import torch
import math

class SGD():
    """
        SGD optimizor class
    """
    def __init__(self, loss, model, nb_epoch, lr):
        # make sure learning rate > 0
        assert lr > 0
        self.model = model
        self.lr = lr
        self.nb_epoch = nb_epoch
        self.loss = loss

    def update_grad(self):
        for module in self.param:
            w, dw = module[0]
            b, db = module[1]
            # make sure that the gradients exist
            if dw is None or db is None:
                continue
            else:
                w -= dw
                b -= db

    def zero_grad(self):
        for module in self.param:
            _, dw = module[0]
            _, db = module[1]
            dw.zeros_()
            db.zeros_()
