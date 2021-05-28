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
        # construct the neural network from the *arg input classes
        for arg in args:
            self.modules.append(arg)
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

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

    def moment(self):
        momentum = []
        for module in self.modules:
            momentum.append(module.moment())
        return momentum

    def fit(self, train_inputs, train_targets, test_inputs, test_targets,
            loss, optimizer, nb_epoch, batch_size, lr, print_5epoch):
        for e in range(nb_epoch):
            # Training
            for b in range(0, train_inputs.size(1), batch_size):
                # forward pass
                outputs = self.forward(train_inputs.narrow(1, b, batch_size))
                # calculate loss of the epoch and gradients of outputs wrt the loss
                grad_wrt_loss = loss.grad_wrt_loss(outputs, train_targets.narrow(0, b, batch_size))
                # backward pass
                optimizer.zero_grad()
                self.backward(grad_wrt_loss)
                # updates the weights and bias with learning rate lr
                optimizer.update(lr)

            tr_output_prob = self.forward(train_inputs)
            tr_predict = torch.where(tr_output_prob > 0.5, 1, 0)
            train_acc = (tr_predict == train_targets).sum() / train_targets.size(0) * 100
            loss_train = loss.calc_loss(tr_output_prob, train_targets)
            self.train_acc.append(train_acc.item())
            self.train_loss.append(loss_train.item())

            # Testing
            test_output_prob = self.forward(test_inputs)
            test_predict = torch.where(test_output_prob > 0.5, 1, 0)
            test_acc = (test_predict == test_targets).sum() / test_targets.size(0) * 100
            loss_test = loss.calc_loss(test_output_prob, test_targets)
            self.test_acc.append(test_acc.item())
            self.test_loss.append(loss_test.item())

            if print_5epoch:
                if e % 5 == 0:
                    print('Epochs: {} \t'.format(e),
                          'Train Loss: {:.02f} \t'.format(loss_train.item()),
                          'Train accuracy: {:.02f}% \t'.format(train_acc.item()),
                          'Test loss: {:.02f} \t'.format(loss_test.item()),
                          'Test accuracy: {:.02f}%'.format(test_acc.item()))

        # Print final training performance
        print('\nTraining loss : {}'.format(self.train_loss[-1]))
        print('Training accuracy : {}\n'.format(self.train_acc[-1]))

        # Print final validation performance
        print('Test loss : {}'.format(self.test_loss[-1]))
        print('Test accuracy : {}'.format(self.test_acc[-1]))

        return self.train_loss, self.train_acc, self.test_loss, self.test_acc
