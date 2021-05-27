import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import dlc_practical_prologue as prologue
import CNN as CNN
import CNN_WS as CNN_WS
import CNN_AL as CNN_AL
import CNN_WS_AL as CNN_WS_AL
import MLP as MLP
#define the lists of models that you want to run
training_models = [CNN,CNN_WS,CNN_AL,CNN_WS_AL]
names = ['Basic CNN','CNN_WS','CNN_AL','CNN_WS_AL']
#set hyperparameters that apply to all models
nb_epochs=45
loss_ratio = 0.5
#please change the number of train_iteration to evaluate models, as to show the result to professor, here I set the iteration to 1 
train_iter = 1
size = 1000
momentum = 0.9
l2 = 5e-5
nb_hidden = 150

losses_lists = {}
train_accuracys = {}
test_accuracys = {}
for m in range (1,5):
    print("Training with model {}".format(names[m-1]))
    if m == 1:
        #for each model, set the optimal hyper-parameters
        mini_batch_size = 50
        lr = 0.01
        model = CNN.CNN(nb_hidden)
        losses_lists[m-1], train_accuracys[m-1], test_accuracys[m-1] = CNN.evaluate_accuracy(model, train_iter, size, mini_batch_size,lr, nb_epochs)
    elif m == 2:
        mini_batch_size = 50
        lr = 0.05
        model = CNN_WS.CNN_WS(nb_hidden)
        losses_lists[m-1], train_accuracys[m-1], test_accuracys[m-1] = CNN_WS.evaluate_accuracy(model, train_iter, size, mini_batch_size, lr, nb_epochs, l2)
    elif m == 3:
        mini_batch_size = 50
        lr = 0.05
        model = CNN_AL.CNN_AL(nb_hidden)
        losses_lists[m-1], train_accuracys[m-1], test_accuracys[m-1] = CNN_AL.evaluate_accuracy(model, train_iter, size, mini_batch_size, lr, nb_epochs,loss_ratio,l2)
    else:
        lr = 0.05
        mini_batch_size = 100
        model = CNN_WS_AL.CNN_WS_AL(nb_hidden)
        losses_lists[m-1], train_accuracys[m-1], test_accuracys[m-1] = CNN_WS_AL.evaluate_accuracy(model, train_iter, size, mini_batch_size, lr, nb_epochs, loss_ratio,momentum)
# print the performance of each model
for m in range(1,5):
    print("The result of the model {}".format(names[m-1]))
    #load the accuracy and losses of specific model
    train_acc = train_accuracys[m-1]
    test_acc = test_accuracys[m-1]
    print("The average train accuracy of model {} is:{}%".format(names[m-1],torch.Tensor(train_acc).mean()*100))
    #as in the test.py, to make sure the test.py run fast enough, we set the iteration to 1, so there is no std
    #print("The average train std of model {} is:{}%".format(names[m-1],torch.Tensor(train_acc).std()*100))
    print("The average test accuracy of model {} is:{}%".format(names[m-1],torch.Tensor(test_acc).mean()*100))
    #print("The average test std of model {} is:{}%".format(names[m-1],torch.Tensor(test_acc).std()*100))
    