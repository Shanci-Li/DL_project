import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import dlc_practical_prologue as prologue

   
"""
define the basic CNN model with two convolutional layers, two maxpooling layers and two full connected layers
"""
class CNN(nn.Module):
    def __init__(self, nb_hidden):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 5, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1) 
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(128, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)
    def forward(self,x):
        """
        Structure of CNN model:
        Input -> Convolution1 -> BatchnNorm1 -> Convolution2 -> BatchNorm2 -> Linear1 -> Linear2
        """
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1,128)))
        x = self.fc2(x)
        return x
    

"""
Train CNN model with cross entropy loss and SGD
    lr: stands for learning rate
    nb_epochs: stands for the number of epochs in a training process
"""
def train_CNN(model, train_input, train_classes, train_target, mini_batch_size = 100, lr = 0.01, nb_epochs = 25):
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    losses = []
    
    for e in range(nb_epochs):
        #print('-> epoch {0}'.format(e))
        train_loss = 0
        #train the model
        for b in range(0, train_input.size(0), mini_batch_size):
            target = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(target, train_target.narrow(0, b, mini_batch_size))
            train_loss = train_loss + loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        #print('Epoch: {} \tTraining Loss: {:.6f}'.format(e+1, train_loss))
        losses.append(train_loss)
    return losses

"""
compute the accuracy of the model
"""
def compute_nb_accuracy(model, test_input, test_target, mini_batch_size):
    nb_target_errors = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        test_output= model(test_input.narrow(0, b, mini_batch_size))
        _, predict_target = torch.max(test_output,1)
        for k in range(mini_batch_size):
            if test_target[b + k] != predict_target[k]:
                nb_target_errors += 1
    target_accuracy = (test_input.size(0)-nb_target_errors)/test_input.size(0)
    return target_accuracy

"""
initialization of weights in convolution and linear layer in the begining of every iteration
conform to a normal distribution with mean 0.0 std 0.01
"""
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        
"""
Function to normalize data input generated from the MNIST
"""
def data_normalization(input):
    input_mean = torch.mean(input)
    input_std = torch.std(input)
    input = input - input_mean
    input = input / input_std
    return input

"""
the function to evaluate the performance of specific model
parameters:
    train_iter: the number of iteration 
    size: the number of data in training and test set
    lr: learning rate
    nb_epochs: number of epochs in every iteration
return:
    losses_vals: lists of training losses returned from the function "train_CNN"
    train_accuray: lists of accuracy for the training set in each iteration
    test_accuracy: lists of accuracy for the test set in each iteration
"""
def evaluate_accuracy(model_class, train_iter = 20, size = 1000, mini_batch_size = 100, lr = 0.005, nb_epochs = 25):
    losses_vals = []
    train_accuracys = []
    test_accuracys = []
    for n in range(train_iter):
        print('>>> Training {0}'.format(n))
        # set the manual seed to make the result reproducible, to guarantee the randomness of data, change the seed every time
        #torch.manual_seed(n)
        model= model_class
        #apply the weight initialization in the begining of every iteration
        model.apply(weights_init)
        #generate_pair_sets generate different 1000 training and test set
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(size)
        #normalize train_Input and test_input
        #train_input = data_normalization(train_input)
        #test_input = data_normalization(test_input)
        train_input, train_target, train_classes = Variable(train_input), Variable(train_target), Variable((train_classes))
        test_input, test_target, test_classes = Variable(test_input), Variable(test_target), Variable(test_classes)
        #train the model
        model.train(True)
        losses = train_CNN(model, train_input, train_classes, train_target, mini_batch_size, lr, nb_epochs)
        losses_vals.append(losses)
        #compute the accuracy of the model
        model.train(False)
        train_accuracy = compute_nb_accuracy(model, train_input, train_target, mini_batch_size)
        test_accuracy = compute_nb_accuracy(model, test_input, test_target, mini_batch_size)
        train_accuracys.append(train_accuracy)
        test_accuracys.append(test_accuracy)
        print("Train accuracy:",train_accuracy)
        print("Test accuracy:",test_accuracy)
    return losses_vals, train_accuracys, test_accuracys
    