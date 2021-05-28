import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import dlc_practical_prologue as prologue

"""
define the CNN model with auxiliary loss
"""
class CNN_AL(nn.Module):
    def __init__(self, nb_hidden):
        super(CNN_AL, self).__init__()
        #convolutional layer and full connected layer for digit1 
        self.conv1a = nn.Conv2d(1, 16, 5, padding=1)  
        self.conv2a = nn.Conv2d(16, 32, 5, padding=1)
        self.bn1a = nn.BatchNorm2d(16)
        self.bn2a = nn.BatchNorm2d(32)
        self.fc1a = nn.Linear(128, nb_hidden) 
        self.fc2a = nn.Linear(nb_hidden, 10)
        
        #convolutional layer and full connected layer for digit2 
        self.conv1b = nn.Conv2d(1, 16, 5, padding=1)  
        self.conv2b = nn.Conv2d(16, 32, 5, padding=1)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2b = nn.BatchNorm2d(32)
        self.fc1b = nn.Linear(128, nb_hidden) 
        self.fc2b = nn.Linear(nb_hidden, 10)
        
        #full connected layer to process concatted result of two  digits
        self.fc3 = nn.Linear(20 , 64)
        self.fc4 = nn.Linear(64 , 2)
    def forward(self,x):
        """
        In auxiliary loss model, separate the original data to digit1 and digit2 first
        into separate network with the same structure:
        input -> Conv1 -> BatchNorm1 -> Conv2 -> BatchNorm2 -> Linear1 -> Linear2 -> digit1/2
        Concatenate the output from the second full connected layer together:
        concat_digit -> Linear3 -> Linear4 -> output
        Output:
        x_output:to compute the principal loss function
        digit1,digit2: to compute the auxiliary loss with the provided digit class
        """
        digit1 = x[:,:1,:,:] 
        digit2 = x[:,1:,:,:]
        
        digit1 = F.max_pool2d(F.relu(self.bn1a(self.conv1a(digit1))), kernel_size=2, stride=2)
        digit1 = F.max_pool2d(F.relu(self.bn2a(self.conv2a(digit1))), kernel_size=2, stride=2)
        digit1 = F.relu(self.fc1a(digit1.view(-1, 128)))
        digit1 = self.fc2a(digit1)
        
        digit2 = F.max_pool2d(F.relu(self.bn1b(self.conv1b(digit2))), kernel_size=2, stride=2)
        digit2 = F.max_pool2d(F.relu(self.bn2b(self.conv2b(digit2))), kernel_size=2, stride=2)
        digit2 = F.relu(self.fc1b(digit2.view(-1, 128)))
        digit2 = self.fc2b(digit2)
        #concat the output of digit1 and digit2
        x_output = torch.cat((digit1.clone(),digit2.clone()),1)
        x_output = F.relu(self.fc3(x_output))
        x_output = self.fc4(x_output)
        return x_output, digit1, digit2
    
"""
Train CNN_AL model with cross entropy loss and SGD
    lr: stands for learning rate
    nb_epochs: stands for the number of epochs in a training process
    l2: weight decay factor(L2 penalty) of SGD
    loss_ratio: the factor of auxiliary losses to compute the total training loss from three loss functions
    Learning rate scheduler(StepLR): decay the learning rate based on the number of epochs
"""    
def train_model(model, train_input, train_classes, train_target, mini_batch_size = 100, lr = 0.1, nb_epochs = 25, loss_ratio = 0.5,l2 = 1e-5):
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    digit1_class = train_classes[:,0]
    digit2_class = train_classes[:,1]
    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay = l2)
    # specify the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    losses = []
    for e in range(nb_epochs):
        #print('-> epoch {0}'.format(e))
        train_loss = 0
        
        #train the model
        for b in range(0, train_input.size(0), mini_batch_size):
            
            train_output, label1, label2 = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(train_output, train_target.narrow(0, b, mini_batch_size))
            loss1 = criterion(label1, digit1_class.narrow(0, b, mini_batch_size))
            loss2 = criterion(label2, digit2_class.narrow(0, b, mini_batch_size))
            
            #calculate the total loss
            loss_aux = loss + loss_ratio*(loss1 + loss2)
            
            optimizer.zero_grad()
            loss_aux.backward()

            train_loss = train_loss + loss_aux.item()
            optimizer.step()
        
        scheduler.step()
        #print('Epoch: {} \tTraining Loss: {:.6f}'.format(e, train_loss))
        losses.append(train_loss)
        
    return losses

"""
compute the accuracy of the model
"""
def compute_nb_accuracy(model, test_input, test_target, mini_batch_size):
    nb_target_errors = 0
 
    for b in range(0, test_input.size(0), mini_batch_size):
        test_output,_,_= model(test_input.narrow(0, b, mini_batch_size))
        _, predict_target = torch.max(test_output.data,1)
        for k in range(mini_batch_size):
            if test_target[b + k] != predict_target[k]:
                nb_target_errors += 1
    target_accuracy = (test_input.size(0)-nb_target_errors)/test_input.size(0)
    return target_accuracy

"""
initialization of weights in convolution and linear layer in the begining of every iteration
conform to a normal distribution with mean 0.0 std 0.1
"""
def weights_init(m):
    # initialization of weights in convolution and linear layer
    # conform to a normal distribution
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        m.weight.data.normal_(0.0, 0.1)
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
    loss_ratio: the factor of auxiliary losses to compute the total training loss from three loss functions
    l2: weight decay factor(L2 penalty) in optimizer
return:
    losses_vals: lists of training losses returned from the function "train_model"
    train_accuray: lists of accuracy for the training set in each iteration
    test_accuracy: lists of accuracy for the test set in each iteration
"""    
def evaluate_accuracy(model_class, train_iter = 20, size = 1000, mini_batch_size = 100, lr = 0.005, nb_epochs = 25, loss_ratio = 0.5 , l2 = 1e-5):
    losses_vals = []
    train_accuracys = []
    test_accuracys = []
    for n in range(train_iter):
        print('>>> Training {0}'.format(n))
        # set the manual seed to make the result reproducible, to guarantee the randomness of data, change the seed every time
        #torch.manual_seed(n+2020)
        model= model_class
        #apply the weight initialization in the begining of every iteration
        model.apply(weights_init)
        #generate_pair_sets generate different 1000 training and test set
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(size)
        #normalize train_Input and test_input
        train_input = data_normalization(train_input)
        test_input = data_normalization(test_input)
        train_input, train_target, train_classes = Variable(train_input), Variable(train_target), Variable((train_classes))
        test_input, test_target, test_classes = Variable(test_input), Variable(test_target), Variable(test_classes)
        #train the model
        model.train(True)
        losses = train_model(model, train_input, train_classes, train_target, mini_batch_size, lr, nb_epochs, loss_ratio, l2)
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