from helpers import Module, LossMSE
import torch
import math
from torch import FloatTensor, LongTensor, Tensor
from activation import Relu, Tanh
from linear import Linear
from optimizor import SGD
from sequential import Sequential
from matplotlib import pyplot as plt


def generate_data(n):
    inputs = torch.empty(n, 2).uniform_(0, 1)
    center = Tensor([0.5, 0.5]).view(1, -1)
    distances = torch.norm((inputs - center).abs(), 2, 1)
    labels = (distances < 1 / math.sqrt(2 * math.pi)).type(LongTensor)
    return inputs, labels


# plt.scatter(inputs[:,0], inputs[:,1],c=labels)
# plt.show()

model = Sequential(Linear(2, 25),
                   Relu(),
                   Linear(25, 25),
                   Relu(),
                   Linear(25, 25),
                   Tanh(),
                   Linear(25, 1),
                   Tanh())

# mu, std = train_input.mean(), train_input.std()
# train_input.sub_(mu).div_(std)

def train(model, train_inputs , train_targets, epochs, lr):
    print("------------Training---------------")
    optimizer = SGD(model.param(), lr)
    criterion = LossMSE()
    l = []
    batch_size = 100
    k = epochs/10
    for e in range(epochs):
        for b in range(0, train_inputs.size(0), batch_size):
            predictions = model.forward(train_inputs.narrow(0, b, batch_size))
            loss = criterion.calculate_mse(predictions, train_targets.narrow(0, b, batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.update_grad()
        if e % k == 0:
            print('Epochs ', e, ': Loss ', loss.item())
            l.append(loss.loss.item())
    print("\n")
    return l


def count_errors(model, data_input, data_target):
    '''
    Compute number of mis-classified data
    '''
    error_count = 0
    data_output = model.forward(data_input)

    _, target_classes = torch.max(data_target.data, 1)
    _, predicted_classes = torch.max(data_output.data, 1)

    for k in range(data_input.size()[0]):
        if target_classes.data[k] != predicted_classes[k]:
            error_count = error_count + 1
    return error_count

lr = 0.01
n = 1000
epochs = 300
train_inputs, train_targets = generate_data(n)
test_inputs, test_targets = generate_data(n)

l = train(model, train_inputs, train_targets, epochs, lr)


print("---------------------- Error ---------------------")
count_train_errors = count_errors(model, train_inputs, train_targets)
count_test_errors = count_errors(model, test_inputs, test_targets)

print('Test error  {:0.2f}% {:d}/{:d}'.format((100 * count_test_errors) / test_inputs.size(0)))
print('Train error {:0.2f}% {:d}/{:d}'.format((100 * count_train_errors) / train_inputs.size(0)))
print("-----------\n")