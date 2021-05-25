from helpers import Module, LossMSE
import torch
import math
from torch import FloatTensor, LongTensor, Tensor
from activation import Relu, Tanh
from linear import Linear
from optimizor import SGD
from sequential import Sequential
from matplotlib import pyplot as plt


def generate_data(n=1000):
    inputs = torch.empty(n, 2).uniform_(0, 1)
    center = Tensor([0.5, 0.5]).view(1, -1)
    distances = torch.norm((inputs - center).abs(), 2, 1)
    labels = (distances < 1 / math.sqrt(2 * math.pi)).type(LongTensor)
    return inputs, labels


train_inputs, train_targets = generate_data(1000)

# plt.scatter(inputs[:,0], inputs[:,1],c=labels)
# plt.show()

lr, nb_epochs, batch_size = 1e-1, 10, 100
model = Sequential(Linear(2, 25),
                   Relu(),
                   Linear(25, 25),
                   Relu(),
                   Linear(25, 25),
                   Tanh(),
                   Linear(25, 1),
                   Tanh())

optimizer = SGD()
criterion = LossMSE()

# mu, std = train_input.mean(), train_input.std()
# train_input.sub_(mu).div_(std)
for e in range(nb_epochs):
    for inputs, targets in zip(train_inputs.split(batch_size), train_targets.split(batch_size)):
        predictions = model.forward(inputs)
        loss = criterion.calculate_mse(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.update_grad()
