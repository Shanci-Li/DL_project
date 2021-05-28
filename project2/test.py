from helpers import LossMSE, generate_data
import torch
from activation import Relu, Tanh
from linear import Linear
from optimizer import SGD
from sequential import Sequential

# disable the auto_grad
torch.set_grad_enabled(False)

# initialize parameters
lr = {'eta': 0.0005,
      'gamma': 0.5}
n = 1000
epochs = 300
batch_size = 20

# Generate dataset
train_inputs, train_targets = generate_data(n)
test_inputs, test_targets = generate_data(n)

# construct the pipeline
criterion = LossMSE()
model = Sequential(Linear(2, 25), Relu(),
                   Linear(25, 25), Relu(),
                   Linear(25, 25), Tanh(),
                   Linear(25, 1), Tanh())

# initialize optimizer and choose the method to update the gradients
# you can choose among 2 methods: 'sgd' and 'sgd_momentum'
optimizer = SGD(model, method='sgd_momentum')

print('optimizer: ', optimizer.method)
model.fit(train_inputs, train_targets, test_inputs, test_targets,
          criterion, optimizer, epochs, batch_size, lr)
