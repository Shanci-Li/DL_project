from helpers import LossMSE, generate_data
import torch
from activation import Relu, Tanh
from linear import Linear
from optimizer import SGD
from sequential import Sequential

# disable the auto_grad
torch.set_grad_enabled(False)

# initialize parameters
lr = {'eta': 0.001,
      'gamma': 0.5,
      'beta1': 0.9,
      'beta2': 0.999,
      'eps': 1.0e-8}
n = 1000
epochs = 500
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
# you can choose among 3 methods: 'sgd', 'sgd_momentum' and 'Adam'
optimizer = SGD(model, method='sgd')

model.fit(train_inputs, train_targets, test_inputs, test_targets,
          criterion, optimizer, epochs, batch_size, lr)
