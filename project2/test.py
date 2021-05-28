from helpers import LossMSE, generate_data
import torch
from activation import Relu, Tanh, Sigmoid
from linear import Linear
from optimizer import SGD
from sequential import Sequential
import matplotlib.pyplot as plt

# disable the auto_grad
torch.set_grad_enabled(False)

# initialize parameters
lr = {'eta': 0.01,
      'gamma': 0.8}
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
                   Linear(25, 1), Sigmoid())

# initialize optimizer and choose the method to update the gradients
# you can choose among 2 methods: 'sgd' and 'sgd_momentum'
optimizer = SGD(model, method='sgd_momentum')

print('optimizer: ', optimizer.method)
train_loss, train_acc, test_loss, test_acc = model.fit(train_inputs, train_targets, test_inputs, test_targets,
                                                       criterion, optimizer, epochs, batch_size, lr, print_5epoch=True)

title = 'Learning curve of the model'
x_label = 'epoch'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(title)
ax1.set_ylabel('MSE')
ax1.set_xlabel(x_label)
ax2.set_ylabel('accuracy [% correct]')
ax2.set_xlabel(x_label)

ax1.plot(train_loss, label="training")
ax1.plot(test_loss, label="test")
ax2.plot(train_acc, label="training")
ax2.plot(test_acc, label="test")

ax1handles, ax1labels = ax1.get_legend_handles_labels()
if len(ax1labels) > 0:
    ax1.legend(ax1handles, ax1labels)

ax2handles, ax2labels = ax2.get_legend_handles_labels()
if len(ax2labels) > 0:
    ax2.legend(ax2handles, ax2labels)

fig.tight_layout()
plt.subplots_adjust(top=0.9)

plt.show()
