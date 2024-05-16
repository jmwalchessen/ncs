import torch as th
from losses import *
from train_functions import *
from nn_architecture import *


classifier = (CNNClassifier()).to(device)


initial_learning_rate = 2e-4
beta1 = .9
beta2 = .999
epsilon = 1e-8
weight_decay = 0
num_epochs = 10
optimizer = optim.Adam(classifier.parameters(), lr=initial_learning_rate)
state = dict(optimizer=optimizer, loss_function = torch.nn.functional.binary_cross_entropy,
             classifier = classifier, step=0)
eval_losses = []
eval_train_losses = []

train_dataloader, eval_dataloader, eval_train_dataloader = get_dataloaders()

