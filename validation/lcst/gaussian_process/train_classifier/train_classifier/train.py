import numpy as np
import torch
from train_functions import *
from nn_architecture import *

num_draws = 50
num_epochs = 100
initial_learning_rate = 2e-4
beta1 = .9
beta2 = .999
epsilon = 1e-8
weight_decay = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = (CNNClassifier()).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=initial_learning_rate)
state = {
    'classifier': classifier,
    'optimizer': optimizer,
    'loss_function': torch.nn.functional.binary_cross_entropy,
    'step': 0
}
variance = .4
lengthscale = 1.6
number_of_train_replicates = 10
number_of_eval_replicates = 5
train_first_class_seed_value = 23423
eval_first_class_seed_value = 23945

number_of_train_replicates_per_call = 5
number_of_eval_replicates_per_call = 5
train_calls = 2
eval_calls = 1
train_second_class_seed_values = [23423, 23945]
eval_second_class_seed_values = [5234]
num_scales = 1000
beta_min = 0.1
beta_max = 20
model_name = "model5_beta_min_max_01_20_random050_masks.pth"
p = .5
batch_size = 4
eval_batch_size = 100
train_and_evaluate(num_draws, num_epochs, state, classifier, beta1, beta2,
                   initial_learning_rate, epsilon, weight_decay, device, variance, lengthscale,
                   number_of_train_replicates, number_of_train_replicates_per_call,
                   train_calls, number_of_eval_replicates, number_of_eval_replicates_per_call,
                   eval_calls, train_first_class_seed_value, eval_first_class_seed_value,
                   train_second_class_seed_values, eval_second_class_seed_values,
                   num_scales, beta_min, beta_max, model_name, p, batch_size, eval_batch_size)