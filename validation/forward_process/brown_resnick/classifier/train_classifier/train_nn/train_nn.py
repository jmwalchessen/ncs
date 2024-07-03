import torch
import torch.optim as optim
import numpy as np
from nn_architecture import *
from losses import *
from append_directories import *
import sys
import os
train_classifier_folder = append_directory(2)
sys.path.append(train_classifier_folder + "/generate_data")
import data_generation_on_the_fly
import os
import sys





def train_nn(num_epochs, data_draws, samples_per_draw, eval_samples_per_draw, num_timesteps,
             betamin, betamax, seed_value, range_value, smooth_value, classifier, weight_decay,
             beta1, beta2, epsilon, loss_function, device, initial_learning_rate,
             batch_size, eval_batch_size):

    optimizer = optim.Adam(classifier.parameters(), lr=initial_learning_rate, betas=(beta1, beta2), eps=epsilon,
                           weight_decay=weight_decay)
    state = dict(optimizer=optimizer, loss_function = loss_function, classifier = classifier, step=0)
    eval_losses = []
    train_losses = []
    for data_draw in range(0, data_draws):

        train_dataloader, eval_dataloader = data_generation_on_the_fly.get_training_and_evaluation_image_datasets(samples_per_draw, eval_samples_per_draw,
                                                                               num_timesteps, betamin, betamax, seed_value, range_value,
                                                                               smooth_value, batch_size, eval_batch_size)
        
        for epoch in range(0, num_epochs):
            train_losses_per_epoch = []
            eval_losses_per_epoch = []
            train_iterator = iter(train_dataloader)
            eval_iterator = iter(eval_dataloader)
            while True:
                try:
                # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
                    batch = next(train_iterator)
                    # Execute one training step
                    loss = step_fn(state, batch, device)
                    train_losses_per_epoch.append(float(loss))
                except StopIteration:
                    train_losses.append((sum(train_losses_per_epoch)/len(train_losses_per_epoch)))
                    break

            eval_iterator =iter(eval_dataloader)
            while True:
                try:
                    eval_batch = next(eval_iterator)
                    eval_loss = step_fn(state, eval_batch, device)
                    eval_losses_per_epoch.append(float(eval_loss))
                except StopIteration:
                    eval_losses.append((sum(eval_losses_per_epoch)/len(eval_losses_per_epoch)))
                    break


            lr = learning_rate_schedule(initial_learning_rate, epoch)
            state['optimizer'] = optimizer = optim.Adam((state['classifier']).parameters(), lr=lr,
                                                    betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        
    return classifier, eval_losses, eval_train_losses

num_samples_per_draw = 1000
eval_num_samples_per_draw = 100
eval_train_batch_size = 100
eval_num_samples = 100
num_timesteps = 1000
betamin = 0.1
betamax = 20
seed_value =43234
range_value = 1.6
smooth_value = 1.6
num_epochs = 20
data_draws = 10
samples_per_draw = 1000
eval_samples_per_draw = 100
weight_decay = 0.001
beta1 = 0.9
beta2 = 0.999
initial_learning_rate = 2e-5
device = "cuda:0"
batch_size = 4
eval_batch_size = 50
epsilon = 1e-8
loss_function = torch.nn.CrossEntropyLoss()


classifier = (SmallCNNClassifier()).to(device)

classifier, eval_losses, eval_train_losses = train_nn(num_epochs, data_draws, samples_per_draw, eval_samples_per_draw, num_timesteps,
                                                      betamin, betamax, seed_value, range_value, smooth_value, classifier, weight_decay,
                                                      beta1, beta2, epsilon, loss_function, device, initial_learning_rate,
                                                      batch_size, eval_batch_size)

lossfig_name = "models/small_classifier/model1_range_1.6_smooth_1.6_epochs_" + str(num_epochs) + "_losses.png"
visualize_loss(num_epochs, eval_losses, eval_train_losses, lossfig_name)
torch.save(classifier.state_dict(), ("models/small_classifier/model1_range_1.6_smooth_1.6_epochs_" + str(num_epochs) + "_parameters.pth"))
