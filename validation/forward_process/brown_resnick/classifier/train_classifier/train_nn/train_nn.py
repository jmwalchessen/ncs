import torch
import torch.optim as optim
import numpy as np
from nn_architecture import *
from losses import *
from append_directories import *
import os
import sys


num_samples_per_draw = 10000
batch_size = 4
eval_batch_size = 2000
eval_train_batch_size = 2000
eval_num_samples = 1000



train_loader, eval_loader = get_training_and_evaluation_image_datasets(total_number_of_samples, eval_total_number_of_samples, 
                                                                       num_timesteps, betamin, betamax, seed_value, range_value,
                                                                       smooth_value, batch_size, eval_batch_size)

def train_nn(num_epochs, classifier, weight_decay, beta1, beta2, epsilon,
             loss_function, train_loader, eval_loader, device, 
             initial_learning_rate):

    optimizer = optim.Adam(classifier.parameters(), lr=initial_learning_rate, betas=(beta1, beta2), eps=epsilon,
                           weight_decay=weight_decay)
    state = dict(optimizer=optimizer, loss_function = loss_function, classifier = classifier, step=0)
    eval_losses = []
    eval_train_losses = []
    for epoch in range(0, num_epochs):
        train_iterator = iter(train_loader)
        while True:
            try:
            # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
                batch = next(train_iterator)
                # Execute one training step
                loss = step_fn(state, batch, device)
            except StopIteration:
                break

        eval_iterator =iter(eval_loader)
        while True:
            try:
                eval_batch = next(eval_iterator)
                eval_loss = step_fn(state, eval_batch, device)
                eval_losses.append(float(eval_loss.detach().cpu().numpy()))
                print(eval_loss)
            except StopIteration:
                break

        eval_train_iterator = iter(evaluation_train_dataloader)
        eval_train_batch = next(eval_train_iterator)
        eval_train_loss = step_fn(state, eval_train_batch, device)
        eval_train_losses.append(float(eval_train_loss.detach().cpu().numpy()))
        print(eval_train_loss)

        lr = learning_rate_schedule(initial_learning_rate, epoch)
        state['optimizer'] = optimizer = optim.Adam((state['classifier']).parameters(), lr=lr,
                                                    betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        
    return classifier, eval_losses, eval_train_losses

num_epochs = 240
weight_decay = 0.001
beta1 = 0.9
beta2 = 0.999
initial_learning_rate = 2e-5
device = "cuda:0"
batch_size = 512

classifier = (SmallCNNClassifier()).to(device)

classifier, eval_losses, eval_train_losses = train_nn(num_epochs = num_epochs, classifier = classifier,
                                                      weight_decay = weight_decay, beta1 = beta1, beta2 = beta2,
                                                      epsilon = 1e-8, loss_function = torch.nn.CrossEntropyLoss(),
                                                      train_loader = train_dataloader, eval_loader = eval_dataloader,
                                                      device = device, initial_learning_rate = initial_learning_rate)

lossfig_name = "models/small_classifier/model3_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_losses.png"
visualize_loss(num_epochs, eval_losses, eval_train_losses, lossfig_name)
torch.save(classifier.state_dict(), ("models/small_classifier/model3_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_parameters.pth"))