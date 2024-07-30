import torch
import torch.optim as optim
import numpy as np
from nn_architecture import *
from losses import *
from append_directories import *
unconditional_folder = append_directory(4)
data_generation_folder = (unconditional_folder + "/generate_data")
import os
import sys
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *
from dataloader_functions import *


classifier_folder = append_directory(2)
train_images_pathname = (classifier_folder + 
                            "/generate_data/data/diffusion/model5_unconditional_lengthscale_1.6_variance_0.4_100000.npy")
eval_images_pathname = (classifier_folder +
                        "/generate_data/data/diffusion/model5_unconditional_lengthscale_1.6_variance_0.4_eval_3000.npy.npy")
num_samples = 100000
batch_size = 64
eval_batch_size = 6000
eval_train_batch_size = 6000
eval_num_samples = 3000
crop_size = 2


train_dataloader = prepare_crop_and_create_dataloader(path = train_images_pathname, num_samples = num_samples,
                                                 minX = -10, maxX = 10, minY = -10, maxY = 10, n = 32,
                                                 variance = 0.4, lengthscale = 1.6, seed_value = 43234,
                                                 batch_size = batch_size, crop_size = crop_size)

eval_dataloader = prepare_crop_and_create_dataloader(path = eval_images_pathname, num_samples = eval_num_samples,
                                                minX = -10, maxX = 10, minY = -10, maxY = 10, n = 32,
                                                variance = 0.4, lengthscale = 1.6, seed_value = 43234,
                                                batch_size = eval_batch_size, crop_size = crop_size)

evaluation_train_dataloader = prepare_crop_and_create_dataloader(path = train_images_pathname,
                                                                 num_samples = num_samples, minX = -10,
                                                                 maxX = 10, minY = -10, maxY = 10, n = 32,
                                                                 variance = 0.4, lengthscale = 1.6, seed_value = 43234,
                                                                 batch_size = eval_train_batch_size,
                                                                 crop_size = crop_size)

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

lossfig_name = "models/small_classifier/model5_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_losses.png"
visualize_loss(num_epochs, eval_losses, eval_train_losses, lossfig_name)
torch.save(classifier.state_dict(), ("models/small_classifier/model5_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_parameters.pth"))