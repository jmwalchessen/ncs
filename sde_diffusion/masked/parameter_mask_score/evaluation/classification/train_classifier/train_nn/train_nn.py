import torch
import torch.optim as optim
import numpy as np
from nn_architecture import *
from losses import *
from append_directories import *
from dataloader_functions import *
classifier_folder = append_directory(2)
sys.path.append((classifier_folder + "/generate_data"))
from true_unconditional_data_generation import *




def train_nn(num_epochs, classifier, weight_decay, beta1, beta2, epsilon,
             loss_function, device, 
             initial_learning_rate, images_pathname, split, num_samples, batch_size,
             eval_batch_size, crop_size, shuffle = False):

    seed_value = int(np.random.randint(0, 1000000))
    train_loader, eval_loader, eval_train_loader = prepare_crop_and_create_dataloaders(path = images_pathname, split = split,
                                                 num_samples = num_samples, minX = -10, maxX = 10, minY = -10,
                                                 maxY = 10, n = 32, variance = 0.4, lengthscale = 1.6, seed_value = seed_value,
                                                 batch_size = batch_size, eval_batch_size = eval_batch_size,
                                                 crop_size = crop_size)

    optimizer = optim.Adam(classifier.parameters(), lr=initial_learning_rate, betas=(beta1, beta2), eps=epsilon,
                           weight_decay=weight_decay)
    state = dict(optimizer=optimizer, loss_function = loss_function, classifier = classifier, step=0)
    eval_losses = []
    eval_train_losses = []
    for epoch in range(0, num_epochs):
        eval_epoch_losses = []
        eval_train_epoch_losses = []
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
                eval_epoch_losses.append(float(eval_loss.detach().cpu().numpy()))
            except StopIteration:
                eval_losses.append(sum(eval_epoch_losses)/len(eval_epoch_losses))
                print(eval_losses)
                break

        eval_train_iterator = iter(eval_train_loader)
        while True:
            try:
                eval_train_batch = next(eval_train_iterator)
                eval_train_loss = step_fn(state, eval_train_batch, device)
                eval_train_epoch_losses.append(float(eval_train_loss.detach().cpu().numpy()))

            except StopIteration:
                eval_train_losses.append(sum(eval_train_epoch_losses)/len(eval_train_epoch_losses))
                break

        lr = learning_rate_schedule(initial_learning_rate, epoch)
        state['optimizer'] = optimizer = optim.Adam((state['classifier']).parameters(), lr=lr,
                                                    betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        
    return classifier, eval_losses, eval_train_losses

def train_nn_on_the_fly(num_epochs, classifier, weight_decay, beta1, beta2, epsilon,
                        loss_function, device, initial_learning_rate, images_pathname, train_starts,
                        train_ends, split, num_samples, batch_size, eval_batch_size, crop_size, shuffle = False):
    

    optimizer = optim.Adam(classifier.parameters(), lr=initial_learning_rate, betas=(beta1, beta2), eps=epsilon,
                           weight_decay=weight_decay)
    state = dict(optimizer=optimizer, loss_function = loss_function, classifier = classifier, step=0)
    eval_losses = []
    eval_train_losses = []
    for epoch in range(0, num_epochs):
        eval_epoch_losses = []
        eval_train_epoch_losses = []
        seed_value = int(np.random.randint(0, 1000000))
        train_loader, eval_loader, eval_train_loader = prepare_crop_and_create_dataloaders_on_the_fly(path = images_pathname,
                                                 train_start = train_starts[epoch], train_end = train_ends[epoch],
                                                 split = split, num_samples = num_samples, minX = -10, maxX = 10, minY = -10,
                                                 maxY = 10, n = 32, variance = 0.4, lengthscale = 1.6, seed_value = seed_value,
                                                 batch_size = batch_size, eval_batch_size = eval_batch_size,
                                                 crop_size = crop_size)
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
                eval_epoch_losses.append(float(eval_loss.detach().cpu().numpy()))
            except StopIteration:
                eval_losses.append(sum(eval_epoch_losses)/len(eval_epoch_losses))
                print(eval_losses)
                break

        eval_train_iterator = iter(eval_train_loader)
        while True:
            try:
                eval_train_batch = next(eval_train_iterator)
                eval_train_loss = step_fn(state, eval_train_batch, device)
                eval_train_epoch_losses.append(float(eval_train_loss.detach().cpu().numpy()))

            except StopIteration:
                eval_train_losses.append(sum(eval_train_epoch_losses)/len(eval_train_epoch_losses))
                break

        lr = learning_rate_schedule(initial_learning_rate, epoch)
        state['optimizer'] = optimizer = optim.Adam((state['classifier']).parameters(), lr=lr,
                                                    betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
        
    return classifier, eval_losses, eval_train_losses
#need to keep batch number low and learning rate low otherwise gradient jumps to only positive or negative solution
num_epochs = 500
weight_decay = 0.001
beta1 = 0.9
beta2 = 0.999
initial_learning_rate = 2e-5
device = "cuda:0"

classifier = (Small1CNNClassifier()).to(device)

images_pathname = (classifier_folder + 
                            "/generate_data/data/model6/unconditional/unconditional_images_variance_.4_lengthscale_1.6_100000.npy")
num_samples = 40000
split = 38000
batch_size = 128
eval_batch_size = 200
eval_train_batch_size = 200
crop_size = 4
train_starts = [i*1900 for i in range(0,20)]
train_ends = [i*1900 for i in range(1,21)]

classifier, eval_losses, eval_train_losses = train_nn(num_epochs = num_epochs, classifier = classifier,
                                                      weight_decay = weight_decay, beta1 = beta1, beta2 = beta2,
                                                      epsilon = 1e-8, loss_function = torch.nn.BCEWithLogitsLoss(),
                                                      device = device, initial_learning_rate = initial_learning_rate,
                                                      images_pathname = images_pathname, split = split, num_samples = num_samples,
                                                      batch_size = batch_size, eval_batch_size = eval_batch_size, crop_size = crop_size,
                                                      shuffle = False)

lossfig_name = "classifiers/classifier14/small1cnnclassifier_maxpool_classifier14_model6_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_losses.png"
visualize_loss(num_epochs, eval_losses, eval_train_losses, lossfig_name)
torch.save(classifier.state_dict(), ("classifiers/classifier14/small1cnnclassifier_maxpool_classifier_crop_4_model6_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_parameters.pth"))