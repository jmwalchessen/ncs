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
             initial_learning_rate, diffusion_images_pathname, split, num_samples, batch_size,
             eval_batch_size, crop_size, shuffle = False, masks_path_name = None):

    seed_value = int(np.random.randint(0, 1000000))
    if(masks_path_name == None):
        train_loader, eval_loader, eval_train_loader = prepare_crop_and_create_dataloaders(diffusion_path = diffusion_images_pathname,
                                                 split = split, num_samples = num_samples, n = 32,
                                                 batch_size = batch_size, eval_batch_size = eval_batch_size,
                                                 crop_size = crop_size, shuffle = shuffle)
    else:
        train_loader, eval_loader, eval_train_loader = prepare_crop_and_create_dataloaders(diffusion_path = diffusion_images_pathname,
                                                 masks_pathname = masks_path_name, split = split,
                                                 num_samples = num_samples, n = 32,
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


#need to keep batch number low and learning rate low otherwise gradient jumps to only positive or negative solution
num_epochs = 500
weight_decay = 0.001
beta1 = 0.9
beta2 = 0.999
initial_learning_rate = 2e-5
device = "cuda:0"
model_number = "model2"
mask_number = "mask1"
diffusion_file_name = "train_conditional_diffusion_random50_variance_.4_lengthscale_1.6_model2_40000.npy"
true_file_name = "train_unconditional_true_variance_.4_lengthscale_1.6_40000.npy"
classifier = (CNNClassifier()).to(device)

diffusion_pathname = (classifier_folder + "/generate_data/data/fixed_mask/" + model_number
                      + "/" + mask_number + "/" + diffusion_file_name)
true_pathname = (classifier_folder + "/generate_data/data/fixed_mask/" + model_number
                      + "/" + mask_number + "/" + true_file_name)
num_samples = 40000
split = 38000
batch_size = 1024
eval_batch_size = 200
eval_train_batch_size = 200
crop_size = 2

classifier, eval_losses, eval_train_losses = train_nn(num_epochs = num_epochs, classifier = classifier,
                                                      weight_decay = weight_decay, beta1 = beta1, beta2 = beta2,
                                                      epsilon = 1e-8, loss_function = torch.nn.BCEWithLogitsLoss(),
                                                      device = device, initial_learning_rate = initial_learning_rate,
                                                      diffusion_images_pathname = diffusion_pathname,
                                                      true_images_pathname = true_pathname, split = split, num_samples = num_samples,
                                                      batch_size = batch_size, eval_batch_size = eval_batch_size, crop_size = crop_size,
                                                      shuffle = False)

lossfig_name = "classifiers/classifier1/classifier1_model2_random50_lengthscale_1.6_variance_.4_epochs_" + str(num_epochs) + "_losses.png"
visualize_loss(num_epochs, eval_losses, eval_train_losses, lossfig_name)
torch.save(classifier.state_dict(), ("classifiers/classifier1/model2_random50_lengthscale_1.6_variance_.4_epochs_" + str(num_epochs) + "_parameters.pth"))