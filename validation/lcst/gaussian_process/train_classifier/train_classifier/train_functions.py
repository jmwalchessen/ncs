import torch
from torch import *
from losses import *
from append_directories import *
classifier_folder = append_directory(2)
import sys
sys.path.append((classifier_folder + "/generate_data"))
import generate_first_and_second_class_data



def train_and_evaluation_step_per_epoch(epoch, state, classifier, beta1, beta2,
                                        initial_learning_rate, train_dataloader,
                                        eval_dataloader,
                                        epsilon, weight_decay, device):

    train_iterator = iter(train_dataloader)
    train_losses_per_epoch = []
    eval_losses_per_epoch = []
    while True:
        try:
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
            batch = next(train_iterator)
            # Execute one training step
            loss = step_fn(state, batch, device)
            train_losses_per_epoch.append(float(loss.detach().cpu().numpy()))
        except StopIteration:
            break

    eval_iterator = iter(eval_dataloader)
    optimizer = torch.optim.Adam(classifier.parameters(),
                           lr=learning_rate_schedule(initial_learning_rate, epoch),
                           betas=(beta1, beta2), eps=epsilon,
                           weight_decay=weight_decay)
    while True:
        try:
            eval_batch = next(eval_iterator)
            eval_loss = evaluate_loss(state, eval_batch, device)
            eval_losses_per_epoch.append(float(eval_loss.detach().cpu().numpy()))
            print(eval_loss)
        except StopIteration:
            break
    
    #not sure why these steps are necessary, will look at later
    lr = learning_rate_schedule(initial_learning_rate, epoch)
    state['optimizer'] = optimizer = torch.optim.Adam((state['classifier']).parameters(), lr=lr,
                                                 betas=(beta1, beta2), eps=epsilon,
                                                 weight_decay=weight_decay)
    train_loss = sum(train_losses_per_epoch)/len(train_losses_per_epoch)
    eval_loss = sum(eval_losses_per_epoch)/len(eval_losses_per_epoch)
    return train_loss, eval_loss

def train_and_evaluation_per_draw(num_epochs, state, classifier, beta1, beta2,
                                  initial_learning_rate, train_dataloader,
                                  eval_dataloader, epsilon, weight_decay, device):
    train_losses_per_draw = []
    eval_losses_per_draw = []
    for epoch in range(num_epochs):
        train_loss, eval_loss = train_and_evaluation_step_per_epoch(epoch, state, classifier, beta1, beta2,
                                                                    initial_learning_rate, train_dataloader,
                                                                    eval_dataloader, epsilon, weight_decay,
                                                                    device)
        train_losses_per_draw.append(train_loss)
        eval_losses_per_draw.append(eval_loss)
    return train_losses_per_draw, eval_losses_per_draw

def train_and_evaluate(num_draws, num_epochs, state, classifier, beta1, beta2,
                       initial_learning_rate,
                       epsilon, weight_decay, device, variance, lengthscale,
                       number_of_train_replicates, number_of_train_replicates_per_call,
                       train_calls, number_of_eval_replicates, number_of_eval_replicates_per_call,
                       eval_calls, train_first_class_seed_value, eval_first_class_seed_value,
                       train_second_class_seed_values, eval_second_class_seed_values,
                       num_scales, beta_min, beta_max, model_name, p, batch_size, eval_batch_size):
    train_losses = []
    eval_losses = []
    for draw in range(num_draws):

        train_dataloader, eval_dataloader = generate_first_and_second_class_data.get_training_and_evaluation_datasets(variance, lengthscale,
                                                                                                                      number_of_train_replicates,
                                                                                                                      number_of_train_replicates_per_call,
                                                                                                                      train_calls, number_of_eval_replicates,
                                                                                                                      number_of_eval_replicates_per_call,
                                                                                                                      eval_calls, train_first_class_seed_value,
                                                                                                                      eval_first_class_seed_value,
                                                                                                                      train_second_class_seed_values,
                                                                                                                      eval_second_class_seed_values,
                                                                                                                      num_scales, beta_min, beta_max, 
                                                                                                                      model_name, p, device, batch_size,
                                                                                                                      eval_batch_size)
        train_losses_per_draw, eval_losses_per_draw = train_and_evaluation_per_draw(num_epochs, state, classifier, beta1, beta2,
                                                                                    initial_learning_rate, train_dataloader,
                                                                                    eval_dataloader, epsilon, weight_decay, device)
        train_losses.extend(train_losses_per_draw)
        eval_losses.extend(eval_losses_per_draw)
    return train_losses, eval_losses

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
    'loss_function': torch.nn.CrossEntropyLoss(),
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
epoch = 0
train_dataloader, eval_dataloader = generate_first_and_second_class_data.get_training_and_evaluation_datasets(variance, lengthscale,
                                                                                                                      number_of_train_replicates,
                                                                                                                      number_of_train_replicates_per_call,
                                                                                                                      train_calls, number_of_eval_replicates,
                                                                                                                      number_of_eval_replicates_per_call,
                                                                                                                      eval_calls, train_first_class_seed_value,
                                                                                                                      eval_first_class_seed_value,
                                                                                                                      train_second_class_seed_values,
                                                                                                                      eval_second_class_seed_values,
                                                                                                                      num_scales, beta_min, beta_max, 
                                                                                                                      model_name, p, device, batch_size,
                                                                                                                      eval_batch_size)
train_and_evaluation_step_per_epoch(epoch, state, classifier, beta1, beta2,
                                        initial_learning_rate, train_dataloader,
                                        eval_dataloader,
                                        epsilon, weight_decay, device)