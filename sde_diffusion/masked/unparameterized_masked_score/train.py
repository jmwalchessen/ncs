import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_generation_on_the_fly import *
from models.ema import ExponentialMovingAverage
from models.ncsnpp import *
import losses
import sde_lib
from configs.vp import ncsnpp_config as vp_ncsnpp_config
import matplotlib.pyplot as plt

def visualize_loss(epochs_and_draws, train_losses, eval_losses, figname):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.plot(epochs_and_draws, train_losses, label = 'Train Loss')
    ax.plot(epochs_and_draws, eval_losses, label = 'Eval Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(figname)



def train_per_mask(config, data_draws, epochs_per_drawn_data, number_of_replicates,
          evaluation_number_of_replicates, batch_size, eval_batch_size, seed_value,
          variance, lengthscale, mask, score_model_path, loss_path):
    """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
    # Initialize model.
    #score_model = mutils.create_model(config)
    score_model = (NCSNpp(config)).to(config.device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    initial_step = int(state['step'])
    eval_losses = []
    train_losses = []

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=config.model.num_scales)
    #vesde 
    else:
        sde = sde_lib.VESDE(sigma_min=0.01, sigma_max=50, N = config.model.num_scales)
        sampling_eps = 1e-3

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
    
    num_train_steps = config.training.n_iters
    for data_draw in range(0, data_draws):
        print(data_draw)
        #if ((epoch % 10) == 0):                    
        train_dataloader, eval_dataloader = get_training_and_evaluation_dataset_per_mask(number_of_replicates,
                                                                                         evaluation_number_of_replicates,
                                                                                         batch_size,
                                                                                         eval_batch_size,
                                                                                         seed_value,
                                                                                         variance, lengthscale,
                                                                                         mask)
        
        for epoch in range(0, epochs_per_drawn_data):
            print(epoch)
            train_iterator = iter(train_dataloader)
            eval_iterator = iter(eval_dataloader)
            train_losses_per_epoch = []
            while True:
                try:
                    batch = get_next_batch(train_iterator, config)
                    loss = train_step_fn(state, batch)
                    train_losses_per_epoch.append(loss)
                except StopIteration:
                    train_losses.append(float(np.mean(np.ndarray(train_losses_per_epoch))))
                    break
                while True:
                    try:
                        eval_batch = get_next_batch(eval_iterator, config)
                        eval_loss = eval_step_fn(state, eval_batch)
                        eval_losses.append(eval_loss)
                    except StopIteration:
                        break

    epochs_and_draws = [i for i in range(0, epochs_per_drawn_data*data_draws)]
    visualize_loss(epochs_and_draws, train_losses, eval_losses, loss_path)
    torch.save(score_model.state_dict(), score_model_path)

def train_per_multiple_random_masks(config, data_draws, epochs_per_drawn_data,
                             random_missingness_percentages,
                             number_of_random_replicates,
                             number_of_evaluation_random_replicates,
                             number_of_masks_per_image, number_of_evaluation_masks_per_image,
                             seed_values, variance, lengthscale, batch_size,
                             eval_batch_size, score_model_path, loss_path):
    
    # Initialize model.
    #score_model = mutils.create_model(config)
    score_model = nn.DataParallel((NCSNpp(config)).to(config.device))
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    initial_step = int(state['step'])
    eval_losses = []
    train_losses = []
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=config.model.num_scales)
    #vesde 
    else:
        sde = sde_lib.VESDE(sigma_min=0.01, sigma_max=50, N = config.model.num_scales)
        sampling_eps = 1e-3

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting,
                                        masked = True)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    masked = True)
    
    num_train_steps = config.training.n_iters
    for data_draw in range(0, data_draws):
        print(data_draw)

        train_dataloader, eval_dataloader = get_training_and_evaluation_random_mask_and_image_datasets(number_of_random_replicates, 
                                                                                                       random_missingness_percentages, 
                                                                                                       number_of_evaluation_random_replicates,
                                                                                                       number_of_masks_per_image,
                                                                                                       number_of_evaluation_masks_per_image,
                                                                                                       batch_size, eval_batch_size, variance,
                                                                                                       lengthscale, seed_values[data_draw])        
        
        
        for epoch in range(0, epochs_per_drawn_data):
            #want to iterate over the same masks and images for each epoch (taking epectation with respect to p(X,M)=p(X)p(M))
            train_losses_per_epoch = []
            eval_losses_per_epoch = []
            train_iterator = iter(train_dataloader)
            eval_iterator = iter(eval_dataloader)
            #train for this epoch, then do eval
            while True:
                try:
                    batch = get_next_batch(train_iterator, config)
                    loss = train_step_fn(state, batch)
                    train_losses_per_epoch.append(float(loss))
                except StopIteration:
                    train_losses.append((sum(train_losses_per_epoch)/len(train_losses_per_epoch)))
                    break

            while True:
                try:
                    batch = get_next_batch(eval_iterator, config)
                    eval_loss = eval_step_fn(state, batch)
                    print(eval_loss)
                    eval_losses_per_epoch.append(float(eval_loss))
                except StopIteration:
                    eval_losses.append((sum(eval_losses_per_epoch)/len(eval_losses_per_epoch)))
                    break



    torch.save(score_model.state_dict(), score_model_path)
    epochs_and_draws = [i for i in range(0, len(train_losses))]
    visualize_loss(epochs_and_draws, train_losses, eval_losses, loss_path)


def train_per_multiple_random_masks_revised_data_generation(config, data_draws, epochs_per_drawn_data,
                             random_missingness_percentages,
                             number_of_random_replicates,
                             number_of_evaluation_random_replicates,
                             number_of_masks_per_image, number_of_evaluation_masks_per_image,
                             seed_values_list, variance, lengthscale, batch_size,
                             eval_batch_size, score_model_path, loss_path):
    
    # Initialize model.
    #score_model = mutils.create_model(config)
    score_model = nn.DataParallel((NCSNpp(config)).to(config.device))
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    initial_step = int(state['step'])
    eval_losses = []
    train_losses = []
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=config.model.num_scales)
    #vesde 
    else:
        sde = sde_lib.VESDE(sigma_min=0.01, sigma_max=50, N = config.model.num_scales)
        sampling_eps = 1e-3

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting,
                                        masked = True)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    masked = True)
    
    num_train_steps = config.training.n_iters
    for data_draw in range(0, data_draws):
        print(data_draw)

        train_dataloader, eval_dataloader = get_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     variance, lengthscale, seed_values_list[data_draw])       
        
        
        for epoch in range(0, epochs_per_drawn_data):
            #want to iterate over the same masks and images for each epoch (taking epectation with respect to p(X,M)=p(X)p(M))
            train_losses_per_epoch = []
            eval_losses_per_epoch = []
            train_iterator = iter(train_dataloader)
            eval_iterator = iter(eval_dataloader)
            #train for this epoch, then do eval
            while True:
                try:
                    batch = get_next_batch(train_iterator, config)
                    loss = train_step_fn(state, batch)
                    train_losses_per_epoch.append(float(loss))
                except StopIteration:
                    train_losses.append((sum(train_losses_per_epoch)/len(train_losses_per_epoch)))
                    break

            while True:
                try:
                    batch = get_next_batch(eval_iterator, config)
                    eval_loss = eval_step_fn(state, batch)
                    print(eval_loss)
                    eval_losses_per_epoch.append(float(eval_loss))
                except StopIteration:
                    eval_losses.append((sum(eval_losses_per_epoch)/len(eval_losses_per_epoch)))
                    break



    torch.save(score_model.state_dict(), score_model_path)
    epochs_and_draws = [i for i in range(0, len(train_losses))]
    visualize_loss(epochs_and_draws, train_losses, eval_losses, loss_path)



vp_ncsnpp_configuration = vp_ncsnpp_config.get_config()
vpconfig = vp_ncsnpp_configuration

#see 5 million in total
"""
data_draws = 5
epochs_per_data_draws = 10
number_of_random_replicates = 10000
number_of_evaluation_random_replicates = 32
number_of_masks_per_image = 100
number_of_evaluation_masks_per_image = 10
#smaller p means less ones which means more observed values
random_missingness_percentages = [.4,.45,.5,.55,.6]
batch_size = 512
eval_batch_size = 32
variance = .4
lengthscale = 1.6
seed_values = [(int(np.random.randint(0, 100000)), int(np.random.randint(0, 100000))) for i in range(0, data_draws)]
score_model_path = "trained_score_models/vpsde/model4_beta_min_max_01_20_random010_channel_mask.pth"
loss_path = "trained_score_models/vpsde/model4_beta_min_max_01_20_random010_parameterized_mask_loss.png"
torch.cuda.empty_cache()
train_per_multiple_random_masks(vpconfig, data_draws, epochs_per_data_draws,
                             random_missingness_percentages,
                             number_of_random_replicates,
                             number_of_evaluation_random_replicates,
                             number_of_masks_per_image, number_of_evaluation_masks_per_image,
                             seed_values, variance, lengthscale, batch_size,
                             eval_batch_size, score_model_path, loss_path)"""

data_draws = 20
epochs_per_data_draws = 10
number_of_random_replicates = 1000
number_of_evaluation_random_replicates = 32
number_of_masks_per_image = 100
number_of_evaluation_masks_per_image = 10
#smaller p means less ones which means more observed values
random_missingness_percentages = [.4,.45,.5,.55,.6]
batch_size = 512
eval_batch_size = 32
variance = .4
lengthscale = 1.6
seed_values_list = [[(int(np.random.randint(0, 100000)), int(np.random.randint(0, 100000))) for j in range(0, len(random_missingness_percentages))] for i in range(0, data_draws)]
score_model_path = "trained_score_models/vpsde/model4_beta_min_max_01_20_random4060_channel_mask.pth"
loss_path = "trained_score_models/vpsde/model4_beta_min_max_01_20_random4060_parameterized_mask_loss.png"
torch.cuda.empty_cache()
train_per_multiple_random_masks_revised_data_generation(vpconfig, data_draws, epochs_per_data_draws,
                                                        random_missingness_percentages,
                                                        number_of_random_replicates,
                                                        number_of_evaluation_random_replicates,
                                                        number_of_masks_per_image, number_of_evaluation_masks_per_image,
                                                        seed_values_list, variance, lengthscale, batch_size,
                                                        eval_batch_size, score_model_path, loss_path)

