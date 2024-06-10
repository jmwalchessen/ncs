import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_generation_on_the_fly import *
import scipy
from scipy.stats.qmc import LatinHypercube
from models.ema import ExponentialMovingAverage
from models.ncsnpp import *
import losses
import sde_lib
from configs.vp import ncsnpp_config as vp_ncsnpp_config
from configs.ve import ncsnpp_config as ve_ncsnpp_config
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

def train_per_multiple_masks(config, mask_draws, epochs_per_mask_draws,
                             number_of_mask_random_replicates,
                             number_of_eval_mask_random_replicates,
                             random_missingness_percentages,
                             number_of_image_replicates_per_mask, 
                             number_of_eval_image_replicates_per_mask,
                             mask_batch_size, eval_mask_batch_size, seed_value,
                             variance, lengthscale, image_batch_size,
                             eval_image_batch_size, score_model_path, loss_path):
    
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
                                        likelihood_weighting=likelihood_weighting,
                                        masked = True)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    masked = True)
    
    num_train_steps = config.training.n_iters
    for mask_draw in range(0, mask_draws):
        print(mask_draw)
        #if ((epoch % 10) == 0):                    
        train_mask_dataloader, eval_mask_dataloader = get_random_masking_training_and_evaluation_dataset(number_of_mask_random_replicates,
                                                                                                         random_missingness_percentages,
                                                                                                         number_of_eval_mask_random_replicates,
                                                                                                         mask_batch_size,
                                                                                                         eval_mask_batch_size)
        
        for epoch in range(0, epochs_per_drawn_data):
            train_mask_iterator = iter(train_mask_dataloader)
            eval_mask_iterator = iter(eval_mask_dataloader)
            train_losses_per_epoch = []
            eval_losses_per_epoch = []
            batch_masks_number = 0
            while True:
                try:
                    batch_masks = get_next_mask_batch(train_mask_iterator, config)
                    train_losses_per_batch_per_epoch = []
                    train_image_dataloader, eval_image_dataloader = get_training_and_evaluation_image_datasets_per_mask(number_of_image_replicates_per_mask,
                                                                                                                        number_of_eval_image_replicates_per_mask,
                                                                                                                        mask_batch_size, eval_mask_batch_size,
                                                                                                                        variance, lengthscale, seed_value,
                                                                                                                        image_batch_size, eval_image_batch_size)
                    train_image_iterator = iter(train_image_dataloader)
                    eval_image_iterator = iter(eval_image_dataloader)
                    while True:
                        try:
                            batch_images = get_next_image_batch(train_image_iterator, config)
                            batch = (batch_images, batch_masks)
                            print(batch_images.shape[0])
                            if((batch_images.shape[0]) == (batch_masks.shape[0])):
                                loss = train_step_fn(state, batch)
                                print(loss)
                                train_losses_per_batch_per_epoch.append(float(loss))
                        except StopIteration:
                            train_losses_per_epoch.append((sum(train_losses_per_batch_per_epoch)/len(train_losses_per_batch_per_epoch)))
                            break

                        while True:
                            try:
                                eval_batch_masks = get_next_mask_batch(eval_mask_iterator, config)
                                eval_losses_per_batch_per_epoch = []
                                while True:
                                    try:
                                        eval_batch_images = get_next_image_batch(eval_image_iterator, config)
                                        eval_batch = (eval_batch_images, eval_batch_masks)
                                        eval_loss = eval_step_fn(state, eval_batch)
                                        eval_losses_per_batch_per_epoch.append(float(eval_loss))
                                    except StopIteration:
                                        eval_losses_per_epoch.append((sum(eval_losses_per_batch_per_epoch)/len(eval_losses_per_batch_per_epoch)))
                                        break
                            except StopIteration:
                                #eval_losses.append((sum(eval_losses_per_epoch)/len(eval_losses_per_epoch)))
                                break
                except StopIteration:
                    train_losses.append(sum(train_losses_per_epoch)/len(train_losses_per_epoch))
                    eval_losses.append(sum(eval_losses_per_epoch)/len(eval_losses_per_epoch))
                    break

    torch.save(score_model.state_dict(), score_model_path)
    epochs_and_draws = [i for i in range(0, len(train_losses))]
    visualize_loss(epochs_and_draws, train_losses, eval_losses, loss_path)


vp_ncsnpp_configuration = vp_ncsnpp_config.get_config()
ve_ncsnpp_configuration = ve_ncsnpp_config.get_config()
vpconfig = vp_ncsnpp_configuration
veconfig = ve_ncsnpp_configuration
epochs_per_drawn_data = 20
data_draws = 10
number_of_replicates = 10000
evaluation_number_of_replicates = 1000
batch_size = 4
eval_batch_size = 1000
seed_value = 43234
variance = .4
lengthscale = 1.6
score_model_path = "trained_score_models/vpsde/model1_beta_min_max_01_20_center_mask.pth"
loss_path = "trained_score_models/vpsde/model1_beta_min_max_01_20_center_mask_loss.png"

mask = torch.zeros((1,1,32,32))
n = 32
mask[:, int(n/4):int(n/4*3), int(n/4):int(n/4*3)] = 1
mask.to("cuda:0")

"""
train_per_mask(vpconfig, data_draws, epochs_per_drawn_data, number_of_replicates,
      evaluation_number_of_replicates, batch_size, eval_batch_size, seed_value,
          variance, lengthscale, mask, score_model_path, loss_path)
"""
mask_draws = 2
epochs_per_mask_draws = 2
number_of_mask_random_replicates = 32
#total_number of masks = number_of_mask_random_replicates*len(random_missingness_percentages)
#total number of images per batch of masks = number_of_image_replicates_per_mask*batch_size 
number_of_eval_mask_random_replicates = 1
random_missingness_percentages = [.5]
number_of_image_replicates_per_mask = 4
number_of_eval_image_replicates_per_mask = 1
#mask batch size and eval mask batch size should match to get correct number of items
#in training_loss and eval_loss
mask_batch_size = 4
eval_mask_batch_size = 4
#image_batch_size needs to match mask_batch_size
image_batch_size = 4
eval_image_batch_size = 32
score_model_path = "trained_score_models/vpsde/model2_beta_min_max_01_20_random50_masks.pth"
loss_path = "trained_score_models/vpsde/model2_beta_min_max_01_20_random50_masks_loss.png"
train_per_multiple_masks(vpconfig, mask_draws, epochs_per_mask_draws,
                             number_of_mask_random_replicates,
                             number_of_eval_mask_random_replicates,
                             random_missingness_percentages,
                             number_of_image_replicates_per_mask, 
                             number_of_eval_image_replicates_per_mask,
                             mask_batch_size, eval_mask_batch_size, seed_value,
                             variance, lengthscale, image_batch_size,
                             eval_image_batch_size, score_model_path, loss_path)