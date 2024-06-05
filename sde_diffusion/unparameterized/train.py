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

def visualize_loss(train_losses, eval_losses, figname):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.plot(train_losses, label = 'Train Loss')
    ax.plot(eval_losses, label = 'Eval Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(figname)


def train(config, data_draws, epochs_per_drawn_data, number_of_replicates,
          evaluation_number_of_replicates, batch_size, eval_batch_size, seed_value,
          variance, lengthscale, score_model_path, loss_path):
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
        train_dataloader, eval_dataloader = get_training_and_evaluation_dataset(number_of_replicates,
                                                                                evaluation_number_of_replicates,
                                                                                batch_size,
                                                                                eval_batch_size,
                                                                                seed_value,
                                                                                variance, lengthscale)
        
        for epoch in range(0, epochs_per_drawn_data):
            train_iterator = iter(train_dataloader)
            eval_iterator = iter(eval_dataloader)
            while True:
                try:
                    batch = get_next_batch(train_iterator, config)
                    loss = train_step_fn(state, batch)
                    train_losses.append(loss)
                    print(loss)
                except StopIteration:
                    break
                while True:
                    try:
                        eval_batch = get_next_batch(eval_iterator, config)
                        eval_loss = eval_step_fn(state, batch)
                        eval_losses.append(eval_loss)
                    except StopIteration:
                        break
    

    visualize_loss(train_losses, eval_losses, loss_path)
    torch.save(score_model.state_dict(), score_model_path)

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
score_model_path = "trained_score_models/vesde/model1_beta_min_max_01_20.pth"
loss_path = "trained_score_models/vespde/model1_beta_min_max_01_20_loss.png"

train(veconfig, data_draws, epochs_per_drawn_data, number_of_replicates,
      evaluation_number_of_replicates, batch_size, eval_batch_size, seed_value,
          variance, lengthscale, score_model_path, loss_path)