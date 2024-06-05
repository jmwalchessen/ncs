import time
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from append_directories import *
data_generation_folder = (append_directory(3) + "/generate_data")
sys.path.append(data_generation_folder)
from twisted_diffusion_data_generation_functions import *
from generate_true_conditional_samples import *


def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - th.logsumexp(log_weights, dim=dim, keepdims=True)
    normalized_weights = th.exp(log_weights)
    return normalized_weights



def plot_weight_distribution(twisted_diffusion_model, mask, ref_image, n, particles, ess_threshold,
                             figname):
    
    finalsamples, partobs, fullobs, m, reindices_trace, ess_trace, log_w_trace, log_proposal_trace, log_potential_xt_trace, log_potential_xtp1_trace, log_p_trans_untwisted_trace = twisted_diffusion_samples_per_call(twisted_diffusion_model,particles,
                                                                                            mask, ref_image, n, ess_threshold)
    
    print(log_w_trace[(twisted_diffusion_model.T-1),:])
    fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (15,5))
    for i in range(0,2):
        for j in range(0, 5):
            timestep = 5*i+j
            log_w = log_w_trace[(twisted_diffusion_model.T-timestep),:]
            normalized_weights = normalize_log_weights(log_w, dim = 0)
            axs[i,j].hist(normalized_weights.numpy(), bins = 10)
            axs[i,j].set_title(("Timestep " + str(timestep)))
    
    plt.savefig(figname)


n = 32
device = "cuda:0"
mask = (th.bernoulli(input = .5*th.ones((1,n,n)), out = th.ones((1,n,n)))).to(device)
mask = mask.to(bool)
minX = minY = -10
maxX = maxY = 10
variance = .4
lengthscale = 1.6
number_of_replicates = 1
seed_value = 43234
particles = 16
ess_threshold = .5
timesteps = [i for i in range(0,10)]
ref_image = ((generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                          number_of_replicates, seed_value))[1]).reshape((1,n,n))
ref_image = th.from_numpy(ref_image).to(device)

for i in range(0, 10):
    timestep = 1
    figname = ("weight_distributions/normalized_weights_particles_" +
               str(particles) + "_var_type_original_ess_" + str(ess_threshold) + "_" + str(i) + ".png")
    plot_weight_distribution(twisted_diffusion, mask, ref_image, n, particles, ess_threshold, figname)
