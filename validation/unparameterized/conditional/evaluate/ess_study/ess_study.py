import time
import torch as th
import matplotlib.pyplot as plt
import os
import sys
from append_directories import *
data_generation_folder = (append_directory(3) + "/generate_data")
sys.path.append(data_generation_folder)
from twisted_diffusion_data_generation_functions import *
from generate_true_conditional_samples import *

n = 32
device = "cuda:0"
mask = (th.bernoulli(input = .5*th.ones((1,n,n)), out = th.ones((1,n,n)))).to(device)
mask = mask.to(bool)

def max_min_weight(log_w_trace):

    w_trace = th.exp(log_w_trace)
    sum_w_trace = th.sum(w_trace, dim = 1)
    sum_w_trace = th.transpose(sum_w_trace.repeat(particles,1), dim0 = 0, dim1 = 1)
    normalized_w = th.div(input = w_trace, other = sum_w_trace)
    normalized_w = th.nan_to_num(normalized_w, nan=0.0)
    w_min_trace = [th.min(normalized_w[i,:]) for i in range(0, (T+1))]
    w_max_trace = [th.max(normalized_w[i,:]) for i in range(0, (T+1))]
    return w_min_trace, w_max_trace




def plot_resampling_ess_w_trace(twisted_diffusion_model, mask, ref_image, n, particles, ess_threshold,
                             figname):

    finalsamples, partobs, fullobs, m, reindices_trace, ess_trace, log_w_trace = twisted_diffusion_samples_per_call(twisted_diffusion_model,particles,
                                                                                            mask, ref_image, n, ess_threshold)
    sampled_particles = [th.unique(reindices_trace[i,:]).shape[0] for i in range(0,twisted_diffusion_model.T)]
    fig, ax = plt.subplots(ncols = 3, figsize = (15,5))
    ax[0].plot([i for i in range(twisted_diffusion_model.T, 0, -1)], sampled_particles)
    ax[0].set_xlabel("Timestep")
    ax[0].set_ylabel("Unique Resampled Particles")
    ax[0].set_title("Resampled Particles per timestep (ess threshold = " + str(ess_threshold) +")")
    ax[0].xaxis.set_inverted(True)
    ax[0].set_ylim(0, (particles + 1))
    ax[1].plot([i for i in range((twisted_diffusion_model.T+1),0,-1)], ess_trace)
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("ESS")
    ax[1].set_title("ESS per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[1].set_ylim(0, (particles+1))
    ax[1].xaxis.set_inverted(True)
    min_w_trace, max_w_trace = max_min_weight(log_w_trace)
    ax[2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], min_w_trace, 'blue')
    ax[2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], max_w_trace, 'green')
    ax[2].set_xlabel("Timestep")
    ax[2].set_ylabel("Weight")
    ax[2].set_title("Weight Trace per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[2].set_ylim(0, 1)
    ax[2].xaxis.set_inverted(True)
    ax[2].legend(['min', 'max'])
    plt.savefig(figname)
    plt.clf()

minX = -10
maxX = 10
minY = -10
maxY = 10
variance = .4
lengthscale = 1.6
number_of_replicates = 1
seed_value = 43234
particles = 32
ess_threshold = 0
ref_image = ((generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                          number_of_replicates, seed_value))[1]).reshape((1,n,n))
ref_image = th.from_numpy(ref_image).to(device)

plot_resampling_ess_w_trace(twisted_diffusion, mask, ref_image, n, particles, ess_threshold,
                            ("resampled_indices/ess_particles_" + str(particles) + "_ess_" + 
                             str(ess_threshold) + ".png"))
    
