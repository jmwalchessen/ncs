import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from helper_functions import *




def visualize_sample(diffusion_sample, n):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(diffusion_sample.detach().cpu().numpy().reshape((n,n)), vmin = -2, vmax = 2)
    plt.show()

def visualize_observed_and_generated_samples(observed, mask, diffusion1, diffusion2, n, figname):

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    im = grid[0].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3)
    grid[0].set_title("Observed")
    grid[1].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3,
                   alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[1].set_title("Partially Observed")
    grid[2].imshow(diffusion1.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3)
    grid[2].set_title("Generated")
    grid[3].imshow(diffusion2.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3)
    grid[3].set_title("Generated")
    grid[0].cax.colorbar(im)
    plt.savefig(figname)

def visualize_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax):


    seed_value = int(np.random.randint(0, 100000))
    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 4),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    number_of_replicates = 8
    n = 32
    if(process_type == "schlather"):

        ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))

    elif(process_type == "brown"):
        ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

   
    for i, ax in enumerate(grid):

        im = ax.imshow(ref_img[i,:,:,:].reshape((n,n)), vmin = vmin, vmax = vmax)

    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)

def generate_visualization(process_type, range_value, smooth_value, p, model_name, i):

        seed_value = int(np.random.randint(0, 100000))
        number_of_replicates = 1
        n = 32
        device = "cuda:0"

        if(process_type == "schlather"):
            ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))
        else:
            ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

        unmasked_y = (th.from_numpy(ref_img)).to(device)
        mask = (th.bernoulli(p*th.ones((1,1,n,n)))).to(device)
        y = ((th.mul(mask, unmasked_y)).to(device)).float()
        num_samples = 2
        diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model,
                                                                    device, mask, y, n,
                                                                    num_samples)

        figname = ("visualizations/" + process_type + "/models/" + model_name + "/random" + str(p) + "_range_" + str(range_value) + 
        "_smooth_" + str(smooth_value) + "_observed_and_generated_samples_" + str(i) + ".png")
        visualize_observed_and_generated_samples(unmasked_y, mask, diffusion_samples[0,:,:,:],
                                            diffusion_samples[1,:,:,:], n, figname)

def generate_multiple_visualizations(process_type, range_value, smooth_value, p, model_name, multiples_beginning, multiples_end):

    for i in range(multiples_beginning, multiples_end):
        generate_visualization(process_type, range_value, smooth_value, p, model_name, i)


"""
process_type = "schlather"
model_name = "model4_beta_min_max_01_20_range_2.2_smooth_1.9_random025_log_parameterized_mask.pth"
mode = "eval"
score_model = load_score_model(process_type, model_name, mode)
beta_min = .1
beta_max = 20
N = 1000
sdevp = load_sde(beta_min = beta_min, beta_max = beta_max, N = N)
range_value = 2.2
smooth_value = 1.9
p = 0
model_name = "model4"
multiples_beginning = 0
multiples_end = 50
generate_multiple_visualizations(process_type, range_value, smooth_value, p, model_name, multiples_beginning, multiples_end)"""


range_value = .4
smooth_value = 1.6
process_type = "smith"
vmax = 6
vmin = -2
figname = "visualizations/smith/true/true_range_" + str(range_value) + "_smooth_" + str(smooth_value) + ".png" 
visualize_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax)