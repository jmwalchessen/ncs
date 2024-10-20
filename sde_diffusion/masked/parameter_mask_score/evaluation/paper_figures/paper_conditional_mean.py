import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from generate_true_conditional_samples import *
from paper_figure_helper_functions import *

def concatenate_observed_and_kriging_sample(observed, conditional_unobserved_sample, mask, n):

    conditional_sample = np.zeros((n**2))
    observed_indices = np.argwhere(mask.flatten() == 1)
    missing_indices = np.argwhere(mask.flatten() == 0)
    m = observed.shape[0]
    conditional_sample[missing_indices] = conditional_unobserved_sample.reshape(((n**2-m),1))
    conditional_sample[observed_indices] = observed.reshape((m,1))
    conditional_sample = conditional_sample.reshape((n,n))
    return conditional_sample

def visualize_conditional_mean_observed_and_diffusion(variance, figname, n, model_name):

    diffusion_means = np.zeros((5,n,n))
    true_conditional_unobserved_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    minX = minY = -10
    maxX = maxY = 10
    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0, 5):
        image_name = "ref_image" + str(i)
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) + "_beta_min_max_01_20_random50_1000")
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        nrep = diffusion_images.shape[0]
        diffusion_means[i,:,:] = (np.mean(diffusion_images, axis = (0,1))).reshape((n,n))
        reference_images[i,:,:] = ref_image
        masks[i,:,:] = mask
        conditional_unobserved_samples = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n,
                                                                        variance, lengthscales[i], y, nrep)
        conditional_unobserved_mean = np.mean(conditional_unobserved_samples, axis = 0)
        true_conditional_unobserved_means[i,:,:] = concatenate_observed_and_kriging_sample(y, conditional_unobserved_mean, mask, n)

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):

        if(i < 5):
            im = ax.imshow(reference_images[(i % 5),:,:], cmap='viridis', vmin = -4, vmax = 4,
                           alpha = (masks[(i % 5),:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            im = ax.imshow(true_conditional_unobserved_means[(i % 5),:,:], cmap='viridis', vmin = -4, vmax = 4,
                           alpha = (1-masks[(i % 5),:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

        else:
            im = ax.imshow(diffusion_means[(i % 5),:,:], cmap='viridis', vmin = -4, vmax = 4,
                           alpha = (1-masks[(i % 5),:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))


    ax.cax.colorbar(im)
    plt.savefig(figname)

variance = 1.5
model_name = "model6"
n = 32
figname = "figures/gp_parameter_conditional_mean_model6.png"
visualize_conditional_mean_observed_and_diffusion(variance, figname, n, model_name)