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

def visualize_observed_and_diffusion(variance, figname, n, model_name):

    diffusion_visualizations = np.zeros((5,n,n))
    true_visualizations = np.zeros((5,n,n))
    reference_visualizations = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    minX = minY = -10
    maxX = maxY = 10
    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0, 5):
        image_name = "ref_image" + str(i)
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        partial_field = np.multiply(mask, ref_image)
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) +
                     "_beta_min_max_01_20_random05_4000")
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        diffusion_visualizations[i,:,:] = (diffusion_images[20,:,:,:]).reshape((n,n))
        reference_visualizations[i,:,:] = ref_image
        masks[i,:,:] = mask
        conditional_unobserved_sample = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n,
                                                                        variance, lengthscales[i], y, 1)
        true_visualizations[i,:,:] = concatenate_observed_and_kriging_sample(y, conditional_unobserved_sample, mask, n)

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):

        if(i < 5):
            im = ax.imshow(reference_visualizations[(i % 5),:,:], cmap='viridis', vmin = -4, vmax = 4,
                           alpha = (masks[i,:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
        elif(i < 10):
            im = ax.imshow(true_visualizations[(i % 5),:,:], cmap='viridis', vmin = -4, vmax = 4)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [8, 16, 24], labels = np.array([-5,0,5]), fontsize = 15)
        else:
            im = ax.imshow(diffusion_visualizations[(i % 5),:,:], cmap='viridis', vmin = -4, vmax = 4)
            if((i % 2) == 0):
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks(ticks = [8, 16, 24], labels = np.array([-5,0,5]), fontsize = 15)
                ax.set_yticks(ticks = [8, 16, 24], labels = np.array([-5,0,5]), fontsize = 15)

    cbar = ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    fig.text(x = .4, y = .95, s = "Parameter U-Net", fontsize = 25)
    plt.tight_layout()
    plt.savefig(figname)

variance = 1.5
model_name = "model7"
image_name = "ref_image0"
n = 32
figname = "figures/gp_parameter_visualization_model7_random05.png"
visualize_observed_and_diffusion(variance, figname, n, model_name)