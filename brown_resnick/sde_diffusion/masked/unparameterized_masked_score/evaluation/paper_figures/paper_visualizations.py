import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *

def load_diffusion_images(model_name, image_name, file_name):

    eval_folder = append_directory(2)
    diffusion_images = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/diffusion/" + file_name + ".npy"))
    return diffusion_images

def load_mask(model_name, image_name):

    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/" + "mask.npy"))
    return mask

def load_reference_image(model_name, image_name):

    eval_folder = append_directory(2)
    ref_image = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy"))
    return ref_image

def load_observations(model_name, image_name, mask, n):

    eval_folder = append_directory(2)
    ref_image = np.load(eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy")
    observations = ref_image[(mask).astype(int) == 1]
    return observations

def produce_figure_name(model_name, image_name, fig_name):

    eval_folder = append_directory(2)
    figname = (eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name +
               "/paper_figures/" + fig_name)
    return figname

def concatenate_observed_and_kriging_sample(observed, conditional_unobserved_sample, mask, n):

    conditional_sample = np.zeros((n**2))
    observed_indices = np.argwhere(mask.flatten() == 1)
    missing_indices = np.argwhere(mask.flatten() == 0)
    m = observed.shape[0]
    conditional_sample[missing_indices] = conditional_unobserved_sample.reshape(((n**2-m),1))
    conditional_sample[observed_indices] = observed.reshape((m,1))
    conditional_sample = conditional_sample.reshape((n,n))
    return conditional_sample

def visualize_observed_and_diffusion(figname, n, model_name):

    diffusion_visualizations = np.zeros((5,n,n))
    reference_visualizations = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    percentages = [.01,.05,.1,.25,.5]
    ref_numbers = [0,1,2,3,4]
    for i in range(0, 5):
        image_name = "ref_image" + str(ref_numbers[i])
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_range_3.0_smooth_1.5_4000_random" + str(percentages[i]))
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        diffusion_visualizations[i,:,:] = (diffusion_images[20,:,:,:]).reshape((n,n))
        reference_visualizations[i,:,:] = ref_image
        masks[i,:,:] = mask

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):

        if(i < 5):
            im = ax.imshow(reference_visualizations[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[(i % 5),:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            im = ax.imshow(reference_visualizations[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

        else:
            im = ax.imshow(diffusion_visualizations[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    #plt.tight_layout()
    ax.cax.colorbar(im)
    plt.savefig(figname)

smooth = 1.5
range_value = 3.0
model_name = "model4"
n = 32
figname = "figures/br_percentage_visualization_model4.png"
visualize_observed_and_diffusion(figname, n, model_name)