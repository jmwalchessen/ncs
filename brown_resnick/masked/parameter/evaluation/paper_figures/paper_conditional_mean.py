import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from paper_figure_helper_functions import *

def visualize_conditional_mean_observed_and_diffusion(variance, figname, n, model_name):

    diffusion_means = np.zeros((5,n,n))
    true_conditional_unobserved_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    minX = minY = -10
    maxX = maxY = 10
    range_values = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0, 5):
        image_name = "ref_image" + str(i)
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        nrep = diffusion_images.shape[0]
        diffusion_means[i,:,:] = (np.mean(diffusion_images, axis = (0,1))).reshape((n,n))
        reference_images[i,:,:] = ref_image
        masks[i,:,:] = mask

    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):

        if(i < 5):
            im = ax.imshow(reference_images[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[(i % 5),:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            im = ax.imshow(diffusion_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))


    ax.cax.colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_conditional_mean_observed_ncs_lcs(figname, n, model_name, lcs_file_name):

    diffusion_means = np.zeros((5,n,n))
    lcs_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    range_values = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0, 5):
        image_name = "ref_image" + str(i)
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        ncs_file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        diffusion_images = load_diffusion_images(model_name, image_name, ncs_file_name)
        lcs_images = load_univariate_lcs_images(model_name, image_name, lcs_file_name)
        nrep = diffusion_images.shape[0]
        diffusion_means[i,:,:] = (np.mean(diffusion_images, axis = (0,1))).reshape((n,n))
        lcs_means[i,:,:] = (np.mean(lcs_images, axis = 0)).reshape((n,n))
        reference_images[i,:,:] = ref_image
        masks[i,:,:] = mask

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):

        if(i < 5):
            im = ax.imshow(reference_images[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[(i % 5),:,:].astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
        elif(i < 10):

            im = ax.imshow(lcs_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = ((1-masks[(i % 5),:,:]).astype(float)))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            ax.set_yticks(ticks = [8, 16, 24], labels = np.array([-5,0,5]), fontsize = 15)

        else:
            im = ax.imshow(diffusion_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = ((1-masks[(i % 5),:,:]).astype(float)))
            if(( i % 2) == 0):
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks(ticks = [8, 16, 24], labels = np.array([-5,0,5]), fontsize = 15)
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)


    cbar=ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    fig.text(x = .3, y = .95, s = "Conditional Mean Field", fontsize = 25)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_conditional_mean_observed_and_diffusion_transposed(variance, figname, n, model_name):

    diffusion_means = np.zeros((5,n,n))
    lcs_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    range_values = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0, 5):
        image_name = "ref_image" + str(i)
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        ncs_file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        diffusion_images = load_diffusion_images(model_name, image_name, ncs_file_name)
        lcs_images = load_univariate_lcs_images(model_name, image_name, lcs_file_name)
        nrep = diffusion_images.shape[0]
        diffusion_means[i,:,:] = (np.mean(diffusion_images, axis = (0,1))).reshape((n,n))
        lcs_means[i,:,:] = (np.mean(lcs_images, axis = 0)).reshape((n,n))
        reference_images[i,:,:] = ref_image
        masks[i,:,:] = mask

    fig = plt.figure(figsize=(6,9))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 3),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    counter = -1
    for i, ax in enumerate(grid):
    
        if((i % 3)==0):
            counter = counter + 1
            im = ax.imshow(reference_images[counter,:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[counter,:,:].astype(float)))
            if(i == 12):
                ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])
        elif((i % 3) == 1):
            im = ax.imshow(lcs_means[counter,:,:], cmap='viridis', vmin = -2, vmax = 6)
            if(i == 13):
                ax.set_xticks(ticks = [7, 15, 23], labels = np.array([-5,0,5]), fontsize = 15)
            else:
                ax.set_xticks([])
        else:
            im = ax.imshow(diffusion_means[counter,:,:], cmap='viridis', vmin = -2, vmax = 6)
            if((i == 14)):
                ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])
            ax.set_yticks([])

    for i,ax in enumerate(grid):
        if((i == 0) | (i==6) | (i==12)):
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
        elif((i == 3) | (i == 9)):
            ax.set_yticks(ticks = [7, 15, 23], labels = np.array([-5,0,5]), fontsize = 15)

    cbar = ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    fig.text(0.44, .94, "LCS", fontsize = 15)
    fig.text(0.69, .94, "NCS", fontsize = 15)
    fig.text(0.1, .94, "Partially Obs.", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname, dpi = 500)

smooth = 1.5
model_name = "model4"
n = 32
smooth_value = 1.5
figname = "figures/br_parameter_conditional_mean_model4_random05.png"
lcs_file_name = "univariate_lcs_4000_neighbors_7_nugget_1e5"
visualize_conditional_mean_observed_ncs_lcs(figname, n, model_name, lcs_file_name)
figname = "figures/br_parameter_conditional_mean_model4_random05_transposed.png"
visualize_conditional_mean_observed_and_diffusion_transposed(smooth_value, figname, n, model_name)