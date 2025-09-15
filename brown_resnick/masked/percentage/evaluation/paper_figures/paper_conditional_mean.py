import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from paper_figure_helper_functions import *
from matplotlib.patches import Rectangle

def visualize_conditional_mean_observed_and_diffusion(range_value, smooth_value, figname, n, model_name):

    diffusion_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    univariate_lcs_means = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    percentages = [.01,.05,.1,.25,.5]
    ref_numbers = [0,1,2,3,4]
    for i in range(0, 5):
        image_name = "ref_image" + str(ref_numbers[i])
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth_value) + "_4000_random" + str(percentages[i]))
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        file_name = "univariate_lcs_4000_neighbors_7_nugget_1e5"
        univariate_lcs_images = load_univariate_lcs_images(model_name, image_name, file_name)
        nrep = diffusion_images.shape[0]
        diffusion_means[i,:,:] = (np.mean(diffusion_images, axis = (0,1))).reshape((n,n))
        univariate_lcs_means[i,:,:] = (np.mean(univariate_lcs_images, axis = 0)).reshape((n,n))
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
            im = ax.imshow(univariate_lcs_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            ax.set_yticks(ticks = [8, 16, 24], labels = np.array([-5,0,5]), fontsize = 15)

        else:
            im = ax.imshow(diffusion_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
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

def visualize_conditional_mean_fcs_ncs(range_value, model_version, figname):

    nrep = 4000
    obs_numbers = [1,2,3,5,7]
    n = 32
    evaluation_folder = append_directory(2)
    fcs_images = np.zeros((len(obs_numbers),nrep,n,n))
    ref_images = np.zeros((len(obs_numbers),n,n))
    ncs_images = np.zeros((len(obs_numbers),nrep,n,n))
    fcs_means = np.zeros((len(obs_numbers),n,n))
    ncs_means = np.zeros((len(obs_numbers),n,n))
    masks = np.zeros((len(obs_numbers),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(obs_numbers)):
        fcs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/processed_log_scale_fcs_range_" + str(range_value)
                                     + "_smooth_1.5_nugget_1e5_obs_" + str(obs_numbers[i]) + "_" + str(nrep) + ".npy"))
        ref_images[i,:,:] = np.log(np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i]) + 
                                     "/ref_image" + str(int(range_value-1)) + "/ref_image.npy")))
        masks[i,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i]) + "/ref_image"
                                + str(int(range_value-1)) + "/mask.npy"))
        ncs_images[i,:,:,:] = (np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/diffusion/model" + str(model_version) + "_range_"
                                     + str(range_value) + "_smooth_1.5_" + str(nrep) + "_random.npy"))).reshape((nrep,n,n))
        ncs_means[i,:,:] = np.mean(ncs_images[i,:,:,:], axis = 0)
        fcs_means[i,:,:] = np.mean(fcs_images[i,:,:,:], axis = 0)

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):

        if(i < 5):
            im = ax.imshow(ref_images[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[(i % 5),:,:].astype(float)))
            observed_indices = np.argwhere(masks[(i%5),:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            im = ax.imshow(fcs_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6)
            observed_indices = np.argwhere(masks[(i%5),:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

        else:
            im = ax.imshow(ncs_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            observed_indices = np.argwhere(masks[(i%5),:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))


    cbar=ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    fig.text(x = .3, y = .95, s = "Conditional Mean Field", fontsize = 25)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_conditional_mean_fcs_ncs_transposed(range_value, model_version, figname):

    nrep = 4000
    obs_numbers = [1,2,3,5,7]
    n = 32
    evaluation_folder = append_directory(2)
    fcs_images = np.zeros((len(obs_numbers),nrep,n,n))
    ref_images = np.zeros((len(obs_numbers),n,n))
    ncs_images = np.zeros((len(obs_numbers),nrep,n,n))
    fcs_means = np.zeros((len(obs_numbers),n,n))
    ncs_means = np.zeros((len(obs_numbers),n,n))
    masks = np.zeros((len(obs_numbers),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(obs_numbers)):
        fcs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/processed_log_scale_fcs_range_" + str(range_value)
                                     + "_smooth_1.5_nugget_1e5_obs_" + str(obs_numbers[i]) + "_" + str(nrep) + ".npy"))
        ref_images[i,:,:] = np.log(np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i]) + 
                                     "/ref_image" + str(int(range_value-1)) + "/ref_image.npy")))
        masks[i,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i]) + "/ref_image"
                                + str(int(range_value-1)) + "/mask.npy"))
        ncs_images[i,:,:,:] = (np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/diffusion/model" + str(model_version) + "_range_"
                                     + str(range_value) + "_smooth_1.5_" + str(nrep) + "_random.npy"))).reshape((nrep,n,n))
        ncs_means[i,:,:] = np.mean(ncs_images[i,:,:,:], axis = 0)
        fcs_means[i,:,:] = np.mean(fcs_images[i,:,:,:], axis = 0)

    fig = plt.figure(figsize=(6,9))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 3),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    counter = -1
    for i, ax in enumerate(grid):

        if((i % 3) == 0):
            counter = counter + 1
            im = ax.imshow(ref_images[counter,:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[counter,:,:].astype(float)))
            observed_indices = np.argwhere(masks[counter,:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]))
        elif((i % 3) == 1):
            im = ax.imshow(fcs_means[counter,:,:], cmap='viridis', vmin = -2, vmax = 6)
            observed_indices = np.argwhere(masks[counter,:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]))

        else:
            im = ax.imshow(ncs_means[counter,:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            observed_indices = np.argwhere(masks[counter,:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]))


    cbar=ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    fig.text(0.57, .87, "LCS", fontsize = 15)
    fig.text(0.75, .87, "NCS", fontsize = 15)
    fig.text(0.13, .87, "Partially Obs.", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_conditional_mean_observed_and_diffusion_transposed(range_value, smooth_value, figname, n, model_name):

    diffusion_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    univariate_lcs_means = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    percentages = [.01,.05,.1,.25,.5]
    ref_numbers = [0,1,2,3,4]
    for i in range(0, 5):
        image_name = "ref_image" + str(ref_numbers[i])
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth_value) + "_4000_random" + str(percentages[i]))
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        file_name = "univariate_lcs_4000_neighbors_7_nugget_1e5"
        univariate_lcs_images = load_univariate_lcs_images(model_name, image_name, file_name)
        nrep = diffusion_images.shape[0]
        diffusion_means[i,:,:] = (np.mean(diffusion_images, axis = (0,1))).reshape((n,n))
        univariate_lcs_means[i,:,:] = (np.mean(univariate_lcs_images, axis = 0)).reshape((n,n))
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

        if((i % 3) == 0):
            counter = counter + 1
            im = ax.imshow(reference_images[counter,:,:], cmap='viridis', vmin = -2, vmax = 6,
                           alpha = (masks[counter,:,:].astype(float)))
            if(i == 12):
                ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])
        elif((i % 3) == 0):
            im = ax.imshow(univariate_lcs_means[counter,:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            if((i == 13)):
                ax.set_xticks(ticks = [7, 15, 23, 31], labels = np.array([-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])
            ax.set_yticks([])
        else:
            im = ax.imshow(diffusion_means[counter,:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            observed_indices = np.argwhere(masks[counter,:,:].reshape((n,n)) > 0)
            if(i == 13):
                ax.set_xticks(ticks = [7, 15, 23], labels = np.array([-5,0,5]), fontsize = 15)
            else:
                ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)

    for i,ax in enumerate(grid):
        if((i == 0) | (i== 6) | (i==12)):
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
        elif((i == 3) | (i == 9)):
            ax.set_yticks(ticks = [7, 15, 23], labels = np.array([-5,0,5]), fontsize = 15)

    cbar=ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    fig.text(0.44, .94, "LCS", fontsize = 15)
    fig.text(0.69, .94, "NCS", fontsize = 15)
    fig.text(0.1, .94, "Partially Obs.", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname, dpi = 500)



def visualize_conditional_mean_field_with_variables():

    range_value = 3.
    smooth_value = 1.5
    figname = "figures/br_percentage_conditional_mean_model4.png"
    n = 32
    model_name = "model4"
    visualize_conditional_mean_observed_and_diffusion(range_value, smooth_value, figname, n, model_name)


def visualize_conditional_mean_field_transposed_with_variables():

    range_value = 3.
    smooth_value = 1.5
    figname = "figures/br_percentage_conditional_mean_model4_transposed.png"
    n = 32
    model_name = "model4"
    visualize_conditional_mean_observed_and_diffusion_transposed(range_value, smooth_value, figname, n, model_name)

visualize_conditional_mean_field_transposed_with_variables()
