import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from matplotlib.patches import Rectangle



def visualize_unconditional_mean_observed_fcs_ncs(figname, m):

    eval_folder = append_directory(2)
    fcs_folder = (eval_folder + "/fcs")
    n = 32
    nrep = 4000
    fcs_images = np.zeros((5,nrep,n,n))
    fcs_unconditional_means = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    range_values = [1.0,2.0,3.0,4.0,5.0]
    ncs_images = np.zeros((5,nrep,n,n))
    true_images = np.zeros((5,nrep,n,n))
    ncs_unconditional_means = np.zeros((5,n,n))
    true_unconditional_means = np.zeros((5,n,n))
    for i in range(0, 5):
        ref_folder = (fcs_folder + "/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image" + str(i))
        masks[i,:,:] = (np.load((ref_folder + "/mask.npy"))).reshape((n,n))
        fcs_images[i,:,:,:] = np.log(np.load((ref_folder + "/processed_unconditional_fcs_fixed_mask_range_" + str(range_values[i]) +
                               "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy")))
        true_images[i,:,:,:] = np.log((np.load((ref_folder + "/true_brown_resnick_images_range_" + str(int(range_values[i])) + "_smooth_1.5_4000.npy")))).reshape((nrep,n,n))
        true_unconditional_means[i,:,:] = (np.mean(true_images[i,:,:,:], axis = (0))).reshape((n,n))
        fcs_unconditional_means[i,:,:] = (np.mean(fcs_images[i,:,:,:], axis = (0))).reshape((n,n))
        ncs_images[i,:,:,:] = (np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy"))).reshape((nrep,n,n))
        ncs_unconditional_means[i,:,:] = (np.mean(ncs_images[i,:,:], axis = (0))).reshape((n,n))

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i, ax in enumerate(grid):
        if(i < 5):
            observed_indices = np.argwhere(masks[(i%5),:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            im = ax.imshow(true_unconditional_means[(i % 5),:,:], cmap='viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            observed_indices = np.argwhere(masks[(i%5),:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            im = ax.imshow(fcs_unconditional_means[(i % 5),:,:], cmap='viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 15):
            observed_indices = np.argwhere(masks[(i%5),:,:].reshape((n,n)) > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            im = ax.imshow(ncs_unconditional_means[(i % 5),:,:], cmap='viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))


    ax.cax.colorbar(im)
    plt.tight_layout()
    figname = (fcs_folder + "/data/unconditional/fixed_locations/obs" + str(m)  + "/" + figname)
    plt.savefig(figname)

def visualize_unconditional_mean_observed_fcs_ncs_with_variables():
    
    for m in range(1,8):
        figname = "unconditional_mean_field_obs" + str(m) + "_nugget_1e5_4000.png"
        visualize_unconditional_mean_observed_fcs_ncs(figname, m)