import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *

def visualize_conditional_mean_observed_and_fcs(figname, m):

    eval_folder = append_directory(2)
    fcs_folder = (eval_folder + "/fcs")
    n = 32
    nrep = 4000
    fcs_images = np.zeros((5,nrep,n,n))
    fcs_conditional_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    range_values = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0, 5):
        ref_folder = (fcs_folder + "/data/model4/obs" + str(m) + "/ref_image" + str(i))
        masks[i,:,:] = np.load((ref_folder + "/mask.npy"))
        fcs_images[i,:,:,:] = np.load((ref_folder + "/processed_log_scale_fcs_range_" + str(range_values[i]) +
                               "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + 
                               str(nrep) + ".npy"))
        fcs_conditional_means[i,:,:] = (np.mean(fcs_images[i,:,:,:], axis = (0))).reshape((n,n))
        reference_images[i,:,:] = np.log(np.load((ref_folder + "/ref_image.npy")))

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
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            im = ax.imshow(reference_images[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 15):
            im = ax.imshow(fcs_conditional_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))


    ax.cax.colorbar(im)
    plt.tight_layout()
    figname = (fcs_folder + "/data/model4/obs" + str(m)  + "/" + figname)
    plt.savefig(figname)


def visualize_conditional_mean_observed_fcs_ncs(figname, m):

    eval_folder = append_directory(2)
    fcs_folder = (eval_folder + "/fcs")
    n = 32
    nrep = 4000
    fcs_images = np.zeros((5,nrep,n,n))
    fcs_conditional_means = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    masks = np.zeros((5,n,n))
    range_values = [1.0,2.0,3.0,4.0,5.0]
    ncs_images = np.zeros((5,nrep,n,n))
    ncs_conditional_means = np.zeros((5,n,n))
    for i in range(0, 5):
        ref_folder = (fcs_folder + "/data/conditional/obs" + str(m) + "/ref_image" + str(i))
        masks[i,:,:] = np.load((ref_folder + "/mask.npy"))
        fcs_images[i,:,:,:] = np.load((ref_folder + "/processed_log_scale_fcs_range_" + str(range_values[i]) +
                               "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + 
                               str(nrep) + ".npy"))
        fcs_conditional_means[i,:,:] = (np.mean(fcs_images[i,:,:,:], axis = (0))).reshape((n,n))
        reference_images[i,:,:] = np.log(np.load((ref_folder + "/ref_image.npy")))
        if(i == 2):
            ncs_images[i,:,:,:] = np.load((ref_folder + "/diffusion/model5_range_3.0_smooth_1.5_4000_random.npy"))
            ncs_conditional_means[i,:,:] = (np.mean(ncs_images[i,:,:,:], axis = (0))).reshape((n,n))

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
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            im = ax.imshow(reference_images[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6,
                           )
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 15):
            im = ax.imshow(fcs_conditional_means[(i % 5),:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))


    ax.cax.colorbar(im)
    plt.tight_layout()
    figname = (fcs_folder + "/data/model4/obs" + str(m)  + "/" + figname)
    plt.savefig(figname)

m = 1
figname = "conditional_mean_field_obs" + str(m) + "_nugget_1e5_4000.png"
visualize_conditional_mean_observed_and_fcs(figname, m)