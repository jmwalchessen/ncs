import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
import subprocess
from studentt_generation import *

home_folder = append_directory(5)
tsde_folder = (home_folder + "/studentnugget/masked/unparameterized")




def plot_unconditional_true_samples(tsamples, figname):

    fig = plt.figure(figsize=(20, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,4),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
 
    for i, ax in enumerate(grid):
        im = ax.imshow(tsamples[i,:,:], vmin = -4, vmax = 4)
        ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([0,20])
    fig.text(0.5, 0.9, 'Unconditional True', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)





def plot_unconditional_true_and_diffusion_samples(diffusion_samples, indices,
                                                  minX, maxX,
                                                  minY, maxY, n, variance,
                                                  lengthscale, df,
                                                  seed_value, figname):

    num_samples = 4
    diffusion_samples = diffusion_samples[indices,:,:,:]
    tvectors, tsamples = generate_student_nugget(minX, maxX, minY, maxY, n,
                                       variance, lengthscale, df,
                                       num_samples, seed_value)
    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,4),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    

    for i, ax in enumerate(grid):
        if(i > 3):
            im = ax.imshow(diffusion_samples[(i-4),:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Unconditional Diffusion", fontsize = 20)
        else:
            im = ax.imshow(tsamples[i,:,:].reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Unconditional True", fontsize = 20)

    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.set_ticks([])
    cbar.set_ticks([-2,0,2])
    #fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)

diffusion_samples = np.load("model5_beta_min_max_01_20_random0_1000.npy")
minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
df = 3
seed_value = 2349923
figname = "tunconditional_visualization_model5.png"
indices = np.array([50,67,90,120])
plot_unconditional_true_and_diffusion_samples(diffusion_samples, indices, minX, maxX,
                                                  minY, maxY, n, variance,
                                                  lengthscale, df, seed_value, figname)
