import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from rtopython_helper_functions import *
from matplotlib.patches import Rectangle

def plot_conditional_true_and_difussion_samples(mask, conditional_simulations, ref_image, n, figname):

    fig = plt.figure(figsize=(10, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,2),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    n = 32
    mask = mask.reshape((n,n))
    ref_image = ref_image.reshape((n,n))

    for i, ax in enumerate(grid):
        if(i == 0):
            im = ax.imshow(np.log(ref_image).reshape((n,n)),
                           alpha = mask, vmin = -2, vmax = 3)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Partial Obs.")
        elif(i==1):
            im = ax.imshow(np.log(ref_image).reshape((n,n)), vmin = -2, vmax = 3)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("True")
        else:

            observed_indices = np.argwhere(mask)
            for j in range(0, observed_indices.shape[0]):
                rect = Rectangle((observed_indices[j,1], observed_indices[j,0]), width=1.5, height=1.5, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            im = ax.imshow(np.log(conditional_simulations[(i-2),:,:]), vmin = -2, vmax = 3)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("MCMC Approx")

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,-1,0,1,2])
    fig.text(0.5, 0.975, 'Schlather range = 3, smooth = 1.6', ha='center', va='center', fontsize = 12)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)


