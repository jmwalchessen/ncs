import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from helper_functions import *
from matplotlib.patches import Rectangle

def visualize_approx_conditional_and_mean(folder_name, conditional_image_file, n, figname, irep):

    mask = np.load((folder_name + "/mask.npy"))
    conditional_images = np.log(np.load((folder_name + "/mcmc/" + conditional_image_file)))
    ref_image = np.log(np.load((folder_name + "/ref_image.npy")))
    matrix_obs_indices = np.argwhere((mask.reshape((n,n))))
    m = matrix_obs_indices.shape[0]

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
    
    for i, ax in enumerate(grid):
        if(i == 0):
            im = ax.imshow(ref_image.reshape((n,n)),
                 vmin = -2, vmax = 6)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.5), (matrix_obs_indices[i,0]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("True")
        
        elif(i == 1):
            ax.imshow(conditional_images[irep,:,:].reshape((n,n)), vmin = -2, vmax = 6)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.5), (matrix_obs_indices[i,0]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Approx")
        elif(i == 2):
            ax.imshow(conditional_images[(irep+1),:,:].reshape((n,n)), vmin = -2, vmax = 6)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.5), (matrix_obs_indices[i,0]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Approx")

        elif(i == 3):
            mcmc_mean = np.mean(conditional_images, axis = 0)
            ax.imshow(mcmc_mean.reshape((n,n)), vmin = -2, vmax = 6)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.55), (matrix_obs_indices[i,0]-.55)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Conditional Mean")
    
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)


folder_name = "diffusion_generation/data/model3/ref_image1"
conditional_image_file = "approximate_conditional_images_range_10_smooth_1_4000.npy"
n = 32
for irep in range(0, 20, 1):
    figname = (folder_name + "/mcmc/visualizations/approximate_conditional_image_range_10_smooth_1_"
               + str(irep) + ".png")
    visualize_approx_conditional_and_mean(folder_name, conditional_image_file, n, figname, irep)