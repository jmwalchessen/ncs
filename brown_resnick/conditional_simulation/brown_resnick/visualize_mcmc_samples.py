import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid


def concatenate_conditional_simulations(folder_name, ref_image_name, mask_name, condsim_name, n, nrep):

    ref_image = (np.load((folder_name + "/" + ref_image_name))).reshape((1,n**2))
    mask = (np.load((folder_name + "/" + mask_name))).reshape((n**2))
    condsim = np.load((folder_name + "/" + condsim_name))
    #shape (nrep,unobs)

    conditional_images = np.zeros((nrep, n**2))
    conditional_images[:,mask == 0] = condsim
    conditional_images[:,mask == 1] = np.tile(ref_image[:,mask == 1], reps = (nrep,1))
    conditional_images = conditional_images.reshape((nrep,n,n))

    return conditional_images

def visualize_approx_cond_and_mean(ref_image, mask, conditional_images, n, figname):

    
    conditional_images = np.log(conditional_images)
    ref_image = np.log(ref_image)
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
                 vmin = -2, vmax = 3)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.5), (matrix_obs_indices[i,0]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("True")
        
        elif(i == 1):
            ax.imshow(conditional_images[0,:,:].reshape((n,n)), vmin = -2, vmax = 3)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.5), (matrix_obs_indices[i,0]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Approx")
        elif(i == 2):
            ax.imshow(conditional_images[1,:,:].reshape((n,n)), vmin = -2, vmax = 3)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.5), (matrix_obs_indices[i,0]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Approx")

        elif(i == 3):
            mcmc_mean = np.mean(conditional_images, axis = 0)
            ax.imshow(mcmc_mean.reshape((n,n)), vmin = -2, vmax = 3)
            for i in range(0,m):
                rect = Rectangle(((matrix_obs_indices[i,1]-.55), (matrix_obs_indices[i,0]-.55)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Conditional Mean")
    
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)


folder_name = "data/25_by_25/all/ref_image13"
ref_image_name = "ref_image.npy"
mask_name = "mask.npy"
condsim_name = "conditional_simulations_range_20_smooth_1_observed_5_20.npy"
n = 25
nrep = 20
ref_image = np.load((folder_name + "/" + ref_image_name))
mask = np.load((folder_name + "/" + mask_name))
figname = (folder_name + "/conditional_visualization_range_20_smooth_1_observed_5_0.png")
conditional_images = concatenate_conditional_simulations(folder_name, ref_image_name, mask_name, condsim_name, n, nrep)
visualize_approx_cond_and_mean(ref_image, mask, conditional_images, n, figname)