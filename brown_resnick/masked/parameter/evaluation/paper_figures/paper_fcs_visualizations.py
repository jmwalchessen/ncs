import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from append_directories import *


def visualie_fcs_multiple_ranges(evaluation_folder, range_values, smooth,
                                       figname, n, nrep):

    fcs = np.zeros((len(range_values),nrep,n,n))
    ncs_images = np.zeros((len(range_values),nrep,n,n))
    masks = np.zeros((len(range_values),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(range_values)):

        fcs[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/model4/ref_image" +
                                           str(i) + "/processed_fcs_range_" + str(range_values[i]) + "_smooth_"
                                           + str(smooth) + "_nugget_1e5_obs_7_10.npy"))
        ncs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/model4/ref_image"
                                       + str(i) + "/ncs_images_range_" + str(range_values[i]) +
                                       "_smooth_" + str(smooth) + "_10.npy"))
        masks[i,:,:] = np.load((evaluation_folder + "/fcs/data/model4/ref_image" + str(i) 
                                + "/mask.npy"))
        
        
    fcs[fcs != 0] = np.log(fcs[fcs != 0])

    for i, ax in enumerate(grid):
        if(i < 5):
            im = ax.imshow(fcs[i,1,:,:], cmap='viridis', vmin = -2, vmax = 6, alpha = (1-masks[i,:,:]))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            im = ax.imshow(ncs_images[(i%5),9,:,:], cmap='viridis', vmin = -2, vmax = 6, alpha = (1-masks[(i%5),:,:]))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    plt.tight_layout()
    ax.cax.colorbar(im)
    plt.savefig(figname)

def visualize_fcs_multiple_ranges_with_variables():

    evaluation_folder = append_directory(2)
    range_values = [1.,2.,3.,4.,5.]
    smooth = 1.5
    figname = "figures/br_parameter_fcs_vs_ncs_nugget_1e5_obs_7.png"
    n = 32
    nrep = 10
    visualie_fcs_multiple_ranges(evaluation_folder, range_values, smooth,
                                        figname, n, nrep)
