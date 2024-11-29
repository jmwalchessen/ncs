import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from brown_resnick_data_generation import *

def visualize_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax):


    seed_value = int(np.random.randint(0, 100000))
    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 4),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    number_of_replicates = 8
    n = 32
    if(process_type == "schlather"):

        ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))

    elif(process_type == "brown"):
        ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

   
    for i, ax in enumerate(grid):

        im = ax.imshow(ref_img[i,:,:,:].reshape((n,n)), vmin = vmin, vmax = vmax)

    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_partially_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax, p):


    mask = (th.bernoulli(p*th.ones((1,1,32,32)))).float().numpy()
    seed_value = int(np.random.randint(0, 100000))
    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 4),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    number_of_replicates = 8
    n = 32
    if(process_type == "schlather"):

        ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))

    elif(process_type == "brown"):
        ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

   
    for i, ax in enumerate(grid):

        im = ax.imshow(ref_img[i,:,:,:].reshape((n,n)), vmin = vmin, vmax = vmax, alpha = mask.reshape((n,n)))

    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)

range_value = 1.6
smooth_value = 1.6
process_type = "brown"
figname = "visualizations/true/observed_brown_resnick_range_" + str(range_value) + "_smooth_" + str(smooth_value) + ".png"
vmin = -2
vmax = 4
p = .25
visualize_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax)
visualize_partially_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax, p)