import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
evaluation_folder = append_directory(2)
data_generation_folder = (evaluation_folder + "/diffusion_generation")
sys.path.append(data_generation_folder)
sys.path.append(evaluation_folder)
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle


#index is assumed to be in i*n+j form where (i,j) is index of matrix
def index_to_spatial_location(minX, maxX, minY, maxY, n, index):

    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    
    xlocation = (X[index])[0]
    ylocation = (Y[index])[0]
    return (xlocation, ylocation)


def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))




def produce_pit_marginal_density(folder_name, pit_file, missing_index, figname):

    mask = np.load((folder_name + "/mask.npy"))
    pit_values = np.load((folder_name + "/" + pit_file + "_" + str(missing_index) + ".npy"))
    n = 32
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    axs[0].imshow(mask)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    axs[1].hist(pit_values, bins = 20)
    plt.savefig((folder_name + "/visualizations/" + figname + "_" + str(missing_index) + ".png"))

def produce_pit_marginal_density_with_variables():
    
    folder_name = (data_generation_folder + "/data/mcmc/mask1")
    pit_file = "pit_values_1000_range_1_smooth_1_neighbors_5_4000"
    missing_indices = [1,100,200,300,400,500,600,800,900,1000]
    figname = "pit_marginal_1000_range_1_smooth_1_neighbors_5_4000"
    for missing_index in missing_indices:
        produce_pit_marginal_density(folder_name, pit_file, missing_index, figname)

