import numpy as np
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle
from append_directories import *


def visualize_mcmc_kriging_marginal_density(ref_image_name, mask_name, mcmc_file_name,
                                                       missing_index, n, figname):

    mask = np.load(mask_name)
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load(ref_image_name)
    mcmc_samples = np.load((mcmc_file_name + "_" + str(missing_index) + ".npy"))
    missing_index = missing_index - 1
    matrix_missing_index = (int(missing_indices[missing_index] % n), int(missing_indices[missing_index] / n))

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

    mask[matrix_missing_index[1],matrix_missing_index[0]] = 1
    print(ref_image[matrix_missing_index[1],matrix_missing_index[0]])
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .8)
    rect = Rectangle(((matrix_missing_index[0]-.5), (matrix_missing_index[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    pdd = pd.DataFrame(np.log(mcmc_samples), columns = None)
    sns.kdeplot(data = pdd, palette=['blue'], ax = ax[1])
    ax[1].axvline(ref_image[matrix_missing_index[1],matrix_missing_index[0]], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['MCMC Kriging'])
    plt.savefig(figname)
    plt.clf()

def visualize_mcmc_kriging_bivariate_density(ref_image_name, mask_name, mcmc_file_name, missing_index1,
                                     missing_index2, n, figname):
    
    mask = np.load(mask_name)
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load(ref_image_name)
    mcmc_samples1 = np.load((mcmc_file_name + "_" + str(missing_index1) + ".npy"))
    mcmc_samples2 = np.load((mcmc_file_name + "_" + str(missing_index2) + ".npy"))
    missing_index1 = missing_index1 - 1
    missing_index2 = missing_index2 - 1
    matrix_missing_index1 = (int(missing_indices[missing_index1] % n), int(missing_indices[missing_index1] / n))
    matrix_missing_index2 = (int(missing_indices[missing_index2] % n), int(missing_indices[missing_index2] / n))
    nrep = mcmc_samples1.shape[0]


    mcmc_bivariate_density = np.concatenate([mcmc_samples1.reshape((nrep,1)),
                                                  mcmc_samples2.reshape((nrep,1))],
                                                  axis = 1)
    mcmc_bivariate_density = np.log(mcmc_bivariate_density)
    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

    mask[matrix_missing_index1[1],matrix_missing_index1[0]] = 1
    print(ref_image[matrix_missing_index1[1],matrix_missing_index1[0]])
    mask[matrix_missing_index2[1],matrix_missing_index2[0]] = 1
    print(ref_image[matrix_missing_index2[1],matrix_missing_index2[0]])
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .9)
    rect = Rectangle(((matrix_missing_index1[0]-.5), (matrix_missing_index1[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    rect = Rectangle(((matrix_missing_index2[0]-.5), (matrix_missing_index2[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    sns.kdeplot(x = mcmc_bivariate_density[:,0], y = mcmc_bivariate_density[:,1],
                ax = ax[1], color = "blue", label = "MCMC")
    ax[1].set_xlim(-2,4)
    ax[1].set_ylim(-2,4)

    plt.axvline(ref_image[int(matrix_missing_index1[1]),int(matrix_missing_index1[0])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_missing_index2[1]),int(matrix_missing_index2[0])], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['MCMC Kriging'])
    plt.savefig(figname)
    plt.clf()



def visualize_approximate_conditional_marginal_density(folder_name, approx_cond_name,
                                                       missing_index, n, figname):

    mask = np.load((folder_name + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((folder_name + "/ref_image.npy"))
    conditional_images = np.load((folder_name + "/mcmc/approximate_conditional/" + approx_cond_name))
    matrix_missing_index = (int(missing_indices[missing_index] % n), int(missing_indices[missing_index] / n))
    marginal_density = conditional_images[:,matrix_missing_index[0],matrix_missing_index[1]]

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

    mask[matrix_missing_index[1],matrix_missing_index[0]] = 1
    print(ref_image[matrix_missing_index[1],matrix_missing_index[0]])
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .8)
    rect = Rectangle(((matrix_missing_index[0]-.5), (matrix_missing_index[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    pdd = pd.DataFrame(np.log(marginal_density), columns = None)
    sns.kdeplot(data = pdd, palette=['blue'], ax = ax[1])
    ax[1].axvline(ref_image[matrix_missing_index[1],matrix_missing_index[0]], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['MCMC'])
    figname = (folder_name + "/mcmc/approximate_conditional/marginal_density/" + figname + "_" + 
               str(missing_index) + ".png")
    plt.savefig(figname)
    plt.clf()

def visualize_multiple_approximate_conditional_marginal_density(folder_name, approx_cond_name,
                                                                missing_indices, n, figname):
    
    for missing_index in missing_indices:

        current_figname = (figname + "_" + str(missing_index) + ".npy")
        visualize_approximate_conditional_marginal_density(folder_name, approx_cond_name,
                                                       missing_index, n, figname)
    


def visualize_approximate_conditional_bivariate_density(folder_name, approx_cond_name, missing_index1,
                                                        missing_index2, n, figname):
    
    mask = np.load((folder_name + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.log(np.load((folder_name + "/ref_image.npy")))
    conditional_images = np.load((folder_name + "/mcmc/approximate_conditional/" + approx_cond_name))
    matrix_missing_index1 = (int(missing_indices[missing_index1] % n), int(missing_indices[missing_index1] / n))
    matrix_missing_index2 = (int(missing_indices[missing_index2] % n), int(missing_indices[missing_index2] / n))
    mcmc_samples1 = conditional_images[:,matrix_missing_index1[0],matrix_missing_index1[1]]
    mcmc_samples2 = conditional_images[:,matrix_missing_index2[0],matrix_missing_index2[1]]
    nrep = conditional_images.shape[0]


    mcmc_bivariate_density = np.concatenate([mcmc_samples1.reshape((nrep,1)),
                                                  mcmc_samples2.reshape((nrep,1))],
                                                  axis = 1)
    mcmc_bivariate_density = np.log(mcmc_bivariate_density)
    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

    mask[matrix_missing_index1[1],matrix_missing_index1[0]] = 1
    print(ref_image[matrix_missing_index1[1],matrix_missing_index1[0]])
    mask[matrix_missing_index2[1],matrix_missing_index2[0]] = 1
    print(ref_image[matrix_missing_index2[1],matrix_missing_index2[0]])
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .9)
    rect = Rectangle(((matrix_missing_index1[0]-.5), (matrix_missing_index1[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    rect = Rectangle(((matrix_missing_index2[0]-.5), (matrix_missing_index2[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    sns.kdeplot(x = mcmc_bivariate_density[:,0], y = mcmc_bivariate_density[:,1],
                ax = ax[1], color = "blue", label = "MCMC")
    ax[1].set_xlim(-2,4)
    ax[1].set_ylim(-2,4)

    plt.axvline(ref_image[int(matrix_missing_index1[1]),int(matrix_missing_index1[0])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_missing_index2[1]),int(matrix_missing_index2[0])], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['MCMC'])
    figname = (folder_name + "/mcmc/approximate_conditional/bivariate_density/" + figname + "_" + 
               str(missing_index1) + "_" + str(missing_index2) + ".png")
    plt.savefig(figname)
    plt.clf()


def produce_multiple_approximate_conditional_bivariate_density(folder_name, approx_cond_name, indices1,
                                                               indices2, n, figname):

    for missing_index1 in indices1:
        for missing_index2 in indices2:

            current_figname = (figname + "_" + str(missing_index1) +
                                "_" + str(missing_index2) + ".png")
            visualize_approximate_conditional_bivariate_density(folder_name, approx_cond_name,
                                                       missing_index1, missing_index2, n, current_figname)


n = 32
evaluation_folder = append_directory(2)
data_generation_folder = (evaluation_folder + "/diffusion_generation")
folder_name = (data_generation_folder + "/data/model3/ref_image2")
mask = np.load((folder_name + "/mask.npy"))
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
m = missing_indices.shape[0]
approx_cond_name = "approximate_conditional_images_range_11_smooth_1_4000.npy"
figname = "approximate_conditional_range_11_smooth_1"
visualize_multiple_approximate_conditional_marginal_density(folder_name, approx_cond_name,
                                                                missing_indices, n, figname)


