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
from mcmc_interpolation_helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle


def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))


def visualize_local_conditional_simulation_marginal_density(ref_image_folder, univariate_lcs_file,
                                                            missing_index, n, figname):

    mask = np.load((data_generation_folder + "/" + ref_image_folder + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((data_generation_folder + "/" + ref_image_folder + "/ref_image.npy"))
    univariate_lcs_samples = np.load((data_generation_folder + "/" + ref_image_folder + "/lcs/univariate/" + univariate_lcs_file))
    matrix_missing_index = (int(missing_indices[missing_index] % n), int(missing_indices[missing_index] / n))
    univariate_lcs_marginal_density = univariate_lcs_samples[matrix_missing_index[1],matrix_missing_index[0]]

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

    mask[matrix_missing_index[1],matrix_missing_index[0]] = 1
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .8)
    rect = Rectangle(((matrix_missing_index[0]-.5), (matrix_missing_index[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    pdd = pd.DataFrame(np.log(univariate_lcs_marginal_density), columns = None)
    sns.kdeplot(data = pdd, palette=['blue'], ax = ax[1])
    ax[1].axvline(ref_image[matrix_missing_index[1],matrix_missing_index[0]], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['Univariate LCS'])
    plt.savefig(figname)
    plt.clf()


def visualize_lcs_bivariate_density(ref_image_folder, bivariate_lcs_file, missing_index1, missing_index2, n, nrep, figname):

    mask = np.load((data_generation_folder + "/" + ref_image_folder + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((data_generation_folder + "/" + ref_image_folder + "/ref_image.npy"))
    bivariate_lcs_density = np.load((data_generation_folder + "/" + ref_image_folder + "/lcs/bivariate/" + 
                                     bivariate_lcs_file + "_" + str(missing_index1) + "_" + str(missing_index2) + ".npy"))
    bivariate_lcs_density = np.log(bivariate_lcs_density)
    matrix_missing_index1 = (int(missing_indices[missing_index1] % n), int(missing_indices[missing_index1] / n))
    matrix_missing_index2 = (int(missing_indices[missing_index2] % n), int(missing_indices[missing_index2] / n))

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .8)
    sns.kdeplot(x = bivariate_lcs_density[:,0], y = bivariate_lcs_density[:,1],
                ax = ax[1])
    plt.axvline(ref_image[int(matrix_missing_index1[0]),int(matrix_missing_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_missing_index2[0]),int(matrix_missing_index2[1])], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['Bivariate LCS'])
    ax[1].set_xlim(-10,10)
    ax[1].set_ylim(-10,10)
    plt.show()
    #plt.savefig(figname)
    #plt.clf()

def produce_generated_and_univariate_lcs_marginal_density(ref_image_folder, n, missing_index,
                                                          ncs_file_name, univariate_lcs_file,
                                                          figname):

    mask = np.load((data_generation_folder + "/" + ref_image_folder + "/mask.npy"))
    ref_image = np.load((data_generation_folder + "/" + ref_image_folder + "/ref_image.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ncs_images = np.load((data_generation_folder + "/" + ref_image_folder + "/diffusion/" + ncs_file_name))
    univariate_lcs_samples = np.load((data_generation_folder + "/" + ref_image_folder + "/lcs/univariate/" + univariate_lcs_file))
    missing_true_index = missing_indices[(missing_index-1)]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    generated_marginal_density = ncs_images[:,0,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
    lcs_marginal_density = univariate_lcs_samples[:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]


    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)
    lcs_pdd = pd.DataFrame(lcs_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    sns.kdeplot(data = lcs_pdd, palette = ["purple"], ax = axs[1])
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], 
                color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(0,1.5)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    purple_patch = mpatches.Patch(color='purple')
    orange_patch = mpatches.Patch(color='orange')
    axs[1].legend(handles = [purple_patch, orange_patch], labels = ['univariate LCS', 'NCS'])
    plt.savefig((data_generation_folder + "/" + ref_image_folder + "/lcs/univariate/marginal_density/" + figname))
    plt.clf()


def produce_generated_and_lcs_bivariate_density(ref_image_folder, n, missing_index1, missing_index2,
                                                ncs_file_name, bivariate_lcs_file, figname, nrep):
    
    mask = np.load((data_generation_folder + "/" + ref_image_folder + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((data_generation_folder + "/" + ref_image_folder + "/ref_image.npy"))
    ncs_images = np.load((data_generation_folder + "/" + ref_image_folder + "/diffusion/" + ncs_file_name))
    bivariate_lcs_density = np.load((data_generation_folder + "/" + ref_image_folder + "/lcs/bivariate/" + 
                                     bivariate_lcs_file + "_" + str(missing_index1) + "_" + str(missing_index2) + ".npy"))
    if(bivariate_lcs_density.size == 1):
        pass
    else:

        bivariate_lcs_density = np.log(bivariate_lcs_density)
        matrix_missing_index1 = (int(missing_index1 % n), int(missing_index2 / n))
        matrix_missing_index2 = (int(missing_index2 % n), int(missing_index2 / n))
        bivariate_ncs_density = np.concatenate([(ncs_images[:,0,int(matrix_missing_index1[0]),int(matrix_missing_index1[1])]).reshape((nrep,1)),
                                            (ncs_images[:,0,int(matrix_missing_index2[0]),int(matrix_missing_index2[1])]).reshape((nrep,1))],
                                            axis = 1)


        #fig, ax = plt.subplots(1)
        #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
        fig, axs = plt.subplots(ncols = 2, figsize = (9,3.5))

        #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
        mask = mask.astype(float).reshape((n,n))
        im = axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
        axs[0].plot(matrix_missing_index1[1], matrix_missing_index1[0], "r+")
        axs[0].plot(matrix_missing_index2[1], matrix_missing_index2[0], "r+")
        plt.colorbar(im, shrink = .8)
        sns.kdeplot(x = bivariate_lcs_density[:,0], y = bivariate_lcs_density[:,1],
                    ax = axs[1], color = 'purple')
        sns.kdeplot(x = bivariate_ncs_density[:,0], y = bivariate_ncs_density[:,1],
                    ax = axs[1], color = 'orange')
        plt.axvline(ref_image[int(matrix_missing_index1[0]),int(matrix_missing_index1[1])], color='red', linestyle = 'dashed')
        plt.axhline(ref_image[int(matrix_missing_index2[0]),int(matrix_missing_index2[1])], color='red', linestyle = 'dashed')
        axs[1].set_title("Bivariate")
        axs[1].set_xlim(-10,10)
        axs[1].set_ylim(-10,10)
        #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
        #rlocation = (round(location[0],2), round(location[1],2))
        #axs[1].set_xlabel("location: " + str(rlocation))
        purple_patch = mpatches.Patch(color='purple')
        orange_patch = mpatches.Patch(color='orange')
        axs[1].legend(handles = [purple_patch, orange_patch], labels = ['bivariate LCS', 'NCS'])
        plt.savefig((data_generation_folder + "/" + ref_image_folder + "/lcs/bivariate/bivariate_density/" + figname))
        plt.clf()


"""
ref_image_folder = "data/model4/ref_image4"
nrep = 4000
neighbors = 7
p = .5
univariate_lcs_file = "univariate_lcs_" + str(nrep) + "_neighbors_" + str(neighbors) + "_nugget_1e5.npy"
n = 32
ncs_file_name = "model4_range_3.0_smooth_1.5_4000_random" + str(p) + ".npy"
for missing_index in range(0,1000):
    figname = ("univariate_lcs_" + str(nrep) + "neighbors_" + str(neighbors) + "_nugget_1e5_marginal_density_missing_index_"
                + str(missing_index) + ".png")

    produce_generated_and_univariate_lcs_marginal_density(ref_image_folder, n, missing_index,
                                                      ncs_file_name, univariate_lcs_file, figname)"""



ref_image_folder = "data/model4/ref_image0"
nrep = 4000
neighbors = 7
range_value = 3.0
bivariate_lcs_file = "bivariate_lcs_" + str(nrep) + "_neighbors_" + str(neighbors) + "_nugget_1e5"
n = 32
p = .01
ncs_file_name = "model4_range_" + str(range_value) + "_smooth_1.5_4000_random" + str(p) + ".npy"
bilcs_folder = (evaluation_folder + "/diffusion_generation/" + ref_image_folder
                 + "/lcs/bivariate")
filenames = [f for f in os.listdir(bilcs_folder) if os.path.isfile(os.path.join(bilcs_folder, f))]


for i in range(len(filenames)):

    figname = ("bivariate_lcs_" + str(nrep) + "neighbors_" + str(neighbors) + "_nugget_1e5_bivariate_density_missing_index_")
    produce_generated_and_lcs_bivariate_density(ref_image_folder, n,
                                                ncs_file_name, filenames[i], figname, nrep)