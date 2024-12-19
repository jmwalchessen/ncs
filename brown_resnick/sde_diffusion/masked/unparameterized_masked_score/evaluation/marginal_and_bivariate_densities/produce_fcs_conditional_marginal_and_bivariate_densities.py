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
fcs_folder = (evaluation_folder + "/fcs")
sys.path.append(fcs_folder)
sys.path.append(evaluation_folder)
from mcmc_interpolation_helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle



def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))


def visualize_fcs_conditional_marginal_density(ref_image_folder, fcs_file,
                                               missing_index, n, figname):

    mask = np.load((fcs_folder + "/" + ref_image_folder + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((fcs_folder + "/" + ref_image_folder + "/ref_image.npy"))
    fcs_images = np.load((fcs_folder + "/" + ref_image_folder + "/" + fcs_file))
    matrix_missing_index = (int(missing_indices[missing_index] / n), int(missing_indices[missing_index] % n))
    fcs_marginal_density = fcs_images[:,matrix_missing_index[0],matrix_missing_index[1]]

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

    mask[matrix_missing_index[0],matrix_missing_index[1]] = 1
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    plt.colorbar(im, shrink = .8)
    rect = Rectangle(((matrix_missing_index[0]-.5), (matrix_missing_index[1]-.5)), width=1, height=1,
                             facecolor='none', edgecolor='r')
    ax[0].add_patch(rect)
    pdd = pd.DataFrame(np.log(fcs_marginal_density), columns = None)
    sns.kdeplot(data = pdd, palette=['blue'], ax = ax[1])
    ax[1].axvline(ref_image[matrix_missing_index[0],matrix_missing_index[1]], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['FCS'])
    plt.savefig(figname)
    plt.clf()


def visualize_fcs_conditional_bivariate_density(ref_image_folder, fcs_file, n, missing_index1,
                                                missing_index2, figname):

    mask = np.load((fcs_folder + "/" + ref_image_folder + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((fcs_folder + "/" + ref_image_folder + "/ref_image.npy"))
    fcs_images = np.load((fcs_folder + "/" + ref_image_folder + "/" + 
                                     fcs_file))
    matrix_missing_index1 = (int(missing_index1 / n), int(missing_index1 % n))
    matrix_missing_index2 = (int(missing_index2 / n), int(missing_index2 % n))
    bivariate_fcs_density = fcs_images[:,matrix_missing_index1,matrix_missing_index2]

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))
    im = ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)).astype(float),
                 vmin = -2, vmax = 4)
    ax[0].plot(matrix_missing_index1[0], matrix_missing_index1[1], "r+")
    ax[0].plot(matrix_missing_index2[0], matrix_missing_index2[1], "r+")
    plt.colorbar(im, shrink = .8)
    sns.kdeplot(x = bivariate_fcs_density[:,0], y = bivariate_fcs_density[:,1],
                ax = ax[1])
    plt.axvline(ref_image[int(matrix_missing_index1[0]),int(matrix_missing_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_missing_index2[0]),int(matrix_missing_index2[1])], color='red', linestyle = 'dashed')
    ax[1].legend(labels = ['FCS'])
    ax[1].set_xlim(-10,10)
    ax[1].set_ylim(-10,10)
    plt.savefig(figname)
    plt.clf()


def produce_ncs_and_fcs_marginal_density(ref_image_folder, n, missing_index,
                                                          ncs_file_name, fcs_file,
                                                          figname):

    mask = np.load((fcs_folder + "/" + ref_image_folder + "/mask.npy"))
    ref_image = np.load((fcs_folder + "/" + ref_image_folder + "/ref_image.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ncs_images = np.load((fcs_folder + "/" + ref_image_folder + "/" + ncs_file_name))
    fcs_images = np.load((fcs_folder + "/" + ref_image_folder + "/" + fcs_file))
    matrix_missing_index = index_to_matrix_index(missing_index, n)
    ncs_marginal_density = ncs_images[:,0,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
    fcs_marginal_density = fcs_images[:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]


    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    ncs_pdd = pd.DataFrame(ncs_marginal_density,
                                    columns = None)
    fcs_pdd = pd.DataFrame(fcs_marginal_density,
                                    columns = None)

    #missing_index1, missing_index2,, missing_index2,ield = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = ncs_pdd, palette = ["orange"], ax = axs[1])
    sns.kdeplot(data = fcs_pdd, palette = ["purple"], ax = axs[1])
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
    plt.savefig((fcs_folder + "/" + ref_image_folder + "/marginal_density/" + figname))
    plt.clf()


def produce_ncs_and_fcs_bivariate_density(ref_image_folder, n,
                                          ncs_file_name, fcs_file,
                                          missing_index1, missing_index2,
                                          figname, nrep):
    
    mask = np.load((fcs_folder + "/" + ref_image_folder + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    ref_image = np.load((fcs_folder + "/" + ref_image_folder + "/ref_image.npy"))
    ncs_images = np.load((fcs_folder + "/" + ref_image_folder + "/" + ncs_file_name))
    fcs_images = np.load((fcs_folder + "/" + ref_image_folder + "/" + fcs_file))
    matrix_missing_index1 = index_to_matrix_index(missing_index1, n)
    matrix_missing_index2 = index_to_matrix_index(missing_index2, n)
    bivariate_fcs_density = np.concatenate([(fcs_images[:,matrix_missing_index1[0],matrix_missing_index1[1]]).reshape((nrep,1)),
                                            (fcs_images[:,matrix_missing_index2[0], matrix_missing_index2[1]]).reshape((nrep,1))],
                                            axis = 1)

    bivariate_fcs_density = np.log(bivariate_fcs_density)
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
    sns.kdeplot(x = bivariate_fcs_density[:,0], y = bivariate_fcs_density[:,1],
                    ax = axs[1], color = 'purple')
    sns.kdeplot(x = bivariate_ncs_density[:,0], y = bivariate_ncs_density[:,1],
                    ax = axs[1], color = 'orange')
    plt.axvline(ref_image[int(matrix_missing_index1[0]),int(matrix_missing_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_missing_index2[0]),int(matrix_missing_index2[1])], color='red', linestyle = 'dashed')
    axs[1].set_xlim(-3,6)
    axs[1].set_ylim(-3,6)
        #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
        #rlocation = (round(location[0],2), round(location[1],2))
        #axs[1].set_xlabel("location: " + str(rlocation))
    purple_patch = mpatches.Patch(color='purple')
    orange_patch = mpatches.Patch(color='orange')
    axs[1].legend(handles = [purple_patch, orange_patch], labels = ['bivariate LCS', 'NCS'])
    plt.savefig((fcs_folder + "/" + ref_image_folder + "/bivariate_density/" +
                 figname + "_" + str(missing_index1) + "_" + str(missing_index2)
                 + ".png"))
    plt.clf()