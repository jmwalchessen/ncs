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

def visualize_spatial_field(observation, min_value, max_value, figname):

    fig, ax = plt.subplots(1)
    plt.imshow(observation, vmin = min_value, vmax = max_value)
    plt.savefig(figname)


def produce_generated_marginal_density(mask, minX, maxX, minY, maxY, n, missing_index, missing_indices,
                                       conditional_generated_samples, ref_image,
                                       figname):


    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    generated_marginal_density = conditional_generated_samples[:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask), vmin = -2, vmax = 4)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], 
                color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-2,4)
    axs[1].set_ylim(0,2)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['generated'])
    plt.savefig(figname)
    plt.clf()

def produce_generated_and_mcmc_interpolation_marginal_density(mask, minX, maxX, minY, maxY, n,
                                                              missing_index, missing_indices,
                                                              conditional_generated_samples,
                                                              conditional_mcmc_samples, ref_image,
                                                              figname):


    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    generated_marginal_density = conditional_generated_samples[:,0,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
    mcmc_marginal_density = conditional_mcmc_samples[:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]


    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)
    mcmc_pdd = pd.DataFrame(mcmc_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask), vmin = -2, vmax = 6)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    sns.kdeplot(data = mcmc_pdd, palette = ["purple"], ax = axs[1])
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
    axs[1].legend(handles = [purple_patch, orange_patch], labels = ['mcmc', 'generated'])
    plt.savefig(figname)
    plt.clf()

def produce_generated_bivariate_density(mask, minX, maxX, minY, maxY, n, range_value, smooth_value,
                                                 number_of_replicates, missing_two_indices,
                                                 missing_indices, observed_vector,
                                                 conditional_generated_samples, ref_image, figname):
    
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    number_of_replicates = conditional_generated_samples.shape[0]
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask_reshaped = (mask.reshape((n,n))).astype(float)
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask_reshaped), vmin = -2, vmax = 6)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                ax = axs[1])
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    #blue_patch = mpatches.Patch(color='blue')
    #orange_patch = mpatches.Patch(color='orange')
    plt.axvline(ref_image[int(matrix_index1[0]),int(matrix_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_index2[0]),int(matrix_index2[1])], color='red', linestyle = 'dashed')
    plt.xlim(-1,2)
    plt.ylim(-1,2)
    axs[1].set_title("Bivariate")
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    #axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

def produce_generated_bivariate_density(mask, minX, maxX, minY, maxY, n, range_value, smooth_value,
                                                 number_of_replicates, missing_two_indices,
                                                 missing_indices, observed_vector,
                                                 conditional_generated_samples, ref_image, figname):
    
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    number_of_replicates = conditional_generated_samples.shape[0]
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask_reshaped = (mask.reshape((n,n))).astype(float)
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask_reshaped), vmin = -2, vmax = 6)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                ax = axs[1])
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    #blue_patch = mpatches.Patch(color='blue')
    #orange_patch = mpatches.Patch(color='orange')
    plt.axvline(ref_image[int(matrix_index1[0]),int(matrix_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_index2[0]),int(matrix_index2[1])], color='red', linestyle = 'dashed')
    plt.xlim(-2,4)
    plt.ylim(-2,4)
    axs[1].set_title("Bivariate")
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    #axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

#really doesn't make sense because mcmc interpolation is independent for each pixel
def produce_generated_and_mcmc_interpolation_bivariate_density(mask, minX, maxX, minY, maxY, n,
                                                               number_of_replicates, missing_two_indices,
                                                               missing_indices,
                                                               conditional_generated_samples,
                                                               conditional_mcmc_samples, ref_image, figname):
    
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    number_of_replicates = conditional_generated_samples.shape[0]
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,0,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,0,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    mcmc_bivariate_density = np.concatenate([(conditional_mcmc_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_mcmc_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask_reshaped = (mask.reshape((n,n))).astype(float)
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask_reshaped, vmin = -2, vmax = 2)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                ax = axs[1], levels = 10, color = 'orange')
    sns.kdeplot(x = mcmc_bivariate_density[:,0], y = mcmc_bivariate_density[:,1],
                ax = axs[1], color = 'purple', levels = 10, label = "mcmc")
    purple_patch = mpatches.Patch(color='purple')
    orange_patch = mpatches.Patch(color='orange')
    plt.axvline(ref_image[int(matrix_index1[0]),int(matrix_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_index2[0]),int(matrix_index2[1])], color='red', linestyle = 'dashed')
    plt.xlim(-4,8)
    plt.ylim(-4,8)
    axs[1].set_title("Bivariate")
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(handles = [purple_patch, orange_patch],labels = ['mcmc', 'generated'])
    plt.savefig(figname)
    plt.clf()

def produce_mcmc_interpolation_visualization(mcmc_images, irep, figname):

    fig, ax = plt.subplots(1)
    ax.imshow(mcmc_images[irep,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    plt.savefig(figname)

def produce_mcmc_interpolation_visualizations(mcmc_images, mask, ref_image, irep, n, figname):

    fig = plt.figure(figsize=(10, 10))

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
            im = ax.imshow(ref_image, vmin = -2, vmax = 2)
        
        if(i == 1):
            ax.imshow(ref_image, alpha = mask.astype(float), vmin = -2, vmax = 2)
        
        if(i == 2):
            ax.imshow((mcmc_images[irep,:,:]).reshape((n,n)), vmin = -2, vmax = 2)
        if(i == 3):
            ax.imshow((mcmc_images[(irep+1),:,:]).reshape((n,n)), vmin = -2, vmax = 2,
                      alpha = (1-mask).astype(float))
            
    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.set_ticks([])
    cbar.set_ticks([-2,1,0,1,2])
    #fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)

def produce_diffusion_and_mcmc_interpolation_visualizations(diffusion_images, mcmc_images, mask,
                                                            ref_image, irep, n, figname):

    fig = plt.figure(figsize=(10, 10))

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
            im = ax.imshow(ref_image, vmin = -2, vmax = 2)
        
        if(i == 1):
            ax.imshow(ref_image, alpha = mask.astype(float), vmin = -2, vmax = 2)
        
        if(i == 2):
            ax.imshow((mcmc_images[irep,:,:]).reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_title("MCMC")
        if(i == 3):
            ax.imshow((diffusion_images[irep,:,:]).reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_title("Diffusion")
            
    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.set_ticks([])
    cbar.set_ticks([-2,1,0,1,2])
    #fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)

def visualize_diffusion_and_mcmc_interpolation_conditional_mean(diffusion_images, mcmc_images, mask,
                                                                ref_image, mcmc_mask, n, figname):
    
    mcmc_mask_interrupted = np.zeros((n,n))
    mcmc_mask_interrupted[mcmc_mask == -1] = 1

    
    fig = plt.figure(figsize=(10, 10))

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

            im = ax.imshow(ref_image, vmin = -2, vmax = 2)

        if(i == 1):
            ax.imshow(ref_image, alpha = mask.astype(float), vmin = -2, vmax = 2)
        
        if(i == 2):
            mcmc_mean = np.mean(mcmc_images, axis = 0)
            ax.imshow(mcmc_mean, vmin = -2, vmax = 2, alpha = (1-mcmc_mask_interrupted).astype(float))
            ax.set_title("MCMC")
        if(i == 3):
            diffusion_mean = np.mean(diffusion_images, axis = 0)
            ax.imshow(diffusion_mean.reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_title("Diffusion")

    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.set_ticks([])
    cbar.set_ticks([-2,1,0,1,2])
    #fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_mcmc_marginal_density(ref_image_name, mask_name, mcmc_file_name, missing_index, n, figname):

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
    ax[1].legend(labels = ['MCMC'])
    plt.savefig(figname)
    plt.clf()


def visualize_mcmc_bivariate_density(ref_image_name, mask_name, mcmc_file_name, missing_index1,
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
    ax[1].legend(labels = ['MCMC'])
    plt.savefig(figname)
    plt.clf()
    
"""
for missing_index in range(200,500):
    nrep = 4000
    print(missing_index)
    ref_image_name = (data_generation_folder + "/data/model2/ref_image2/ref_image.npy")
    mask_name = (data_generation_folder + "/data/model2/ref_image2/mask.npy")
    mcmc_file_name = (data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/mcmc_interpolation_missing_index")
    n = 32
    mcmc_samples = np.load((mcmc_file_name + "_" + str(missing_index) + ".npy"))
    if(mcmc_samples.size == nrep):
        figname = (data_generation_folder + 
           "/data/model2/ref_image2/mcmc_interpolation/marginal_density/mcmc_interpolation_neighbors_7_4000_"
           + str(missing_index) + ".png")
        print(figname)

        visualize_mcmc_marginal_density(ref_image_name, mask_name, mcmc_file_name, missing_index, n, figname)"""

diffusion_images = np.load((data_generation_folder + "/data/model2/ref_image2/diffusion/model2_random025_beta_min_max_01_20_1000.npy"))
mcmc_images = np.load((data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/mcmc_interpolation_neighbors_7_4000.npy"))
mcmc_mask = np.load((data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/mcmc_interpolation_neighbors_7_4000_mask.npy"))
mask = np.load((data_generation_folder + "/data/model2/ref_image2/mask.npy"))
n = 32
ref_image = np.load((data_generation_folder + "/data/model2/ref_image2/ref_image.npy"))
minX = -10
maxX = 10
minY = -10
maxY = 10
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
m = missing_indices.shape[0]
number_of_replicates = 4000

"""
for missing_index in range(0,m):

    figname = (data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/marginal_density/mcmc_interpolation_vs_diffusion_neighbors_7_4000_"
               + "missing_index_" + str(missing_index) + ".png")
    produce_generated_and_mcmc_interpolation_marginal_density(mask, minX, maxX, minY, maxY, n,
                                                              missing_index, missing_indices,
                                                              diffusion_images,
                                                              mcmc_images, ref_image,
                                                              figname)"""
    
for i in range(415,416):
    for j in [385,388,390,392,440,442,445,447,470,472,475]:

        missing_two_indices = [i,j]
        figname = (data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/bivariate_density/mcmc_interpolation_vs_diffusion_neighbors_7_4000_"
               + "missing_index_" + str(i) + "_" + str(j) + ".png")
        produce_generated_and_mcmc_interpolation_bivariate_density(mask, minX, maxX, minY, maxY, n,
                                                               number_of_replicates, missing_two_indices,
                                                               missing_indices,
                                                               diffusion_images,
                                                               mcmc_images, ref_image, figname)
        
for irep in range(0, 20, 2):

    figname = (data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/visualizations/mcmc_interpolation_vs_diffusion_neighbors_7_4000_"
               + str(irep) + ".png")
    produce_diffusion_and_mcmc_interpolation_visualizations(diffusion_images, mcmc_images, mask,
                                                            ref_image, irep, n, figname)
    
figname = (data_generation_folder + "/data/model2/ref_image2/mcmc_interpolation/visualizations/mcmc_interpolation_vs_diffusion_conditional_mean_neighbors_7_4000_"
               + ".png")    
visualize_diffusion_and_mcmc_interpolation_conditional_mean(diffusion_images, mcmc_images, mask,
                                                                ref_image, mcmc_mask, n, figname)

"""
for missing_index1 in range(156,157):
    for missing_index2 in range(161,185):

        ref_image_name = (data_generation_folder + "/data/model1/ref_image1/ref_image.npy")
        mask_name = (data_generation_folder + "/data/model1/ref_image1/mask.npy")
        mcmc_file_name = (data_generation_folder + "/data/model1/ref_image1/mcmc_interpolation/mcmc_interpolation_neighbors_7_4000_missing_index")
        n = 32
        figname = (data_generation_folder + 
           "/data/model1/ref_image1/mcmc_interpolation/bivariate_density/mcmc_interpolation_neighbors_7_4000_"
           + str(missing_index1) +"_" + str(missing_index2) + ".png")
        visualize_mcmc_bivariate_density(ref_image_name, mask_name, mcmc_file_name, missing_index1,
                                     missing_index2, n, figname)"""

"""

n = 32
number_of_replicates = 4000 
conditional_samples = np.load((data_generation_folder + "/data/schlather/model2/ref_image2/diffusion/model2_random025_range_3_smooth_1.6_beta_min_max_01_20_1000.npy"))
conditional_samples = conditional_samples.reshape((number_of_replicates,n,n))
#mask = np.load((data_generation_folder + "/data/ref_image1/mask.npy"), allow_pickle = True)
n = 32
#mask = th.zeros((1,n,n))
#mask[:, int(n/4):int(n/4*3), int(n/4):int(n/4*3)] = 1
device = "cuda:0"
p = .025
mask = np.load((data_generation_folder + "/data/schlather/model2/ref_image2/mask.npy"))
ref_image = (np.load((data_generation_folder + "/data/schlather/model2/ref_image2/ref_image.npy")))
range_value = 3.
smooth_value = 1.6                                                                                        
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
mask_type = "random50"
folder_name = (data_generation_folder + "/data/schlather/model2/ref_image2/marginal_density")
m = missing_indices.shape[0]
observed_vector = ref_image.reshape((n**2))
observed_vector = np.delete(observed_vector, missing_indices)
minX = -10
maxX = 10
minY = -10
maxY = 10


for i in range(0, m, 10):
    missing_index = i
    true_missing_index = missing_indices[missing_index]
    true_missing_matrix_index = index_to_matrix_index(true_missing_index, n)
    figname = (folder_name + "/marginal_density_model2_" + str(int(true_missing_matrix_index[0]))
               + "_" + str(int(true_missing_matrix_index[1])) + ".png")
    produce_generated_marginal_density((1-mask), minX, minY, maxX, maxY, n, missing_index, missing_indices,
                                       conditional_samples, ref_image,
                                       figname)


"""

"""
indices1 = [300]
indices2 = [282,285,288,289,290,291,292,299,300,301,302,303,315,317,318,319,320]

for i in indices1:
    for j in indices2:
        missing_index1 = i
        missing_index2 = j
        folder_name = (data_generation_folder + "/data/schlather/model1/ref_image1/bivariate_density")
        true_missing_index1 = i
        true_missing_matrix_index1 = index_to_matrix_index(true_missing_index1, n)
        true_missing_index2 = j
        true_missing_matrix_index2 = index_to_matrix_index(true_missing_index2, n)
        figname = (folder_name + "/bivariate_density_model2_" + str(int(true_missing_matrix_index1[0]))
                + "_" + str(int(true_missing_matrix_index1[1])) + "_" +
                str(int(true_missing_matrix_index2[0])) + "_" + str(int(true_missing_matrix_index2[1]))
                    + ".png")
        missing_two_indices = [i,j]
        produce_generated_bivariate_density((1-mask), minX, maxX, minY, maxY, n, range_value, smooth_value,
                                                 number_of_replicates, missing_two_indices,
                                                 missing_indices, observed_vector,
                                                 conditional_samples, ref_image, figname)"""

"""
n = 32
number_of_replicates = 4000 
folder_name = (evaluation_folder + "/diffusion_generation/data/schlather/model2/ref_image3")
mcmc_file_name = "mcmc_interpolation/mcmc_interpolation_simulations_range_3_smooth_1.6_4000.npy"
conditional_mcmc_images = load_mcmc_interpolation_images(folder_name, mcmc_file_name, number_of_replicates, n)
diffusion_images = np.load((data_generation_folder + "/data/schlather/model2/ref_image1/diffusion/model2_random025_range_3_smooth_1.6_beta_min_max_01_20_1000.npy"))
diffusion_images = diffusion_images.reshape((number_of_replicates,n,n))
#mask = np.load((data_generation_folder + "/data/ref_image1/mask.npy"), allow_pickle = True)
n = 32
#mask = th.zeros((1,n,n))
#mask[:, int(n/4):int(n/4*3), int(n/4):int(n/4*3)] = 1
device = "cuda:0"
p = .5
mask = np.load((data_generation_folder + "/data/schlather/model2/ref_image1/mask.npy"))
ref_image = (np.load((data_generation_folder + "/data/schlather/model2/ref_image1/ref_image.npy")))
range_value = 3.
smooth_value = 1.6                                                                                        
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
mask_type = "random50"
folder_name = (data_generation_folder + "/data/schlather/model2/ref_image1/mcmc_interpolation/marginal_density")
m = missing_indices.shape[0]
observed_vector = ref_image.reshape((n**2))
observed_vector = np.delete(observed_vector, missing_indices)
minX = -10
maxX = 10
minY = -10
maxY = 10
"""

"""
for i in range(0, m, 1):
    missing_index = i
    true_missing_index = missing_indices[missing_index]
    true_missing_matrix_index = index_to_matrix_index(true_missing_index, n)
    figname = (folder_name + "/marginal_density_mcmc_and_diffusion_model1_" + str(int(true_missing_matrix_index[0]))
               + "_" + str(int(true_missing_matrix_index[1])) + ".png")
    produce_generated_and_mcmc_interpolation_marginal_density((1-mask), minX, minY, maxX, maxY, n, missing_index, missing_indices,
                                       diffusion_images, conditional_mcmc_images, ref_image, figname)

indices1 = [300]
indices2 = [282,285,288,289,290,291,292,299,300,301,302,303,315,317,318,319,320]

for i in indices1:
    for j in indices2:
        missing_index1 = i
        missing_index2 = j
        folder_name = (data_generation_folder + "/data/schlather/model1/ref_image1/mcmc_interpolation/bivariate_density")
        true_missing_index1 = i
        true_missing_matrix_index1 = index_to_matrix_index(true_missing_index1, n)
        true_missing_index2 = j
        true_missing_matrix_index2 = index_to_matrix_index(true_missing_index2, n)
        figname = (folder_name + "/bivariate_density_mcmc_and_diffusion_model1_" + str(int(true_missing_matrix_index1[0]))
                + "_" + str(int(true_missing_matrix_index1[1])) + "_" +
                str(int(true_missing_matrix_index2[0])) + "_" + str(int(true_missing_matrix_index2[1]))
                    + ".png")
        missing_two_indices = [i,j]
        produce_generated_and_mcmc_interpolation_bivariate_density((1-mask), minX, maxX, minY, maxY, n, range_value, smooth_value,
                                                 number_of_replicates, missing_two_indices,
                                                 missing_indices, observed_vector,
                                                 diffusion_images, conditional_mcmc_images, ref_image, figname)"""

"""
for irep in range(0, number_of_replicates, 100):
    figname = (folder_name + "/visualization_mcmc_and_diffusion_model1_" + str(irep) + ".png")
    folder_name = (data_generation_folder + "/data/schlather/model1/ref_image1/mcmc_interpolation/visualizations")
    produce_diffusion_and_mcmc_interpolation_visualizations(diffusion_images, conditional_mcmc_images,
                                                            mask, ref_image, irep, n, figname)"""
"""
figname = (data_generation_folder + "/data/schlather/model1/ref_image1/mcmc_interpolation/conditional_mean/diffusion_and_mcmc_interpolation_conditional_mean.png")
visualize_diffusion_and_mcmc_interpolation_conditional_mean(diffusion_images, conditional_mcmc_images,
                                                            mask, ref_image, n, figname)"""