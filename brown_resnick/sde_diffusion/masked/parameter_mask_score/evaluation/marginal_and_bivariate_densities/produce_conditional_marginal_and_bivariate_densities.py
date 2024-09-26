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


def produce_generated_marginal_density(mask, n, missing_index, missing_indices,
                                       conditional_generated_samples, ref_image,
                                       figname):


    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    generated_marginal_density = conditional_generated_samples[:,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 4)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], 
                color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(0,1.5)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['diffusion'])
    plt.savefig(figname)
    plt.clf()

def produce_multiple_generated_marginal_density(mask, n, indices,
                                                conditional_generated_samples, ref_image,
                                                figname):

    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    for missing_index in indices:

        current_figname = (figname + "_" + str(missing_index) + ".png")
        produce_generated_marginal_density(mask, n, missing_index, missing_indices,
                                       conditional_generated_samples, ref_image,
                                       current_figname)
    

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


def produce_generated_bivariate_density(mask, n, range_value, smooth_value,
                                                 number_of_replicates, missing_two_indices,
                                                 missing_indices, observed_vector,
                                                 conditional_generated_samples, ref_image, figname):
    
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    number_of_replicates = conditional_generated_samples.shape[0]
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
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
    #axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

def produce_multiple_generated_bivariate_density(mask, n, range_value, smooth_value,
                                                 number_of_replicates, missing_indices1, missing_indices2,
                                                 conditional_generated_samples, ref_image, figname):

    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    observed_vector = (ref_image.reshape((n**2)))[-missing_indices]
    for missing_index1 in missing_indices1:
        for missing_index2 in missing_indices2:

            missing_two_indices = [missing_index1, missing_index2]
            current_figname = (figname + "missing_indices_" + str(missing_index1) + "_" + str(missing_index2) + ".png")
            produce_generated_bivariate_density(mask, n, range_value, smooth_value,
                                                number_of_replicates, missing_two_indices,
                                                missing_indices, observed_vector,
                                                conditional_generated_samples, ref_image, current_figname)

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

range_values = [1.4, 1.6, 1.8, 2.]
missing_indices1 = [250]
missing_indices2 = [230,235,240,241,243,243,244,245,246,247,248,249,251,252,253,254,255,256,257,258,259,260,265,270]
smooth_value = 1.6
for i, range_value in enumerate(range_values):

    ref_folder = (data_generation_folder + "/data/model2/ref_image" + str(i+3))
    mask = np.load((ref_folder + "/mask.npy"))
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    n = 32
    conditional_generated_samples = np.load((ref_folder + "/diffusion/model2_range_" + str(range_value) + "_smooth_1.6_random0.5_4000.npy"))
    figname = (ref_folder + "/marginal_density/diffusion_marginal_density")
    indices = [i for i in range(0, 505, 10)]
    produce_multiple_generated_marginal_density(mask, n, indices, conditional_generated_samples, ref_image, figname)
    figname = (ref_folder + "/bivariate_density/diffusion_bivariate_density")
    number_of_replicates = 4000
    produce_multiple_generated_bivariate_density(mask, n, range_value, smooth_value,
                                                 number_of_replicates, missing_indices1, missing_indices2,
                                                 conditional_generated_samples, ref_image, figname)