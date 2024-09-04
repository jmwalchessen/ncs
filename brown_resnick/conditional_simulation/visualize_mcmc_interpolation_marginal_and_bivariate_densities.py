import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import patches as mpatches
from append_directories import *


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


def produce_mcmc_marginal_density(mask, minX, maxX, minY, maxY, n, missing_index, missing_indices,
                                  mcmc_samples, ref_image, figname):


    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    mcmc_marginal_density = mcmc_samples[missing_index,:]


    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    mcmc_pdd = pd.DataFrame(mcmc_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask), vmin = -2, vmax = 4)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = mcmc_pdd, palette = ["blue"], ax = axs[1])
    ref_image = ref_image.reshape((n,n))
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], 
                color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-2,4)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    blue_patch = mpatches.Patch(color='blue')
    axs[1].legend(handles = [blue_patch],labels = ['mcmc'])
    plt.savefig(figname)
    plt.clf()


def produce_diffusion_and_mcmc_marginal_density(mask, minX, maxX, minY, maxY, n, missing_index, missing_indices,
                                       mcmc_samples, diffusion_samples, ref_image, figname):


    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    generated_marginal_density = diffusion_samples[:,:,matrix_missing_index[0],matrix_missing_index[1]]
    mcmc_marginal_density = mcmc_samples[missing_index,:]


    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)
    mcmc_pdd = pd.DataFrame(mcmc_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask), vmin = -2, vmax = 4)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    sns.kdeplot(data = mcmc_pdd, palette = ["blue"], ax = axs[1])
    ref_image = ref_image.reshape((n,n))
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], 
                color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-2,4)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()


def produce_mcmc_bivariate_density(mask, minX, maxX, minY, maxY, n, number_of_replicates, missing_two_indices,
                                   missing_indices, mcmc_samples, ref_image, figname):
    
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)

    mcmc_bivariate_density = mcmc_samples[missing_two_indices,:]
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask_reshaped = (mask.reshape((n,n))).astype(float)
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask_reshaped), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    sns.kdeplot(x = mcmc_bivariate_density[:,0], y = mcmc_bivariate_density[:,1],
                ax = axs[1])
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    blue_patch = mpatches.Patch(color='blue')
    ref_image = ref_image.reshape((n,n))
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
    axs[1].legend(handles = [blue_patch],labels = ['true'])
    plt.savefig(figname)
    plt.clf()

def produce_diffusion_and_mcmc_bivariate_density(mask, minX, maxX, minY, maxY, n, number_of_replicates, missing_two_indices,
                                        missing_indices, mcmc_samples, diffusion_samples, ref_image, figname):
    
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    number_of_replicates = diffusion_samples.shape[0]
    generated_bivariate_density = np.concatenate([(diffusion_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (diffusion_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    mcmc_bivariate_density = mcmc_samples[missing_two_indices,:]
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask_reshaped = (mask.reshape((n,n))).astype(float)
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask_reshaped), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                ax = axs[1])
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    ref_image = ref_image.reshape((n,n))
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
    axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
mask = np.load("data/powexp/MCMC_interpolation/ref_image1/mask.npy")
missing_index = 124
missing_indices =  np.squeeze(np.argwhere(mask.reshape((n**2,))))
mcmc_samples = np.load("data/powexp/MCMC_interpolation/ref_image1/conditional_simulations_neighbors5_powexp_range_3_smooth_1.6_4000.npy")
ref_image = np.load("data/powexp/MCMC_interpolation/ref_image1/observed_simulation_powexp_range_3_smooth_1.6.npy")
figname = ("data/powexp/MCMC_interpolation/ref_image1/marginal_density/marginal_density_missing_index_" + str(missing_index) + ".png")
brfolder = append_directory(2)
diffusion_samples = np.load(brfolder + "/sde_diffusion/masked/unparameterized_masked_score/evaluation/diffusion_generation/data/model1/ref_image1/diffusion/model1_random50_beta_min_max_01_20_1000.npy")
produce_diffusion_and_mcmc_marginal_density(mask, minX, maxX, minY, maxY, n, missing_index, missing_indices,
                                            mcmc_samples, diffusion_samples, ref_image, figname)