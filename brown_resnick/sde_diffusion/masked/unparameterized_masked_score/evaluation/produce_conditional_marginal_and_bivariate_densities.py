import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
data_generation_folder = (append_directory(2) + "/evaluation/diffusion_generation")
print(data_generation_folder)
sys.path.append(data_generation_folder)

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
    axs[0].imshow(ref_image.reshape((n,n)), alpha = (1-mask), vmin = -2, vmax = 6)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], 
                color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-2,6)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'generated'])
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
    plt.xlim(-2,6)
    plt.ylim(-2,6)
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



n = 32
number_of_replicates = 4000 
conditional_samples = np.load((data_generation_folder + "/data/model1/ref_image1/diffusion/model1_random50_beta_min_max_01_20_1000.npy"))
conditional_samples = conditional_samples.reshape((number_of_replicates,n,n))
#mask = np.load((data_generation_folder + "/data/ref_image1/mask.npy"), allow_pickle = True)
n = 32
#mask = th.zeros((1,n,n))
#mask[:, int(n/4):int(n/4*3), int(n/4):int(n/4*3)] = 1
device = "cuda:0"
p = .5
mask = np.load((data_generation_folder + "/data/model1/ref_image1/mask.npy"))
ref_image = (np.load((data_generation_folder + "/data/model1/ref_image1/ref_image.npy")))
range_value = 1.6
smooth_value = 1.6                                                                                        
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
mask_type = "random50"
folder_name = (data_generation_folder + "/data/model1/ref_image1/marginal_density")
m = missing_indices.shape[0]
observed_vector = ref_image.reshape((n**2))
observed_vector = np.delete(observed_vector, missing_indices)
minX = -10
maxX = 10
minY = -10
maxY = 10

"""
for i in range(0, m, 10):
    missing_index = i
    true_missing_index = missing_indices[missing_index]
    true_missing_matrix_index = index_to_matrix_index(true_missing_index, n)
    figname = (folder_name + "/marginal_density_model1_" + str(int(true_missing_matrix_index[0]))
               + "_" + str(int(true_missing_matrix_index[1])) + ".png")
    produce_generated_marginal_density((1-mask), minX, minY, maxX, maxY, n, missing_index, missing_indices,
                                       conditional_samples, ref_image,
                                       figname)"""



                   
indices1 = [212]
indices2 = [195,211,213,214,215,225,226,227,228,229,230]

for i in indices1:
    for j in indices2:
        missing_index1 = i
        missing_index2 = j
        folder_name = (data_generation_folder + "/data/model1/ref_image1/bivariate_density")
        true_missing_index1 = i
        true_missing_matrix_index1 = index_to_matrix_index(true_missing_index1, n)
        true_missing_index2 = j
        true_missing_matrix_index2 = index_to_matrix_index(true_missing_index2, n)
        figname = (folder_name + "/bivariate_density_model6_" + str(int(true_missing_matrix_index1[0]))
                + "_" + str(int(true_missing_matrix_index1[1])) + "_" +
                str(int(true_missing_matrix_index2[0])) + "_" + str(int(true_missing_matrix_index2[1]))
                    + ".png")
        missing_two_indices = [i,j]
        produce_generated_bivariate_density((1-mask), minX, maxX, minY, maxY, n, range_value, smooth_value,
                                                 number_of_replicates, missing_two_indices,
                                                 missing_indices, observed_vector,
                                                 conditional_samples, ref_image, figname)