import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
home_folder = (append_directory(3))
sys.path.append(home_folder)
from student_t_true_conditional_data_generation import *


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

def produce_bivariate_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                              masked_vector, number_of_replicates, missing_two_indices,
                              missing_indices, observed_vector):
    
    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    conditional_vectors = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n,
                                                     variance, lengthscale, masked_vector,
                                                     number_of_replicates)
    #conditional_vectors is shape (number of replicates, m)
    bivariate_density = (conditional_vectors[:,missing_two_indices]).reshape((number_of_replicates,2))
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(observed_vector.reshape((n,n)), alpha = (1-mask), vmin = -2, vmax = 2)
    missing_true_index1 = missing_indices[missing_two_indices[0]]
    missing_true_index2 = missing_indices[missing_two_indices[1]]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    axs[0].plot(matrix_index1[0], matrix_index1[1], "r+")
    axs[0].plot(matrix_index2[0], matrix_index2[1], "r+")
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1],
                ax = axs[1])
    plt.axvline(observed_vector[missing_true_index1], color='red', linestyle = 'dashed')
    plt.axhline(observed_vector[missing_true_index2], color='red', linestyle = 'dashed')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1].set_title("Marginal")
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(labels = ['true'])
    plt.show()

def produce_marginal_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                             number_of_replicates, missing_index, missing_indices,
                             observed_vector, ref_image):

    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    m = (n**2) - observed_vector.shape[0]
    nminusm = observed_vector.shape[0]
    unobserved_unconditional_mean = np.zeros((m,1))
    observed_unconditional_mean = np.zeros((nminusm,1))
    conditional_vectors = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance,
                                                          lengthscale, observed_vector,
                                                          observed_unconditional_mean,
                                                          unobserved_unconditional_mean, df, 
                                                          number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (conditional_vectors[:,missing_index]).reshape((number_of_replicates,1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    emp_mean = round(np.mean(marginal_density), 2)
    emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(ref_image, alpha = (1-mask), vmin = -8, vmax = 8)
    missing_true_index = missing_indices[missing_index]
    matrix_index = index_to_matrix_index(missing_true_index, n)
    axs[0].plot(matrix_index[0], matrix_index[1], "r+")
    sns.kdeplot(data = pdd, ax = axs[1])
    plt.axvline(ref_image[int(matrix_index[0]),int(matrix_index[1])], color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true'])
    plt.show()

def produce_true_and_generated_marginal_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                                number_of_replicates, missing_index, missing_indices,
                                                observed_vector, ref_image, conditional_generated_samples,
                                                figname):

    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    m = (n**2) - observed_vector.shape[0]
    nminusm = observed_vector.shape[0]
    unobserved_unconditional_mean = np.zeros((m,1))
    observed_unconditional_mean = np.zeros((nminusm,1))
    conditional_vectors = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance,
                                                          lengthscale, observed_vector,
                                                          observed_unconditional_mean,
                                                          unobserved_unconditional_mean, df, 
                                                          number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (conditional_vectors[:,missing_index]).reshape((number_of_replicates,1))
    missing_true_index = missing_indices[missing_index]
    matrix_missing_index = index_to_matrix_index(missing_true_index, n)
    generated_marginal_density = conditional_generated_samples[:,0,int(matrix_missing_index[0]),int(matrix_missing_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask = mask.astype(float).reshape((n,n))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -8, vmax = 8)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    plt.axvline(ref_image[int(matrix_missing_index[0]),int(matrix_missing_index[1])], color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-12,12)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

"""
minX = minY = -10
maxX = maxY = 10
n = 32
variance = 10
lengthscale = 1.6
number_of_replicates = 1000
df = 1
p = .5
mask = np.load((home_folder + "/evaluation/diffusion_generation/data/model1/ref_image1/mask.npy"))
missing_indices = (np.squeeze(np.argwhere((1-mask).reshape((n**2,)))))
observed_indices = (np.squeeze(np.argwhere(mask.reshape((n**2,)))))
m = missing_indices.shape[0]

for missing_index in range(0, m):
    nminusm = (n**2)-m
    ref_image = np.load((home_folder + "/evaluation/diffusion_generation/data/model1/ref_image1/ref_image1.npy"))
    #ref_vector = ref_vectors[0,:]
    ref_vector = ref_image.reshape((n**2))
    observed_vector = ((ref_vector[observed_indices])).reshape((nminusm,1))
    conditional_generated_samples = np.load((home_folder + 
                                            "/evaluation/diffusion_generation/data/model1/ref_image1/diffusion/model1_beta_min_max_01_25_random50_1000.npy"))
    figname = "marginal/model1/ref_image1/true_and_diffusion_marginal_1000_" + str(missing_index) + ".png"
    produce_true_and_generated_marginal_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                                    number_of_replicates, missing_index, missing_indices,
                                                    observed_vector, ref_image, conditional_generated_samples,
                                                    figname)

"""

def produce_true_and_generated_bivariate_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                                number_of_replicates, missing_index1, missing_index2,
                                                missing_indices, observed_vector, ref_image,
                                                conditional_generated_samples, figname):
    
    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    m = (n**2) - observed_vector.shape[0]
    nminusm = observed_vector.shape[0]
    unobserved_unconditional_mean = np.zeros((m,1))
    observed_unconditional_mean = np.zeros((nminusm,1))
    conditional_vectors = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance,
                                                          lengthscale, observed_vector,
                                                          observed_unconditional_mean,
                                                          unobserved_unconditional_mean, df, 
                                                          number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    missing_two_indices = np.array([missing_index1, missing_index2])
    bivariate_density = (conditional_vectors[:,missing_two_indices]).reshape((number_of_replicates,2))
    missing_true_index1 = missing_indices[missing_index1]
    missing_true_index2 = missing_indices[missing_index2]
    matrix_index1 = index_to_matrix_index(missing_true_index1, n)
    matrix_index2 = index_to_matrix_index(missing_true_index2, n)
    number_of_replicates = conditional_generated_samples.shape[0]
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,0,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,0,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    bivariate_density = np.concatenate([bivariate_density, generated_bivariate_density], axis = 0)
    class_vector = np.concatenate([(np.repeat('true', number_of_replicates)).reshape((number_of_replicates,1)),
                                   (np.repeat('generated', number_of_replicates)).reshape((number_of_replicates,1))], axis = 0)
    bivariate_density = np.concatenate([bivariate_density, class_vector], axis = 1)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    mask_reshaped = (mask.reshape((n,n))).astype(float)
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask_reshaped, vmin = -8, vmax = 8)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1], hue = 'class', shade = True, levels = 5, alpha = .5)
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.axvline(ref_image[int(matrix_index1[0]),int(matrix_index1[1])], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[int(matrix_index2[0]),int(matrix_index2[1])], color='red', linestyle = 'dashed')
    plt.xlim(-12,12)
    plt.ylim(-12,12)
    axs[1].set_title("Marginal")
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

minX = minY = -10
maxX = maxY = 10
n = 32
variance = 10
lengthscale = 1.6
number_of_replicates = 1000
df = 1
p = .5
mask = np.load((home_folder + "/evaluation/diffusion_generation/data/model1/ref_image1/mask.npy"))
missing_indices = (np.squeeze(np.argwhere((1-mask).reshape((n**2,)))))
observed_indices = (np.squeeze(np.argwhere(mask.reshape((n**2,)))))
m = missing_indices.shape[0]
ref_image = np.load((home_folder + "/evaluation/diffusion_generation/data/model1/ref_image1/ref_image1.npy"))
#ref_vector = ref_vectors[0,:]
ref_vector = ref_image.reshape((n**2))
nminusm = (n**2)-m
observed_vector = ((ref_vector[observed_indices])).reshape((nminusm,1))
conditional_generated_samples = np.load((home_folder + 
                                            "/evaluation/diffusion_generation/data/model1/ref_image1/diffusion/model1_beta_min_max_01_25_random50_1000.npy"))

indices1 = [2,4,10,30,40,50,87,93,96,100,104,107,113,125,134,150,175]
indices2 = [1,3,9,15,19,23,25,29,31,43,87,89,93,105,114,132,140,145]
for i in indices1:
    for j in indices2:
        missing_index1 = i
        missing_index2 = j
        true_missing_index1 = missing_indices[missing_index1]
        true_missing_matrix_index1 = index_to_matrix_index(true_missing_index1, n)
        true_missing_index2 = missing_indices[missing_index2]
        true_missing_matrix_index2 = index_to_matrix_index(true_missing_index2, n)
        figname = ("bivariate/model1/ref_image1/true_and_diffusion_bivariate_1000_" + str(int(true_missing_matrix_index1[0]))
                + "_" + str(int(true_missing_matrix_index1[1])) + "_" +
                str(int(true_missing_matrix_index2[0])) + "_" + str(int(true_missing_matrix_index2[1]))
                    + ".png")
        missing_two_indices = [i,j]
        produce_true_and_generated_bivariate_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                                number_of_replicates, missing_index1, missing_index2,
                                                missing_indices, observed_vector, ref_image,
                                                conditional_generated_samples, figname)