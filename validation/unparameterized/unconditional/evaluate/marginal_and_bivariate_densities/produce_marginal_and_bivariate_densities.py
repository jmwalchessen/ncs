import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
data_generation_folder = (append_directory(3) + "/generate_data")
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *

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

def matrix_index_to_index(matrix_index, n):

    return (matrix_index[0]*n+matrix_index[1])


def index_to_matrix_index(index, n):

    return (int(index % n), int(index / n))

def visualize_spatial_field(observation, min_value, max_value, figname):

    fig, ax = plt.subplots(1)
    plt.imshow(observation, vmin = min_value, vmax = max_value)
    plt.savefig(figname)

def produce_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, missing_two_indices, seed_value):
    
    unconditional_gp_vector, unconditional_gp_matrix = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                     variance, lengthscale,
                                                     number_of_replicates, seed_value)

    bivariate_density = (unconditional_gp_vector[missing_two_indices,:]).reshape((2,number_of_replicates)).T
    fig, axs = plt.subplots(ncols =2, figsize = (10,5))
    pdd = pd.DataFrame(bivariate_density,
                                    columns = None)
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_two_indices[0])
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_two_indices[1])
    matrix_index1 = index_to_matrix_index(missing_two_indices[0], n)
    matrix_index2 = index_to_matrix_index(missing_two_indices[1], n)
    axs[0].imshow(unconditional_gp_matrix[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index1[0], matrix_index1[1], "r+")
    axs[0].plot(matrix_index2[0], matrix_index2[1], "r+")
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1], ax = axs[1], thresh = .25)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1].set_title("Bivariate")
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(labels = ['true'])
    plt.show()

def produce_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                             number_of_replicates, missing_index, seed_value):

    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    unconditional_gpv, unconditional_gpm = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                     variance, lengthscale,
                                                     number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_gpv[missing_index,:]).reshape((number_of_replicates,1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    matrix_index = index_to_matrix_index(missing_index, n)
    axs[0].imshow(unconditional_gpm[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index[0], matrix_index[1], "r+")
    sns.kdeplot(data = pdd, ax = axs[1])
    axs[1].set_title("Marginal")
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true'])
    plt.show()

def produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                                  number_of_replicates, missing_index,
                                  unconditional_generated_samples, seed_value,
                                  figname):

    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    uncond_gpv, uncond_gpm = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                     variance, lengthscale,
                                                     number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    print(uncond_gpv.shape)
    marginal_density = (uncond_gpv[missing_index,:]).reshape((number_of_replicates,1))
    matrix_index = index_to_matrix_index(missing_index, n)
    generated_marginal_density = unconditional_generated_samples[:,int(matrix_index[0]),int(matrix_index[1])]

    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)

    axs[0].imshow(uncond_gpm[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index[0], matrix_index[1], "r+")
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-2,2)
    axs[1].set_ylim(0,.8)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, missing_two_indices,
                                                 unconditional_generated_samples, seed_value, figname):
    
    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    uncond_gpv, uncond_gpm = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                       variance, lengthscale, number_of_replicates,
                                                       seed_value)
    #conditional_vectors is shape (number of replicates, m)
    bivariate_density = (uncond_gpv[missing_two_indices,:]).reshape((2,number_of_replicates)).T
    matrix_index1 = index_to_matrix_index(missing_indices[0], n)
    matrix_index2 = index_to_matrix_index(missing_indices[1], n)
    generated_bivariate_density = np.concatenate([(unconditional_generated_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
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
    axs[0].imshow(uncond_gpm[0,:,:,:].reshape((n,n)),vmin = -2, vmax = 2)
    axs[0].plot(matrix_index1[0], matrix_index1[1], "r+")
    axs[0].plot(matrix_index2[0], matrix_index2[1], "r+")
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1], hue = 'class', shade = True, levels = 5, alpha = .5)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1].set_title("Bivariate")
    location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_indices[0])
    rlocation1 = (round(location1[0],2), round(location1[1],2))
    location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_indices[1])
    rlocation2 = (round(location2[0],2), round(location2[1],2))
    axs[1].set_xlabel("location: " + str(rlocation1))
    axs[1].set_ylabel("location: " + str(rlocation2))
    plt.savefig(figname)
    plt.clf()


minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
number_of_replicates = 1000
missing_index = 700
missing_indices = [100,101]
home_folder = append_directory(3)
uncond_samples = np.load((home_folder + "/generate_data/data/diffusion/unconditional_lengthscale_1.6_variance_0.4_1000.npy"))

for missing_index in range(0,1):
    produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                             number_of_replicates, missing_index, uncond_samples, 43234,
                             ("marginal_density/true_and_generated_marginal_density_" 
                              + str(number_of_replicates) + "_" + str(missing_index) + ".png"))

indices1 = np.random.randint(0, n**2, 2)
indices2 = np.random.randint(0, n**2, 2)
indices1 = [500]
indices2 = [i for i in range(495,505)]
matrix_indices1 = [(27,27)]
matrix_indices2 = [(i,j) for i in range(22,32) for j in range(22,32)]
for i in range(len(matrix_indices1)):
    for j in range(len(matrix_indices2)):
        missing_indices = [matrix_index_to_index(matrix_indices1[i], n),
                           matrix_index_to_index(matrix_indices2[j],n)]
        produce_true_and_generated_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, missing_indices,
                                                 uncond_samples, 43234,("bivariate_density/true_and_generated_bivariate_density_"
                                                                        + str(number_of_replicates) + "_" + str(missing_indices[0])
                                                                        + "_" + str(missing_indices[1]) + ".png"))

