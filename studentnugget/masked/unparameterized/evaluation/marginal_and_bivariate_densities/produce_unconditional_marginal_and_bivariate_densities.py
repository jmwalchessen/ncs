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

def produce_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale, df, seed_value,
                              number_of_replicates, missing_two_indices):
    
    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    unconditional_vectors, unconditonal_matrices = generate_student_nugget(minX, maxX, minY, maxY, n, variance,
                                                                           lengthscale, df, number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    bivariate_density = (unconditional_vectors[:,missing_two_indices]).reshape((number_of_replicates,2))
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditonal_matrices[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    matrix_index1 = index_to_matrix_index(missing_two_indices[0], n)
    matrix_index2 = index_to_matrix_index(missing_two_indices[1], n)
    axs[0].plot(matrix_index1[0], matrix_index1[1], "r+")
    axs[0].plot(matrix_index2[0], matrix_index2[1], "r+")
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1],
                ax = axs[1])
    plt.xlim(-12,12)
    plt.ylim(-12,12)
    axs[1].set_title("Marginal")
    #location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    #rlocation1 = (round(location1[0],2), round(location1[1],2))
    #location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    #rlocation2 = (round(location2[0],2), round(location2[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation1))
    #axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(labels = ['true'])
    plt.show()

def produce_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df, seed_value,
                             number_of_replicates, missing_index, figname):

    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    unconditional_vectors, unconditional_matrices = generate_student_nugget(minX, maxX, minY, maxY, n,
                                                                            variance, lengthscale, df,
                                                                            number_of_replicates,
                                                                            seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_vectors[:,missing_index]).reshape((number_of_replicates,1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow((unconditional_matrices[0,:,:,:]).reshape((n,n)), vmin = -8, vmax = 8)
    matrix_index = index_to_matrix_index(missing_index, n)
    axs[0].plot(matrix_index[0], matrix_index[1], "r+")
    sns.kdeplot(data = pdd, ax = axs[1])
    axs[1].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true'])
    plt.savefig(figname)
    plt.clf()


def produce_t_vs_normal_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                         seed_value, number_of_replicates, missing_index, figname):

    #missing_index is in between 0 and m, it's not the original missing index from n x n field
    unconditional_vectors, unconditional_matrices = generate_student_nugget(minX, maxX, minY, maxY, n,
                                                                            variance, lengthscale, df,
                                                                            number_of_replicates,
                                                                            seed_value)
    print(np.max(unconditional_vectors))
    
    unconditional_gpvectors, unconditional_gpmatrices = generate_gaussian_process(minX, maxX, minY, maxY,
                                                                                  n, variance, lengthscale, number_of_replicates,
                                                                                  seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_vectors[:,missing_index]).reshape((number_of_replicates,1))
    marginal_gpdensity = (unconditional_gpvectors[missing_index,:]).reshape((number_of_replicates,1))
    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    npdd = pd.DataFrame(marginal_gpdensity,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow((unconditional_matrices[0,:,:,:]).reshape((n,n)), vmin = -2, vmax = 2)
    matrix_index = index_to_matrix_index(missing_index, n)
    axs[0].plot(matrix_index[0], matrix_index[1], "r+")
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = npdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['student', 'gaussian'])
    plt.savefig(figname)
    plt.clf()



def produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                                number_of_replicates, missing_index, unconditional_generated_samples,
                                                figname):

    unconditional_vectors, unconditional_matrices = generate_student_nugget(minX, maxX, minY, maxY, n, 
                                                                            variance, lengthscale, df, number_of_replicates,
                                                                            seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_vectors[:,missing_index]).reshape((number_of_replicates,1))
    matrix_missing_index = index_to_matrix_index(missing_index, n)
    generated_marginal_density = unconditional_generated_samples[:,0,int(matrix_missing_index[0]),int(matrix_missing_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow((unconditional_matrices[0,:,:,:]).reshape((n,n)), vmin = -8, vmax = 8)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-20,20)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()

n = 32
minX = -10
maxX = 10
minY = -10
maxY = 10
variance = .2
lengthscale = 1.6
number_of_replicates = 1000
missing_index = 200
df = 2
seed_value = np.random.randint(0, 100000, 1)[0]
figname = "student_t_variance_.2_lengthscale_1.6_df_2_vs_gp_marginal_density.png"
produce_t_vs_normal_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                     seed_value, number_of_replicates, missing_index, figname)

""""
for i in range(0, 25):
    missing_index = i
    n = 32
    minX = -10
    maxX = 10
    minY = -10
    maxY = 10
    variance = 10
    lengthscale = 1.6
    number_of_replicates = 1000
    df = 1

    unconditional_generated_samples = np.load((home_folder + 
                                            "/evaluation/diffusion_generation/data/model1/ref_image2/diffusion/model1_beta_min_max_01_25_random0_1000.npy"))
    figname = "marginal/model1/ref_image2/true_vs_generated_1000_marginal_density_ " + str(missing_index) + ".png"
    produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                                    number_of_replicates, missing_index, unconditional_generated_samples,
                                                    figname)"""