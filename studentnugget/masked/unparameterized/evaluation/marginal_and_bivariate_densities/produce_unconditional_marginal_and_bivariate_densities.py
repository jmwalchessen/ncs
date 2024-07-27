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
                                                                            variance, lengthscale, df, 10*number_of_replicates,
                                                                            seed_value)
    unconditional_gpvectors, unconditional_gpmatrices = generate_gaussian_process(minX, maxX, minY, maxY,
                                                                                  n, variance, lengthscale, 10*number_of_replicates,
                                                                                  seed_value)

    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_vectors[:,missing_index]).reshape((10*number_of_replicates,1))
    matrix_missing_index = index_to_matrix_index(missing_index, n)
    generated_marginal_density = unconditional_generated_samples[:,0,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
    gpmarginaldensity = (unconditional_gpvectors[missing_index,:]).reshape((10*number_of_replicates,1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)
    
    gpdd = pd.DataFrame(gpmarginaldensity, columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow((unconditional_matrices[0,:,:,:]).reshape((n,n)), vmin = -8, vmax = 8)
    axs[0].plot(matrix_missing_index[1], matrix_missing_index[0], "r+")
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    sns.kdeplot(data = gpdd, palette = ["green"], ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-5,5)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'generated', 'gaussian'])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                                number_of_replicates, missingmatrixindex1,
                                                missingmatrixindex2, unconditional_generated_samples,
                                                figname):

    unconditional_vectors, unconditional_matrices = generate_student_nugget(minX, maxX, minY, maxY, n, 
                                                                            variance, lengthscale, df, 10*number_of_replicates,
                                                                            seed_value)
    unconditional_gpvectors, unconditional_gpmatrices = generate_gaussian_process(minX, maxX, minY, maxY,
                                                                                  n, variance, lengthscale, 10*number_of_replicates,
                                                                                  seed_value)

    #conditional_vectors is shape (number of replicates, m)
    bivariate_density = np.concatenate([(unconditional_matrices[:,0,missingmatrixindex1[0],missingmatrixindex1[1]]).reshape((10*number_of_replicates,1)),
    (unconditional_matrices[:,0,missingmatrixindex2[0],missingmatrixindex2[1]]).reshape((10*number_of_replicates,1))], axis = 1)

    generated1 = (unconditional_generated_samples[:,0,int(missingmatrixindex1[0]),int(missingmatrixindex1[1])]).reshape((number_of_replicates,1))
    generated2 = (unconditional_generated_samples[:,0,int(missingmatrixindex2[0]),int(missingmatrixindex2[1])]).reshape((number_of_replicates,1))
    generated_bivariate_density = np.concatenate([generated1,generated2], axis = 1)
    gpbivariatedensity = np.concatenate([(unconditional_gpmatrices[:,0,missingmatrixindex1[0], missingmatrixindex1[1]]).reshape((10*number_of_replicates,1)),
    (unconditional_gpmatrices[:,0,missingmatrixindex2[0], missingmatrixindex2[1]]).reshape((10*number_of_replicates,1))], axis = 1)


    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow((unconditional_matrices[0,:,:,:]).reshape((n,n)), vmin = -8, vmax = 8)
    axs[0].plot(missingmatrixindex1[0], missingmatrixindex1[1], "r+")
    axs[0].plot(missingmatrixindex2[0], missingmatrixindex2[1], "r+")
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1],
                ax = axs[1], levels = 20)
    sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                ax = axs[1], levels = 20)
    sns.kdeplot(x = gpbivariatedensity[:,0], y = gpbivariatedensity[:,1],
                ax = axs[1], levels = 20)
    axs[1].set_title("Bivariate")
    axs[1].set_xlim(-10,10)
    axs[1].set_ylim(-10,10)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'generated', 'gaussian'])
    plt.savefig(figname)
    plt.clf()



n = 32
minX = -10
maxX = 10
minY = -10
maxY = 10
variance = .4
lengthscale = 1.6
number_of_replicates = 5000
missing_index = 200
df = 3
seed_value = np.random.randint(0, 100000, 1)[0]
figname = "student_t_variance_.4_lengthscale_1.6_df_3_vs_gp_marginal_density.png"
produce_t_vs_normal_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                     seed_value, number_of_replicates, missing_index, figname)


"""
for i in range(342,400):
    print(i)
    missing_index = i
    n = 32
    minX = -10
    maxX = 10
    minY = -10
    maxY = 10
    variance = .4
    lengthscale = 1.6
    number_of_replicates = 1000
    df = 3

    unconditional_generated_samples = np.load((home_folder + 
                                            "/evaluation/diffusion_generation/data/model5/ref_image1/diffusion/model5_beta_min_max_01_20_random0_1000.npy"))
    figname = "marginal/model5/ref_image1/true_vs_generated_1000_marginal_density_ " + str(missing_index) + ".png"
    produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                                    number_of_replicates, missing_index, unconditional_generated_samples,
                                                    figname)
                                                    """
number_of_replicates = 1000
fixedindex = (15,15)
movingindex = [(i,j) for i in range(10,20) for j in range(10,20)]
unconditional_generated_samples = np.load((home_folder + 
                                            "/evaluation/diffusion_generation/data/model5/ref_image1/diffusion/model5_beta_min_max_01_20_random0_1000.npy"))
for movingind in movingindex:
    missing_index = movingind[0]*16+movingind[1]
    figname = "bivariate/model5/ref_image1/true_vs_generated_1000_marginal_density_ " + str(missing_index) + ".png"
    produce_true_and_generated_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                                number_of_replicates, fixedindex,
                                                movingind, unconditional_generated_samples,
                                                figname)