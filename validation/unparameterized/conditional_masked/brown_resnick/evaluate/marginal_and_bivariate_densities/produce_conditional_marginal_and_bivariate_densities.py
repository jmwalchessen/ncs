import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import numpy as np
from append_directories import *
data_generation_folder = (append_directory(3) + "/generate_data")
sys.path.append(data_generation_folder)
import generate_true_unconditional_samples

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
    plt.show()

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images


def produce_generated_log_marginal_density(minX, maxX, minY, maxY, n,
                                                number_of_replicates, missing_index,
                                                conditional_generated_samples, mask,
                                                ref_img, figname):
    
    matrix_index = index_to_matrix_index(missing_index, n)
    generated_marginal_density = (conditional_generated_samples[:,0,int(matrix_index[0]),int(matrix_index[1])]).astype(float)
    mask = mask.astype(float)
    ref_img = ref_img.astype(float)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = ["generated"])
    axs[0].imshow(ref_img, vmin = -4, vmax = 10, alpha = mask.reshape((n,n)))
    #using scott's method to compute bandwidth which depends on number of data points and dimension of data so as long as
    #bw_adjust is the same between true and generated and true and generated have same number of data points
    #and the same across pixels
    axs[0].plot(matrix_index[0], matrix_index[1], "r+")
    sns.kdeplot(data = generated_pdd["generated"], palette = ["orange"], bw_adjust = 1, ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,10)
    axs[1].set_ylim(0,1)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['generated'])
    print(figname)
    plt.savefig(figname)
    plt.clf()

def produce_generated_log_bivariate_density(minX, maxX, minY, maxY, n, missing_indices,
                                            number_of_replicates, ref_img, mask,
                                            figname):
    
    #log transformation
    conditional_generated_samples = log_transformation(conditional_generated_samples)
    matrix_index1 = index_to_matrix_index(missing_indices[0], n)
    matrix_index2 = index_to_matrix_index(missing_indices[1], n)
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,0,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,0,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(generated_bivariate_density, columns = ['x', 'y'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(ref_img.reshape((n,n)), vmin = -2, vmax = 2, alpha = mask.numpy().reshape((n,n)))
    axs[0].plot(matrix_index1[0], matrix_index1[1], "r+")
    axs[0].plot(matrix_index2[0], matrix_index2[1], "r+")
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', bw_method = "scott", bw_adjust = 1,
                ax = axs[1], shade = True, levels = 5, alpha = .5)
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
number_of_replicates = 1000
missing_index = 528
home_folder = append_directory(3)
conditional_generated_samples = np.load((home_folder + "/generate_data/data/conditional/ref_img1/model2_beta_min_max_01_25_random050_1000.npy"))
ref_img = np.load((home_folder + "/generate_data/data/conditional/ref_img1/ref_image1.npy"))
mask = np.load((home_folder + "/generate_data/data/conditional/ref_img1/mask.npy"))
figname = (home_folder + "/generate_data/data/conditional/ref_img1/marginal_density/model2_marginal_density_" + str(missing_index) + ".png")
"""
produce_generated_log_marginal_density(minX, maxX, minY, maxY, n,
                                                number_of_replicates, missing_index,
                                                conditional_generated_samples, mask,
                                                ref_img, figname)
                                                """
visualize_spatial_field(conditional_generated_samples[0,:,:,:].reshape((n,n)), -4, 10, figname)