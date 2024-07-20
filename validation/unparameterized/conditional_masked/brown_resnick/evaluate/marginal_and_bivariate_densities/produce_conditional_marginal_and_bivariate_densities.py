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
    return (ylocation, xlocation)

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

def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def global_boundary_process(images, minvalue, maxvalue):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - .5
    log01cs = 6*log01c
    return log01cs

def log_and_normalize(images):

    images = np.log(images)
    images = (images - np.mean(images))/np.std(images)
    return images

def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs

def global_quantile_boundary_inverse(images, minvalue, maxvalue, quantvalue01):

    images = (images/6)+quantvalue01
    images = (maxvalue-minvalue)*images
    images = images + minvalue
    return images


def produce_generated_marginal_density(minX, maxX, minY, maxY, n,
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
    axs[0].imshow(ref_img, vmin = -3, vmax = 3, alpha = mask.reshape((n,n)))
    #using scott's method to compute bandwidth which depends on number of data points and dimension of data so as long as
    #bw_adjust is the same between true and generated and true and generated have same number of data points
    #and the same across pixels
    axs[0].plot(matrix_index[1], matrix_index[0], "r+")
    sns.kdeplot(data = generated_pdd["generated"], palette = ["orange"], bw_adjust = 1, ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,10)
    axs[1].set_ylim(0,2)
    location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_index)
    rlocation = (round(location[0],2), round(location[1],2))
    axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['generated'])
    print(figname)
    plt.savefig(figname)
    plt.clf()

def produce_generated_bivariate_density(minX, maxX, minY, maxY, n, missing_indices,
                                            number_of_replicates, ref_img, mask,
                                            figname):
    
    matrix_index1 = index_to_matrix_index(missing_indices[0], n)
    matrix_index2 = index_to_matrix_index(missing_indices[1], n)
    print("matrix index")
    print(matrix_index1)
    print(matrix_index2)
    generated_bivariate_density = np.concatenate([(conditional_generated_samples[:,0,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                                   (conditional_generated_samples[:,0,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)

    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(generated_bivariate_density, columns = ['x', 'y'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(ref_img.reshape((n,n)), vmin = -3, vmax = 3,
                  alpha = (mask.reshape((n,n))).astype(float))
    axs[0].plot(matrix_index1[1], matrix_index1[0], "r+")
    axs[0].plot(matrix_index2[1], matrix_index2[0], "r+")
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', bw_method = "scott", bw_adjust = 1,
                ax = axs[1], fill=True)
    plt.xlim(-4,10)
    plt.ylim(-4,10)
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
home_folder = append_directory(3)
br_folder = (append_directory(7) + "/brown_resnick/sde_diffusion/masked/unparameterized")
trainlogmaxmin = np.load((br_folder + "/trained_score_models/vpsde/model10_train_logminmax.npy"))
conditional_generated_samples = np.load((home_folder + "/generate_data/data/conditional/model10/ref_img2/model10_random50_beta_min_max_01_20_1000_random50_1000.npy"))
ref_img = np.load((home_folder + "/generate_data/data/conditional/model10/ref_img2/ref_image2.npy"))
mask = np.load((home_folder + "/generate_data/data/conditional/model10/ref_img2/mask.npy"))
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
m = missing_indices.shape[0]
conditional_generated_samples = global_quantile_boundary_inverse(conditional_generated_samples, trainlogmaxmin[0],
                                                                 trainlogmaxmin[1], trainlogmaxmin[2])

"""

for missing_index in range(0,1024):
    print(missing_index)
    matrix_index = index_to_matrix_index(missing_index, n)
    generated_marginal_density = (conditional_generated_samples[:,0,int(matrix_index[0]),int(matrix_index[1])]).astype(float)
    if(np.unique(generated_marginal_density).shape[0] > 1):
        figname = (home_folder + "/generate_data/data/conditional/model10/ref_img2/marginal_density/model10_rebounded_log_scale_marginal_density_" + 
               str(missing_index) + ".png")
        produce_generated_marginal_density(minX, maxX, minY, maxY, n,
                                           number_of_replicates, missing_index,
                                           conditional_generated_samples, mask,
                                           ref_img, figname)
    



"""
missing_indices1 = [603,  604,  605,  606,  607,  608,  609]
missing_indices2 = [603,  604,  605,  606,  607,  608,  609]
print(missing_indices)

for i in missing_indices1:
   for j in missing_indices2:
        
        matrix_index1 = index_to_matrix_index(i, n)
        matrix_index2 = index_to_matrix_index(j, n)
        generated_marginal_density1 = (conditional_generated_samples[:,0,int(matrix_index1[0]),int(matrix_index1[1])]).astype(float)
        generated_marginal_density2 = (conditional_generated_samples[:,0,int(matrix_index2[0]),int(matrix_index2[1])]).astype(float)
        if(np.unique(generated_marginal_density1).shape[0] > 1):
            if(np.unique(generated_marginal_density2).shape[0] > 1):
                figname = (home_folder + "/generate_data/data/conditional/model10/ref_img2/bivariate_density/model10_rebounded_log_scale_bivariate_density_" + 
                    str(i) + "_" + str(j) + ".png")
                produce_generated_bivariate_density(minX, maxX, minY, maxY, n, (i,j),
                                            number_of_replicates, ref_img, mask,
                                            figname)
                


