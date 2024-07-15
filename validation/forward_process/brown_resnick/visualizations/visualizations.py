import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from brown_resnick_data_generation import *
home_folder = append_directory(5)
sys.path.append((home_folder + "/brown_resnick/sde_diffusion/masked/unparameterized"))
from sde_lib import *
import seaborn as sns
import pandas as pd

sdevp = VPSDE(beta_min=0.1, beta_max=20, N=1000)

range_value = 1.6
smooth_value = 1.6
seed_value = 234231
n = 32
br_samples = np.load("brown_resnick_samples_1000.npy")


def construct_norm_matrix(minX, maxX, minY, maxY, n):
    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    X_matrix = (np.repeat(X, n**2, axis = 0)).reshape((n**2, n**2))
    Y_matrix = (np.repeat(Y, n**2, axis = 0)).reshape((n**2, n**2))
    longitude_squared = np.square(np.subtract(X_matrix, np.transpose(X_matrix)))
    latitude_squared = np.square(np.subtract(Y_matrix, np.transpose(Y_matrix)))
    norm_matrix = np.sqrt(np.add(longitude_squared, latitude_squared))
    return norm_matrix

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return exp_kernel


def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    C = np.linalg.cholesky(kernel)
    y_matrix = (np.flip(np.matmul(np.transpose(C),
                                  np.transpose(z_matrix))))
    
    gp_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, y_matrix.shape[1]):
        gp_matrix[i,:,:,:] = y_matrix[:,i].reshape((1,n,n))
    return gp_matrix

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log_and_normalize(images):

    images = np.log(images)
    images = (images - np.mean(images))/np.std(images)
    return images

def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def forward_process(sde, br_sample, t):

    noise = th.randn_like(br_sample)
    perturbed_data = sde.sqrt_alphas_cumprod[t, None, None, None] * br_sample + \
                     sde.sqrt_1m_alphas_cumprod[t, None, None, None] * noise
    return perturbed_data

def visualize(sde, br_sample, t, n):

    perturbed_data = forward_process(sde, br_sample, t)
    perturbed_data = perturbed_data.numpy().reshape((n,n))
    plt.imshow(perturbed_data, vmin = -2, vmax = 2)
    plt.show()

def plot_marginal_density(sde, br_samples, t, n, number_of_replicates, matrix_index, figname):
    fig, ax = plt.subplots(1)
    br_samples = th.from_numpy(log_and_boundary_process(br_samples.reshape((number_of_replicates,n,n))))
    perturbed_data = forward_process(sde, br_samples, t)
    marginal_density = perturbed_data[:,matrix_index[0],matrix_index[1]]
    pdd = pd.DataFrame(marginal_density, columns = ['true'])
    sns.kdeplot(data = pdd["true"], palette = ["orange"], bw_adjust = 1, ax = ax)
    ax.set_xlim(-4,4)
    plt.savefig(figname)

def plot_bivariate_density(sde, br_samples, t, n, number_of_replicates, matrixindex1, matrixindex2, figname):

    fig,ax = plt.subplots(1)
    br_samples = th.from_numpy(log_and_boundary_process(br_samples.reshape((number_of_replicates,n,n))))
    perturbed_data = forward_process(sde, br_samples, t)
    generated_bivariate_density = np.concatenate([(perturbed_data[:,int(matrixindex1[0]),int(matrixindex1[1])]).reshape((number_of_replicates,1)),
                                                   (perturbed_data[:,int(matrixindex2[0]),int(matrixindex2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    pdd = pd.DataFrame(generated_bivariate_density, columns = ['x', 'y'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', bw_method = "scott", bw_adjust = 1,
                ax = ax, fill=True)
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    plt.savefig(figname)

def plot_gp_marginal_density(sde, n, t, number_of_replicates, matrix_index, figname):
    fig, ax = plt.subplots(1)
    gp_samples = generate_gaussian_process(minX = -10, maxX = 10, minY = -10, maxY = 10, n = n, variance = .4,
                                           lengthscale = 1.6, number_of_replicates = number_of_replicates,
                                           seed_value = 234234)
    gp_samples = th.from_numpy(gp_samples)
    perturbed_data = forward_process(sde, gp_samples, t)
    marginal_density = perturbed_data[:,0,matrix_index[0],matrix_index[1]]
    pdd = pd.DataFrame(marginal_density, columns = ['true'])
    sns.kdeplot(data = pdd["true"], palette = ["orange"], bw_adjust = 1, ax = ax)
    plt.savefig(figname)

def plot_gp_bivariate_density(sde, n, t, number_of_replicates, matrixindex1, matrixindex2, figname):

    fig,ax = plt.subplots(1)
    gp_samples = generate_gaussian_process(minX = -10, maxX = 10, minY = -10, maxY = 10, n = n, variance = .4,
                                           lengthscale = 1.6, number_of_replicates = number_of_replicates,
                                           seed_value = 23423)
    print(gp_samples.shape)
    gp_samples = (th.from_numpy(gp_samples))
    perturbed_data = forward_process(sde, gp_samples, t)
    generated_bivariate_density = np.concatenate([(perturbed_data[:,0,int(matrixindex1[0]),int(matrixindex1[1])]).reshape((number_of_replicates,1)),
                                                   (perturbed_data[:,0,int(matrixindex2[0]),int(matrixindex2[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    pdd = pd.DataFrame(generated_bivariate_density, columns = ['x', 'y'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', bw_method = "scott", bw_adjust = 1,
                ax = ax, fill=True)
    plt.savefig(figname)




number_of_replicates = 1000
matrix_index1 = (8,16)
matrix_index2 = (8,17)
sdevp = VPSDE(beta_min=0.1, beta_max=20, N = 1000)
for t in range(0, 1000, 10):
    figname = ("brown_resnick/logbounded/bivariate_density/br_bivariate_density_" + str(t) + "_index_" + str(matrix_index1[0]) 
           + "_" + str(matrix_index1[1]) + "_" + str(matrix_index2[0]) + "_" + str(matrix_index2[1]) + ".png")
    plot_bivariate_density(sdevp, br_samples, t, n, number_of_replicates, matrix_index1, matrix_index2, figname)
    figname = ("brown_resnick/logbounded/marginal_density/br_marginal_density_" + str(t) + "_index_" + str(matrix_index1[0]) 
           + "_" + str(matrix_index1[1]) + ".png")
    plot_marginal_density(sdevp, br_samples, t, n, number_of_replicates, matrix_index1, figname)
