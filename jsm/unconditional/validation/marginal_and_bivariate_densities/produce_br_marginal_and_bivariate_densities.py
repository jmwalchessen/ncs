import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
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

def matrix_index_to_index(matrix_index, n):

    index = matrix_index[0]*n+matrix_index[1]
    return index

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def visualize_spatial_field(observation, min_value, max_value, figname):

    fig, ax = plt.subplots(1)
    plt.imshow(observation, vmin = min_value, vmax = max_value)
    plt.savefig(figname)

def produce_true_and_generated_marginal_densities1(n, number_of_replicates, matrix_index1,
                                                   matrix_index2, matrix_index3, matrix_index4,
                                                   unconditional_generated_samples,
                                                   br_samples, figname):

    #conditional_vectors is shape (number of replicates, m)
    marginal_density1 = (br_samples[:,0,matrix_index1[0],matrix_index1[1]]).reshape((number_of_replicates,1))
    marginal_density2 = (br_samples[:,0,matrix_index2[0],matrix_index2[1]]).reshape((number_of_replicates,1))
    marginal_density3 = (br_samples[:,0,matrix_index3[0],matrix_index3[1]]).reshape((number_of_replicates,1))
    marginal_density4 = (br_samples[:,0,matrix_index4[0],matrix_index4[1]]).reshape((number_of_replicates,1))
    generated_marginal_density1 = unconditional_generated_samples[:,0,int(matrix_index1[0]),int(matrix_index1[1])]
    generated_marginal_density2 = unconditional_generated_samples[:,0,int(matrix_index2[0]),int(matrix_index2[1])]
    generated_marginal_density3 = unconditional_generated_samples[:,0,int(matrix_index3[0]),int(matrix_index3[1])]
    generated_marginal_density4 = unconditional_generated_samples[:,0,int(matrix_index4[0]),int(matrix_index4[1])]
    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (20,10))
    pdd = pd.DataFrame(marginal_density1,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density1,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,0].imshow(br_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[0,0].plot(matrix_index1[1], matrix_index1[0], "ro", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[0,1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[0,1])
    axs[0,1].set_xticks(ticks = [-4,-2,0,2,4,6,8,10], labels = [-4,-2,0,2,4,6,8,10])
    axs[0,1].set_yticks(ticks = [0,.1,.2,.3,.4], labels = [0,.1,.2,.3,.4])
    axs[0,1].set_title("Marginal")
    axs[0,1].set(xlabel = None, ylabel = None)
    axs[0,1].set_xlim(-4,10)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[0,1].legend(labels = ['true', 'diffusion'])
    axs[0,0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0,0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])

    #second set
    pdd = pd.DataFrame(marginal_density2,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density2,
                                    columns = None)
    axs[0,2].imshow(br_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[0,2].plot(matrix_index2[1], matrix_index2[0], "ro", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[0,3], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[0,3])
    axs[0,3].set_xticks(ticks = [-4,-2,0,2,4,6,8,10], labels = [-4,-2,0,2,4,6,8,10])
    axs[0,3].set_yticks(ticks = [0,.1,.2,.3,.4], labels = [0,.1,.2,.3,.4])
    axs[0,3].set_title("Marginal")
    axs[0,3].set(xlabel = None, ylabel = None)
    axs[0,3].set_xlim(-4,10)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[0,3].legend(labels = ['true', 'diffusion'])
    axs[0,2].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0,2].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])

    #third set
    pdd = pd.DataFrame(marginal_density3,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density3,
                                    columns = None)
    axs[1,0].imshow(br_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[1,0].plot(matrix_index3[1], matrix_index3[0], "ro", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1,1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1,1])
    axs[1,1].set_xticks(ticks = [-4,-2,0,2,4,6,8,10], labels = [-4,-2,0,2,4,6,8,10])
    axs[1,1].set_yticks(ticks = [0,.1,.2,.3,.4], labels = [0,.1,.2,.3,.4])
    axs[1,1].set_title("Marginal")
    axs[1,1].set(xlabel = None, ylabel = None)
    axs[1,1].set_xlim(-4,10)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1,1].legend(labels = ['true', 'diffusion'])
    axs[1,0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])

    #fourth set
    pdd = pd.DataFrame(marginal_density4,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density4,
                                    columns = None)
    axs[1,2].imshow(br_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[1,2].plot(matrix_index4[1], matrix_index4[0], "ro", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1,3], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1,3])
    axs[1,3].set_xticks(ticks = [-4,-2,0,2,4,6,8,10], labels = [-4,-2,0,2,4,6,8,10])
    axs[1,3].set_yticks(ticks = [0,.1,.2,.3,.4], labels = [0,.1,.2,.3,.4])
    axs[1,3].set_title("Marginal")
    axs[1,3].set(xlabel = None, ylabel = None)
    axs[1,3].set_xlim(-4,10)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1,3].legend(labels = ['true', 'diffusion'])
    axs[1,2].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,2].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_bivariate_densities1(n, number_of_replicates, matrixindex11,
                                                    matrixindex12, matrixindex21, matrixindex22,
                                                    matrixindex31, matrixindex32, matrixindex41,
                                                    matrixindex42, unconditional_generated_samples,
                                                    br_samples, figname):
    
    

    bivariate_density1 = np.concatenate([(br_samples[:,0,int(matrixindex11[0]),int(matrixindex11[1])]).reshape((number_of_replicates,1)),
                                        (br_samples[:,0,int(matrixindex12[0]),int(matrixindex12[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density2 = np.concatenate([(br_samples[:,0,int(matrixindex21[0]),int(matrixindex21[1])]).reshape((number_of_replicates,1)),
                                        (br_samples[:,0,int(matrixindex22[0]),int(matrixindex22[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density3 = np.concatenate([(br_samples[:,0,int(matrixindex31[0]),int(matrixindex31[1])]).reshape((number_of_replicates,1)),
                                        (br_samples[:,0,int(matrixindex32[0]),int(matrixindex32[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density4 = np.concatenate([(br_samples[:,0,int(matrixindex41[0]),int(matrixindex41[1])]).reshape((number_of_replicates,1)),
                                        (br_samples[:,0,int(matrixindex42[0]),int(matrixindex42[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    
    generated_bivariate_density1 = np.concatenate([(unconditional_generated_samples[:,0,int(matrixindex11[0]),int(matrixindex11[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,0,int(matrixindex12[0]),int(matrixindex12[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density2 = np.concatenate([(unconditional_generated_samples[:,0,int(matrixindex21[0]),int(matrixindex21[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,0,int(matrixindex22[0]),int(matrixindex22[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density3 = np.concatenate([(unconditional_generated_samples[:,0,int(matrixindex31[0]),int(matrixindex31[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,0,int(matrixindex32[0]),int(matrixindex32[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density4 = np.concatenate([(unconditional_generated_samples[:,0,int(matrixindex41[0]),int(matrixindex41[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,0,int(matrixindex42[0]),int(matrixindex42[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    bivariate_density1 = np.concatenate([bivariate_density1, generated_bivariate_density1], axis = 0)
    bivariate_density2 = np.concatenate([bivariate_density2, generated_bivariate_density2], axis = 0)
    bivariate_density3 = np.concatenate([bivariate_density3, generated_bivariate_density3], axis = 0)
    bivariate_density4 = np.concatenate([bivariate_density4, generated_bivariate_density4], axis = 0)
    class_vector = np.concatenate([(np.repeat('true', number_of_replicates)).reshape((number_of_replicates,1)),
                                   (np.repeat('generated', number_of_replicates)).reshape((number_of_replicates,1))], axis = 0)
    bivariate_density1 = np.concatenate([bivariate_density1, class_vector], axis = 1)
    bivariate_density2 = np.concatenate([bivariate_density2, class_vector], axis = 1)
    bivariate_density3 = np.concatenate([bivariate_density3, class_vector], axis = 1)
    bivariate_density4 = np.concatenate([bivariate_density4, class_vector], axis = 1)
    fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (20,10))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density1,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,0].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[0,0].plot(matrixindex11[1], matrixindex11[0], "k^", markersize = 20, linewidth = 20)
    axs[0,0].plot(matrixindex12[1], matrixindex12[0], "r^", markersize = 20, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[0,1], hue = 'class', fill = False, levels = 8, alpha = .8)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    axs[0,1].set_xlim(-4,10)
    axs[0,1].set_ylim(-4,10)
    axs[0,1].set_title("Bivariate")
    axs[0,1].set(xlabel = None, ylabel = None)

    axs[0,0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0,0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    #location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, index1)
    #rlocation1 = (round(location1[0],2), round(location1[1],2))
    #location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, index2)
    #rlocation2 = (round(location2[0],2), round(location2[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation1))
    #axs[1].set_ylabel("location: " + str(rlocation2))
    axs[0,1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])

    #second set
    pdd = pd.DataFrame(bivariate_density2,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,2].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[0,2].plot(matrixindex21[1], matrixindex21[0], "k^", markersize = 20, linewidth = 20)
    axs[0,2].plot(matrixindex22[1], matrixindex22[0], "r^", markersize = 20, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[0,3], hue = 'class', fill = False, levels = 8, alpha = .8)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    axs[0,3].set_xlim(-4,10)
    axs[0,3].set_ylim(-4,10)
    axs[0,3].set_title("Bivariate")
    axs[0,3].set(xlabel = None, ylabel = None)

    axs[0,2].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0,2].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0,3].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])

    #third set
    pdd = pd.DataFrame(bivariate_density3,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[1,0].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[1,0].plot(matrixindex31[1], matrixindex31[0], "k^", markersize = 20, linewidth = 20)
    axs[1,0].plot(matrixindex32[1], matrixindex32[0], "r^", markersize = 20, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1,1], hue = 'class', fill = False, levels = 8, alpha = .8)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    axs[1,1].set_xlim(-4,10)
    axs[1,1].set_ylim(-4,10)
    axs[1,1].set_title("Bivariate")
    axs[1,1].set(xlabel = None, ylabel = None)

    axs[1,0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])

    #fourth set
    pdd = pd.DataFrame(bivariate_density4,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[1,2].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[1,2].plot(matrixindex41[1], matrixindex41[0], "k^", markersize = 20, linewidth = 20)
    axs[1,2].plot(matrixindex42[1], matrixindex42[0], "r^", markersize = 20, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1,3], hue = 'class', fill = False, levels = 8, alpha = .8)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    axs[1,3].set_xlim(-4,10)
    axs[1,3].set_ylim(-4,10)
    axs[1,3].set_title("Bivariate")
    axs[1,3].set(xlabel = None, ylabel = None)

    axs[1,2].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,2].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,3].legend(handles = [blue_patch, orange_patch],labels = ['true', 'diffusion'])
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
matrix_index1 = (15,15)
matrix_index2 = (8,8)
matrix_index3 = (24,24)
matrix_index4 = (24,8)
seed_value = 2343
home_folder = "/home/julia/Dropbox/diffusion/twisted_diffusion/validation/unparameterized/conditional_masked/brown_resnick/generate_data/data/"
unconditional_generated_samples = np.load((home_folder + "unconditional/diffusion/model3_beta_min_max_01_20_random0_1000.npy"))
br_samples = (log_transformation(np.load("brown_resnick_samples_1024_1000.npy"))).reshape((number_of_replicates, 1, n, n))
figname = "br_unconditional_marginal_density.png"
produce_true_and_generated_marginal_densities1(n, number_of_replicates, matrix_index1,
                                               matrix_index2, matrix_index3, matrix_index4,
                                               unconditional_generated_samples,
                                               br_samples, figname)

matrixindex11 = (16,16)
matrixindex12 = (15,16)
matrixindex21 = (8,24)
matrixindex22 = (24,8)
matrixindex31 = (3,3)
matrixindex32 = (3,4)
matrixindex41 = (16,7)
matrixindex42 = (16,8)
figname = "br_unconditional_bivariate_1000.png"

produce_true_and_generated_bivariate_densities1(n, number_of_replicates, matrixindex11,
                                                    matrixindex12, matrixindex21, matrixindex22,
                                                    matrixindex31, matrixindex32, matrixindex41,
                                                    matrixindex42, unconditional_generated_samples,
                                                    br_samples, figname)