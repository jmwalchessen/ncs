import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from brown_resnick_data_generation import *
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

def visualize_spatial_field(observation, min_value, max_value, figname):

    fig, ax = plt.subplots(1)
    plt.imshow(observation, vmin = min_value, vmax = max_value)
    plt.savefig(figname)

def produce_true_and_generated_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                matrix_index, seed_value, unconditional_generated_samples,
                                                unconditional_true_samples,
                                                figname):

    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_matrices[:,0,matrix_index[0],matrix_index[1]]).reshape((number_of_replicates,1))
    generated_marginal_density = unconditional_generated_samples[:,0,int(matrix_index[0]),int(matrix_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    emp_mean = round(np.mean(marginal_density), 2)
    emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(0,.5)
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'diffusion'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_min_max_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                        seed_value, unconditional_generated_samples,
                                                        unconditional_true_samples,
                                                        figname):


    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    max_density = np.max(unconditional_matrices.reshape(number_of_replicates, n**2), axis = 1).reshape((number_of_replicates,1))
    min_density = np.min(unconditional_matrices.reshape(number_of_replicates, n**2), axis = 1).reshape((number_of_replicates,1))
    generated_max_density = (np.max(unconditional_generated_samples.reshape(number_of_replicates, n**2), axis = 1)).reshape((number_of_replicates, 1))
    generated_min_density = (np.min(unconditional_generated_samples.reshape(number_of_replicates, n**2), axis = 1)).reshape((number_of_replicates, 1))
    diff_density = np.subtract(max_density, min_density)
    generated_diff_density = np.subtract(generated_max_density, generated_min_density)

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(diff_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_diff_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title(" Min Max Difference Marginal")
    axs[1].set_xlim(2,12)
    axs[1].set_ylim(0,.5)
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'diffusion'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_min_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                        seed_value, unconditional_generated_samples,
                                                        unconditional_true_samples,
                                                        figname):


    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    min_density = np.min(unconditional_matrices.reshape(number_of_replicates, n**2), axis = 1).reshape((number_of_replicates,1))
    generated_min_density = (np.min(unconditional_generated_samples.reshape(number_of_replicates, n**2), axis = 1)).reshape((number_of_replicates, 1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(min_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_min_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title(" Min Marginal")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(0,1)
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'diffusion'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_max_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                        seed_value, unconditional_generated_samples,
                                                        unconditional_true_samples,
                                                        figname):


    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    max_density = np.max(unconditional_matrices.reshape(number_of_replicates, n**2), axis = 1).reshape((number_of_replicates,1))
    generated_max_density = (np.max(unconditional_generated_samples.reshape(number_of_replicates, n**2), axis = 1)).reshape((number_of_replicates, 1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(max_density,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_max_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title(" Max Marginal")
    axs[1].set_xlim(2,12)
    axs[1].set_ylim(0,1)
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'diffusion'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()


def produce_true_and_generated_bivariate_density(n, range_value, smooth_value,
                                                 number_of_replicates, matrixindex1, matrixindex2, seed_value,
                                                 unconditional_generated_samples, unconditional_true_samples,
                                                 figname):
    
    unconditional_matrices = np.log(unconditional_true_samples).reshape((number_of_replicates, 1, n, n))
    bivariate_density = np.concatenate([(unconditional_matrices[:,0,int(matrixindex1[0]),int(matrixindex1[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex2[0]),int(matrixindex2[1])]).reshape((number_of_replicates,1))], axis = 1).reshape((number_of_replicates,2))
    number_of_replicates = unconditional_matrices.shape[0]
    generated_bivariate_density = np.concatenate([(unconditional_generated_samples[:,0,int(matrixindex1[0]),int(matrixindex1[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,0,int(matrixindex2[0]),int(matrixindex2[1])]).reshape((number_of_replicates,1))],
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
    axs[0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 6)
    axs[0].plot(matrixindex1[1], matrixindex1[0], "r+")
    axs[0].plot(matrixindex2[1], matrixindex2[0], "r+")
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1], hue = 'class', shade = True, levels = 5, alpha = .5)
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-4,8)
    plt.ylim(-4,8)
    axs[1].set_title("Bivariate")
    index1 = matrix_index_to_index(matrixindex1, n)
    index2 = matrix_index_to_index(matrixindex2, n)
    #location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, index1)
    #rlocation1 = (round(location1[0],2), round(location1[1],2))
    #location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, index2)
    #rlocation2 = (round(location2[0],2), round(location2[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation1))
    #axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()


def produce_unconditional_metrics_with_variables():


    eval_folder = append_directory(2)
    diffusion_generation_folder = (eval_folder + "/diffusion_generation")
    n = 32
    range_value = 1.6
    smooth_value = 1.6
    number_of_replicates = 4000
    matrix_index = (15,15)
    seed_value = 201852
    unconditional_true_samples = np.log(np.load("brown_resnick_samples_range_1.6_smooth_1.6_4000.npy"))
    unconditional_generated_samples = np.load(diffusion_generation_folder + "/data/model2/ref_image1/diffusion/model2_random0_beta_min_max_01_20_1000.npy")
    figname = (diffusion_generation_folder + "/data/model2/ref_image1/marginal_density/model2_diff_marginal_density.png")
    produce_true_and_generated_min_max_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                            seed_value, unconditional_generated_samples,
                                                            unconditional_true_samples,
                                                            figname)

    figname = (diffusion_generation_folder + "/data/model2/ref_image1/marginal_density/model2_min_marginal_density.png")
    produce_true_and_generated_min_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                            seed_value, unconditional_generated_samples,
                                                            unconditional_true_samples,
                                                            figname)

    figname = (diffusion_generation_folder + "/data/model2/ref_image1/marginal_density/model2_max_marginal_density.png")
    produce_true_and_generated_max_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                            seed_value, unconditional_generated_samples,
                                                            unconditional_true_samples,
                                                            figname)
    for i in range(0, n, 4):
        for j in range(0, n, 4):

            matrix_index = (i,j)
            seed_value = np.random.randint(0, 100000, 1)[0]
            figname = (diffusion_generation_folder + "/data/model2/ref_image1/marginal_density/model2_marginal_density_" + str(i) + "_" + str(j) + ".png")
            produce_true_and_generated_marginal_density(n, range_value, smooth_value, number_of_replicates,
                                                        matrix_index, seed_value, unconditional_generated_samples,
                                                        unconditional_true_samples, figname)

