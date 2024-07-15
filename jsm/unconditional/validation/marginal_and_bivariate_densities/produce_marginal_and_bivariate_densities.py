import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
from true_unconditioanl_gp_data_generation import *


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

def produce_bivariate_density(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, matrix_index1, matrix_index2, seed_value):
    
    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    bivariate_density = (unconditional_matrices[:,matrix_index1[0],matrix_index2[1]]).reshape((number_of_replicates,2))
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:], alpha = (1-mask), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index1[0], matrix_index1[1], "r+")
    axs[0].plot(matrix_index2[0], matrix_index2[1], "r+")
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1],
                ax = axs[1])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1].set_title("Marginal")
    #location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    #rlocation1 = (round(location1[0],2), round(location1[1],2))
    #location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    #rlocation2 = (round(location2[0],2), round(location2[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation1))
    #axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(labels = ['true'])
    plt.show()

def produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                                  number_of_replicates, matrix_index, seed_value,
                                  unconditional_generated_samples,
                                  figname):

    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (unconditional_matrices[:,0,matrix_index[0],matrix_index[1]]).reshape((number_of_replicates,1))
    generated_marginal_density = unconditional_generated_samples[:,int(matrix_index[0]),int(matrix_index[1])]

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
    axs[0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'diffusion'])
    axs[1].patch.set_facecolor('red')
    axs[1].patch.set_alpha(0.2)
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_marginal_densities(minX, maxX, minY, maxY, n, variance, lengthscale,
                                  number_of_replicates, matrix_index1, matrix_index2, matrix_index3,
                                  seed_value, unconditional_generated_samples, figname):

    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density1 = (unconditional_matrices[:,0,matrix_index1[0],matrix_index1[1]]).reshape((number_of_replicates,1))
    marginal_density2 = (unconditional_matrices[:,0,matrix_index2[0],matrix_index2[1]]).reshape((number_of_replicates,1))
    marginal_density3 = (unconditional_matrices[:,0,matrix_index3[0],matrix_index3[1]]).reshape((number_of_replicates,1))
    generated_marginal_density1 = unconditional_generated_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]
    generated_marginal_density2 = unconditional_generated_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]
    generated_marginal_density3 = unconditional_generated_samples[:,int(matrix_index3[0]),int(matrix_index3[1])]
    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 4, figsize = (20,5))
    pdd = pd.DataFrame(marginal_density1,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density1,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "rx", markersize = 20, linewidth = 20)
    axs[0].plot(matrix_index2[1], matrix_index2[0], "yx", markersize = 20, linewidth = 20)
    axs[0].plot(matrix_index3[1], matrix_index3[0], "mx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'diffusion'])
    axs[1].patch.set_facecolor('red')
    axs[1].patch.set_alpha(0.2)
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])

    pdd = pd.DataFrame(marginal_density2,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density2,
                                    columns = None)
    sns.kdeplot(data = pdd, ax = axs[2], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[2])
    axs[2].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[2].legend(labels = ['true', 'diffusion'])
    axs[2].patch.set_facecolor('yellow')
    axs[2].patch.set_alpha(0.2)

    pdd = pd.DataFrame(marginal_density3,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density3,
                                    columns = None)
    sns.kdeplot(data = pdd, ax = axs[3], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[3])
    axs[3].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[3].legend(labels = ['true', 'diffusion'])
    axs[3].patch.set_facecolor('purple')
    axs[3].patch.set_alpha(0.2)
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_marginal_densities1(minX, maxX, minY, maxY, n, variance, lengthscale,
                                  number_of_replicates, matrix_index1, matrix_index2, matrix_index3,
                                  matrix_index4, seed_value, unconditional_generated_samples, figname):

    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    #conditional_vectors is shape (number of replicates, m)
    marginal_density1 = (unconditional_matrices[:,0,matrix_index1[0],matrix_index1[1]]).reshape((number_of_replicates,1))
    marginal_density2 = (unconditional_matrices[:,0,matrix_index2[0],matrix_index2[1]]).reshape((number_of_replicates,1))
    marginal_density3 = (unconditional_matrices[:,0,matrix_index3[0],matrix_index3[1]]).reshape((number_of_replicates,1))
    marginal_density4 = (unconditional_matrices[:,0,matrix_index4[0],matrix_index4[1]]).reshape((number_of_replicates,1))
    generated_marginal_density1 = unconditional_generated_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]
    generated_marginal_density2 = unconditional_generated_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]
    generated_marginal_density3 = unconditional_generated_samples[:,int(matrix_index3[0]),int(matrix_index3[1])]
    generated_marginal_density4 = unconditional_generated_samples[:,int(matrix_index4[0]),int(matrix_index4[1])]
    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (20,10))
    pdd = pd.DataFrame(marginal_density1,
                                    columns = None)
    generated_pdd = pd.DataFrame(generated_marginal_density1,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0,0].plot(matrix_index1[1], matrix_index1[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[0,1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[0,1])
    axs[0,1].set_title("Marginal")
    axs[0,1].set_xlim(-2,2)
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
    axs[0,2].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0,2].plot(matrix_index2[1], matrix_index2[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[0,3], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[0,3])
    axs[0,3].set_title("Marginal")
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
    axs[1,0].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[1,0].plot(matrix_index3[1], matrix_index3[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1,1], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1,1])
    axs[1,1].set_title("Marginal")
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
    axs[1,2].imshow(unconditional_matrices[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[1,2].plot(matrix_index4[1], matrix_index4[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1,3], palette=['blue'])
    sns.kdeplot(data = generated_pdd, palette = ["orange"], ax = axs[1,3])
    axs[1,3].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1,3].legend(labels = ['true', 'diffusion'])
    axs[1,2].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[1,2].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_true_and_generated_bivariate_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, matrixindex1, matrixindex2, seed_value,
                                                 unconditional_generated_samples, figname):
    
    
    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    bivariate_density = np.concatenate([(unconditional_matrices[:,0,int(matrixindex1[0]),int(matrixindex1[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex2[0]),int(matrixindex2[1])]).reshape((number_of_replicates,1))], axis = 1).reshape((number_of_replicates,2))
    number_of_replicates = unconditional_generated_samples.shape[0]
    generated_bivariate_density = np.concatenate([(unconditional_generated_samples[:,int(matrixindex1[0]),int(matrixindex1[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex2[0]),int(matrixindex2[1])]).reshape((number_of_replicates,1))],
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
    axs[0].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrixindex1[1], matrixindex1[0], "r+")
    axs[0].plot(matrixindex2[1], matrixindex2[0], "r+")
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1], hue = 'class', shade = True, levels = 5, alpha = .5)
    #kde2 = sns.kdeplot(x = generated_bivariate_density[:,0], y = generated_bivariate_density[:,1],
                #ax = axs[1], color = 'orange', levels = 5, label = "generated")
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1].set_title("Marginal")
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



def produce_true_and_generated_bivariate_densities(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, matrixindex11, matrixindex12,
                                                 matrixindex21, matrixindex22, seed_value,
                                                 unconditional_generated_samples, figname):
    
    
    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    bivariate_density1 = np.concatenate([(unconditional_matrices[:,0,int(matrixindex11[0]),int(matrixindex11[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex12[0]),int(matrixindex12[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density2 = np.concatenate([(unconditional_matrices[:,0,int(matrixindex21[0]),int(matrixindex21[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex22[0]),int(matrixindex22[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    

    generated_bivariate_density1 = np.concatenate([(unconditional_generated_samples[:,int(matrixindex11[0]),int(matrixindex11[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex12[0]),int(matrixindex12[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density2 = np.concatenate([(unconditional_generated_samples[:,int(matrixindex21[0]),int(matrixindex21[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex22[0]),int(matrixindex22[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    bivariate_density1 = np.concatenate([bivariate_density1, generated_bivariate_density1], axis = 0)
    bivariate_density2 = np.concatenate([bivariate_density2, generated_bivariate_density2], axis = 0)
    class_vector = np.concatenate([(np.repeat('true', number_of_replicates)).reshape((number_of_replicates,1)),
                                   (np.repeat('generated', number_of_replicates)).reshape((number_of_replicates,1))], axis = 0)
    bivariate_density1 = np.concatenate([bivariate_density1, class_vector], axis = 1)
    bivariate_density2 = np.concatenate([bivariate_density2, class_vector], axis = 1)
    fig, axs = plt.subplots(ncols = 4, figsize = (20,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density1,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0].plot(matrixindex11[1], matrixindex11[0], "rx", markersize = 10, linewidth = 20)
    axs[0].plot(matrixindex12[1], matrixindex12[0], "rx", markersize = 10, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1], hue = 'class', shade = True, levels = 5, alpha = .5)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1].set_title("Bivariate")
    #location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, index1)
    #rlocation1 = (round(location1[0],2), round(location1[1],2))
    #location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, index2)
    #rlocation2 = (round(location2[0],2), round(location2[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation1))
    #axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])


    pdd = pd.DataFrame(bivariate_density2,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[2].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[2].plot(matrixindex21[1], matrixindex21[0], "rx", markersize = 10, linewidth = 20)
    axs[2].plot(matrixindex22[1], matrixindex22[0], "rx", markersize = 10, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[3], hue = 'class', shade = True, levels = 5, alpha = .5)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[3].set_title("Bivariate")
    axs[3].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()


def produce_true_and_generated_bivariate_densities1(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, matrixindex11, matrixindex12,
                                                 matrixindex21, matrixindex22, matrixindex31, matrixindex32,
                                                 matrixindex41, matrixindex42, seed_value,
                                                 unconditional_generated_samples, figname):
    
    
    unconditional_vectors, unconditional_matrices = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                                              variance, lengthscale,
                                                                              number_of_replicates, seed_value)
    bivariate_density1 = np.concatenate([(unconditional_matrices[:,0,int(matrixindex11[0]),int(matrixindex11[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex12[0]),int(matrixindex12[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density2 = np.concatenate([(unconditional_matrices[:,0,int(matrixindex21[0]),int(matrixindex21[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex22[0]),int(matrixindex22[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density3 = np.concatenate([(unconditional_matrices[:,0,int(matrixindex31[0]),int(matrixindex31[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex32[0]),int(matrixindex32[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    bivariate_density4 = np.concatenate([(unconditional_matrices[:,0,int(matrixindex41[0]),int(matrixindex41[1])]).reshape((number_of_replicates,1)),
                                        (unconditional_matrices[:,0,int(matrixindex42[0]),int(matrixindex42[1])]).reshape((number_of_replicates,1))],
                                        axis = 1).reshape((number_of_replicates,2))
    

    generated_bivariate_density1 = np.concatenate([(unconditional_generated_samples[:,int(matrixindex11[0]),int(matrixindex11[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex12[0]),int(matrixindex12[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density2 = np.concatenate([(unconditional_generated_samples[:,int(matrixindex21[0]),int(matrixindex21[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex22[0]),int(matrixindex22[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density3 = np.concatenate([(unconditional_generated_samples[:,int(matrixindex31[0]),int(matrixindex31[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex32[0]),int(matrixindex32[1])]).reshape((number_of_replicates,1))],
                                                   axis = 1)
    generated_bivariate_density4 = np.concatenate([(unconditional_generated_samples[:,int(matrixindex41[0]),int(matrixindex41[1])]).reshape((number_of_replicates,1)),
                                                   (unconditional_generated_samples[:,int(matrixindex42[0]),int(matrixindex42[1])]).reshape((number_of_replicates,1))],
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
    axs[0,0].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0,0].plot(matrixindex11[1], matrixindex11[0], "rx", markersize = 10, linewidth = 20)
    axs[0,0].plot(matrixindex12[1], matrixindex12[0], "rx", markersize = 10, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[0,1], hue = 'class', shade = True, levels = 5, alpha = .5)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[0,1].set_title("Bivariate")
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
    axs[0,2].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[0,2].plot(matrixindex21[1], matrixindex21[0], "rx", markersize = 10, linewidth = 20)
    axs[0,2].plot(matrixindex22[1], matrixindex22[0], "rx", markersize = 10, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[0,3], hue = 'class', shade = True, levels = 5, alpha = .5)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[0,3].set_title("Bivariate")
    axs[0,3].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])

    #third set
    pdd = pd.DataFrame(bivariate_density3,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[1,0].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[1,0].plot(matrixindex31[1], matrixindex31[0], "rx", markersize = 10, linewidth = 20)
    axs[1,0].plot(matrixindex32[1], matrixindex32[0], "rx", markersize = 10, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1,1], hue = 'class', shade = True, levels = 5, alpha = .5)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1,1].set_title("Bivariate")
    axs[1,1].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])

    #fourth set
    pdd = pd.DataFrame(bivariate_density4,
                                    columns = ['x', 'y', 'class'])
    pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[1,2].imshow(unconditional_generated_samples[0,:,:].reshape((n,n)), vmin = -2, vmax = 2)
    axs[1,2].plot(matrixindex41[1], matrixindex41[0], "rx", markersize = 10, linewidth = 20)
    axs[1,2].plot(matrixindex42[1], matrixindex42[0], "rx", markersize = 10, linewidth = 20)
    kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y',
                ax = axs[1,3], hue = 'class', shade = True, levels = 5, alpha = .5)
    blue_patch = mpatches.Patch(color='blue')
    orange_patch = mpatches.Patch(color='orange')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    axs[1,3].set_title("Bivariate")
    axs[1,3].legend(handles = [blue_patch, orange_patch],labels = ['true', 'generated'])
    plt.savefig(figname)
    plt.clf()





minX = -10
minY = -10
maxX = 10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
number_of_replicates = 1000
matrix_index = (15,15)
seed_value = 94234
home_folder = "/home/julia/Dropbox/diffusion/twisted_diffusion/validation/unparameterized/unconditional/generate_data/data/"
unconditional_generated_samples = np.load((home_folder + "diffusion/unconditional_lengthscale_1.6_variance_0.4_1000.npy"))
figname = "gp_unconditional_marginal_1000_15_15.png"
produce_true_and_generated_marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale,
                                  number_of_replicates, matrix_index, seed_value,
                                  unconditional_generated_samples,
                                  figname)
matrix_index1 = (15,15)
matrix_index2 = (23,23)
matrix_index3 = (7,23)
matrix_index4 = (23,7)
produce_true_and_generated_marginal_densities1(minX, maxX, minY, maxY, n, variance, lengthscale,
                                  number_of_replicates, matrix_index1, matrix_index2, matrix_index3,
                                  matrix_index4, seed_value, unconditional_generated_samples, figname)


matrixindex11 = (16,16)
matrixindex12 = (15,16)
matrixindex21 = (8,24)
matrixindex22 = (24,8)
matrixindex31 = (3,3)
matrixindex32 = (3,4)
matrixindex41 = (16,7)
matrixindex42 = (16,8)
figname = "gp_unconditional_bivariate_1000.png"


produce_true_and_generated_bivariate_densities1(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, matrixindex11, matrixindex12,
                                                 matrixindex21, matrixindex22, matrixindex31, matrixindex32,
                                                 matrixindex41, matrixindex42, seed_value,
                                                 unconditional_generated_samples, figname)