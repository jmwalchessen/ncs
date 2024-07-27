import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
home_folder = append_directory(4)
data_generation_folder = (home_folder + "/validation/unparameterized/conditional_masked/brown_resnick/generate_data")
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
    return (xlocation, ylocation)


def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log10_transformation(images):

    images = np.log10(np.where(images !=0, images, np.min(images[images != 0])))

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

#.99999 quantile, .00001 quantile, .7 quantile of the transformed by .99999 and .00001 quantile
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

def visualize_spatial_field(observation, min_value, max_value, figname):

    fig, ax = plt.subplots(1)
    plt.imshow(observation, vmin = min_value, vmax = max_value)
    plt.savefig(figname)

def produce_generated_bivariate_density(conditional_generated_samples,
                                        number_of_replicates, missing_matrix_index1,
                                        missing_matrix_index2, ref_image, mask, n, figname):
    

    #conditional_vectors is shape (number of replicates, m)
    bivariate_density = np.concatenate([(conditional_generated_samples[:,:,missing_matrix_index1[0],missing_matrix_index1[1]]).reshape((number_of_replicates,1)),
                                        (conditional_generated_samples[:,:,missing_matrix_index2[0],missing_matrix_index2[1]]).reshape((number_of_replicates,1))], axis = 1)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    #emp_mean = round(np.mean(marg), 2)
    #emp_var = round(np.std(marginal_density)**2, 2)
    pdd = pd.DataFrame(bivariate_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 4)
    axs[0].plot(missing_matrix_index1[0], missing_matrix_index1[1], "r+")
    axs[0].plot(missing_matrix_index2[0], missing_matrix_index2[1], "r+")
    axs[0].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1],
                ax = axs[1])
    plt.axvline(ref_image[missing_matrix_index1[0],missing_matrix_index1[1]], color='red', linestyle = 'dashed')
    plt.axhline(ref_image[missing_matrix_index2[0],missing_matrix_index2[1]], color='red', linestyle = 'dashed')
    plt.xlim(-2,4)
    plt.ylim(-2,4)
    axs[1].set_title("Marginal")
    #location1 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index1)
    #rlocation1 = (round(location1[0],2), round(location1[1],2))
    #location2 = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index2)
    #rlocation2 = (round(location2[0],2), round(location2[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation1))
    #axs[1].set_ylabel("location: " + str(rlocation2))
    axs[1].legend(labels = ['diffusion'])
    plt.savefig(figname)

def produce_generated_marginal_density(conditional_generated_samples,
                                       number_of_replicates, missing_matrix_index,
                                       ref_image, mask, n, figname):

    #conditional_vectors is shape (number of replicates, m)
    marginal_density = (conditional_generated_samples[:,:,missing_matrix_index[0],missing_matrix_index[1]]).reshape((number_of_replicates,1))

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 4)
    axs[0].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0].plot(missing_matrix_index[0], missing_matrix_index[1], "r+")
    sns.kdeplot(data = pdd, ax = axs[1])
    plt.axvline(ref_image[int(missing_matrix_index[0]),int(missing_matrix_index[1])], color='red', linestyle = 'dashed')
    axs[1].set_title("Marginal")
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['diffusion'])
    plt.savefig(figname)

def produce_generated_marginal_densities(conditional_generated_samples,
                                       number_of_replicates, missing_matrix_index1,
                                       missing_matrix_index2, missing_matrix_index3,
                                       missing_matrix_index4, ref_image, mask, n, figname):

    #conditional_vectors is shape (number of replicates, m)
    marginal_density1 = (conditional_generated_samples[:,:,missing_matrix_index1[0],missing_matrix_index1[1]]).reshape((number_of_replicates,1))
    marginal_density2 = (conditional_generated_samples[:,:,missing_matrix_index2[0],missing_matrix_index2[1]]).reshape((number_of_replicates,1))
    marginal_density3 = (conditional_generated_samples[:,:,missing_matrix_index3[0],missing_matrix_index3[1]]).reshape((number_of_replicates,1))
    marginal_density4 = (conditional_generated_samples[:,:,missing_matrix_index4[0],missing_matrix_index4[1]]).reshape((number_of_replicates,1))
    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (20,10))
    pdd1 = pd.DataFrame(marginal_density1,
                                    columns = None)
    pdd2 = pd.DataFrame(marginal_density2,
                                    columns = None)
    pdd3 = pd.DataFrame(marginal_density3,
                                    columns = None)
    pdd4 = pd.DataFrame(marginal_density4,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[0,0].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,0].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,0].plot(missing_matrix_index1[0], missing_matrix_index1[1], "ro", markersize = 20)
    sns.kdeplot(data = pdd1, ax = axs[0,1], palette = ["orange"])
    axs[0,1].axvline(ref_image[int(missing_matrix_index1[1]),int(missing_matrix_index1[0])], color='red', linestyle = 'dashed')
    axs[0,1].set_title("Marginal")
    axs[0,1].set_xlim(-4,6)
    axs[0,1].set_ylim(0,.8)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[0,1].legend(labels = ['diffusion'])

    axs[0,2].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[0,2].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,2].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,2].plot(missing_matrix_index2[0], missing_matrix_index2[1], "ro", markersize = 20)
    sns.kdeplot(data = pdd2, ax = axs[0,3], palette = ["orange"])
    axs[0,3].axvline(ref_image[int(missing_matrix_index2[1]),int(missing_matrix_index2[0])], color='red', linestyle = 'dashed')
    axs[0,3].set_title("Marginal")
    axs[0,3].set_xlim(-4,6)
    axs[0,3].set_ylim(0,.8)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[0,3].legend(labels = ['diffusion'])

    axs[1,0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[1,0].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[1,0].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[1,0].plot(missing_matrix_index3[0], missing_matrix_index3[1], "ro", markersize = 20)
    sns.kdeplot(data = pdd3, ax = axs[1,1], palette = ["orange"])
    axs[1,1].axvline(ref_image[int(missing_matrix_index3[1]),int(missing_matrix_index3[0])], color='red', linestyle = 'dashed')
    axs[1,1].set_title("Marginal")
    axs[1,1].set_xlim(-4,6)
    axs[1,1].set_ylim(0,.8)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1,1].legend(labels = ['diffusion'])

    axs[1,2].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[1,2].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[1,2].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[1,2].plot(missing_matrix_index4[1], missing_matrix_index4[0], "ro", markersize = 20)
    sns.kdeplot(data = pdd4, ax = axs[1,3], palette = ["orange"])
    axs[1,3].axvline(ref_image[int(missing_matrix_index4[1]),int(missing_matrix_index4[0])], color='red', linestyle = 'dashed')
    axs[1,3].set_title("Marginal")
    axs[1,3].set_xlim(-4,6)
    axs[1,3].set_ylim(0,.8)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, missing_true_index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1,3].legend(labels = ['diffusion'])
    plt.savefig(figname)

def produce_generated_bivariate_densities(conditional_generated_samples,
                                        number_of_replicates, missing_matrix_index11,
                                        missing_matrix_index12, missing_matrix_index21,
                                        missing_matrix_index22, missing_matrix_index31,
                                        missing_matrix_index32, missing_matrix_index41,
                                        missing_matrix_index42, ref_image, mask, n, figname):
    

    #conditional_vectors is shape (number of replicates, m)
    bivariate_density1 = np.concatenate([(conditional_generated_samples[:,:,missing_matrix_index11[0],missing_matrix_index11[1]]).reshape((number_of_replicates,1)),
                                        (conditional_generated_samples[:,:,missing_matrix_index12[0],missing_matrix_index12[1]]).reshape((number_of_replicates,1))], axis = 1)
    bivariate_density2 = np.concatenate([(conditional_generated_samples[:,:,missing_matrix_index21[0],missing_matrix_index21[1]]).reshape((number_of_replicates,1)),
                                        (conditional_generated_samples[:,:,missing_matrix_index22[0],missing_matrix_index22[1]]).reshape((number_of_replicates,1))], axis = 1)
    bivariate_density3 = np.concatenate([(conditional_generated_samples[:,:,missing_matrix_index31[0],missing_matrix_index31[1]]).reshape((number_of_replicates,1)),
                                        (conditional_generated_samples[:,:,missing_matrix_index32[0],missing_matrix_index32[1]]).reshape((number_of_replicates,1))], axis = 1)
    bivariate_density4 = np.concatenate([(conditional_generated_samples[:,:,missing_matrix_index41[0],missing_matrix_index41[1]]).reshape((number_of_replicates,1)),
                                        (conditional_generated_samples[:,:,missing_matrix_index42[0],missing_matrix_index42[1]]).reshape((number_of_replicates,1))], axis = 1)
    fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize = (20,10))
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[0,0].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,0].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,0].plot(missing_matrix_index11[1], missing_matrix_index11[0], "r^", markersize = 20)
    axs[0,0].plot(missing_matrix_index12[1], missing_matrix_index12[0], "k^", markersize = 20)
    sns.kdeplot(x = bivariate_density1[:,0], y = bivariate_density1[:,1],
                ax = axs[0,1], color = "orange", fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
    axs[0,1].axvline(ref_image[missing_matrix_index11[0],missing_matrix_index11[1]], color='red', linestyle = 'dashed')
    axs[0,1].axhline(ref_image[missing_matrix_index12[0],missing_matrix_index12[1]], color='red', linestyle = 'dashed')
    axs[0,1].set_xlim(-4,8)
    axs[0,1].set_ylim(-4,8)
    axs[0,1].set_title("Bivariate")
    orange_patch = mpatches.Patch(color='orange')
    axs[0,1].legend(handles = [orange_patch], labels = ['diffusion'])

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0,2].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[0,2].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,2].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[0,2].plot(missing_matrix_index21[1], missing_matrix_index21[0], "r^", markersize = 20)
    axs[0,2].plot(missing_matrix_index22[1], missing_matrix_index22[0], "k^", markersize = 20)
    sns.kdeplot(x = bivariate_density2[:,0], y = bivariate_density2[:,1],
                ax = axs[0,3], color = "orange", fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
    axs[0,3].axvline(ref_image[missing_matrix_index21[0],missing_matrix_index21[1]], color='red', linestyle = 'dashed')
    axs[0,3].axhline(ref_image[missing_matrix_index22[0],missing_matrix_index22[1]], color='red', linestyle = 'dashed')
    axs[0,3].set_xlim(-4,8)
    axs[0,3].set_ylim(-4,8)
    axs[0,3].set_title("Bivariate")
    orange_patch = mpatches.Patch(color='orange')
    axs[0,3].legend(handles = [orange_patch], labels = ['diffusion'])


    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[1,0].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[1,0].plot(missing_matrix_index31[1], missing_matrix_index31[0], "r^", markersize = 20)
    axs[1,0].plot(missing_matrix_index32[1], missing_matrix_index32[0], "k^", markersize = 20)
    axs[1,0].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[1,0].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    sns.kdeplot(x = bivariate_density3[:,0], y = bivariate_density3[:,1],
                ax = axs[1,1], color = "orange", fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
    axs[1,1].axvline(ref_image[missing_matrix_index31[0],missing_matrix_index31[1]], color='red', linestyle = 'dashed')
    axs[1,1].axhline(ref_image[missing_matrix_index32[0],missing_matrix_index32[1]], color='red', linestyle = 'dashed')
    axs[1,1].set_xlim(-4,8)
    axs[1,1].set_ylim(-4,8)
    axs[1,1].set_title("Bivariate")
    orange_patch = mpatches.Patch(color='orange')
    axs[1,1].legend(handles = [orange_patch], labels = ['diffusion'])

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[1,2].imshow(ref_image.reshape((n,n)), alpha = mask, vmin = -2, vmax = 6)
    axs[1,2].plot(missing_matrix_index41[1], missing_matrix_index41[0], "r^", markersize = 20)
    axs[1,2].plot(missing_matrix_index42[1], missing_matrix_index42[0], "k^", markersize = 20)
    axs[1,2].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    axs[1,2].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    sns.kdeplot(x = bivariate_density4[:,0], y = bivariate_density4[:,1],
                ax = axs[1,3], color = "orange", fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
    axs[1,3].axvline(ref_image[missing_matrix_index41[0],missing_matrix_index41[1]], color='red', linestyle = 'dashed')
    axs[1,3].axhline(ref_image[missing_matrix_index42[0],missing_matrix_index42[1]], color='red', linestyle = 'dashed')
    axs[1,3].set_xlim(-4,8)
    axs[1,3].set_ylim(-4,8)
    axs[1,3].set_title("Bivariate")
    orange_patch = mpatches.Patch(color='orange')
    axs[1,3].legend(handles = [orange_patch], labels = ['diffusion'])


    plt.savefig(figname)



filename = (data_generation_folder + "/data/conditional/model11/ref_img2/model11_random50_beta_min_max_01_20_1000_random0_1000.npy")
conditional_generated_samples = np.load(filename)
number_of_replicates = 1000
missing_matrix_index = (8,9)
ref_image = np.load(data_generation_folder + "/data/conditional/model11/ref_img2/ref_image2.npy")
trainlogmaxminquant = np.load((home_folder + "/brown_resnick/sde_diffusion/masked/unparameterized/trained_score_models/vpsde/"
                               + "model11_train_logminmax.npy"))
ref_image = global_quantile_boundary_inverse(ref_image, trainlogmaxminquant[0],
                                             trainlogmaxminquant[1], trainlogmaxminquant[2])
conditional_generated_samples = global_quantile_boundary_inverse(conditional_generated_samples,
                                                                 trainlogmaxminquant[0],
                                                                 trainlogmaxminquant[1],
                                                                 trainlogmaxminquant[2])
#0 in mask means missing
mask = ((th.from_numpy(np.load(data_generation_folder + "/data/conditional/model11/ref_img2/mask.npy"))).float()).numpy()
n = 32
figname = "br_conditional_marginal_density_model11.png"
produce_generated_marginal_density(conditional_generated_samples, number_of_replicates, missing_matrix_index,
                                   ref_image, mask, n, figname)
figname = "br_conditional_bivariate_density_model11.png"
missing_matrix_index1 = (16,15)
missing_matrix_index2 = (5,5)
produce_generated_bivariate_density(conditional_generated_samples,
                                        number_of_replicates, missing_matrix_index1,
                                        missing_matrix_index2, ref_image, mask, n, figname)

missing_matrix_index3 = (10,27)
missing_matrix_index4 = (7,21)
figname = "br_conditional_marginal_densities_model11.png"
produce_generated_marginal_densities(conditional_generated_samples,
                                       number_of_replicates, missing_matrix_index1,
                                       missing_matrix_index2, missing_matrix_index3,
                                       missing_matrix_index4, ref_image, mask, n, figname)

missing_matrix_index11 = (25,12)
missing_matrix_index12 = (25,26)
missing_matrix_index21 = (18,14)
missing_matrix_index22 = (18,16)
missing_matrix_index31 = (5,26)
missing_matrix_index32 = (7,23)
missing_matrix_index41 = (24,12)
missing_matrix_index42 = (24,13)
figname = "br_conditional_bivariate_densities_model11.png"
produce_generated_bivariate_densities(conditional_generated_samples,
                                        number_of_replicates, missing_matrix_index11,
                                        missing_matrix_index12, missing_matrix_index21,
                                        missing_matrix_index22, missing_matrix_index31,
                                        missing_matrix_index32, missing_matrix_index41,
                                        missing_matrix_index42, ref_image, mask, n, figname)