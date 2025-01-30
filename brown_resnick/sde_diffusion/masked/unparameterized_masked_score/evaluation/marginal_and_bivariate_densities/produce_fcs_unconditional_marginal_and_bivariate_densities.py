import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import seaborn as sns
import pandas as pd
import os
import sys
from append_directories import *
import scipy

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


def produce_true_and_ncs_unconditional_marginal_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        figname):

    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    matrix_index = index_to_matrix_index(missing_index, n)
    marginal_density = (unconditional_matrices[:,0,matrix_index[0],matrix_index[1]]).reshape((number_of_replicates,1))
    fcs_marginal_density = unconditional_fcs_samples[:,int(matrix_index[0]),int(matrix_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    fcs_pdd = pd.DataFrame(fcs_marginal_density,
                                    columns = None)

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = fcs_pdd, palette = ["orange"], ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(0,.5)
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'FCS'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()


def produce_true_and_ncs_unconditional_bivariate_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index1,
                                                        missing_index2,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        figname):

    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    matrix_index1 = index_to_matrix_index(missing_index1, n)
    matrix_index2 = index_to_matrix_index(missing_index2, n)
    biv_density = np.concatenate([(unconditional_matrices[:,0,matrix_index1[0],matrix_index1[1]]).reshape((number_of_replicates,1)),
                                  (unconditional_matrices[:,0,matrix_index2[0],matrix_index2[1]]).reshape((number_of_replicates,1))], axis = 1)
    fcs_biv_density = np.concatenate([(unconditional_fcs_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                       (unconditional_fcs_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))], axis = 1)

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))

    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    axs[0].imshow(unconditional_matrices[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "rx", markersize = 20, linewidth = 20)
    axs[0].plot(matrix_index2[1], matrix_index2[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(x = biv_density[:,0], y = biv_density[:,1],
                ax = axs[1])
    sns.kdeplot(x = fcs_biv_density[:,0], y = fcs_biv_density[:,1],
                ax = axs[1])
    axs[1].set_title("Bivariate")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(-4,8)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'FCS'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()


def produce_true_and_fcs_unconditional_marginal_density_files(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index,
                                                        unconditional_fcs_file,
                                                        unconditional_true_file,
                                                        figname):
    eval_folder = append_directory(2)
    ref_folder = (eval_folder + "/extremal_coefficient_and_high_dimensional_metrics")
    unconditional_fcs_samples = np.log(np.load((ref_folder + "/data/fcs/" + unconditional_fcs_file)))
    unconditional_true_samples = np.log(np.load((ref_folder + "/data/true/" + unconditional_true_file)))
    ref_figname = (eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/" + figname)
    produce_true_and_ncs_unconditional_marginal_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        ref_figname)
    
def produce_true_and_fcs_unconditional_bivariate_density_files(n, range_value, smooth_value,
                                                               number_of_replicates, missing_index1,
                                                               missing_index2, unconditional_fcs_file,
                                                               unconditional_true_file, figname):
    
    eval_folder = append_directory(2)
    ref_folder = (eval_folder + "/extremal_coefficient_and_high_dimensional_metrics")
    unconditional_fcs_samples = np.log(np.load((ref_folder + "/data/fcs/" + unconditional_fcs_file)))
    unconditional_true_samples = np.log(np.load((ref_folder + "/data/true/" + unconditional_true_file)))
    ref_figname = (eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/" + figname)
    produce_true_and_ncs_unconditional_bivariate_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index1,
                                                        missing_index2,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        ref_figname)

    
def produce_multiple_true_and_fcs_unconditional_marginal_densities(m, missing_indices):
    
    n = 32
    range_value = 3.
    smooth_value = 1.5
    number_of_replicates = 4000
    unconditional_fcs_file = ("processed_unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_4000.npy")
    unconditional_true_file = "brown_resnick_images_range_3.0_smooth_1.5_4000.npy"
    for missing_index in missing_indices:
        figname = ("unconditional_marginal_and_bivariate_densities/marginal_density/obs" + str(m) + "/unconditional_true_vs_fcs_obs_" + str(m) + "_range_3.0_smooth_1.5_" + str(missing_index) + ".png")
        produce_true_and_fcs_unconditional_marginal_density_files(n, range_value, smooth_value,
                                                            number_of_replicates, missing_index,
                                                            unconditional_fcs_file,
                                                            unconditional_true_file,
                                                            figname)
        
def produce_multiple_true_and_fcs_unconditional_bivariate_densities(m, missing_indices1, missing_indices2):
    
    n = 32
    range_value = 3.
    smooth_value = 1.5
    number_of_replicates = 4000
    unconditional_fcs_file = ("processed_unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_4000.npy")
    unconditional_true_file = "brown_resnick_images_range_3.0_smooth_1.5_4000.npy"
    for missing_index1 in missing_indices1:
        for missing_index2 in missing_indices2:
            figname = ("unconditional_marginal_and_bivariate_densities/bivariate_density/obs" + str(m) + "/unconditional_true_vs_fcs_obs_" + str(m) + "_range_3.0_smooth_1.5_" + str(missing_index1) + "_" + str(missing_index2) + ".png")
            produce_true_and_fcs_unconditional_bivariate_density_files(n, range_value, smooth_value,
                                                            number_of_replicates, missing_index1,
                                                            missing_index2,
                                                            unconditional_fcs_file,
                                                            unconditional_true_file,
                                                            figname)
            

            
def goodness_of_fit_unconditional_true_vs_fcs_marginal_ks():

    n = 32
    range_values = [float(i) for i in range(1,6)]
    smooth_value = 1.5
    number_of_replicates = 4000
    ms = [i for i in range(1,8)]
    extr_folder = append_directory(2) + "/extremal_coefficient_and_high_dimensional_metrics"
    ks_gof_statistic = np.zeros((len(range_values),len(ms),n,n))
    ks_gof_pvalue = np.zeros((len(range_values),len(ms),n,n))
    avg_ks_gof_statistic = np.zeros((len(range_values), len(ms)))
    for i,range_value in enumerate(range_values):
        for j,m in enumerate(ms):
            unconditional_fcs_file = (extr_folder + "/data/fcs/processed_unconditional_fcs_range_" + str(range_value)
                                      + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_4000.npy")
            unconditional_true_file = (extr_folder + "/data/true/brown_resnick_images_range_" 
                                       + str(range_value) + "_smooth_1.5_4000.npy")
            unconditional_fcs_images = (np.load(unconditional_fcs_file)).reshape((number_of_replicates,n**2))
            unconditional_true_images = (np.load(unconditional_true_file)).reshape((number_of_replicates,n**2))
            ks_gof_product = scipy.stats.kstest(unconditional_fcs_images, unconditional_true_images, axis = 0)
            ks_gof_statistic[i,j,:,:] = (ks_gof_product.statistic).reshape((n,n))
            ks_gof_pvalue[i,j,:,:] = (ks_gof_product.pvalue).reshape((n,n))
            avg_ks_gof_statistic[i,j] = np.mean(ks_gof_statistic[i,j,:,:], axis = (0,1))

    ks_gof_statistic_file = (extr_folder + "/data/fcs/ks_gof_statistic_obs_1_7_range_1_5_smooth_1.5_nugget_1e5_obs_" + str(m) + ".npy")
    np.save(ks_gof_statistic_file, ks_gof_statistic)
    ks_gof_pvalue_file = (extr_folder + "/data/fcs/ks_gof_pvalue_obs_1_7_range_1_5_smooth_1.5_nugget_1e5_obs_" + str(m) + ".npy")
    np.save(ks_gof_pvalue_file, ks_gof_pvalue)
    avg_ks_gof_statistic_file = (extr_folder + "/data/fcs/avg_ks_gof_statistic_obs_1_7_range_1_5_smooth_1.5_nugget_1e5_obs_" + str(m) + ".npy")
    np.save(avg_ks_gof_statistic_file, avg_ks_gof_statistic)


goodness_of_fit_unconditional_true_vs_fcs_marginal_ks()


            


