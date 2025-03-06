import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from append_directories import *
from matplotlib.patches import Rectangle

def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))

def matrix_index_to_index(matrix_index, n):

    index = matrix_index[0]*n+matrix_index[1]
    return index

def produce_true_fcs_ncs_unconditional_marginal_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        unconditional_ncs_samples,
                                                        mask,
                                                        figname):

    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    matrix_index = index_to_matrix_index(missing_index, n)
    marginal_density = (unconditional_matrices[:,0,matrix_index[0],matrix_index[1]]).reshape((number_of_replicates,1))
    fcs_marginal_density = unconditional_fcs_samples[:,int(matrix_index[0]),int(matrix_index[1])]
    ncs_marginal_density = unconditional_ncs_samples[:,int(matrix_index[0]),int(matrix_index[1])]

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    fcs_pdd = pd.DataFrame(fcs_marginal_density,
                                    columns = None)
    ncs_pdd = pd.DataFrame(ncs_marginal_density,
                                    columns = None)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
    axs[0].imshow(unconditional_matrices[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    for j in range(observed_indices.shape[0]):
        rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
        axs[0].add_patch(rect)
    axs[0].plot(matrix_index[1], matrix_index[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(data = pdd, ax = axs[1], palette=['blue'])
    sns.kdeplot(data = ncs_pdd, palette = ["orange"], ax = axs[1])
    sns.kdeplot(data = fcs_pdd, palette = ["purple"], ax = axs[1])
    axs[1].set_title("Marginal")
    axs[1].set_xlim(-4,8)
    axs[1].set_ylim(0,.5)
    index = matrix_index_to_index(matrix_index, n)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'NCS', 'FCS'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()


def produce_true_fcs_ncs_unconditional_bivariate_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index1,
                                                        missing_index2,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        unconditional_ncs_samples,
                                                        mask,
                                                        figname):

    unconditional_matrices = unconditional_true_samples.reshape((number_of_replicates,1,n,n))
    #conditional_vectors is shape (number of replicates, m)
    matrix_index1 = index_to_matrix_index(missing_index1, n)
    matrix_index2 = index_to_matrix_index(missing_index2, n)
    biv_density = np.concatenate([(unconditional_matrices[:,0,matrix_index1[0],matrix_index1[1]]).reshape((number_of_replicates,1)),
                                  (unconditional_matrices[:,0,matrix_index2[0],matrix_index2[1]]).reshape((number_of_replicates,1))], axis = 1)
    fcs_biv_density = np.concatenate([(unconditional_fcs_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                       (unconditional_fcs_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))], axis = 1)
    ncs_biv_density = np.concatenate([(unconditional_ncs_samples[:,int(matrix_index1[0]),int(matrix_index1[1])]).reshape((number_of_replicates,1)),
                                       (unconditional_ncs_samples[:,int(matrix_index2[0]),int(matrix_index2[1])]).reshape((number_of_replicates,1))], axis = 1)

    #fig, ax = plt.subplots(1)
    #ax.hist(marginal_disalsotribution, density = True, histtype = 'step', bins = 100)
    fig, axs = plt.subplots(ncols = 2, figsize = (10,5))
    observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
    #partially_observed_field = np.multiply(mask.astype(bool), observed_vector.reshape((n,n)))
    for j in range(observed_indices.shape[0]):
        rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
        axs[0].add_patch(rect)
    axs[0].imshow(unconditional_matrices[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 4)
    axs[0].plot(matrix_index1[1], matrix_index1[0], "rx", markersize = 20, linewidth = 20)
    axs[0].plot(matrix_index2[1], matrix_index2[0], "rx", markersize = 20, linewidth = 20)
    sns.kdeplot(x = biv_density[:,0], y = biv_density[:,1],
                ax = axs[1], color = "blue")
    sns.kdeplot(x = fcs_biv_density[:,0], y = fcs_biv_density[:,1],
                ax = axs[1], color = "purple")
    sns.kdeplot(x = ncs_biv_density[:,0], y = ncs_biv_density[:,1],
                ax = axs[1], color = "orange")
    axs[1].set_title("Bivariate")
    axs[1].set_xlim(-32,8)
    axs[1].set_ylim(-32,8)
    #location = index_to_spatial_location(minX, maxX, minY, maxY, n, index)
    #rlocation = (round(location[0],2), round(location[1],2))
    #axs[1].set_xlabel("location: " + str(rlocation))
    axs[1].legend(labels = ['true', 'FCS', 'NCS'])
    axs[0].set_xticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    axs[0].set_yticks(ticks = [0,7,15,23,31], labels = [-10,-5,0,5,10])
    plt.savefig(figname)
    plt.clf()

def produce_multiple_true_fcs_ncs_unconditional_marginal_densities_with_variables():

    n = 32
    eval_folder = append_directory(2)
    range_values = [float(i) for i in range(3,4)]
    smooth_value = 1.5
    number_of_replicates = 4000
    missing_indices = [i for i in range(0,1000,35)]
    ms = [i for i in range(1,8)]
    for m in ms:
        for range_value in range_values:
            for missing_index in missing_indices:
                fcs_folder = (eval_folder + "/fcs")
                ref_folder = (fcs_folder + "/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image" + str(int(range_value)-1))
                unconditional_fcs_samples = np.log(np.load((ref_folder + "/processed_unconditional_fcs_fixed_mask_range_" + str(range_value) +
                               "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(number_of_replicates) + ".npy")))
                unconditional_true_samples = np.log((np.load((ref_folder + "/true_brown_resnick_images_range_" + str(int(range_value)) + "_smooth_1.5_4000.npy"))))
                unconditional_ncs_samples = (np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + str(range_value) + "_smooth_1.5_model5_4000.npy")))
                figname = (ref_folder + "/marginal_density/unconditional_fixed_location_marginal_density_" + str(missing_index) + ".png")
                mask = np.load((ref_folder + "/mask.npy"))
                produce_true_fcs_ncs_unconditional_marginal_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        unconditional_ncs_samples,
                                                        mask,
                                                        figname)
                

def produce_multiple_true_fcs_ncs_unconditional_bivariate_densities_with_variables():

    n = 32
    eval_folder = append_directory(2)
    range_values = [float(i) for i in range(3,4)]
    smooth_value = 1.5
    number_of_replicates = 4000
    missing_indices1 = (np.random.randint(0,1024,20)).tolist()
    ms = [i for i in range(1,8)]
    for m in ms:
        for range_value in range_values:
            for missing_index1 in missing_indices1:
                fcs_folder = (eval_folder + "/fcs")
                ref_folder = (fcs_folder + "/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image" + str(int(range_value)-1))
                unconditional_fcs_samples = np.log(np.load((ref_folder + "/processed_unconditional_fcs_fixed_mask_range_" + str(range_value) +
                            "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(number_of_replicates) + ".npy")))
                unconditional_true_samples = np.log((np.load((ref_folder + "/true_brown_resnick_images_range_" + str(int(range_value)) + "_smooth_1.5_4000.npy"))))
                unconditional_ncs_samples = (np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + str(range_value) + "_smooth_1.5_model5_4000.npy")))
                mask = np.load((ref_folder + "/mask.npy"))
                missing_indices2 = (np.array((np.where(mask.reshape((n**2)) == 1)))).reshape((m)).tolist()
                for missing_index2 in missing_indices2:
                    figname = (ref_folder + "/bivariate_density/unconditional_fixed_location_bivariate_density_" + str(missing_index1) + "_" + str(missing_index2) + ".png")
                    produce_true_fcs_ncs_unconditional_bivariate_density(n, range_value, smooth_value,
                                                        number_of_replicates, missing_index1,
                                                        missing_index2,
                                                        unconditional_fcs_samples,
                                                        unconditional_true_samples,
                                                        unconditional_ncs_samples,
                                                        mask,
                                                        figname)
                    
produce_multiple_true_fcs_ncs_unconditional_bivariate_densities_with_variables()