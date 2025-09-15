import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
from matplotlib import gridspec
import pandas as pd
from matplotlib import patches as mpatches

evaluation_folder = append_directory(2)
data_generation_folder = (evaluation_folder + "/diffusion_generation")

def produce_ncs_marginal_density(model_name, image_name, file_name, missing_index, n, nrep):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    return diffusion_marginal_density

def produce_univariate_lcs_marginal_density(model_name, image_name, lcs_file_name,
                                            missing_index, n, nrep):
    univariate_lcs_images = load_univariate_lcs_images(model_name, image_name, lcs_file_name)
    univariate_lcs_marginal_density = (univariate_lcs_images.reshape((nrep,n**2)))[:,missing_index]
    return univariate_lcs_marginal_density

def produce_bivariate_densities(model_name, image_name, nrep,missing_index1, missing_index2, file_name):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    diffusion_images = diffusion_images.reshape((nrep,n**2))
    diffusion_bivariate_densities = np.concatenate([(diffusion_images[:,missing_index1]).reshape((nrep,1)),
                                          (diffusion_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return diffusion_bivariate_densities

def visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_density(model_name, univariate_lcs_file_name, missing_indices,
                                                                   missing_indices1, missing_indices2, bivariate_lcs_file,
                                                                   n, nrep, figname):

    range_values = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    univariate_lcs_marginal_densities = np.zeros((5,nrep))
    bivariate_densities = np.zeros((5,nrep,2))
    lcs_bivariate_density = np.zeros((5,nrep,2))
    class_vector = np.ones((nrep)).reshape((nrep, 1))
    
    for i in range(0, 5):
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        masked_indices = np.squeeze(np.argwhere((1-masks[i,:,:]).reshape((n**2,))))
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_marginal_density = produce_ncs_marginal_density(model_name, image_name, file_name, missing_indices[i], n, nrep)
        univariate_lcs_marginal_density = produce_univariate_lcs_marginal_density(model_name, image_name, univariate_lcs_file_name, missing_indices[i], n, nrep)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_marginal_densities[i,:] = univariate_lcs_marginal_density
        dbdensities = produce_bivariate_densities(model_name, image_name, nrep, missing_indices1[i], missing_indices2[i], file_name)
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                                                 bivariate_lcs_file + "_" + str(missing_indices1[i]) + "_" + str(missing_indices2[i]) + ".npy")))
        bivariate_densities[i,:,:] = dbdensities
        lcs_bivariate_density[i,:,:] = bilcs
    


    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=3,
                         width_ratios=[1,1,1,1,1], wspace=0.15,
                         hspace=0.25, height_ratios=[1, 1, 1])
    
    for i in range(0,15):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            if(i == 0):
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_yticks([])
            if(((i == 0)) | (i == 4)):
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
        elif(i < 10):
            sns.kdeplot(univariate_lcs_marginal_densities[(i % 5),:], ax = ax, color = 'purple')
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange', linestyle = "dashed")
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]))
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            if(i == 5):
                ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 15)
                ax.legend(labels = ['NCS', 'LCS'], fontsize = 12)
            else:
                ax.set_yticks([])
            if((i == 5) | (i == 9)):
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            else:
                ax.set_xticks([])

        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            kde1 = sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                               ax = ax, color = 'purple', alpha = .5)
            kde2 = sns.kdeplot(x = bivariate_densities[(i%5),:,0], y = bivariate_densities[(i%5),:,1],
                               ax = ax, color = 'orange', alpha = .5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            if(i == 10):
                op = mpatches.Patch(color='orange')
                pp = mpatches.Patch(color='purple')
                ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
                ax.legend(handles = [op,pp], labels = ['NCS', 'LCS'], fontsize = 12)
            else:
                ax.set_yticks([])
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)

    fig.text(x = .38, y = .9, s = "Partially Observed Field", fontsize = 15)
    fig.text(x = .35, y = .62, s = "Conditional Marginal Density", fontsize = 15)
    fig.text(x = .35, y = .34, s = "Conditional Bivariate Density", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()


def visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_density_transformed(model_name, univariate_lcs_file_name, missing_indices,
                                                                   missing_indices1, missing_indices2, bivariate_lcs_file,
                                                                   n, nrep, figname):

    range_values = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    univariate_lcs_marginal_densities = np.zeros((5,nrep))
    bivariate_densities = np.zeros((5,nrep,2))
    lcs_bivariate_density = np.zeros((5,nrep,2))
    class_vector = np.ones((nrep)).reshape((nrep, 1))
    
    for i in range(0, 5):
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        masked_indices = np.squeeze(np.argwhere((1-masks[i,:,:]).reshape((n**2,))))
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_marginal_density = produce_ncs_marginal_density(model_name, image_name, file_name, missing_indices[i], n, nrep)
        univariate_lcs_marginal_density = produce_univariate_lcs_marginal_density(model_name, image_name, univariate_lcs_file_name, missing_indices[i], n, nrep)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_marginal_densities[i,:] = univariate_lcs_marginal_density
        dbdensities = produce_bivariate_densities(model_name, image_name, nrep, missing_indices1[i], missing_indices2[i], file_name)
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                                                 bivariate_lcs_file + "_" + str(missing_indices1[i]) + "_" + str(missing_indices2[i]) + ".npy")))
        bivariate_densities[i,:,:] = dbdensities
        lcs_bivariate_density[i,:,:] = bilcs
    


    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(9)
 
    # set width of each subplot as 8
    fig.set_figwidth(5.5)
    spec = gridspec.GridSpec(ncols=3, nrows=5,
                         width_ratios=[1,1,1], wspace=0.3,
                         hspace=0.25, height_ratios=[1, 1, 1,1,1])
    
    counter = -1
    for i in range(0,15):
        ax = fig.add_subplot(spec[i])
        if((i%3)==0):
            counter = counter + 1
            matrix_index = index_to_matrix_index(missing_indices[counter], n)
            missing_index1 = missing_indices1[counter]
            missing_index2 = missing_indices2[counter]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = ax.imshow(reference_images[counter,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[counter,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 12)
            #else:
                #ax.set_yticks([])
            #if(((i == 0)) | (i == 4)):
            ax.set_xticks(ticks = [0,7,15,23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 12)
            #else:
                #ax.set_xticks([])
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
        elif((i%3)==1):
            sns.kdeplot(univariate_lcs_marginal_densities[counter,:], ax = ax, color = 'purple')
            sns.kdeplot(diffusion_marginal_densities[counter,:], ax = ax, color = 'orange', linestyle = "dashed")
            ax.axvline(reference_images[counter,matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]), fontsize = 12)
            ax.set_xticks(ticks = [-2,0,2,4,6], labels = np.array([-2,0,2,4,6]), fontsize = 12)
            #ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            #if(counter == 5):
                #ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 15)
            if(i == 1):
                ax.legend(labels = ['NCS', 'LCS'], fontsize = 12)
            #else:
                #ax.set_yticks([])
            #if((i == 5) | (i == 9)):
                #ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            #else:
                #ax.set_xticks([])

        else:
            missing_index1 = missing_indices1[counter]
            missing_index2 = missing_indices2[counter]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            kde1 = sns.kdeplot(x = lcs_bivariate_density[counter,:,0], y = lcs_bivariate_density[counter,:,1],
                               ax = ax, color = 'purple', alpha = .5)
            kde2 = sns.kdeplot(x = bivariate_densities[counter,:,0], y = bivariate_densities[counter,:,1],
                               ax = ax, color = 'orange', alpha = .5)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.axvline(reference_images[counter,matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[counter,matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 12)
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 12)
            if(i == 2):
                ol = mpatches.Patch(color='orange', lw = .2)
                pl = mpatches.Patch(color='purple', lw = .2)
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
            #if(i == 10):
                #op = mpatches.Patch(color='orange')
                #pp = mpatches.Patch(color='purple')
                #ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
                #ax.legend(handles = [op,pp], labels = ['NCS', 'LCS'], fontsize = 12)
            #else:
                #ax.set_yticks([])
            #ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)

    #fig.text(x = .38, y = .9, s = "Partially Observed Field", fontsize = 15)
    #fig.text(x = .35, y = .62, s = "Conditional Marginal Density", fontsize = 15)
    #fig.text(x = .35, y = .34, s = "Conditional Bivariate Density", fontsize = 15)
    fig.text(x = .32, y = .89, s = "Parameter U-Net", fontsize = 18)
    plt.tight_layout()
    plt.savefig(figname, dpi = 500)
    plt.clf()

def visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_densities_with_variables():

    n = 32
    range_values = [1.,2.,3.,4.,5.]
    model_name = "model4"
    #401, 597
    missing_indices1 = [292,242,849,724,682]
    missing_indices2 = [235,301,846,310,696]
    figname = "figures/br_parameter_lcs_vs_ncs_conditional_marginal_bivariate_density.png"
    nrep = 4000
    bivariate_lcs_file = "bivariate_lcs_4000_neighbors_7_nugget_1e5"
    univariate_lcs_file_name = "univariate_lcs_4000_neighbors_7_nugget_1e5"
    missing_indices = [642,129,392,497,829]
    visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_density(model_name, univariate_lcs_file_name, missing_indices,
                                                                    missing_indices1, missing_indices2, bivariate_lcs_file,
                                                                    n, nrep, figname)
    

def visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_densities_transformed_with_variables():

    n = 32
    range_values = [1.,2.,3.,4.,5.]
    model_name = "model4"
    #401, 597
    missing_indices1 = [292,242,849,724,682]
    missing_indices2 = [235,301,846,310,696]
    figname = "figures/br_parameter_lcs_vs_ncs_conditional_marginal_bivariate_density_transposed.png"
    nrep = 4000
    bivariate_lcs_file = "bivariate_lcs_4000_neighbors_7_nugget_1e5"
    univariate_lcs_file_name = "univariate_lcs_4000_neighbors_7_nugget_1e5"
    missing_indices = [642,129,392,497,829]
    visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_density_transformed(model_name, univariate_lcs_file_name, missing_indices,
                                                                    missing_indices1, missing_indices2, bivariate_lcs_file,
                                                                    n, nrep, figname)
    

visualize_ncs_vs_univariate_lcs_marginal_and_bivariate_densities_transformed_with_variables()