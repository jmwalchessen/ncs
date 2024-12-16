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

def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))

def produce_marginal_density(model_name, image_name, file_name, missing_index, n, nrep, variance, lengthscale):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    return diffusion_marginal_density


def produce_bivariate_densities(model_name, lengthscale, variance, image_name, nrep,
                                missing_index1, missing_index2, file_name):

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


def visualize_conditional_marginal_bivariate_density(model_name, range_value, smooth, nrep, missing_indices,
                                                     missing_indices1, missing_indices2, univariate_lcs_file,
                                                     bivariate_lcs_file, n, figname):
    
    percentages = [.01,.05,.1,.25,.5]
    diffusion_bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    lcs_marginal_density = np.zeros((len(percentages),nrep))
    univariate_lcs_images = np.zeros((5,nrep,n,n))
    lcs_bivariate_density = np.zeros((5,nrep,2))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        missing_index = missing_indices[i]
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_4000_random" + str(percentages[i]))
        dbdensities = produce_bivariate_densities(model_name, range_value, smooth, 
                                                              image_name, nrep, missing_indices1[i],
                                                              missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_bivariate_densities[i,:,:] = dbdensities
        diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                              missing_indices[i], n, nrep, range_value,
                                                              smooth)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_images[i,:,:,:] = (np.load((data_generation_folder + ref_image_folder + "/lcs/univariate/" + univariate_lcs_file))).reshape((nrep,n,n))
        lcs_marginal_density[i,:] = univariate_lcs_images[i,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                               bivariate_lcs_file + "_random" + str(ps[i]) + "_" + str(missing_indices1[i])
                               + "_" + str(missing_indices2[i]) + ".npy")))
        lcs_bivariate_density[i,:,:] = bilcs


#fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=3,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.25, height_ratios=[1, 1, 1])

    for i in range(0, 15):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "r^", markersize = 10, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r^", markersize = 10, linewidth = 20)
            ax.plot(matrix_index[1], matrix_index[0], "ro", markersize = 10, linewidth = 20)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

        elif(i < 10):
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(lcs_marginal_density[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]))
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            ax.legend(labels = ['NCS', 'LCS'], fontsize = 6)
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            sns.kdeplot(x = diffusion_bivariate_densities[(i%5),:,0], y = diffusion_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [-2,0,2,4,6], labels = np.array([-2,0,2,4,6]))
            purple_patch = mpatches.Patch(color='purple')
            orange_patch = mpatches.Patch(color='orange')
            ax.legend(handles = [purple_patch, orange_patch], labels = ['LCS', 'NCS'], fontsize = 7)
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)

    plt.savefig(figname)
    plt.clf()


n = 32
model_name = "model4"
missing_indices1 = [267,407,746,203,226]
missing_indices2 = [353,452,241,186,827]
figname = "figures/br_percentage_lcs_vs_ncs_conditional_marginal_bivariate_density.png"
nrep = 4000
ps = [.01,.05,.1,.25,.5]
bivariate_lcs_file = "bivariate_lcs_4000_neighbors_7_nugget_1e5"
range_value = 3.0
smooth = 1.5
missing_indices = [650,460,392,497,829]
univariate_lcs_file = "univariate_lcs_4000_neighbors_7_nugget_1e5.npy"
visualize_conditional_marginal_bivariate_density(model_name, range_value, smooth, nrep, missing_indices,
                                                 missing_indices1, missing_indices2, univariate_lcs_file,
                                                 bivariate_lcs_file, n, figname)