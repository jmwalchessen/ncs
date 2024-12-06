import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
from matplotlib import gridspec

def produce_ncs_marginal_density(model_name, image_name, file_name, missing_index, n, nrep):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    return diffusion_marginal_density

def produce_univariate_lcs_marginal_density(model_name, image_name, lcs_file_name,
                                            missing_index, n, nrep):
    univariate_lcs_images = load_univariate_lcs_images(model_name, image_name, lcs_file_name)
    univariate_lcs_marginal_density = (univariate_lcs_images.reshape((nrep,n**2)))[:,missing_index]
    return univariate_lcs_marginal_density

def visualize_marginal_density(model_name, missing_indices, n, nrep, figname):

    range_values = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_marginal_density = produce_ncs_marginal_density(model_name, image_name, file_name, missing_indices[i], n, nrep)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density


    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(4)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=2,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.25, height_ratios=[1, 1])
    for i in range(0,10):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "ro", markersize = 10, linewidth = 20)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]))
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            ax.legend(labels = ['NCS'], fontsize = 6)

    plt.savefig(figname)
    plt.clf()


def visualize_ncs_vs_univariate_lcs_marginal_density(model_name, univariate_lcs_file_name, missing_indices,
                                                     n, nrep, figname):

    range_values = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    univariate_lcs_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        masked_indices = np.squeeze(np.argwhere((1-masks[i,:,:]).reshape((n**2,))))
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_marginal_density = produce_ncs_marginal_density(model_name, image_name, file_name, missing_indices[i], n, nrep)
        univariate_lcs_marginal_density = produce_univariate_lcs_marginal_density(model_name, image_name, univariate_lcs_file_name, missing_indices[i], n, nrep)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_marginal_densities[i,:] = univariate_lcs_marginal_density


    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(4)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=2,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.25, height_ratios=[1, 1])
    for i in range(0,10):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "ro", markersize = 10, linewidth = 20)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(univariate_lcs_marginal_densities[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]))
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            ax.legend(labels = ['NCS'], fontsize = 6)

    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()




model_name = "model4"
missing_indices = [203, 338, 203, 303, 204]
n = 32
nrep = 4000
smooth = 1.5
univariate_lcs_file_name = "univariate_lcs_4000_neighbors_7_nugget_1e5"
figname = "figures/br_parameter_marginal_density_model4_random05.png"
visualize_marginal_density(model_name, missing_indices, n, nrep, figname)
figname = "figures/br_parameter_lcs_vs_ncs_marginal_density_model4_random05.png"
visualize_ncs_vs_univariate_lcs_marginal_density(model_name, univariate_lcs_file_name,
                                                 missing_indices, n, nrep, figname)