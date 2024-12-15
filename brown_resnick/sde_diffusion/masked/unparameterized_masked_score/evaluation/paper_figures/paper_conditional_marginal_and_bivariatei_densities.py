import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns

def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))

def produce_marginal_density(model_name, image_name, file_name, missing_index, n, nrep, variance, lengthscale):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    return diffusion_marginal_density

def visualize_marginal_density(model_name, missing_indices, n, nrep, range_value, smooth_value, figname):

    percentages = [.01,.05,.1,.25,.5]
    reference_numbers = [0,1,2,3,4]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    marginal_densities = np.zeros((5,nrep))
    diffusion_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth_value) + "_4000_random" + str(percentages[i]))
        image_name = "ref_image" + str(reference_numbers[i])
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                              missing_indices[i], n, nrep, range_value,
                                                              smooth_value)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density


    fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (20,8))
    for i in range(0,10):
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            axs[int(i/5),int(i%5)].imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            axs[int(i/5),int(i%5)].plot(matrix_index[1], matrix_index[0], "ro", markersize = 10, linewidth = 20)
        else:
            matrix_index = index_to_matrix_index(missing_indices[(i%5)], n)
            sns.kdeplot(marginal_densities[(i % 5),:], ax = axs[int(i/5),int(i%5)], color = "blue")
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = axs[int(i/5),int(i%5)], color = 'orange', linestyle = '--')
            axs[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index[0],matrix_index[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_ylim([0,1])
            axs[int(i/5),int(i%5)].legend(labels = ['true', 'diffusion'])


    plt.savefig(figname)
    plt.clf()


def visualize_ncs_vs_univariate_lcs_marginal_density(model_name, univariate_lcs_file_name, missing_indices,
                                                     n, nrep, range_value, smooth, p, figname):

    ps = [.01,.05,.1,.25,.5]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    univariate_lcs_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_" + str(nrep) + "_random" + str(p) + ".npy")
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
    spec = gridspec.GridSpec(ncols=5, nrows=3,
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
        elif(i < 10):
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(univariate_lcs_marginal_densities[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]))
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            ax.legend(labels = ['NCS'], fontsize = 6)
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            pdd = pd.DataFrame(np.concatenate([bivariate_densities[(i%5),:,:], class_vector],axis = 1),
                                    columns = ['x', 'y', 'class'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64', 'class': 'float64'})
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], hue = 'class',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = 1)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            #axs[int(i/5),int(i%5)].legend(handles = [blue_patch, orange_patch], labels = ['true', 'NCS'])
            axs[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_ylim([-4.5,4.5])


    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()


def visualize_ncs_vs_lcs_marginal_and_bivariate_density(model_name, univariate_lcs_file_name, missing_indices,
                                                        n, nrep, range_value, smooth, figname):

    ps = [.01,.05,.1,.25,.5]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    univariate_lcs_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_" + str(nrep) + "_random" + str(p) + ".npy")
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
missing_indices = [600, 700, 200, 343, 495]
n = 32
nrep = 4000
range_value = 3.0
figname = "figures/br_percentages_ncs_vs_lcs_marginal_density.png"
smooth_value = 1.5
visualize_ncs_vs_lcs_marginal_and_bivariate_density(model_name, univariate_lcs_file_name, missing_indices,
                                                     n, nrep, range_value, smooth, p, figname)