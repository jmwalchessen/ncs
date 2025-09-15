import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
from generate_true_conditional_samples import *
from paper_figure_helper_functions import *
from matplotlib import patches as mpatches
from matplotlib import gridspec


def produce_bivariate_densities(model_name, lengthscale, variance, image_name, nrep,
                                missing_index1, missing_index2, file_name):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    conditional_unobserved_samples = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance,
                                                  lengthscale, observations, nrep)
    true_images = concatenate_observed_and_kriging_samples(observations, conditional_unobserved_samples, mask, n)
    true_images = true_images.reshape((nrep,n**2))
    diffusion_images = diffusion_images.reshape((nrep,n**2))
    bivariate_densities = np.concatenate([(true_images[:,missing_index1]).reshape((nrep,1)),
                                          (true_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    diffusion_bivariate_densities = np.concatenate([(diffusion_images[:,missing_index1]).reshape((nrep,1)),
                                          (diffusion_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return bivariate_densities, diffusion_bivariate_densities

def visualize_bivariate_density(model_name, variance, nrep,
                                missing_indices1, missing_indices2, n, figname):
    
    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    bivariate_densities = np.zeros((5,2*nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) + "_beta_min_max_01_20_random50_1000")
        bdensities, dbdensities = produce_bivariate_densities(model_name, lengthscales[i], variance,
                                                                                         image_name, nrep, missing_indices1[i],
                                                                                         missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        bivariate_densities[i,:,:] = np.concatenate([bdensities,dbdensities], axis = 0)
    
    class_vector = np.concatenate([(np.ones((nrep))).reshape((nrep,1)),
                                   (np.zeros((nrep)).reshape((nrep,1)))], axis = 0)
    
    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(2.5)
 
    # set width of each subplot as 8
    fig.set_figwidth(2.5)
    spec = gridspec.GridSpec(ncols=5, nrows=2,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.5, height_ratios=[1, 1])
    for i in range(0,10):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            missing_index1 = missing_indices1[i]
            missing_index2 = missing_indices2[i]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = ax[int(i/5),int(i%5)].imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            ax[int(i/5),int(i%5)].plot(matrix_index1[1], matrix_index1[0], "r^", markersize = 10, linewidth = 20)
            ax[int(i/5),int(i%5)].plot(matrix_index2[1], matrix_index2[0], "k^", markersize = 10, linewidth = 20)
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            pdd = pd.DataFrame(np.concatenate([bivariate_densities[(i%5),:,:], class_vector],axis = 1),
                                    columns = ['x', 'y', 'class'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64', 'class': 'float64'})
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], hue = 'class',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            ax[int(i/5),int(i%5)].legend(handles = [blue_patch, orange_patch], labels = ['true', 'diffusion'])
            ax[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax[int(i/5),int(i%5)].axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            ax[int(i/5),int(i%5)].set_ylim([-4.5,4.5])

    #fig.colorbar(im, ax=ax, shrink = .6)
    plt.savefig(figname)
    plt.clf()


def visualize_close_bivariate_density(model_name, variance, nrep,
                                      missing_indices1, missing_indices2,
                                      n, figname):
    
    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    bivariate_densities = np.zeros((5,2*nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) + "_beta_min_max_01_20_random05_4000")
        bdensities, dbdensities = produce_bivariate_densities(model_name, lengthscales[i], variance,
                                                                                         image_name, nrep, missing_indices1[i],
                                                                                         missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        bivariate_densities[i,:,:] = np.concatenate([bdensities,dbdensities], axis = 0)
    
    class_vector = np.concatenate([(np.ones((nrep))).reshape((nrep,1)),
                                   (np.zeros((nrep)).reshape((nrep,1)))], axis = 0)
    
    fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (20,6))
    for i in range(0,10):
        if(i < 5):
            missing_index1 = missing_indices1[i]
            missing_index2 = missing_indices2[i]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = axs[int(i/5),int(i%5)].imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            axs[int(i/5),int(i%5)].plot(matrix_index1[1], matrix_index1[0], "rx", markersize = 10, linewidth = 20)
            axs[int(i/5),int(i%5)].plot(matrix_index2[1], matrix_index2[0], "kx", markersize = 10, linewidth = 20)
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            pdd = pd.DataFrame(np.concatenate([bivariate_densities[(i%5),:,:], class_vector],axis = 1),
                                    columns = ['x', 'y', 'class'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64', 'class': 'float64'})
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], hue = 'class',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            axs[int(i/5),int(i%5)].legend(handles = [blue_patch, orange_patch], labels = ['true', 'diffusion'])
            axs[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_ylim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_xlabel("")
            axs[int(i/5),int(i%5)].set_ylabel("")
            axs[int(i/5),int(i%5)].set_xticks(ticks = [-4,-2,0,2,4], labels = [-4,-2,0,2,4])
            axs[int(i/5),int(i%5)].set_yticks(ticks = [-4,-2,0,2,4], labels = [-4,-2,0,2,4])

    fig.colorbar(im, ax=axs, shrink = .6)
    plt.savefig(figname)
    plt.clf()
model_name = "model7"
variance = 1.5
nrep = 4000
missing_indices1 = [232,772,810,493,567]
missing_indices2 = [233,835,874,505,568]
figname = "figures/gp_parameter_close_bivairate_density_model7_random05.png" 
n = 32
visualize_close_bivariate_density(model_name, variance, nrep,
                                  missing_indices1, missing_indices2,
                                  n, figname) 