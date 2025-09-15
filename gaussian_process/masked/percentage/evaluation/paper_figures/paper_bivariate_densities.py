import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
from generate_true_conditional_samples import *
from paper_figure_helper_functions import *
from matplotlib import patches as mpatches


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
    print(true_images.shape)
    true_images = true_images.reshape((nrep,n**2))
    diffusion_images = diffusion_images.reshape((nrep,n**2))
    bivariate_densities = np.concatenate([(true_images[:,missing_index1]).reshape((nrep,1)),
                                          (true_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    diffusion_bivariate_densities = np.concatenate([(diffusion_images[:,missing_index1]).reshape((nrep,1)),
                                          (diffusion_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return bivariate_densities, diffusion_bivariate_densities

def visualize_bivariate_density(model_name, lengthscale, variance, nrep,
                                missing_indices1, missing_indices2, n, figname):
    
    percentages = [.01,.05,.1,.25,.5]
    bivariate_densities = np.zeros((5,nrep,2))
    diffusion_bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    reference_numbers = [0,1,2,4,7]

    for i in range(0,5):

        image_name = "ref_image" + str(reference_numbers[i])
        file_name = (model_name + "_beta_min_max_01_20_1000_" + str(percentages[i]))
        bdensities, dbdensities = produce_bivariate_densities(model_name, lengthscale, variance,
                                                                                         image_name, nrep, missing_indices1[i],
                                                                                         missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        bivariate_densities[i,:,:] = bdensities
        diffusion_bivariate_densities[i,:,:] = dbdensities
    
    
    fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (20,6))
    for i in range(0,10):
        if(i < 5):
            missing_index1 = missing_indices1[i]
            missing_index2 = missing_indices2[i]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = axs[int(i/5),int(i%5)].imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            axs[int(i/5),int(i%5)].plot(matrix_index1[1], matrix_index1[0], "r^", markersize = 10, linewidth = 20)
            axs[int(i/5),int(i%5)].plot(matrix_index2[1], matrix_index2[0], "k^", markersize = 10, linewidth = 20)
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            pdd = pd.DataFrame(bivariate_densities[(i%5),:,:],
                                    columns = ['x', 'y'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
            print(pdd)
            diffusion_pdd = pd.DataFrame(diffusion_bivariate_densities[(i%5),:,:],
                                    columns = ['x', 'y'])
            diffusion_pdd = pdd.astype({'x': 'float64', 'y': 'float64'})

            kde2 = sns.kdeplot(data = diffusion_pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], color = 'orange',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], color = 'blue',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            axs[int(i/5),int(i%5)].legend(handles = [blue_patch, orange_patch], labels = ['true', 'diffusion'])
            axs[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_ylim([-4.5,4.5])

    fig.colorbar(im, ax=axs, shrink = .6)
    plt.savefig(figname)
    plt.clf()

def visualize_close_bivariate_density(model_name, lengthscale, variance, nrep,
                                missing_indices1, missing_indices2, n, figname):
    
    percentages = [.01,.05,.1,.25,.5]
    bivariate_densities = np.zeros((5,nrep,2))
    diffusion_bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    reference_numbers = [0,1,2,4,7]

    for i in range(0,5):

        image_name = "ref_image" + str(reference_numbers[i])
        file_name = (model_name + "_beta_min_max_01_20_1000_" + str(percentages[i]))
        bdensities, dbdensities = produce_bivariate_densities(model_name, lengthscale, variance,
                                                                                         image_name, nrep, missing_indices1[i],
                                                                                         missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        bivariate_densities[i,:,:] = bdensities
        diffusion_bivariate_densities[i,:,:] = dbdensities
    
    
    fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (20,6))
    for i in range(0,10):
        if(i < 5):
            missing_index1 = missing_indices1[i]
            missing_index2 = missing_indices2[i]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = axs[int(i/5),int(i%5)].imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            axs[int(i/5),int(i%5)].plot(matrix_index1[1], matrix_index1[0], "rx", markersize = 10, linewidth = 30)
            axs[int(i/5),int(i%5)].plot(matrix_index2[1], matrix_index2[0], "kx", markersize = 10, linewidth = 30)
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            pdd = pd.DataFrame(bivariate_densities[(i%5),:,:],
                                    columns = ['x', 'y'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
            print(pdd)
            diffusion_pdd = pd.DataFrame(diffusion_bivariate_densities[(i%5),:,:],
                                    columns = ['x', 'y'])
            diffusion_pdd = pdd.astype({'x': 'float64', 'y': 'float64'})

            kde2 = sns.kdeplot(data = diffusion_pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], color = 'orange',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = axs[int(i/5),int(i%5)], color = 'blue',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            axs[int(i/5),int(i%5)].legend(handles = [blue_patch, orange_patch], labels = ['true', 'diffusion'])
            axs[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_ylim([-4.5,4.5])

    fig.colorbar(im, ax=axs, shrink = .6)
    plt.savefig(figname)
    plt.clf()



model_name = "model7"
lengthscale = 3.0
variance = 1.5
nrep = 4000
missing_indices1 = [117,484,220,170,162]
missing_indices2 = [934,921,230,856,869]
figname = "figures/gp_percentage_bivairate_density.png" 
n = 32
visualize_bivariate_density(model_name, lengthscale, variance, nrep, missing_indices1,
                            missing_indices2, n, figname)

missing_indices1 = [756,495,147,132,803]
missing_indices2 = [760,498,150,196,837]
figname = "figures/gp_percentage_close_bivairate_density.png" 
visualize_close_bivariate_density(model_name, lengthscale, variance, nrep,
                                missing_indices1, missing_indices2, n, figname)  