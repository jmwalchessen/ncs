import numpy as np
from generate_true_conditional_samples import *
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
from matplotlib import gridspec

def produce_marginal_density(model_name, image_name, file_name, missing_index, n, nrep, variance, lengthscale):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    minX = minY = -10
    maxX = maxY = 10
    true_images = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations, nrep)
    true_images = concatenate_observed_and_kriging_samples(observations, true_images, mask, n)
    true_marginal_density = (true_images.reshape((nrep,n**2)))[:,missing_index]
    return true_marginal_density, diffusion_marginal_density

def visualize_marginal_density(model_name, missing_indices, n, nrep, variance, figname):

    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    marginal_densities = np.zeros((5,nrep))
    diffusion_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) + "_beta_min_max_01_20_random05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        true_marginal_density, diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                                                     missing_indices[i], n, nrep, variance,
                                                                                     lengthscales[i])
        marginal_densities[i,:] = true_marginal_density
        diffusion_marginal_densities[i,:] = diffusion_marginal_density


    fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (20,8))
    for i in range(0,10):
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            im = axs[int(i/5),int(i%5)].imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            axs[int(i/5),int(i%5)].plot(matrix_index[1], matrix_index[0], "ro", markersize = 10, linewidth = 20)
            axs[int(i/5),int(i%5)].set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            axs[int(i/5),int(i%5)].set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            sns.kdeplot(marginal_densities[(i % 5),:], ax = axs[int(i/5),int(i%5)], color = "blue")
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = axs[int(i/5),int(i%5)], color = 'orange', linestyle = '--')
            axs[int(i/5),int(i%5)].axvline(reference_images[(i%5),matrix_index[0],matrix_index[1]], color='red', linestyle = 'dashed')
            axs[int(i/5),int(i%5)].set_xlim([-4.5,4.5])
            axs[int(i/5),int(i%5)].set_ylim([0,1])
            axs[int(i/5),int(i%5)].legend(labels = ['true', 'NCS'])

    fig.colorbar(im, ax=axs, shrink = 1)
    plt.savefig(figname)
    plt.clf()

model_name = "model7"
missing_indices = [845, 700, 200, 301, 118]
n = 32
nrep = 4000
variance = 1.5
figname = "figures/gp_parameter_marginal_density_model7_random05.png"
visualize_marginal_density(model_name, missing_indices, n, nrep, variance, figname)