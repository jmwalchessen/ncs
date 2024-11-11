import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from generate_true_conditional_samples import *
from paper_figure_helper_functions import *


def produce_true_and_diffusion_correlation_map_per_pixel(diffusion_images, mask, missing_index, n, nrep,
                                                         variance, lengthscale, ref_image):

    flatten_diffusion_images = diffusion_images.reshape((nrep,n**2))
    flatten_masks = mask.reshape(((n**2)))
    unobserved_diffusion_images = flatten_diffusion_images[:,flatten_masks == 0]
    m = unobserved_diffusion_images.shape[1]
    empirical_covariance = np.corrcoef(unobserved_diffusion_images, rowvar = False)
    diffusion_cov = empirical_covariance[:,missing_index]
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    diffusion_cov_image = np.zeros((n**2))
    diffusion_cov_image[missing_indices] = diffusion_cov

    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    observations = np.multiply(ref_image, mask)
    observations = np.delete(observations.reshape((n**2)), missing_indices)
    minX = minY = -10
    maxX = maxY = 10

    true_cond_images = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                                       nrep)
    unobserved_true_images = true_cond_images.reshape((nrep,m))
    empirical_true_covariance = np.corrcoef(unobserved_true_images, rowvar = False)
    true_cov = empirical_true_covariance[:,missing_index]
    true_cov_image = np.zeros((n**2))
    true_cov_image[missing_indices] = true_cov

    return true_cov_image, diffusion_cov_image

def visualize_true_and_diffusion_correlation_maps(missing_indices, n, nrep,
                                                  lengthscale, variance, model_name, figname):
    
    n = 32
    masks = np.zeros((5,n,n))
    minX = minY = -10
    maxX = maxY = 10
    percentages = [.01,.05,.1,.25,.5]
    ref_numbers = [0,1,2,4,7]
    true_cov_images = np.zeros((5,n,n))
    diffusion_cov_images = np.zeros((5,n,n))
    for i in range(0, 5):
        image_name = "ref_image" + str(ref_numbers[i])
        ref_image = load_reference_image(model_name, image_name)
        mask = load_mask(model_name, image_name)
        masks[i,:,:] = mask
        y = load_observations(model_name, image_name, mask, n)
        file_name = (model_name + "_beta_min_max_01_20_1000_" + str(percentages[i]))
        diffusion_images = load_diffusion_images(model_name, image_name, file_name)
        true_cov_image, diffusion_cov_image = produce_true_and_diffusion_correlation_map_per_pixel(diffusion_images, mask, missing_indices[i], n, nrep,
                                                                                                   variance, lengthscale, ref_image)
        true_cov_images[i,:,:] = true_cov_image.reshape((n,n))
        diffusion_cov_images[i,:,:] = diffusion_cov_image.reshape((n,n))

        
        fig = plt.figure(figsize=(10,4))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                    axes_pad=0.1,  # pad between Axes in inch.
                    cbar_mode="single"
                    )
        
        for i, ax in enumerate(grid):

            if(i < 5):
                im = ax.imshow(true_cov_images[i,:,:], cmap='viridis', vmin = 0, vmax = 1,
                            alpha = (1-masks[(i % 5),:,:].astype(float)))
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            elif(i < 10):
                im = ax.imshow(diffusion_cov_images[(i % 5),:,:], cmap='viridis', vmin = 0, vmax = 1,
                            alpha = (1-masks[(i % 5),:,:].astype(float)))
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

            else:
                pass


        ax.cax.colorbar(im)
        plt.savefig(figname)
    

n = 32
nrep = 4000
variance = 1.5
lengthscale = 3.0
model_name = "model7"
figname = "figures/paper_correlation_map_far_from_observed.png"
missing_indices = [455, 103, 500, 200, 305]
visualize_true_and_diffusion_correlation_maps(missing_indices, n, nrep, lengthscale, variance, model_name, figname)