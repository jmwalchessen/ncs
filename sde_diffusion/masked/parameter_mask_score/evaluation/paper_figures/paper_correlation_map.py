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
                                                  variance):
    
    n = 32
    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    pass


    