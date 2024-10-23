import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
from generate_true_conditional_samples import *
from paper_figure_helper_functions import *


def produce_bivariate_densities(model_name, lengthscale, variance, image_name, nrep,
                                missing_index1, missing_index2, file_name):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    true_images = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance,
                                                  lengthscale, observations, nrep)
    
    bivariate_densities = np.concatenate([(true_images.flatten()[missing_index1]).reshape((nrep,1)),
                                          (true_images.flatten()[missing_index2]).reshape((nrep,1))], axis = 1)
    diffusion_bivariate_densities = np.concatenate([(true_images.flatten()[missing_index1]).reshape((nrep,1)),
                                          (true_images.flatten()[missing_index2]).reshape((nrep,1))], axis = 1)
    return bivariate_densities, diffusion_bivariate_densities

def visualize_bivariate_density(model_name, lengthscale, variance, image_name, nrep,
                                missing_index1, missing_index2, file_name):
    
    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    for i in range(0,5):

        ref_image = "ref_image" + str(i)
        produce_bivariate_densities()
    