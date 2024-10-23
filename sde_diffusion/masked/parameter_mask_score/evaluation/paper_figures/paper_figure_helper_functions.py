from append_directories import *
import numpy as np

def load_diffusion_images(model_name, image_name, file_name):

    eval_folder = append_directory(2)
    diffusion_images = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/diffusion/" + file_name + ".npy"))
    return diffusion_images

def load_mask(model_name, image_name):

    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/" + "mask.npy"))
    return mask

def load_reference_image(model_name, image_name):

    eval_folder = append_directory(2)
    ref_image = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy"))
    return ref_image

def load_observations(model_name, image_name, mask, n):

    eval_folder = append_directory(2)
    ref_image = np.load(eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy")
    observations = ref_image[(mask).astype(int) == 1]
    return observations

def produce_figure_name(model_name, image_name, fig_name):

    eval_folder = append_directory(2)
    figname = (eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name +
               "/paper_figures/" + fig_name)
    return figname


def concatenate_observed_and_kriging_sample(observed, conditional_unobserved_sample, mask, n):

    conditional_sample = np.zeros((n**2))
    observed_indices = np.argwhere(mask.flatten() == 1)
    missing_indices = np.argwhere(mask.flatten() == 0)
    m = observed.shape[0]
    conditional_sample[missing_indices] = conditional_unobserved_sample.reshape(((n**2-m),1))
    conditional_sample[observed_indices] = observed.reshape((m,1))
    conditional_sample = conditional_sample.reshape((n,n))
    return conditional_sample

def concatenate_observed_and_kriging_samples(observed, conditional_unobserved_samples, mask, n):

    nrep = conditional_unobserved_samples.shape[0]
    conditional_samples = np.zeros((nrep,n**2))
    m = observed.shape[0]
    observed_indices = (np.argwhere(mask.flatten() == 1)).reshape((m))
    missing_indices = (np.argwhere(mask.flatten() == 0)).reshape((n**2-m))
    conditional_samples[:,missing_indices] = conditional_unobserved_samples.reshape((nrep,(n**2-m)))
    conditional_samples[:,observed_indices] = np.tile(observed.reshape((1,m)), (nrep,1))
    conditional_samples = conditional_samples.reshape((nrep,n,n))
    return conditional_samples

def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))