import numpy as np
import torch as th

def concatenate_conditional_simulation(folder_name, condsim_name, n, nrep):

    condsim = np.load((folder_name + "/mcmc/approximate_conditional/" + condsim_name))
    ref_image = (np.load((folder_name + "/ref_image.npy"))).reshape((1,n**2))
    mask = (np.load((folder_name + "/mask.npy"))).reshape((n**2))

    conditional_images = np.zeros((nrep, n**2))
    conditional_images[:,mask == 0] = condsim
    conditional_images[:,mask == 1] = np.tile(ref_image[:,mask == 1], reps = (nrep,1))

    conditional_images = conditional_images.reshape((nrep,n,n))
    return conditional_images

n = 32
nrep = 4000
folder_name = "data/model3/ref_image2"
condsim_name = "approximate_conditional_simulation_range_11_smooth_1_4000.npy"
conditional_images = concatenate_conditional_simulation(folder_name, condsim_name, n, nrep)
condimage_name = "approximate_conditional_images_range_11_smooth_1_4000.npy"
np.save((folder_name + "/mcmc/approximate_conditional/" + condimage_name), conditional_images)