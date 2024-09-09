import numpy as np
import torch as th

def load_mcmc_interpolation_images(folder_name, mcmc_file_name, nreps, n):

    ref_image = np.load((folder_name + "/ref_image.npy"))
    mask = np.load((folder_name + "/mask.npy"))
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    m = missing_indices.shape[0]
    mcmc_partial_images = np.log(np.load((folder_name +
                                   "/mcmc_interpolation/mcmc_interpolation_simulations_range_3_smooth_1.6_4000.npy")))
    mcmc_partial_images = np.swapaxes(mcmc_partial_images, 0, 1)
    mcmc_images = np.tile(ref_image.reshape((1,(n**2))), reps = (nreps,1))
    mcmc_images[:,missing_indices] = mcmc_partial_images
    mcmc_images = mcmc_images.reshape((nreps,n,n))
    return mcmc_images




