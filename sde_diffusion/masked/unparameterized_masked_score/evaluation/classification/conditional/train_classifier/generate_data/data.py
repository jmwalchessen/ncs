import numpy as np

a = np.load("data/fixed_mask/model2/mask1/mask.npy")
b = np.load("data/fixed_mask/model2/mask1/conditional_diffusion_random50_variance_.4_lengthscale_1.6_model2_40000.npy")

def mask_from_diffusion_images(diffusion_images):

    n = 32
    mask_std = (np.std(diffusion_images, axis = 0)).reshape((n,n))
    mask = np.zeros((n,n))
    mask[mask_std == 0] = 1

mask_from_diffusion_images(b)