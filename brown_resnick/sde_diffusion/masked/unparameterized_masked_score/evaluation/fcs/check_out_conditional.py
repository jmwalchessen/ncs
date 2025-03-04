import numpy as np


ncs_images = np.load("data/conditional/obs1/ref_image2/diffusion/model5_range_3.0_smooth_1.5_4000_random.npy")
mask = np.load("data/conditional/obs1/ref_image2/mask.npy")
ref_img = np.log(np.load("data/conditional/obs1/ref_image2/ref_image.npy"))
obs= np.multiply(ref_img, mask)
print(ncs_images)