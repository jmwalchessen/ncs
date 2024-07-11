import numpy as np

n = 32
images = np.zeros((0,1,n,n))
for i in range(0, 4):
    current = np.load("data/gpmodel7/ref_image1/diffusion/model7_beta_min_max_01_25_random80_250_" + str(i) + ".npy")
    images = np.concatenate([images, current], axis = 0)

np.save("data/gpmodel7/ref_image1/diffusion/model7_beta_min_max_01_25_random80_1000.npy", images)
