import numpy as np

n = 32
samples = np.zeros((0,1,n,n))
for i in range(0,4):
    current_samples = np.load("data/model1/ref_image2/diffusion/model1_beta_min_max_01_20_random50_variance_1.6_lengthscale_.4_250_" + str(i) + ".npy")
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/model1/ref_image2/diffusion/model1_random50_variance_1.6_lengthscale_.4_beta_min_max_01_20_1000.npy", samples)