import numpy as np

n = 32
samples = np.zeros((0,1,n,n))
for i in range(0,3):
    current_samples = np.load("data/model2/ref_image3/diffusion/model2_beta_min_max_01_20_random50_variance_.4_lengthscale_1.2_250_" + str(i) + ".npy")
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/model2/ref_image3/diffusion/model2_random50_beta_min_max_01_20_variance_.4_lengthscale_1.2_750.npy", samples)