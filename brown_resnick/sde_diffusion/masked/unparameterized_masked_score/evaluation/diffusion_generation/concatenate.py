import numpy as np

n = 32
samples = np.zeros((0,1,n,n))
for i in range(0,4):
    current_samples = np.load("data/schlather/model1/ref_image1/diffusion/model1_beta_min_max_01_20_random50_250_" + str(i) + ".npy")
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/schlather/model1/ref_image1/diffusion/model1_random50_beta_min_max_01_20_1000.npy", samples)