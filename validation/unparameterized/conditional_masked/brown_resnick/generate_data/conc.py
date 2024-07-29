import numpy as np

n = 32
samples = np.zeros((0,1,n,n))
for i in range(0,4):
    current_samples = np.load("data/conditional/model19/ref_img2/model19_random50_smooth_1.2_beta_min_max_01_20_1000_random0_250_" + str(i) + ".npy")
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/conditional/model19/ref_img2/model19_random50_smooth_1.2_beta_min_max_01_20_1000_random0_1000.npy", samples)