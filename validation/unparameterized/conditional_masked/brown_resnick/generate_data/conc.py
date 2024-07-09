import numpy as np

n = 32
samples = np.zeros((0,1,n,n))
for i in range(2, 8):
    current_samples = np.load("data/conditional/ref_img4/model6_random3050_beta_min_max_01_20_1000_random50_250_" + str(i) + ".npy")
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/conditional/ref_img4/model6_random3050_beta_min_max_01_20_1000_random50_1500.npy", samples)