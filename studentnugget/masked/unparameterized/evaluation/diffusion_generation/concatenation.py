import numpy as np

n = 32
conc = np.zeros((0,1,n,n))
for i in range(0, 4):
    a = np.load("data/model1/ref_image1/diffusion/model1_beta_min_max_01_25_random50_250_" + str(i) + ".npy")
    conc = np.concatenate([conc,a], axis = 0)

np.save("data/model1/ref_image1/diffusion/model1_beta_min_max_01_25_random50_1000.npy", conc)