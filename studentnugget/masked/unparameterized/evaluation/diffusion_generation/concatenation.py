import numpy as np

n = 32
conc = np.zeros((0,1,n,n))
for i in range(0, 4):
    a = np.load("data/model6/ref_image3/diffusion/model6_beta_min_max_01_20_random50_250_" + str(i) + ".npy")
    conc = np.concatenate([conc,a], axis = 0)

np.save("data/model6/ref_image3/diffusion/model6_beta_min_max_01_20_random50_1000.npy", conc)