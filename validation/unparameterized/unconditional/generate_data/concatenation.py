import numpy as np

n = 32
conc = np.zeros((0,1,n,n))
for i in range(0, 100):
    a = np.load("data/diffusion/model5_unconditional_lengthscale_1.6_variance_0.4_1000_" + str(i) + ".npy")
    conc = np.concatenate([conc,a], axis = 0)

np.save("data/diffusion/model5_unconditional_lengthscale_1.6_variance_0.4_100000.npy", conc)