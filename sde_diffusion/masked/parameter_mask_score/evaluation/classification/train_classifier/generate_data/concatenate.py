import numpy as np

n = 32
samples = np.zeros((0,1,n,n))
for i in range(0,4):
    current_samples = np.load("data/model6/unconditional/unconditional_images_variance_.4_lengthscale_1.6_10000_" + str(i) + ".npy")
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/model6/unconditional/unconditional_images_variance_.4_lengthscale_1.6_100000.npy", samples)