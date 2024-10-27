import numpy as np
import os

n = 32
samples = np.zeros((0,1,n,n))
pathname = "data/model6"
for i in range(0,4):
    filenames = [f for f in os.listdir(pathname)]
    current_samples = np.load((pathname + "/" + filenames[i]))
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/model6/train_images_conditional_variance_.4_lengthscale_1_5_uniform_100_5000.npy", samples)