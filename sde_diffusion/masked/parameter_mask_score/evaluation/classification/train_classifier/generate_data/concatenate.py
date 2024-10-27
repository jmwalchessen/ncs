import numpy as np
import os

n = 32
samples = np.zeros((0,1,n,n))
pathname = "data/model6/train"
variance = 1.5
filenames = [f for f in os.listdir(pathname)]

def extract_lengthscale(filename):

    l = filename.split("lengthscale_")[1]
    lengthscale = float(l.split(".npy")[0])
    return lengthscale

lengthscales = [extract_lengthscale(f) for f in filenames]
print(lengthscales)
for i in range(len(lengthscales)):
    current_samples = np.load((pathname + "/" + filenames[i]))
    samples = np.concatenate([samples, current_samples], axis = 0)

np.save("data/model6/train/train_images_conditional_variance_1.5_lengthscale_1_5_uniform_100_5000.npy", samples)
np.save("data/model6/train/train_variance_1.5_lengthscales_1_5_uniform_100.npy", np.array(lengthscales))