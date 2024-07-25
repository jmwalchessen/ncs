import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

br_samples = np.load("brown_resnick_samples_5000.npy")

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log10_transformation(images):

    images = np.log10(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def log_and_normalize(images):

    images = np.log(images)
    images = (images - np.mean(images))/np.std(images)
    return images

def global_boundary_process(images, minvalue, maxvalue):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - .5
    log01cs = 6*log01c
    return log01cs

def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs


log10samples = log10_transformation(br_samples)
#log10samples = (log10samples-np.mean(log10samples))/np.std(log10samples)
logsamples = log_transformation(br_samples)
print(np.mean(log10samples))
#logsamples = (logsamples - np.mean(logsamples))/np.std(log10samples)
#log10samples = (log10samples - np.quantile(log10samples, [.9]))
print(np.min(log10samples[0,:]))
print(np.max(log10samples[0,:]))
log10density = log10samples[:,343]
print(np.min(log10density))
print(np.max(log10density))
logdensity = logsamples[:,343]
fig, ax = plt.subplots(1)
pdd = pd.DataFrame(log10density, columns = ["log10"])
logpdd = pd.DataFrame(logdensity, columns = ["log"])
sns.kdeplot(data = pdd["log10"], palette = ["orange"], bw_adjust = 1, ax = ax)
sns.kdeplot(data = logpdd["log"], palette = ["blue"], bw_adjust = 1, ax = ax)
plt.show()