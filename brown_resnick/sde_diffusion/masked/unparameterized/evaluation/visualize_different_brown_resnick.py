import numpy as np
import matplotlib.pyplot as plt

def visualize_field(image):

    fig, ax = plt.subplots(1)
    plt.imshow(image, vmin = -2, vmax = 4)
    plt.show()

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log10_transformation(images):

    images = np.log10(np.where(images !=0, images, np.min(images[images != 0])))

    return images


n = 32
br_samples = log_transformation(np.load("brown_resnick_samples_range_1.6_smooth_1.2_500.npy"))
visualize_field(br_samples[10,:].reshape((n,n)))