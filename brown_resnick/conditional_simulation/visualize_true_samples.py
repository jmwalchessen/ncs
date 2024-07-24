import numpy as np
import matplotlib.pyplot as plt

refimg_folder = "data/ref_img2"
ref_img = np.load((refimg_folder + "/ref_img.npy"))
mask = np.load((refimg_folder + "/mask01.npy"))
conditional_samples = np.load((refimg_folder + "/conditional_simulations_100.npy"))

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log10_transformation(images):

    images = np.log10(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def visualize_field(image, n, figname):

    fig, ax = plt.subplots(1)
    ax.imshow(image.reshape((n,n)), vmin = 0, vmax = 10)
    plt.savefig(figname)

def visualize_log_field(image, n, figname):

    fig, ax = plt.subplots(1)
    logimage = log_transformation(image)
    ax.imshow(logimage.reshape((n,n)), vmin = -4, vmax = 6)
    plt.savefig(figname)

def visualize_mask_field(image, n, mask, figname):

    fig, ax = plt.subplots(1)
    #logimage = log_transformation(image)
    ax.imshow(image.reshape((n,n)), vmin = 0, vmax = 10, alpha = mask.astype(float).reshape((n,n)))
    plt.savefig(figname)
"""
figname = (refimg_folder + "/logref_img.png")
n = 32
visualize_log_field(ref_img, n, figname)
figname = (refimg_folder + "/ref_img.png")
n = 32
visualize_field(ref_img, n, figname)
figname = (refimg_folder + "/masked_refimg.png")
visualize_mask_field(ref_img, n, mask, figname)
for i in range(0, 100, 10):
    figname = "data/ref_img2/conditional_sample_" + str(i) + ".png"
    n = 32
    visualize_field(conditional_samples[i,:,:,:], n, figname)
    figname = "data/ref_img2/conditional_log_sample_" + str(i) + ".png"
    visualize_log_field(conditional_samples[i,:,:,:], n, figname)
"""
n = 32
observed_indices = (mask==1).reshape((n**2))
print(conditional_samples[:,observed_indices])