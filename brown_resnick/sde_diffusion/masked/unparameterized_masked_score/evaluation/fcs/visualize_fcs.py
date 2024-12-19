import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.patches import Rectangle

def preprocessing_fcs_file(ref_folder, fcs_file, nrep, n):
    
    fcs_unobserved = np.load((ref_folder + "/" + fcs_file))
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load((ref_folder + "/mask.npy"))
    print(mask.shape)
    fcs_images = np.zeros((nrep,n**2))
    repeated_refimage = np.repeat(ref_image, nrep)
    fcs_images = (np.tile(ref_image.reshape((1,n**2)), reps = nrep)).reshape((n,nrep**2))
    print((fcs_images).shape)
    fcs_images[:,mask.flatten() == 0] = np.log(fcs_unobserved)
    return fcs_images

def visualize_fcs(ref_folder, fcs_file, figname, irep):
    
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load((ref_folder + "/mask.npy"))
    observed_indices = np.argwhere(mask > 0)
    fcs_images = np.load((ref_folder + "/" + fcs_file))
    print(np.log(ref_image[mask == 1]))
    print(fcs_images[:,mask == 1])
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
    im = ax[0,0].imshow(np.log(ref_image), alpha = mask.astype(float), vmin = -2, vmax = 6)
    plt.colorbar(im, shrink = .6)
    im = ax[0,1].imshow(fcs_images[irep,:,:], alpha = mask.astype(float), vmin = -2, vmax = 6)
    plt.colorbar(im, shrink = .6)
    im = ax[1,0].imshow(np.log(ref_image), vmin = -2, vmax = 6)
    plt.colorbar(im, shrink = .6)
    im = ax[1,1].imshow(fcs_images[irep,:,:], vmin = -2, vmax = 6)
    plt.colorbar(im, shrink = .6)
    for i in range(observed_indices.shape[0]):
        rect = Rectangle(((observed_indices[i,1]-.55), (observed_indices[i,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
        ax[0,0].add_patch(rect)
        rect = Rectangle(((observed_indices[i,1]-.55), (observed_indices[i,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
        ax[0,1].add_patch(rect)
        rect = Rectangle(((observed_indices[i,1]-.55), (observed_indices[i,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
        ax[1,0].add_patch(rect)
        rect = Rectangle(((observed_indices[i,1]-.55), (observed_indices[i,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
        ax[1,1].add_patch(rect)
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()

def multiple_visualize_fcs(ref_folder, fcs_file, figname, nrep):

    for irep in range(nrep):

        visualize_fcs(ref_folder, fcs_file, (figname + "_" + str(irep) + ".png"),
                            irep)


ref_folder = "data/model4/ref_image0"
irep = 0
nrep = 4000
n = 32
m = 1
nrep = 4000
fcs_file = "processed_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
figname = (ref_folder + "/visualizations/fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m))
multiple_visualize_fcs(ref_folder, fcs_file, figname, nrep)