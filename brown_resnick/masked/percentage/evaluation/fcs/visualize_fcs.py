import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.patches import Rectangle
from append_directories import *

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

def visualize_fcs(ref_folder, mask_file, fcs_file, figname, irep):
    
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load(mask_file)
    observed_indices = np.argwhere(mask > 0)
    fcs_images = np.load((ref_folder + "/" + fcs_file))
    print(mask.shape)
    print(fcs_images.shape)
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
        
def visualize_unconditional_fcs(ref_folder, fcs_file, mask_file, obs_file, irep, nrep, n, figname):
    
    ref_images = np.log((np.load((ref_folder + "/" + obs_file))).reshape((nrep,n,n)))
    masks = (np.load((ref_folder + "/" + mask_file))).reshape((nrep,n,n))
    observed_indices = np.argwhere(masks[irep,:,:] > 0)
    fcs_images = np.log((np.load((ref_folder + "/" + fcs_file))).reshape((nrep,n,n)))
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))
    im = ax[0,0].imshow(ref_images[irep,:,:], alpha = masks[irep,:,:].astype(float), vmin = -2, vmax = 6)
    plt.colorbar(im, shrink = .6)
    im = ax[0,1].imshow(fcs_images[irep,:,:], alpha = masks[irep,:,:].astype(float), vmin = -2, vmax = 6)
    plt.colorbar(im, shrink = .6)
    im = ax[1,0].imshow(ref_images[irep,:,:], vmin = -2, vmax = 6)
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

def visualize_unconditional_fcs_with_variables(irep, m):

    evaluation_folder = append_directory(2)
    ref_folder = (evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs")
    nrep = 4000
    n = 32
    nrep = 4000
    fcs_file = "processed_unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
    mask_file = "unconditional_mask_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
    obs_file = "unconditional_obs_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
    figname = "visualizations/unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) +  "_" + str(irep) + ".png"
    visualize_unconditional_fcs(ref_folder, fcs_file, mask_file, obs_file, irep, nrep, n, figname)

def visualize_fcs_multiple_ranges_and_locations(ireps):

    range_values = [i for i in range(1,6)]
    ms = [i for i in range(1,8)]
    nrep = 10
    for range_value in range_values:
        ref_folder = "data/ranges/ref_image" + str((range_value-1))
        for m in ms:
            for irep in ireps:
                fcs_file = ("processed_log_scale_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" +
                            str(m) +  "_" + str(nrep) + ".npy")
                mask_file = (ref_folder + "/mask_obs_" + str(m) + ".npy")
                figname = ("visualizations/fcs_range_" + str(range_value) +
                           "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep)
                           + "_" + str(irep) + ".png")
                visualize_fcs(ref_folder, mask_file, fcs_file, figname, irep)


ireps = [i for i in range(0,10)]
visualize_fcs_multiple_ranges_and_locations(ireps)