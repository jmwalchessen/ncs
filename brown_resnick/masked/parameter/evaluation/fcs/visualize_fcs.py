import matplotlib.pyplot as plt
import numpy as np
import torch as th

def preprocessing_fcs_file(ref_folder, fcs_file, nrep, n):
    
    fcs_unobserved = np.load((ref_folder + "/" + fcs_file))
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load((ref_folder + "/mask.npy"))
    fcs_images = np.zeros((nrep,n,n))
    repeated_refimage = np.repeat(ref_image, nrep)
    fcs_images[:,mask == 1] = np.tile(ref_image[:], reps = nrep)
    fcs_images[:,mask == 0] = fcs_unobserved

def visualize_fcs(ref_folder, fcs_file, figname, irep):

    fcs_images = np.load((ref_folder + "/" + fcs_file))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    ax.imshow(fcs_images[irep,:,:], vmin = -2, vmax = 6)
    plt.savefig(figname)

def visualize_fcs_with_variables():
    
    ref_folder = "data/model4/ref_image1"
    fcs_file = "fcs_range_1.0_smooth_1.5_nugget_1e5_4000.npy"
    figname = "visualizations/fcs_range_1.0_smooth_1.5_nugget_1e5_0.png"
    irep = 0
    visualize_fcs(ref_folder, fcs_file, figname, irep)