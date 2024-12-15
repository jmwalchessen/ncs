import matplotlib.pyplot as plt
import numpy as np
import torch as th

def preprocessing_joint_lcs_file(ref_folder, joint_lcs_file, nrep, n):
    
    joint_lcs_unobserved = np.load((ref_folder + "/" + joint_lcs_file))
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load((ref_folder + "/mask.npy"))
    joint_lcs_images = np.zeros((nrep,n,n))
    repeated_refimage = np.repeat(ref_image, nrep)
    print(repeated_refimage.shape)
    joint_lcs_images[:,mask == 1] = np.tile(ref_image[:], reps = nrep)
    joint_lcs_images[:,mask == 0] = joint_lcs_unobserved

def visualize_joint_lcs(ref_folder, joint_lcs_file, figname, irep):

    joint_lcs_images = np.load((ref_folder + "/" + joint_lcs_file))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
    ax.imshow(joint_lcs_images[irep,:,:], vmin = -2, vmax = 6)
    plt.savefig(figname)


ref_folder = "data/model4/ref_image0"
joint_lcs_file = "processed_joint_lcs_range_3.0_smooth_1.5_nugget_1e5_4000.npy"
figname = "visualizations/joint_lcs_range_3.0_smooth_1.5_nugget_1e5_0.png"
irep = 0
visualize_joint_lcs(ref_folder, joint_lcs_file, figname, irep)