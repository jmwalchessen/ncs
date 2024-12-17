import matplotlib.pyplot as plt
import numpy as np
import torch as th

def preprocessing_joint_lcs_file(ref_folder, joint_lcs_file, nrep, n):
    
    joint_lcs_unobserved = np.load((ref_folder + "/" + joint_lcs_file))
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load((ref_folder + "/mask.npy"))
    print(mask.shape)
    joint_lcs_images = np.zeros((nrep,n**2))
    repeated_refimage = np.repeat(ref_image, nrep)
    joint_lcs_images = (np.tile(ref_image.reshape((1,n**2)), reps = nrep)).reshape((n,nrep**2))
    print((joint_lcs_images).shape)
    joint_lcs_images[:,mask.flatten() == 0] = np.log(joint_lcs_unobserved)
    return joint_lcs_images

def visualize_joint_lcs(ref_folder, joint_lcs_file, figname, irep):
    
    ref_image = np.load((ref_folder + "/ref_image.npy"))
    mask = np.load((ref_folder + "/mask.npy"))
    joint_lcs_images = np.load((ref_folder + "/" + joint_lcs_file))
    print(ref_image[mask == 1])
    print(joint_lcs_images[1,mask == 1])
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10,10))
    ax[0].imshow(np.log(ref_image), alpha = mask.astype(float), vmin = -2, vmax = 6)
    ax[1].imshow(np.log(ref_image), vmin = -2, vmax = 6)
    ax[2].imshow(joint_lcs_images[irep,:,:], vmin = -2, vmax = 6)
    plt.savefig(figname)
    plt.clf()

def multiple_visualize_joint_lcs(ref_folder, joint_lcs_file, figname, nrep):

    for irep in range(nrep):

        visualize_joint_lcs(ref_folder, joint_lcs_file, (figname + "_" + str(irep) + ".png"),
                            irep)


ref_folder = "data/model4/ref_image0"
irep = 5
nrep = 10
n = 4
joint_lcs_file = "processed_joint_lcs_range_3.0_smooth_1.5_nugget_1e5_obs_7_10.npy"
figname = (ref_folder + "/visualizations/joint_lcs_range_3.0_smooth_1.5_nugget_1e5_obs_7")
multiple_visualize_joint_lcs(ref_folder, joint_lcs_file, figname, nrep)