import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from append_directories import *


def visualie_joint_lcs_multiple_percentages(evaluation_folder, ps,
                                       figname, n, nrep):

    joint_lcs = np.zeros((len(ps),nrep,n,n))
    ncs_images = np.zeros((len(ps),nrep,n,n))
    masks = np.zeros((len(ps),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(ps)):

        joint_lcs[i,:,:,:] = np.load((evaluation_folder + "/joint_lcs/data/model4/ref_image" +
                                           str(i) + "/processed_joint_lcs_range_3.0_smooth_1.5_nugget_1e5_obs_7_10.npy"))
        ncs_images[i,:,:,:] = np.load((evaluation_folder + "/joint_lcs/data/model4/ref_image"
                                       + str(i) + "/ncs_images_range_3.0_smooth_1.5_10.npy"))
        masks[i,:,:] = np.load((evaluation_folder + "/joint_lcs/data/model4/ref_image" + str(i) 
                                + "/mask.npy"))
        
        
    joint_lcs[joint_lcs != 0] = np.log(joint_lcs[joint_lcs != 0])

    for i, ax in enumerate(grid):
        if(i < 5):
            im = ax.imshow(joint_lcs[i,3,:,:], cmap='viridis', vmin = -2, vmax = 6, alpha = (1-masks[i,:,:]))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            im = ax.imshow(ncs_images[(i%5),1,:,:], cmap='viridis', vmin = -2, vmax = 6, alpha = (1-masks[(i%5),:,:]))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    plt.tight_layout()
    ax.cax.colorbar(im)
    plt.savefig(figname)

evaluation_folder = append_directory(2)
smooth = 1.5
ps = [.01,.05,.1,.25,.5]
figname = "figures/br_percentage_joint_lcs_vs_ncs_nugget_1e5_obs_7.png"
n = 32
nrep = 10
visualie_joint_lcs_multiple_percentages(evaluation_folder, ps,
                                       figname, n, nrep)