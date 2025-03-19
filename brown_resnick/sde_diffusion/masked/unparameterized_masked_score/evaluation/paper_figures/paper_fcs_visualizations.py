import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from append_directories import *
from matplotlib.patches import Rectangle


def visualie_fcs_multiple_percentages(evaluation_folder, ps,
                                       figname, n, nrep):

    fcs_images = np.zeros((len(ps),nrep,n,n))
    ncs_images = np.zeros((len(ps),nrep,n,n))
    masks = np.zeros((len(ps),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(ps)):

        fcs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/model4/ref_image" +
                                           str(i) + "/processed_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_7_10.npy"))
        ncs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/model4/ref_image"
                                       + str(i) + "/ncs_images_range_3.0_smooth_1.5_10.npy"))
        masks[i,:,:] = np.load((evaluation_folder + "/ncs/data/model4/ref_image" + str(i) 
                                + "/mask.npy"))
        
        
    fcs_images[fcs_images != 0] = np.log(fcs_images[fcs_images != 0])

    for i, ax in enumerate(grid):
        if(i < 5):
            im = ax.imshow(fcs_images[i,3,:,:], cmap='viridis', vmin = -2, vmax = 6, alpha = (1-masks[i,:,:]))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            im = ax.imshow(ncs_images[(i%5),1,:,:], cmap='viridis', vmin = -2, vmax = 6, alpha = (1-masks[(i%5),:,:]))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    plt.tight_layout()
    ax.cax.colorbar(im)
    plt.savefig(figname)


def visualize_fcs_multiple_ranges(evaluation_folder, range_values, m, n, figname):

    fcs_images = np.zeros((len(range_values),nrep,n,n))
    ref_images = np.zeros((len(range_values),n,n))
    masks = np.zeros((len(range_values),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(range_values)):

        fcs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/ranges/ref_image" +
                                           str(i) + "/processed_log_scale_fcs_range_" + str(range_values[i])
                                           + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_10.npy"))
        ref_images[i,:,:] = np.load((evaluation_folder + "/fcs/data/ranges/ref_image"
                                       + str(i) + "/ref_image.npy"))
        masks[i,:,:] = np.load((evaluation_folder + "/fcs/data/ranges/ref_image" + str(i) 
                                + "/mask_obs_" + str(m) + ".npy"))
        

    ref_images[ref_images != 0] = np.log(ref_images[ref_images != 0])
    print(fcs_images.shape)

    for i, ax in enumerate(grid):
        if(i < 5):
            im = ax.imshow(ref_images[i,:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            observed_indices = np.argwhere(masks[i,:,:] > 0)
            for i in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[i,1]-.55), (observed_indices[i,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
        else:
            im = ax.imshow(fcs_images[(i%5),5,:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            observed_indices = np.argwhere(masks[(i%5),:,:] > 0)
            for i in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[i,1]-.55), (observed_indices[i,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)

    plt.tight_layout()
    ax.cax.colorbar(im)
    plt.savefig(figname)


def visualize_fcs_ncs_multiple_obs_fixed_range(evaluation_folder, range_value, obs_numbers, n, figname, folder_name,
                                               model_version):

    nrep = 4000
    fcs_images = np.zeros((len(obs_numbers),nrep,n,n))
    ref_images = np.zeros((len(obs_numbers),n,n))
    ncs_images = np.zeros((len(obs_numbers),nrep,n,n))
    masks = np.zeros((len(obs_numbers),n,n))
    fig = plt.figure(figsize=(10,4))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    for i in range(len(obs_numbers)):
        print(range_value)
        fcs_images[i,:,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/processed_log_scale_fcs_range_" + str(range_value)
                                     + "_smooth_1.5_nugget_1e5_obs_" + str(obs_numbers[i]) + "_" + str(nrep) + ".npy"))
        ref_images[i,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i]) + 
                                     "/ref_image" + str(int(range_value-1)) + "/ref_image.npy"))
        masks[i,:,:] = np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i]) + "/ref_image"
                                + str(int(range_value-1)) + "/mask.npy"))
        print((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/diffusion/model" + str(model_version) + "_range_"
                                     + str(range_value) + "_smooth_1.5_" + str(nrep) + "_random.npy"))
        ncs_images[i,:,:,:] = (np.load((evaluation_folder + "/fcs/data/conditional/obs" + str(obs_numbers[i])
                                     + "/ref_image" + str(int(range_value-1)) + "/diffusion/model" + str(model_version) + "_range_"
                                     + str(range_value) + "_smooth_1.5_" + str(nrep) + "_random.npy"))).reshape((nrep,n,n))
        

    ref_images[ref_images != 0] = np.log(ref_images[ref_images != 0])

    for i, ax in enumerate(grid):
        if(i < 5):
            im = ax.imshow(ref_images[i,:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            observed_indices = np.argwhere(masks[i,:,:] > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
        elif(i<10):
            im = ax.imshow(fcs_images[(i%5),5,:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            observed_indices = np.argwhere(masks[(i%5),:,:] > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
        else:
            im = ax.imshow(ncs_images[(i%5),5,:,:], cmap='viridis', vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            observed_indices = np.argwhere(masks[(i%5),:,:] > 0)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)

    plt.tight_layout()
    ax.cax.colorbar(im)
    plt.savefig((folder_name + "/" + figname))

def visualize_fcs_ncs_multiple_obs_fixed_range_with_variables():

    evaluation_folder = append_directory(2)
    obs_numbers = [1,2,3,5,7]
    n = 32
    range_values = [3.0]
    model_versions = [5]
    for i,range_value in enumerate(range_values):
        figname = "paper_fcs_ncs_conditional_visualizations_range_" + str(range_value) + ".png"
        folder_name = "figures/one_to_seven"
        visualize_fcs_ncs_multiple_obs_fixed_range(evaluation_folder, range_value, obs_numbers, n, figname, folder_name,
                                                   model_versions[i])
        
visualize_fcs_ncs_multiple_obs_fixed_range_with_variables()