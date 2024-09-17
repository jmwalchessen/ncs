import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from helper_functions import *
from matplotlib.patches import Rectangle




def visualize_sample(diffusion_sample, n):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(diffusion_sample.detach().cpu().numpy().reshape((n,n)), vmin = -2, vmax = 2)
    plt.show()

def visualize_observed_and_generated_samples(observed, mask, diffusion1, diffusion2, n, figname):

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    im = grid[0].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3)
    grid[0].set_title("Observed")
    grid[1].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3,
                   alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[1].set_title("Partially Observed")
    grid[2].imshow(diffusion1.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3)
    grid[2].set_title("Generated")
    grid[3].imshow(diffusion2.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=3)
    grid[3].set_title("Generated")
    grid[0].cax.colorbar(im)
    plt.savefig(figname)


def visualize_mcmc_approx_and_mean(ref_image_name, mask_name, mcmc_folder, missing_index, n, figname):

    mask = np.load(mask_name)
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    matrix_missing_index = (int(missing_indices[(missing_index-1)] % n), int(missing_indices[(missing_index-1)] / n))
    ref_image = np.load(ref_image_name)
    mcmc_samples = np.load((mcmc_folder + "_" + str(missing_index) + ".npy"))
    mcmc_samples = np.log(mcmc_samples)
    #print(ref_image[:,missing_indices[missing_index]])

    fig = plt.figure(figsize=(10, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,2),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    for i, ax in enumerate(grid):
        if(i == 0):
            mask[matrix_missing_index[1],matrix_missing_index[0]] = 1
            im = ax.imshow(ref_image.reshape((n,n)), alpha = (mask.astype(float)).reshape((n,n)),
                 vmin = -2, vmax = 4)
            rect = Rectangle(((matrix_missing_index[0]-.35), (matrix_missing_index[1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            ax.set_title("True")
        
        elif(i == 1):
            mcmc_image = ref_image.reshape((n**2))
            mcmc_image[missing_indices[(missing_index-1)]] = mcmc_samples[0]
            mask = mask.reshape((n,n))
            mask[matrix_missing_index[1],matrix_missing_index[0]] = 1
            ax.imshow(mcmc_image.reshape((n,n)), alpha = (mask.astype(float)).reshape((n,n)), vmin = -2, vmax = 4)
            rect = Rectangle(((matrix_missing_index[0]-.35), (matrix_missing_index[1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            ax.set_title("MCMC Approx")
        elif(i == 2):
            mcmc_image = ref_image.reshape((n**2))
            mcmc_image[missing_indices[(missing_index-1)]] = mcmc_samples[1]
            mask = mask.reshape((n,n))
            mask[matrix_missing_index[1],matrix_missing_index[0]] = 1
            ax.imshow(mcmc_image.reshape((n,n)), alpha = (mask.astype(float)).reshape((n,n)), vmin = -2, vmax = 4)
            rect = Rectangle(((matrix_missing_index[0]-.35), (matrix_missing_index[1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            ax.set_title("MCMC Approx")

        elif(i == 3):
            mcmc_image = ref_image.reshape((n**2))
            mcmc_mean = np.mean(mcmc_samples)
            mcmc_image[missing_indices[(missing_index-1)]] = mcmc_mean
            ax.imshow(mcmc_image.reshape((n,n)), alpha = (mask.astype(float)).reshape((n,n)), vmin = -2, vmax = 4)
            rect = Rectangle(((matrix_missing_index[0]-.35), (matrix_missing_index[1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            ax.set_title("MCMC Conditional Mean")
    
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)

def produce_mask_missing_indices(mask, n):

    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    m = missing_indices.shape[0]
    matrix_missing_indices = np.zeros((m,2))

    for missing_index in range(0, m):
        matrix_missing_indices[missing_index,:] = (int(missing_indices[missing_index] % n), int(missing_indices[missing_index] / n))

    return missing_indices, matrix_missing_indices



def visualize_mcmc_approx_and_mean_pixels(ref_image_name, mask_name, mcmc_file_name, n, figname):

    mask = np.load(mask_name)
    missing_indices, matrix_missing_indices = produce_mask_missing_indices(mask, n)
    m = missing_indices.shape[0]
    ref_image = np.load(ref_image_name)
    mcmc_samples = np.load(mcmc_file_name)
    mcmc_samples = np.log(mcmc_samples)
    #print(ref_image[:,missing_indices[missing_index]])

    fig = plt.figure(figsize=(10, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,2),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    for i, ax in enumerate(grid):
        if(i == 0):
            im = ax.imshow(ref_image.reshape((n,n)),
                 vmin = -2, vmax = 4)
            for i in range(0, m):
                rect = Rectangle(((matrix_missing_indices[i,0]-.35), (matrix_missing_indices[i,1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("True")
        
        elif(i == 1):
            mcmc_image = ref_image.reshape((n**2))
            mcmc_image[missing_indices] = mcmc_samples[0,:]
            ax.imshow(mcmc_image.reshape((n,n)), vmin = -2, vmax = 4)
            for i in range(0, m):
                rect = Rectangle(((matrix_missing_indices[i,0]-.35), (matrix_missing_indices[i,1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.add_patch(rect)
            ax.set_title("MCMC Approx")
        elif(i == 2):
            mcmc_image = ref_image.reshape((n**2))
            mcmc_image[missing_indices] = mcmc_samples[1,:]
            mask = mask.reshape((n,n))
            ax.imshow(mcmc_image.reshape((n,n)), vmin = -2, vmax = 4)
            for i in range(0, m):
                rect = Rectangle(((matrix_missing_indices[i,0]-.35), (matrix_missing_indices[i,1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.add_patch(rect)
            ax.set_title("MCMC Approx")

        elif(i == 3):
            mcmc_image = ref_image.reshape((n**2))
            mcmc_mean = np.mean(mcmc_samples, axis = 1)
            mcmc_image[missing_indices] = mcmc_mean
            ax.imshow(mcmc_image.reshape((n,n)), vmin = -2, vmax = 4)
            for i in range(0, m):
                rect = Rectangle(((matrix_missing_indices[i,0]-.35), (matrix_missing_indices[i,1]-.35)), width=1, height=1,
                             facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            ax.set_title("MCMC Conditional Mean")
    
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)

def visualize_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax):


    seed_value = int(np.random.randint(0, 100000))
    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 4),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    number_of_replicates = 8
    n = 32
    if(process_type == "schlather"):

        ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))

    elif(process_type == "brown"):
        ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

   
    for i, ax in enumerate(grid):

        im = ax.imshow(ref_img[i,:,:,:].reshape((n,n)), vmin = vmin, vmax = vmax)

    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)

def generate_visualization(process_type, range_value, smooth_value, p, model_name, i):

        seed_value = int(np.random.randint(0, 100000))
        number_of_replicates = 1
        n = 32
        device = "cuda:0"

        if(process_type == "schlather"):
            ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))
        else:
            ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

        unmasked_y = (th.from_numpy(ref_img)).to(device)
        mask = (th.bernoulli(p*th.ones((1,1,n,n)))).to(device)
        y = ((th.mul(mask, unmasked_y)).to(device)).float()
        num_samples = 2
        sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)
        score_model = load_score_model(process_type, model_name, "eval")
        diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model,
                                                                    device, mask, y, n,
                                                                    num_samples)

        figname = ("visualizations/" + process_type + "/models/" + model_name + "/random" + str(p) + "_range_" + str(range_value) + 
        "_smooth_" + str(smooth_value) + "_observed_and_generated_samples_" + str(i) + ".png")
        visualize_observed_and_generated_samples(unmasked_y, mask, diffusion_samples[0,:,:,:],
                                            diffusion_samples[1,:,:,:], n, figname)

def generate_multiple_visualizations(process_type, range_value, smooth_value, p, model_name, multiples_beginning, multiples_end):

    for i in range(multiples_beginning, multiples_end):
        generate_visualization(process_type, range_value, smooth_value, p, model_name, i)


"""
process_type = "schlather"
model_name = "model4_beta_min_max_01_20_range_2.2_smooth_1.9_random025_log_parameterized_mask.pth"
mode = "eval"
score_model = load_score_model(process_type, model_name, mode)
beta_min = .1
beta_max = 20
N = 1000
sdevp = load_sde(beta_min = beta_min, beta_max = beta_max, N = N)
range_value = 2.2
smooth_value = 1.9
p = 0
model_name = "model4"
multiples_beginning = 0
multiples_end = 50
generate_multiple_visualizations(process_type, range_value, smooth_value, p, model_name, multiples_beginning, multiples_end)"""

"""
range_value = .4
smooth_value = 1.6
process_type = "smith"
vmax = 6
vmin = -2
figname = "visualizations/smith/true/true_range_" + str(range_value) + "_smooth_" + str(smooth_value) + ".png" 
visualize_observed_samples(range_value, smooth_value, process_type, figname, vmin, vmax)

for missing_index in range(0,120, 4):
    ref_image_name = "diffusion_generation/data/model1/ref_image1/ref_image.npy"
    mask_name = "diffusion_generation/data/model1/ref_image1/mask.npy"
    mcmc_folder = "diffusion_generation/data/model1/ref_image1/mcmc_interpolation/mcmc_interpolation_missing_index"
    n = 32
    figname = ("visualizations/models/model1/ref_image1/mcmc_interpolation_missing_index_" +
           str(missing_index) + ".png")
    visualize_mcmc_approx_and_mean(ref_image_name, mask_name, mcmc_folder, missing_index, n, figname)"""