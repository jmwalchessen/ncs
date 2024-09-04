import numpy as np
import matplotlib.pyplot as plt
from generate_true_conditional_samples import *
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid

def load_diffusion_images(model_name, image_name, file_name):

    eval_folder = append_directory(2)
    diffusion_images = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/diffusion/" + file_name + ".npy"))
    return diffusion_images

def visualize_kriging_mean_and_diffusion_mean(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                              diffusion_samples, convergence_numbers, figname):



    conditional_mean, conditional_variance = construct_kriging_mean_variance(mask, minX, maxX, minY, maxY, n,
                                                                             variance, lengthscale, observations)
    kriging_mean = np.zeros(((n**2),1))
    flatten_mask = mask.reshape((n**2,1))
    kriging_mean[flatten_mask[:,0] == 1,:] = conditional_mean
    kriging_mean = kriging_mean.reshape((n,n))
    
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
            im = ax.imshow(kriging_mean, cmap='viridis', alpha = mask.astype(float))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Kriging Mean")
        else:
            diffusion_mean = np.mean(diffusion_samples[0:convergence_numbers[i-1],:,:,:], axis = (0,1))
            im = ax.imshow(diffusion_mean, cmap = 'viridis', alpha = mask.astype(float))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion Mean, n = " + str(convergence_numbers[i-1]))

    cbar = ax.cax.colorbar(im)
        #cbar.set_label_text("Value")
    plt.tight_layout()
    plt.savefig(figname)

minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
eval_folder = append_directory(2)
mask = np.load((eval_folder + "/diffusion_generation/data/model6/ref_image1/mask.npy"))
ref_image = np.load((eval_folder + "/diffusion_generation/data/model6/ref_image1/ref_image.npy"))
observations = ref_image.flatten()[mask.flatten() == 1]
m = observations.shape[0]
observations = observations.reshape(m,1)
diffusion_samples = np.load((eval_folder + "/diffusion_generation/data/model6/ref_image1/diffusion/model6_random025_beta_min_max_01_20_1000.npy"))
convergence_numbers = [1000, 2000, 4000]
figname = "kriging_mean_and_diffusion_mean.png"
visualize_kriging_mean_and_diffusion_mean((1-mask), minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                              diffusion_samples, convergence_numbers, figname)