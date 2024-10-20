import numpy as np
import matplotlib.pyplot as plt
from generate_true_conditional_samples import *
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
from append_directories import *

def load_diffusion_images(model_name, image_name, file_name):

    eval_folder = append_directory(2)
    diffusion_images = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/diffusion/" + file_name + ".npy"))
    return diffusion_images

def load_mask(model_name, image_name):

    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" +
                                image_name + "/" + "mask.npy"))
    return mask

def produce_figure_name(model_name, image_name, fig_name):

    eval_folder = append_directory(2)
    figname = (eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name +
               "/conditional_mean_and_correlation/" + fig_name)
    return figname

def load_reference_image(model_name, image_name):

    eval_folder = append_directory(2)
    ref_image = np.load((eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy"))
    return ref_image

def load_observations(model_name, image_name, mask, n):

    eval_folder = append_directory(2)
    ref_image = np.load(eval_folder + "/diffusion_generation/data/" + model_name + "/" + image_name + "/ref_image.npy")
    observations = ref_image[(mask).astype(int) == 1]
    return observations

def visualize_empirical_mean_and_diffusion_mean_convergence(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                              diffusion_samples, convergence_numbers, figname):



    conditional_mean, conditional_variance = construct_empirical_mean_variance(mask, minX, maxX, minY, maxY, n,
                                                                             variance, lengthscale, observations)
    empirical_mean = np.zeros((n**2))
    flatten_mask = mask.reshape((n**2))
    empirical_mean[flatten_mask == 0] = conditional_mean
    empirical_mean[flatten_mask == 1] = observations
    empirical_mean = empirical_mean.reshape((n,n))
    
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
            im = ax.imshow(empirical_mean, cmap='viridis', vmin = -2, vmax = 2, alpha = (1-mask))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Kriging Mean")
        else:
            diffusion_mean = np.mean(diffusion_samples[0:convergence_numbers[i-1],:,:,:], axis = (0,1))
            im = ax.imshow(diffusion_mean, cmap = 'viridis', vmin = -2, vmax = 2, alpha = (1-mask))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion Mean, n = " + str(convergence_numbers[i-1]))

    cbar = ax.cax.colorbar(im)
        #cbar.set_label_text("Value")
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()


def visualize_empirical_mean_and_diffusion_mean(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                              diffusion_samples, figname):



    conditional_mean, conditional_variance = construct_empirical_mean_variance(mask, minX, maxX, minY, maxY, n,
                                                                             variance, lengthscale, observations)
    empirical_mean = np.zeros((n**2))
    flatten_mask = mask.reshape((n**2))
    empirical_mean[flatten_mask == 0] = conditional_mean
    empirical_mean[flatten_mask == 1] = observations
    empirical_mean = empirical_mean.reshape((n,n))
    n = diffusion_images.shape[0]
    
    fig = plt.figure(figsize=(10, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,2),
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
            im = ax.imshow(empirical_mean, cmap='viridis', vmin = -2, vmax = 2, alpha = (1-mask))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Kriging Mean")
        else:
            diffusion_mean = np.mean(diffusion_samples, axis = (0,1))
            im = ax.imshow(diffusion_mean, cmap = 'viridis', vmin = -2, vmax = 2, alpha = (1-mask))
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion Mean, n = " + str(n))

    cbar = ax.cax.colorbar(im)
        #cbar.set_label_text("Value")
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()

def unconditional_covariance_map_per_pixel(images, pixel_location, n, nrep, figname):

    empirical_covariance =np.cov(images.reshape((nrep, (n**2))), rowvar = False)
    fig, ax = plt.subplots()
    pixel_cov = empirical_covariance[:,pixel_location]
    ax.imshow(pixel_cov.reshape((n,n)), vmin = -.5, vmax = .5)
    plt.savefig(figname)

def visualize_conditional_covariance_map_per_pixel(images, mask, missing_index, n, nrep, figname):

    flatten_images = images.reshape((nrep,n**2))
    flatten_masks = mask.reshape(((n**2)))
    unobserved_images = flatten_images[:,flatten_masks == 0]
    m = unobserved_images.shape[1]
    empirical_covariance = np.cov(unobserved_images, rowvar = False)
    pixel_cov = empirical_covariance[:,missing_index]
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    pixel_cov_image = np.zeros((n**2))
    pixel_cov_image[missing_indices] = pixel_cov

    fig, ax = plt.subplots()
    ax.imshow(pixel_cov_image.reshape((n,n)), alpha = (1-mask).astype(float), vmin = -.5, vmax = .5)
    plt.show()


def visualize_correlation_map_per_pixel(images, mask, missing_index, n, nrep, figname):

    flatten_images = images.reshape((nrep,n**2))
    flatten_masks = mask.reshape(((n**2)))
    unobserved_images = flatten_images[:,flatten_masks == 0]
    m = unobserved_images.shape[1]
    empirical_covariance = np.corrcoef(unobserved_images, rowvar = False)
    pixel_cov = empirical_covariance[:,missing_index]
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    pixel_cov_image = np.zeros((n**2))
    pixel_cov_image[missing_indices] = pixel_cov

    fig, ax = plt.subplots()
    ax.imshow(pixel_cov_image.reshape((n,n)), alpha = (1-mask).astype(float), vmin = -1, vmax = 1)
    plt.show()

def compare_correlation_map_per_pixel(diffusion_images, mask, missing_index, n, nrep,
                                                              ref_image, figname):

    flatten_diffusion_images = diffusion_images.reshape((nrep,n**2))
    flatten_masks = mask.reshape(((n**2)))
    unobserved_diffusion_images = flatten_diffusion_images[:,flatten_masks == 0]
    m = unobserved_diffusion_images.shape[1]
    empirical_covariance = np.corrcoef(unobserved_diffusion_images, rowvar = False)
    diffusion_cov = empirical_covariance[:,missing_index]
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    diffusion_cov_image = np.zeros((n**2))
    diffusion_cov_image[missing_indices] = diffusion_cov

    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    observations = np.multiply(ref_image, mask)
    observations = np.delete(observations.reshape((n**2)), missing_indices)
    true_cond_images = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                                       nrep)
    unobserved_true_images = true_cond_images.reshape((nrep,m))
    empirical_true_covariance = np.corrcoef(unobserved_true_images, rowvar = False)
    true_cov = empirical_true_covariance[:,missing_index]
    true_cov_image = np.zeros((n**2))
    true_cov_image[missing_indices] = true_cov

    fig = plt.figure(figsize=(10, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,2),
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
            im = ax.imshow(true_cov_image.reshape((n,n)), alpha = (1-mask).astype(float), vmin = 0, vmax = 1)
            ax.set_title("True")
        if(i == 1):
            im = ax.imshow(diffusion_cov_image.reshape((n,n)), alpha = (1-mask).astype(float), vmin = 0, vmax = 1)
            ax.set_title("Diffusion")
    cbar = ax.cax.colorbar(im)
    fig.text(0.5, 0.95, 'Correlation', ha='center', va='center', fontsize = 10)
    plt.savefig(figname)
    plt.clf()

def correlations_maps_per_pixels(diffusion_images, mask, missing_indices, n, nrep, figname, model_name, image_name):

    for missing_index in missing_indices:
        current_figname = produce_figure_name(model_name, image_name, figname)
        current_figname = (current_figname + "_" + str(missing_index) + ".png")
        ref_image = load_reference_image(model_name, image_name)
        compare_correlation_map_per_pixel(diffusion_images, mask, missing_index, n, nrep,
                                                              ref_image, current_figname)


model_name = "model7"
image_name = "ref_image6"
file_name = "model7_beta_min_max_01_20_1000_0.4"
diffusion_images = load_diffusion_images(model_name, image_name, file_name)
minX = minY = -10
maxX = maxY = 10
n = 32
variance = 1.5
lengthscale = 3
figname = "conditional_correlation_map"
mask = load_mask(model_name, image_name)
observations = load_observations(model_name, image_name, mask, n)
missing_indices = [i for i in range(0, 1020, 5)]
nrep = 4000
correlations_maps_per_pixels(diffusion_images, mask, missing_indices, n, nrep, figname, model_name, image_name)

