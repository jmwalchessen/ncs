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
            im = ax.imshow(kriging_mean, cmap='viridis', alpha = mask.astype(float), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Kriging Mean")
        else:
            diffusion_mean = np.mean(diffusion_samples[0:convergence_numbers[i-1],:,:,:], axis = (0,1))
            im = ax.imshow(diffusion_mean, cmap = 'viridis', alpha = mask.astype(float), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion Mean, n = " + str(convergence_numbers[i-1]))

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
    print(m)
    empirical_covariance = np.cov(unobserved_images, rowvar = False)
    pixel_cov = empirical_covariance[:,missing_index]
    missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    pixel_cov_image = np.zeros((n**2))
    pixel_cov_image[missing_indices] = pixel_cov

    fig, ax = plt.subplots()
    ax.imshow(pixel_cov_image.reshape((n,n)), alpha = (1-mask).astype(float), vmin = -.5, vmax = .5)
    plt.show()


def visualize_standardized_conditional_covariance_map_per_pixel(images, mask, missing_index, n, nrep, figname):

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

def compare_standardized_conditional_covariance_map_per_pixel(diffusion_images, mask, missing_index, n, nrep,
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
    true_cond_images = sample_conditional_distribution((1-mask), minX, maxX, minY, maxY, n, variance, lengthscale, observations,
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
    fig.text(0.5, 0.95, 'Std. Cov', ha='center', va='center', fontsize = 10)
    plt.savefig(figname)
    plt.clf()
"""
def generated_and_true_covariance_per_pixel(diffusion_images, pixel_location, n, nrep, figname):

    true_empirical_covariance = np.cov()"""



minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
eval_folder = append_directory(2)
mask = np.load((eval_folder + "/diffusion_generation/data/model4/ref_image3/mask.npy"))
ref_image = np.load((eval_folder + "/diffusion_generation/data/model4/ref_image3/ref_image.npy"))
observations = ref_image.flatten()[mask.flatten() == 1]
m = observations.shape[0]
missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
observations = observations.reshape(m,1)
diffusion_samples = np.load((eval_folder + "/diffusion_generation/data/model4/ref_image3/diffusion/model4_random50_beta_min_max_01_20_1000.npy"))
convergence_numbers = [1000, 2000, 4000]
figname = "kriging_mean_and_diffusion_mean.png"
"""
visualize_kriging_mean_and_diffusion_mean((1-mask), minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                              diffusion_samples, convergence_numbers, figname)"""

nrep = 4000
m = missing_indices.shape[0]
print(m)

for i in range(0, 2):

    missing_index = int(missing_indices[np.random.randint(0, m, 1)])
    print(missing_index)
    figname = ("visualizations/model4/ref_image3/covariance/true_and_generated_empirical_covariance_map_" + str(missing_index) + ".png")
    compare_standardized_conditional_covariance_map_per_pixel(diffusion_samples, mask, missing_index, n, nrep,
                                                              ref_image, figname)
    
figname = ("visualizations/model4/ref_image3/mean/true_and_generated_empirical_diffusion_mean_and_kriging_mean_map.png")
visualize_kriging_mean_and_diffusion_mean((1-mask), minX, maxX, minY, maxY, n, variance, lengthscale, observations,
                                          diffusion_samples, convergence_numbers, figname)
    
