import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *

home_folder = append_directory(4)
gp_sde_folder = (home_folder + "/sde_diffusion/masked/unparameterized")
sys.path.append(gp_sde_folder)
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config
from block_mask_generation import *

device = "cuda:0"
config = ncsnpp_config.get_config()
config.model.beta_max = .2
config.model.num_scales = 1000
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(torch.load((gp_sde_folder + "/trained_score_models/vpsde/model5_beta_min_max_01_20_random050_masks.pth")))
score_model.eval()

def index_to_matrix_index(index,n):

    return (int(index / n), int(index % n))

def construct_norm_matrix(minX, maxX, minY, maxY, n):
    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    X_matrix = (np.repeat(X, n**2, axis = 0)).reshape((n**2, n**2))
    Y_matrix = (np.repeat(Y, n**2, axis = 0)).reshape((n**2, n**2))
    longitude_squared = np.square(np.subtract(X_matrix, np.transpose(X_matrix)))
    latitude_squared = np.square(np.subtract(Y_matrix, np.transpose(Y_matrix)))
    norm_matrix = np.sqrt(np.add(longitude_squared, latitude_squared))
    return norm_matrix

#mask needs to be type numpy of 0s/1s, returns norm matrix with rows and columns associated with
#masked locations in original nxn matrix masked out in n**2xn**2 matrix (now (n**2-m)x(n**2-m))
def construct_masked_norm_matrix(mask, minX, maxX, minY, maxY, n):
    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    #X is a matrix of nxn which is latitudes of all nxn obs, same for Y
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    missing_indices = (np.argwhere(mask.reshape((n**2))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    X = np.delete(X, missing_indices, axis = 0)
    Y = np.delete(Y, missing_indices, axis = 0)
    #reshape X and Y to (n**2-mx1) vectors of latitude and longitude respectively
    #repeat longitudes and latitudes such that you get n**2 x n**2 matrix
    X_matrix = (np.repeat(X, (n**2-m), axis = 0)).reshape((n**2-m), (n**2-m))
    Y_matrix = (np.repeat(Y, (n**2-m), axis = 0)).reshape((n**2-m), (n**2-m))

    longitude_squared = np.square(np.subtract(X_matrix, np.transpose(X_matrix)))
    latitude_squared = np.square(np.subtract(Y_matrix, np.transpose(Y_matrix)))
    masked_norm_matrix = np.sqrt(np.add(longitude_squared, latitude_squared))
    return masked_norm_matrix

def construct_masked_norm_matrix1(mask, minX, maxX, minY, maxY, n):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    missing_indices = (np.argwhere(mask.reshape((n**2))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    masked_norm_matrix = np.delete(norm_matrix, missing_indices, axis = 0)
    masked_norm_matrix = np.delete(masked_norm_matrix, missing_indices, axis = 1)
    return masked_norm_matrix

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return exp_kernel

def construct_masked_exp_kernel(mask, minX, maxX, minY, maxY, n, variance, lengthscale):

    masked_norm_matrix = construct_masked_norm_matrix(mask, minX, maxX, minY, maxY, n)
    masked_exp_kernel = variance*np.exp((-1/lengthscale)*masked_norm_matrix)
    return masked_exp_kernel


def construct_masked_norm_vector(mask, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    missing_indices = (np.argwhere(mask.reshape((n**2))))
    missing_indices = missing_indices.reshape((missing_indices.shape[0]))
    m = missing_indices.shape[0]
    missing_xlocations = X[missing_indices]
    missing_ylocations = Y[missing_indices]
    missing_locations = np.zeros((m,2))
    missing_locations[:,0] = missing_xlocations.reshape((m))
    missing_locations[:,1] = missing_ylocations.reshape((m))
    X = np.delete(X, missing_indices)
    Y = np.delete(Y, missing_indices)
    masked_norm_vector = np.zeros(((n**2-m), m))

    for i in range(0, m):
        norm_vector = (np.sqrt(np.add(np.square(X-missing_locations[i,0]),
                                      np.square(Y-missing_locations[i,1]))))
        masked_norm_vector[:,i] = norm_vector.reshape((n**2-m))

    return masked_norm_vector

def construct_masked_norm_vector1(mask, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    missing_indices = (np.argwhere(mask.reshape((n**2))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    missing_xlocations = X[missing_indices]
    missing_ylocations = Y[missing_indices]
    missing_locations = np.zeros((m,2))
    missing_locations[:,0] = missing_xlocations.reshape((m))
    missing_locations[:,1] = missing_ylocations.reshape((m))
    masked_norm_vector = np.zeros(((n**2), m))
    for i in range(0, m):
        norm_vector = (np.sqrt(np.add(np.square(X-missing_locations[i,0]),
                                      np.square(Y-missing_locations[i,1]))))
        masked_norm_vector[:,i] = norm_vector.reshape((n**2))
    
    masked_norm_vector = np.delete(masked_norm_vector, missing_indices, axis = 0)
    return masked_norm_vector


def construct_masked_exp_kernel_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale):

    masked_norm_vector = construct_masked_norm_vector(mask, minX, maxX, minY, maxY, n)
    masked_exp_vector = variance*np.exp((-1/lengthscale)*masked_norm_vector)
    return masked_exp_vector

def construct_kriging_mean_variance(mask, minX, maxX, minY, maxY, n, variance, lengthscale, y):

    #(n**2-m)x(n**2-m) matrix
    masked_exp_kernel = construct_masked_exp_kernel(mask, minX, maxX, minY, maxY, n, variance,
                                                    lengthscale)
    
    #(n**2-m)xm vector
    masked_exp_kernel_vector = construct_masked_exp_kernel_vector(mask, minX, maxX, minY, maxY,
                                                                  n, variance, lengthscale)
    #mx(n**2-m) vector
    kriging_matrix = np.matmul(np.transpose(masked_exp_kernel_vector),
                               np.linalg.inv(masked_exp_kernel))
    #construct a mx1 vector, m is the number of fixed locations vector
    conditional_mean =  np.matmul(kriging_matrix, y)
    unmask = (1-mask)
    variance_matrix = construct_masked_exp_kernel(unmask, minX, maxX, minY, maxY, n, variance, lengthscale)
    cov_part = np.matmul(kriging_matrix, masked_exp_kernel_vector)
    #conditional_variance = variance - cov_part
    #cvupper = np.triu(conditional_variance)
    #cvreflected = cvupper.T + cvupper
    #np.fill_diagonal(cvreflected, np.diag(cvupper))
    #conditional_variance = cvreflected
    conditional_variance = variance_matrix - cov_part
    return conditional_mean, conditional_variance

def sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale, y,
                                    number_of_replicates):
    
    conditional_mean, conditional_variance = construct_kriging_mean_variance(mask, minX, maxX, minY,
                                                                             maxY, n, variance,
                                                                             lengthscale, y)
    if(np.all(np.linalg.eigvals(conditional_variance) >= 0)):
        conditional_samples = np.random.multivariate_normal(conditional_mean, conditional_variance,
                                                        number_of_replicates)
        return conditional_samples
    else:
        return None

def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    C = np.linalg.cholesky(kernel)
    y_matrix = (np.flip(np.matmul(np.transpose(C),
                                  np.transpose(z_matrix))))
    
    gp_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, y_matrix.shape[1]):
        gp_matrix[i,:,:,:] = y_matrix[:,i].reshape((1,n,n))
    return y_matrix, gp_matrix

#y is observed part of field
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t):

    num_samples = masked_xt.shape[0]
    timestep = ((torch.tensor([t])).repeat(num_samples)).to(device)
    with torch.no_grad():
        score = score_model(masked_xt, timestep)
    unmasked_p_mean = (1/torch.sqrt(torch.tensor(vpsde.alphas[t])))*(masked_xt + torch.square(torch.tensor(vpsde.sigmas[t]))*score)
    masked_p_mean = torch.mul((1-mask), unmasked_p_mean) + torch.mul(mask, y)
    unmasked_p_variance = (torch.square(torch.tensor(vpsde.sigmas[t])))*torch.ones_like(masked_xt)
    masked_p_variance = torch.mul((1-mask), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, num_samples):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t)
    std = torch.exp(0.5 * torch.log(p_variance))
    noise = torch.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = torch.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample

def true_conditional_image_sampling(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                    observed_vector, observed_matrix,
                                    number_of_replicates, missing_indices):
    
    cond_unobserved_samples = sample_conditional_distribution(mask, minX, maxX, minY,
                                                              maxY, n, variance, lengthscale,
                                                              observed_vector, number_of_replicates)
    m = missing_indices.shape[0]
    conditional_samples = np.zeros((number_of_replicates, n, n))
    for i in range(0,m):
        matrix_index = index_to_matrix_index(missing_indices[i], n)
        conditional_samples[:,matrix_index[0],matrix_index[1]] = cond_unobserved_samples[:,i]

    observed_matrix = np.repeat(observed_matrix.reshape((1,n,n)), repeats = number_of_replicates, axis = 0)
    conditional_samples = np.add(observed_matrix, conditional_samples)
    #conditional_samples = observed_matrix
    return conditional_samples

    


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask, y, n, num_samples):

    unmasked_xT = torch.randn((num_samples, 1, n, n)).to(device)
    masked_xT = torch.mul((1-mask), unmasked_xT) + torch.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples)

    return masked_xt

def plot_conditional_true_samples(mask, observed_vector, observed_matrix,
                                  missing_indices, n, figname):
    
    fig = plt.figure(figsize=(20, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,3),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    minX = minY = -10
    maxX = maxY = 10
    variance = .4
    lengthscale = 1.6
    number_of_replicates = 3
    seed_value = 23423
    n = 32
    gaussian_samples = true_conditional_image_sampling((1-mask).detach().cpu().numpy(), minX,
                                                       maxX, minY, maxY, n,
                                                       variance, lengthscale, observed_vector,
                                                       observed_matrix, number_of_replicates,
                                                       missing_indices)
    gaussian_samples = gaussian_samples.reshape((3,n,n))
    for i, ax in enumerate(grid):
        if(i < 3):
            im = ax.imshow(gaussian_samples[i,:,:], vmin = -2, vmax = 2)
            #ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            #ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,-1,0,1,2])
    #fig.text(0.5, 0.9, 'Unconditional True', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)
    

def plot_conditional_true_and_difussion_samples(vpsde, score_model, device, mask, observed_vector,
                                                observed_matrix, ref_image, missing_indices, n,
                                                figname):

    fig = plt.figure(figsize=(20, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,4),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    minX = minY = -10
    maxX = maxY = 10
    variance = .4
    lengthscale = 1.6
    number_of_replicates = 3
    seed_value = 23423
    n = 32
    gaussian_samples = true_conditional_image_sampling((1-mask).detach().cpu().numpy(), minX,
                                                       maxX, minY, maxY, n,
                                                       variance, lengthscale, observed_vector,
                                                       observed_matrix, number_of_replicates,
                                                       missing_indices)
    gaussian_samples = gaussian_samples.reshape((3,n,n))
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device,
                                                                       mask, ref_image, n,
                                                                       number_of_replicates)
    diffusion_samples = diffusion_samples.detach().cpu().numpy().reshape((3,n,n))
    for i, ax in enumerate(grid):
        if(i == 0):
            im = ax.imshow(ref_image.detach().cpu().numpy().reshape((n,n)),
                           alpha = mask.detach().cpu().numpy().reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Partially Observed")
        elif(i < 4):
            im = ax.imshow(gaussian_samples[(i-1),:,:], vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("True")
        
        elif(i==4):
            im = ax.imshow(ref_image.detach().cpu().numpy().reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Fully Observed")
        else:
            im = ax.imshow(diffusion_samples[(i-5),:,:], vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion")

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,-1,0,1,2])
    #fig.text(0.5, 0.9, 'Unconditional True', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)


vpsde = VPSDE(beta_min=0.1, beta_max=20, N=1000)
p = .5
n = 32
mask = ((torch.bernoulli(p*torch.ones((1,1,n,n)))).to(device)).float()
minX = minY = -10
maxX = maxY = 10
variance = .4
lengthscale = 1.6
number_of_replicates = 1
seed_value = 23423
y, gp_matrix = (generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                               lengthscale, number_of_replicates, seed_value))
observed_vector = y.reshape((n**2))
ref_image = (torch.from_numpy(gp_matrix).to(device)).float()
observed_matrix = ((torch.mul(mask, torch.from_numpy(gp_matrix).to(device)))).detach().cpu().numpy()
missing_indices = np.squeeze(np.argwhere((1-mask).detach().cpu().numpy().reshape((n**2,))))
observed_vector = np.delete(observed_vector, missing_indices)
figname = "gp_conditional_random50.png"
number_of_replicates = 3
#plot_conditional_true_and_difussion_samples(vpsde, score_model, device, mask, observed_vector,
                                            #observed_matrix, ref_image, missing_indices, n, figname)
#tc = true_conditional_image_sampling(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
#                                    observed_vector, observed_matrix,
#                                    number_of_replicates, missing_indices)
plot_conditional_true_samples(mask, observed_vector, observed_matrix,
                              missing_indices, n, figname)

plot_conditional_true_and_difussion_samples(vpsde, score_model, device, mask, observed_vector,
                                                observed_matrix, ref_image, missing_indices, n,
                                                figname)
