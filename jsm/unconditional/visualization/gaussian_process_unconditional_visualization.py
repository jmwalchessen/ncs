import numpy as np
import torch
import matplotlib.pyplot as plt
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
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(torch.load((gp_sde_folder + "/trained_score_models/vpsde/model6_beta_min_max_01_25_250_random0_block_masks.pth")))
score_model.eval()

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

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel)

def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_replicates,
                              seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    C = np.linalg.cholesky(kernel)
    y_matrix = np.matmul(np.transpose(C),
                                  np.transpose(z_matrix))
    
    gp_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, y_matrix.shape[1]):
        gp_matrix[i,:,:,:] = y_matrix[:,i].reshape((1,n,n))
    return gp_matrix

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


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask, y, n, num_samples):

    unmasked_xT = torch.randn((num_samples, 1, n, n)).to(device)
    masked_xT = torch.mul((1-mask), unmasked_xT) + torch.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples)

    return masked_xt

def plot_unconditional_true_samples(figname):

    fig = plt.figure(figsize=(20, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,4),
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
    number_of_replicates = 4
    seed_value = 23423
    n = 32
    gaussian_samples = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                 variance, lengthscale,
                                                 number_of_replicates, seed_value)
    gaussian_samples = gaussian_samples.reshape((4,n,n))
    for i, ax in enumerate(grid):
        im = ax.imshow(gaussian_samples[i,:,:], vmin = -2, vmax = 2)
        ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,-1,0,1,2])
    fig.text(0.5, 0.9, 'Unconditional True', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)



def plot_unconditional_diffusion_samples(vpsde, score_model, device, mask, y, n, figname):

    num_samples = 4
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device,
                                                                       mask, y, n, num_samples)
    fig = plt.figure(figsize=(20, 7.2))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,4),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )
    
    diffusion_samples = diffusion_samples.detach().cpu().numpy().reshape((4,n,n))
    for i, ax in enumerate(grid):
        im = ax.imshow(diffusion_samples[i,:,:], vmin = -2, vmax = 2)
        ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,-1,0,1,2])
    fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)


vpsde = VPSDE(beta_min=0.1, beta_max=25, N=250)
p = 0
n = 32
mask = ((torch.bernoulli(p*torch.ones((1,1,n,n)))).to(device)).float()
minX = minY = -10
maxX = maxY = 10
variance = .4
lengthscale = 1.6
number_of_replicates = 1
seed_value = 43234

y = (torch.from_numpy(generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                               lengthscale, number_of_replicates, seed_value)))
y = y.to(device).float()
plot_unconditional_diffusion_samples(vpsde, score_model, device, mask, y, n, "diffusion.png")  


