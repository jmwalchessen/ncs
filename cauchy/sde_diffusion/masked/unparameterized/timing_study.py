import numpy as np
import torch as th
import matplotlib.pyplot as plt
import os
import sys
from append_directories import *
home_folder = append_directory(2)
sys.path.append(home_folder)
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config
import time

device = "cuda:0"
config = ncsnpp_config.get_config()
print("T", config.model.num_scales)
print("beta max", config.model.beta_max)
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((home_folder + "/trained_score_models/vpsde/model7_beta_min_max_01_25_250_random8090_block_masks.pth")))
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
    with th.no_grad():
        score = score_model(masked_xt, timestep)
    unmasked_p_mean = (1/th.sqrt(th.tensor(vpsde.alphas[t])))*(masked_xt + th.square(th.tensor(vpsde.sigmas[t]))*score)
    masked_p_mean = torch.mul((1-mask), unmasked_p_mean) + torch.mul(mask, y)
    unmasked_p_variance = (th.square(th.tensor(vpsde.sigmas[t])))*th.ones_like(masked_xt)
    masked_p_variance = torch.mul((1-mask), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, num_samples):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = torch.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask, y, n, num_samples):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + torch.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples)

    return masked_xt

def timing_per_diffusion_sampling(vpsde, score_model, device, mask, y, n, num_samples):

    start = time.time()
    posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask, y, n, num_samples)
    end = time.time()
    time_elapsed = end - start
    return time_elapsed

def timing_visualization(vpsde, score_model, device, mask, y, n, samples_list):

    timing_list = []
    for num_samples in samples_list:

        timing_list.append(timing_per_diffusion_sampling(vpsde, score_model, device, mask, y, n,
                                                         num_samples))
        
    fig, ax = plt.subplots(1)
    ax.plot(samples_list, timing_list)
    plt.show()

    
    

sdevp = VPSDE(beta_min=0.1, beta_max=25, N=250)
n = 32
p = .8
mask = ((th.bernoulli(p*th.ones(1,1,n,n)))).to(device)
minX = -10
maxX = 10
minY = -10
maxY = 10
lengthscale = 1.6
variance = .4
number_of_replicates = 250
seed_value = int(np.random.randint(0, 100000))
unmasked_y = (th.from_numpy(generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                                        lengthscale, number_of_replicates,
                                                        seed_value))).to(device)
y = ((torch.mul(mask, unmasked_y)).to(device)).float()
samples_list = [1,2,3]
timing_visualization(sdevp, score_model, device, mask, y, n, samples_list)

