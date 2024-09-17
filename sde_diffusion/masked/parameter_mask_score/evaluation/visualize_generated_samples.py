import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
home_folder = append_directory(2)
sys.path.append(home_folder)
print("a")
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config
from block_mask_generation import *
print("b")

device = "cuda:0"
config = ncsnpp_config.get_config()
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((home_folder + "/trained_score_models/vpsde/model1_variance_.8_lengthscale_1_2_beta_min_max_01_20_random50_channel_mask.pth")))
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

#y is observed part of field, modified to incorporate the mask as channel
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, variance, lengthscale):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    #need mask to be same size as masked_xt
    mask = mask.repeat((reps,1,1,1))
    mask = lengthscale*mask
    masked_xt_and_mask = th.cat([masked_xt, mask], dim = 1)
    with th.no_grad():
        parameter = (torch.tensor([[variance, lengthscale]])).to(device)
        score_and_mask = score_model(masked_xt_and_mask, timestep)
    
    #first channel is score, second channel is mask
    score = score_and_mask[:,0:1,:,:]
    #reduce dimension of mask
    mask = mask[0:1,:,:,:]
    unmasked_p_mean = (1/th.sqrt(th.tensor(vpsde.alphas[t])))*(masked_xt + th.square(th.tensor(vpsde.sigmas[t]))*score)
    masked_p_mean = th.mul((1-mask), unmasked_p_mean) + torch.mul(mask, y)
    unmasked_p_variance = (th.square(th.tensor(vpsde.sigmas[t])))*th.ones_like(masked_xt)
    masked_p_variance = torch.mul((1-mask), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, variance, lengthscale):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt,
                                                                 mask, y, t, variance, lengthscale)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = torch.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask, y, n, num_samples, variance, lengthscale):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + torch.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, variance, lengthscale)

    return masked_xt



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
    
    im = grid[0].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-3, vmax=3)
    grid[0].set_title("Observed")
    grid[1].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-3, vmax=3,
                   alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[1].set_title("Partially Observed")
    grid[2].imshow(diffusion1.detach().cpu().numpy().reshape((n,n)), vmin=-3, vmax=3)
    grid[2].set_title("Generated")
    grid[3].imshow(diffusion2.detach().cpu().numpy().reshape((n,n)), vmin=-3, vmax=3)
    grid[3].set_title("Generated")
    grid[0].cax.colorbar(im)
    plt.savefig(figname)


sdevp = VPSDE(beta_min=0.1, beta_max=20, N=1000)
n = 32
#mask = torch.ones((1,1,n,n)).to(device)
#mask[:,:,int(n/4):int(3*n/4),int(n/4):int(3*n/4)] = 0
#mask = (th.from_numpy(((produce_checkered_mask(n))[0,:,:]).reshape((1,1,n,n))).to(device)).float()
num_samples = 1
minX = -10
maxX = 10
minY = -10
maxY = 10
variance = .8
lengthscale = 1
number_of_replicates = 1

for i in range(0,5):
    p = .5
    mask = (th.bernoulli(p*th.ones(1,1,n,n))).to(device)
    seed_value = int(np.random.randint(0, 100000))
    unmasked_y = (th.from_numpy(generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                                        lengthscale, number_of_replicates,
                                                        seed_value))).to(device)
    print(unmasked_y.min())
    y = ((torch.mul(mask, unmasked_y)).to(device)).float()
    num_samples = 2
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model,
                                                                    device, mask, y, n,
                                                                    num_samples, variance, lengthscale)

    figname = ("visualizations/models/model1/random50_variance_.8_lengthscale_1_observed_and_generated_samples_" + str(i) + ".png")
    visualize_observed_and_generated_samples(unmasked_y, mask, diffusion_samples[0,:,:,:],
                                            diffusion_samples[1,:,:,:], n, figname)