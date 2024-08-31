import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from brown_resnick_data_generation import *

home_folder = append_directory(2)
sys.path.append(home_folder)
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config
from block_mask_generation import *

device = "cuda:0"
config = ncsnpp_config.get_config()
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((home_folder + "/trained_score_models/vpsde/model1_beta_min_max_01_20_random50_log_channel_mask.pth")))
score_model.eval()

#y is observed part of field, modified to incorporate the mask as channel
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    #need mask to be same size as masked_xt
    mask = mask.repeat((reps,1,1,1))
    masked_xt_and_mask = th.cat([masked_xt, mask], dim = 1)
    with th.no_grad():
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
    
    im = grid[0].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=2)
    grid[0].set_title("Observed")
    grid[1].imshow(observed.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=2,
                   alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[1].set_title("Partially Observed")
    grid[2].imshow(diffusion1.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=2)
    grid[2].set_title("Generated")
    grid[3].imshow(diffusion2.detach().cpu().numpy().reshape((n,n)), vmin=-2, vmax=2)
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
range_value = 1.6
smooth_value = 1.6
number_of_replicates = 1


for i in range(0,10):
    p = .5
    mask = (th.bernoulli(p*th.ones(1,1,n,n))).to(device)
    seed_value = int(np.random.randint(0, 100000))
    brsamples = (generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n)).reshape((1,1,n,n))
    unmasked_y = (th.from_numpy(brsamples)).to(device)
    print(unmasked_y.min())
    y = ((torch.mul(mask, unmasked_y)).to(device)).float()
    num_samples = 2
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model,
                                                                    device, mask, y, n,
                                                                    num_samples)

    figname = ("visualizations/models/model1/random50_range_1.6_smooth_1.6_observed_and_generated_samples_" + str(i) + ".png")
    visualize_observed_and_generated_samples(unmasked_y, mask, diffusion_samples[0,:,:,:],
                                            diffusion_samples[1,:,:,:], n, figname)