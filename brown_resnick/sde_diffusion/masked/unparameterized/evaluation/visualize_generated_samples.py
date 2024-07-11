import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
home_folder = append_directory(2)
sys.path.append(home_folder)
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config

device = "cuda:0"
config = ncsnpp_config.get_config()
print("T", config.model.num_scales)
print("beta max", config.model.beta_max)
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((home_folder + "/trained_score_models/vpsde/model6_beta_min_max_01_20_1000_1.6_1.6_random3050_bounded_masks.pth")))
score_model.eval()


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

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def visualize_sample(diffusion_sample, n):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(diffusion_sample.detach().cpu().numpy().reshape((n,n)), vmin = 0, vmax = 10)
    plt.show()

def visualize_observed_and_generated_samples(observed, mask, diffusion1, diffusion2, n, figname):

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    observed = observed.detach().cpu().numpy().reshape((n,n))
    max_value = np.quantile(observed, [.99])[0]
    min_value = np.quantile(observed, [.01])[0]
    im = grid[0].imshow(observed, vmin=min_value, vmax=max_value)
    grid[0].set_title("Observed")
    grid[1].imshow(observed, vmin=min_value,
                   vmax=max_value,
                   alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[1].set_title("Partially Observed")
    diffusion1 = diffusion1.detach().cpu().numpy().reshape((n,n))
    max_value = np.quantile(diffusion1, [.99])[0]
    min_value = np.quantile(diffusion1, [.01])[0]
    grid[2].imshow(diffusion1, vmin=min_value,
                   vmax=max_value)
    grid[2].set_title("Generated")
    diffusion2 = diffusion2.detach().cpu().numpy().reshape((n,n))
    max_value = np.quantile(diffusion2, [.99])[0]
    min_value = np.quantile(diffusion2, [.01])[0]
    grid[3].imshow(diffusion2,
                   vmin=min_value, vmax=max_value)
    grid[3].set_title("Generated")
    grid[0].cax.colorbar(im)
    plt.savefig(figname)

def visualize_observed_and_generated_sample(observed, mask, diffusion, n, figname):

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 2),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    
    observed = observed.detach().cpu().numpy()
    diffusion = diffusion.detach().cpu().numpy()
    max_value = np.quantile(observed, [.995])[0]
    min_value = np.quantile(observed, [.005])[0]
    observed = observed.reshape((n,n))
    diffusion = diffusion.reshape((n,n))
    im = grid[0].imshow(observed, vmin=min_value, vmax=max_value)
    grid[0].set_title("Observed")
    grid[1].imshow(observed, vmin=min_value,
                   vmax=max_value,
                   alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[1].set_title("Partially Observed")
    grid[2].imshow(diffusion, vmin=min_value,
                   vmax=max_value)
    grid[2].set_title("Generated")
    grid[3].imshow(diffusion, vmin=min_value, vmax=max_value, alpha = mask.detach().cpu().numpy().reshape((n,n)))
    grid[3].set_title("Generated Partially Observed")
    grid[0].cax.colorbar(im)
    plt.savefig(figname)


sdevp = VPSDE(beta_min=0.1, beta_max=20, N=1000)
n = 32
#mask = torch.ones((1,1,n,n)).to(device)
#mask[:,:,int(n/4):int(3*n/4),int(n/4):int(3*n/4)] = 0
p = .4
mask = ((th.bernoulli(p*th.ones(1,1,n,n)))).to(device)
print(torch.sum(mask))
num_samples = 1
minX = -10
maxX = 10
minY = -10
maxY = 10
range_value = 1.6
smooth_value = 1.6
number_of_replicates = 250
seed_value = int(np.random.randint(0, 100000))
from brown_resnick_data_generation import *
#unmasked_ys = generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n)
unmasked_ys = np.load("brown_resnick_samples_250.npy")
unmasked_ys = log_and_boundary_process(unmasked_ys)
unmasked_ys = (unmasked_ys.reshape(number_of_replicates,1,n,n))

for i in range(10,20):
    print(i)
    n = 32
    unmasked_y = torch.from_numpy(ndary_process(uunmasked_ys[i,:,:,:])).to(device).float()
    y = ((torch.mul(mask, unmasked_y)).to(device)).float()
    num_samples = 1
    n = 32
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model,
                                                                    device, mask, unmasked_y, n,
                                                                    num_samples)

    figname = ("visualizations/models/model6/random40_observed_and_generated_samples_" + str(i) + ".png")
    visualize_observed_and_generated_sample(unmasked_y, mask, diffusion_samples[0,:,:,:],
                                            n, figname)
    
    figname = ("visualizations/models/model6/random40_observed_and_generated_exp_samples_" + str(i) + ".png")
    visualize_observed_and_generated_sample(torch.exp(unmasked_y), mask,
                                             torch.exp(diffusion_samples[0,:,:,:]), n, figname)