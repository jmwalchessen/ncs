import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
import subprocess

home_folder = append_directory(4)
br_sde_folder = (home_folder + "/brown_resnick/sde_diffusion/masked/unparameterized")
sys.path.append(br_sde_folder)
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config
from block_mask_generation import *

device = "cuda:0"
config = ncsnpp_config.get_config()
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(torch.load((br_sde_folder + "/trained_score_models/vpsde/model4_beta_min_max_01_20_1000_1.6_1.6_random050_masks.pth")))
score_model.eval()

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images


def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    return images


def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def inverse_boundary_process(images, logbrmin, logbrmax):

    images = (logbrmax - logbrmin)*((images/6)+.5)+logbrmin
    return images

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

def plot_unconditional_true_samples(br_samples, figname):

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
    
    range_value = 1.6
    smooth_value = 1.6
    seed_value = 42343
    br_samples = np.exp(br_samples)
    for i, ax in enumerate(grid):
        im = ax.imshow(br_samples[i,:,:], vmin = 0, vmax = 30)
        ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([0,20])
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
    
    diffusion_samples = (diffusion_samples.detach().cpu().numpy().reshape((4,n,n)))
    diffusion_samples = np.exp(diffusion_samples)
    #diffusion_samples = inverse_boundary_process(diffusion_samples, logbrmin, logbrmax)
    for i, ax in enumerate(grid):
        im = ax.imshow(diffusion_samples[i,:], vmin = 0, vmax = 30)
        ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))

    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.set_ticks([])
    cbar.set_ticks([0,20])
    fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)

def plot_unconditional_true_and_diffusion_samples(br_samples, vpsde, score_model, device, mask, y, n, figname):

    num_samples = 4
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device,
                                                                       mask, y, n, num_samples)
    fig = plt.figure(figsize=(20, 10))

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
    
    diffusion_samples = (diffusion_samples.detach().cpu().numpy().reshape((4,n,n)))
    #diffusion_samples = np.exp(diffusion_samples)
    

    for i, ax in enumerate(grid):
        if(i > 3):
            im = ax.imshow(diffusion_samples[(i-4),:,:], vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Unconditional Diffusion (log)", fontsize = 20)
        else:
            im = ax.imunconditional_lengthscale_1.6_variance_0.4_1000.npyshow(br_samples[i,:,:], vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Unconditional True (log)", fontsize = 20)

    cbar = grid.cbar_axes[0].colorbar(im)
    #cbar.set_ticks([])
    cbar.set_ticks([-2,0,2,4,6])
    #fig.text(0.5, 0.9, 'Unconditional Diffusion', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)


vpsde = VPSDE(beta_min=0.1, beta_max=20, N=1000)
p = 0
n = 32
mask = ((torch.bernoulli(p*torch.ones((1,1,n,n)))).to(device)).float()
number_of_replicates = 4
seed_value = int(np.random.randint(0, 10000, 1))
range_value = 1.6
smooth_value = 1.6
br_samples = log_transformation(np.load("brown_resnick_samples_1024_512.npy"))
print(np.quantile(np.exp(br_samples[0,:]), [.99]))
y = ((torch.from_numpy(br_samples))[0,:]).reshape((1,n,n))
y = y.to(device).float()
figname = "br_unconditional_true_and_diffusion_samples_1000.png"
br_samples = (br_samples[0:number_of_replicates,:]).reshape((number_of_replicates, n, n))
plot_unconditional_true_and_diffusion_samples(br_samples, vpsde, score_model, device, mask, y, n, figname)