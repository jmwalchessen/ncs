import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
home_folder = append_directory(4)
br_sde_folder = (home_folder + "/brown_resnick/sde_diffusion/masked/unparameterized")
sys.path.append(br_sde_folder)
from models import ncsnpp
from sde_lib import *
from configs.vp import ncsnpp_config
from block_mask_generation import *
device = "cuda:0"
config = ncsnpp_config.get_config()
config.model.beta_max = 20.
config.model.num_scales = 1000
#if trained parallelized, need to be evaluated that way too
score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(torch.load((br_sde_folder + "/trained_score_models/vpsde/model11_beta_min_max_01_20_1000_1.6_1.6_random050_logglobalbound_200000_40_masks.pth")))
score_model.eval()

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))
    return images

def log10_transformation(images):

    images = np.log10(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def global_boundary_process(images, minvalue, maxvalue):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - .5
    log01cs = 6*log01c
    return log01cs

def log_and_normalize(images):

    images = np.log(images)
    images = (images - np.mean(images))/np.std(images)
    return images

#.99999 quantile, .00001 quantile, .7 quantile of the transformed by .99999 and .00001 quantile
def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs

#.99999 quantile, .00001 quantile, .7 quantile of the transformed by .99999 and .00001 quantile
def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs

#.99999 quantile, .00001 quantile, .75? quantile of the transformed by .99999 and .00001 quantile
def global_quantile_boundary_inverse(images, minvalue, maxvalue, quantvalue01):

    log01 = (images/6)+quantvalue01
    logimages = (maxvalue-minvalue)*log01+minvalue
    return logimages



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

    

def plot_conditional_difussion_samples(vpsde, score_model, device, mask,
                                       ref_image,  n,
                                       figname):

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
    
    minX = minY = -10
    maxX = maxY = 10
    variance = .4
    lengthscale = 1.6
    number_of_replicates = 6
    seed_value = 23423
    n = 32
    diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device,
                                                                       mask, ref_image, n,
                                                                       number_of_replicates)
    diffusion_samples = diffusion_samples.detach().cpu().numpy().reshape((number_of_replicates,n,n))
    for i, ax in enumerate(grid):
        if(i == 0):
            im = ax.imshow(ref_image.detach().cpu().numpy().reshape((n,n)),
                           alpha = mask.detach().cpu().numpy().reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Partially Observed")
        elif(i < 4):
            im = ax.imshow(diffusion_samples[(i-1),:,:], vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion")
        
        elif(i==4):
            im = ax.imshow(ref_image.detach().cpu().numpy().reshape((n,n)), vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Fully Observed")
        else:
            im = ax.imshow(diffusion_samples[(i-2),:,:], vmin = -2, vmax = 2)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion")

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,-1,0,1,2])
    #fig.text(0.5, 0.9, 'Unconditional True', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)

def plot_fixed_conditional_difussion_samples(diffusion_samples, indices, ref_image, mask, n, figname,
                                             trainmaxmin):

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
    
    ref_image = global_quantile_boundary_inverse(ref_image, trainmaxmin[0],
                                                 trainmaxmin[1], trainmaxmin[2])
    diffusion_samples = global_quantile_boundary_inverse(diffusion_samples, trainmaxmin[0],
                                                         trainmaxmin[1], trainmaxmin[2])
    diffusion_samples = diffusion_samples[indices,:,:,:]
    for i, ax in enumerate(grid):
        if(i == 0):
            im = ax.imshow(ref_image,
                           alpha = mask.astype(float), vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Partially Observed")
        elif(i < 4):
            im = ax.imshow(diffusion_samples[(i-1),:,:,:].reshape((n,n)), vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion")
        
        elif(i==4):
            im = ax.imshow(ref_image.reshape((n,n)), vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Fully Observed")
        else:
            im = ax.imshow(diffusion_samples[(i-2),:,:,:].reshape((n,n)), vmin = -2, vmax = 6)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_title("Diffusion")

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_ticks([-2,0,2,4,6])
    #fig.text(0.5, 0.9, 'Unconditional True', ha='center', va='center', fontsize = 25)
    #fig.text(0.1, 0.5, 'range', ha='center', va='center', rotation = 'vertical', fontsize = 40)
    plt.tight_layout()
    plt.savefig(figname)


vpsde = VPSDE(beta_min=0.1, beta_max=20, N=1000)
p = .5
n = 32
"""
mask = ((torch.bernoulli(p*torch.ones((1,1,n,n)))).to(device)).float()
ref_image = (((torch.from_numpy(np.load("brown_resnick_samples_1024_1000.npy").reshape((1000,1,n,n))))[0:1,:,:,:]).to(device)).float()
ref_image = log10_transformation(ref_image)
trainquant = np.load((br_sde_folder + "/trained_score_models/vpsde/model11_train_logminmax.npy"))
ref_image = ref_image - float(trainquant[0])
figname = "br_conditional_diffusion.png"
plot_conditional_difussion_samples(vpsde, score_model, device, mask, ref_image, n,
                                       figname)"""

diffusion_samples = np.load("model11_random50_beta_min_max_01_20_1000_random0_1000.npy")
ref_image = np.load("brmodel11ref_image2.npy")
indices = np.array([6,92,8,455,677,919])
trainmaxmin = np.load((br_sde_folder + "/trained_score_models/vpsde/model11_train_logminmax.npy"))
figname = "br_conditional_diffusion_model11.png"
mask = np.load("brmodel11mask.npy")
n = 32
plot_fixed_conditional_difussion_samples(diffusion_samples, indices, ref_image, mask, n, figname,
                                             trainmaxmin)