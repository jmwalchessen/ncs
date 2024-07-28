import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import sys
from append_directories import *
from studentt_generation import *

home_folder = append_directory(5)
tsde_folder = (home_folder + "/studentnugget/masked/unparameterized")
sys.path.append(tsde_folder)
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
score_model.load_state_dict(torch.load((tsde_folder + "/trained_score_models/vpsde/model5_variance_.4_lengthscale_1.6_df_3_beta_min_max_01_20_1000_random050_masks.pth")))
score_model.eval()

def index_to_matrix_index(index,n):

    return (int(index / n), int(index % n))


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
    df = 3
    nminusm = observed_vector.shape[0]
    m = (n**2)-m
    observed_unconditional_mean = np.zeros((nminusm, 1))
    unobserved_unconditional_mean = np.zeros((m,1))
    tsamples = true_conditional_image_sampling((1-mask).detach().cpu().numpy(), minX, maxX, minY, maxY, n,
                                    variance, lengthscale,
                                    observed_vector, observed_unconditional_mean,
                                    unobserved_unconditional_mean, df, number_of_replicates,
                                    seed_value, observed_matrix)
    tsamples = tsamples.reshape((3,n,n))
    for i, ax in enumerate(grid):
        if(i < 3):
            im = ax.imshow(tsamples[i,:,:], vmin = -2, vmax = 2)
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
    df = 3
    nminusm = observed_vector.shape[0]
    m = (n**2)-nminusm
    observed_unconditional_mean = np.zeros((nminusm, 1))
    unobserved_unconditional_mean = np.zeros((m,1))
    tsamples = true_conditional_image_sampling(mask.detach().cpu().numpy(), minX, maxX, minY, maxY, n,
                                    variance, lengthscale,
                                    observed_vector, observed_unconditional_mean,
                                    unobserved_unconditional_mean, df, number_of_replicates,
                                    seed_value, observed_matrix, missing_indices)
    tsamples = tsamples.reshape((3,n,n))
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
            im = ax.imshow(tsamples[(i-1),:,:], vmin = -2, vmax = 2)
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
mask = (torch.bernoulli(p*torch.ones((1,1,n,n)))).float().to(device)
minX = -10
maxX = 10
minY = -10
maxY = 10
variance = .4
lengthscale = 1.6
df = 3
number_of_replicates = 2
seed_value = 35923
refvec, ref_img = generate_student_nugget(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates,
                            seed_value)
ref_img = (ref_img[0:1,:,:,:])
refvec = refvec[0,:]
observed_vector = refvec.reshape((n**2,1))
observed_matrix = ((torch.mul(mask, torch.from_numpy(ref_img).to(device)))).detach().cpu().numpy()
missing_indices = np.squeeze(np.argwhere((1-mask).detach().cpu().numpy().reshape((n**2,))))
m = missing_indices.shape[0]
observed_vector = np.delete(observed_vector, missing_indices)
observed_vector = observed_vector.reshape(((n**2)-m),1)
figname = "tconditional_true_vs_diffusion_model5.png"
ref_img = (torch.from_numpy(ref_img)).float().to(device)
plot_conditional_true_and_difussion_samples(vpsde, score_model, device, mask, observed_vector,
                                                observed_matrix, ref_img, missing_indices, n,
                                                figname)
