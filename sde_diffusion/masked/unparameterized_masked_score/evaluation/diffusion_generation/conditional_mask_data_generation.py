import torch as th
import numpy as np
from append_directories import *
from functools import partial
from generate_true_conditional_samples import *
import matplotlib.pyplot as plt




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
    masked_p_mean = th.mul((1-mask), unmasked_p_mean) + th.mul(mask, y)
    unmasked_p_variance = (th.square(th.tensor(vpsde.sigmas[t])))*th.ones_like(masked_xt)
    masked_p_variance = th.mul((1-mask), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, num_samples):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + th.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples)

    return masked_xt

def sample_unconditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
                                          num_samples_per_call, calls):
    
    diffusion_samples = th.zeros((0, 1, n, n))
    for call in range(0, calls):
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, mask, y, n,
                                                                                   num_samples_per_call)
        diffusion_samples = th.cat([current_diffusion_samples.detach().cpu(),
                                    diffusion_samples],
                                    dim = 0)
    return diffusion_samples
    


def plot_spatial_field(spatial_field, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax)
    plt.savefig(figname)

def plot_masked_spatial_field(spatial_field, mask, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax, alpha = mask)
    plt.savefig(figname)

def generate_validation_data(folder_name, n, variance, lengthscale, replicates_per_call, calls, p, validation_data_name):


    if(os.path.exists(os.path.join(os.getcwd(), folder_name)) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name))

    if(os.path.exists(os.path.join(os.getcwd(), folder_name, "diffusion")) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name, diffusion))

    minX = -10
    maxX = 10
    minY = -10
    maxY = 10
    n = 32
    number_of_replicates = 1
    seed_value = int(np.random.randint(0, 1000000))
    ref_vec, ref_img = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, seed_value):

    partially_observed = (mask*ref_img).detach().cpu().numpy().reshape((n,n))
    np.save((folder_name + "/ref_image.npy"), ref_img.detach().cpu().numpy().reshape((n,n)))

    conditional_samples = np.zeros((0,1,n,n))
    np.save((folder_name + "/partially_observed_field.npy"), partially_observed.reshape((n,n)))
    np.save((folder_name + "/mask.npy"), mask.int().detach().cpu().numpy().reshape((n,n)))
    np.save((folder_name + "seed_value.npy"), np.array([int(seed_value)]))

    for i in range(0, calls):
        y = ((th.mul(mask, ref_img)).to(device)).float()
        conditional_samples = np.concatenate([conditional_samples, sample_unconditionally_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls)], axis = 0)

    np.save((folder_name + "/diffusion/" + validation_data_name), conditional_samples)

    plot_spatial_field(ref_img.detach().cpu().numpy().reshape((n,n)), -2, 2, (folder_name + "/ref_image.png"))
    plot_spatial_field((conditional_samples[0,:,:,:]).numpy().reshape((n,n)), -2, 2, (folder_name + "/diffusion_sample.png"))
    plot_masked_spatial_field(spatial_field = ref_img.detach().cpu().numpy().reshape((n,n)),
                   vmin = -2, vmax = 2, mask = mask.int().float().detach().cpu().numpy().reshape((n,n)), figname = (folder_name + "/partially_observed_field.png"))
    


minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
folder_name = "data/model6/ref_image11"
replicates_per_call = 250
calls = 4
p = .125
generate_validation_data(folder_name, n, variance, lengthscale, replicates_per_call, calls, p, validation_data_name)
