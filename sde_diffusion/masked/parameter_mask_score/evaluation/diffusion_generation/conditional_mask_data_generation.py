import torch as th
import numpy as np
from append_directories import *
from functools import partial
from generate_true_conditional_samples import *
import matplotlib.pyplot as plt

home_folder = append_directory(6)
sde_folder = home_folder + "/sde_diffusion/masked/parameter_mask_score"
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
from models import ncsnpp
import sde_lib

n = 32
T = 1000
device = "cuda:0"



#get trained score model
config = ncsnpp_config.get_config()
config.model.num_scales = 1000
config.model.beta_max = 20

score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/model4_variance_1.5_lengthscale_3_5_beta_min_max_01_20_random50_channel_mask.pth")))
score_model.eval()

sdevp = sde_lib.VPSDE(beta_min=0.1, beta_max=20, N=1000)

#mask is a True/False (1,32,32) vector with .5 randomly missing pixels
#function gen_mask is in image_utils.py, 50 at end of random50 denotes
#50 percent missing
minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .8
lengthscale = 1.6

#y is observed part of field, modified to incorporate the mask as channel
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, variance, lengthscale):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    #need mask to be same size as masked_xt
    mask = mask.repeat((reps,1,1,1))
    parameter_mask = lengthscale*mask
    masked_xt_and_mask = th.cat([masked_xt, parameter_mask], dim = 1)
    with th.no_grad():
        parameter = th.tensor([variance, lengthscale]).to(device)
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

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, variance, lengthscale):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t,
                                                                 variance, lengthscale)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples, variance, lengthscale):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + th.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, variance, lengthscale)

    return masked_xt

def sample_unconditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
                                          num_samples_per_call, calls, variance, lengthscale):
    
    diffusion_samples = th.zeros((0, 1, n, n))
    for call in range(0, calls):
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, mask, y, n,
                                                                                   num_samples_per_call, variance, lengthscale)
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


def generate_validation_data(folder_name, n, variance, lengthscale, replicates_per_call, calls,
                             p, validation_data_name):


    if(os.path.exists(os.path.join(os.getcwd(), folder_name)) == False):
        os.makedirs(os.path.join(os.getcwd(), folder_name))

    if(os.path.exists(os.path.join(os.getcwd(), folder_name, "diffusion")) == False):
        os.makedirs(os.path.join(os.getcwd(), folder_name, "diffusion"))

    minX = -10
    maxX = 10
    minY = -10
    maxY = 10
    n = 32
    number_of_replicates = 1
    seed_value = int(np.random.randint(0, 1000000))
    ref_vec, ref_img = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                 number_of_replicates, seed_value)

    device = "cuda:0"
    ref_img = (th.from_numpy(ref_img.reshape((1,1,n,n)))).to(device)
    mask = (th.bernoulli(p*th.ones((1,1,n,n)))).to(device)
    y = ((th.mul(mask, ref_img)).to(device)).float()

    conditional_samples = np.zeros((0,1,n,n))

    for i in range(0, calls):

        conditional_samples = np.concatenate([conditional_samples, sample_unconditionally_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls, variance, lengthscale)], axis = 0)

    np.save((folder_name + "/diffusion/" + validation_data_name), conditional_samples)
    partially_observed = (mask*ref_img).detach().cpu().numpy().reshape((n,n))
    np.save((folder_name + "/ref_image.npy"), ref_img.detach().cpu().numpy().reshape((n,n)))
    np.save((folder_name + "/partially_observed_field.npy"), partially_observed.reshape((n,n)))
    np.save((folder_name + "/mask.npy"), mask.int().detach().cpu().numpy().reshape((n,n)))
    np.save((folder_name + "/seed_value.npy"), np.array([int(seed_value)]))

    #plot_spatial_field(ref_img.detach().cpu().numpy().reshape((n,n)), -3, 3, (folder_name + "/ref_image.png"))
    #plot_spatial_field((conditional_samples[0,:,:,:]).reshape((n,n)), -3, 3, (folder_name + "/diffusion_sample.png"))
    #plot_masked_spatial_field(spatial_field = ref_img.detach().cpu().numpy().reshape((n,n)),
                   #vmin = -3, vmax = 3, mask = mask.int().float().detach().cpu().numpy().reshape((n,n)), figname = (folder_name + "/partially_observed_field.png"))


replicates_per_call = 1000
calls = 1
variance = 1.5
p = .5
lengthscales = [3.,4.,5.]
for i, lengthscale in enumerate(lengthscales):

    folder_name = "data/model4/ref_image" + str(i+1)
    validation_data_name = "model4_variance_1.5_lengthscale_" + str(lengthscale) + "_beta_min_max_01_20_random50_1000.npy"
    generate_validation_data(folder_name, n, variance, lengthscale, replicates_per_call, calls,
                         p, validation_data_name)