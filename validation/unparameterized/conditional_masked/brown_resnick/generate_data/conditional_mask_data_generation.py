import torch as th
import numpy as np
from append_directories import *
from functools import partial
import generate_true_unconditional_samples
import matplotlib.pyplot as plt

home_folder = append_directory(6)
sde_folder = home_folder + "/brown_resnick/sde_diffusion/masked/unparameterized"
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
score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/model10_beta_min_max_01_20_1000_1.6_1.6_random050_logglobalbound_masks.pth")))
score_model.eval()
sdevp = sde_lib.VPSDE(beta_min=0.1, beta_max=20, N=1000)

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs

#y is observed part of field
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    with th.no_grad():
        score = score_model(masked_xt, timestep)
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

def sample_conditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
                                          num_samples_per_call, calls):
    
    diffusion_samples = th.zeros((0, 1, n, n))
    for call in range(0, calls):
        print(call)
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
    #plt.show()

def plot_masked_spatial_field(spatial_field, mask, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax, alpha = mask)
    plt.savefig(figname)

#sample unconditionally
minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
range_value = 1.6
smooth_value = 1.6
seed_value = 43423
number_of_replicates = 1000

trainlogmaxmin = np.load((sde_folder + "/trained_score_models/vpsde/model10_train_logminmax.npy"))
#ref_img = generate_true_unconditional_samples.generate_brown_resnick_process(range_value, smooth_value,
                                                                             #seed_value, 256,
                                                                             #n)
ref_img = log_transformation(np.load("brown_resnick_samples_256.npy"))
print(ref_img.shape)

ref_img = (th.from_numpy(global_quantile_boundary_process(ref_img, trainlogmaxmin[0],
trainlogmaxmin[1], trainlogmaxmin[2])).reshape((250,1,n,n))[0,:,:,:]).to(device)
p = 0
mask = (th.bernoulli(p*th.ones(1,1,n,n))).to(device)
replicates_per_call = 250
calls = 1
for i in range(0,4):
    y = ((th.mul(mask, ref_img)).to(device)).float()
    conditional_samples = sample_conditionally_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls)
    np.save("data/conditional/model10/ref_img1/model10_random0_beta_min_max_01_20_1000_random0_250_" + str(i) + ".npy", conditional_samples)
    partially_observed = (mask*ref_img).detach().cpu().numpy().reshape((n,n))
    np.save("data/conditional/model10/ref_img1/ref_image1.npy", ref_img.detach().cpu().numpy().reshape((n,n)))
    np.save("data/conditional/model10/ref_img1/partially_observed_field.npy", partially_observed.reshape((n,n)))
    np.save("data/conditional/model10/ref_img1/mask.npy", mask.int().detach().cpu().numpy().reshape((n,n)))
    np.save("data/conditional/model10/ref_img1/seed_value.npy", np.array([int(seed_value)]))

    plot_spatial_field(ref_img.detach().cpu().numpy().reshape((n,n)), -2, 4, "data/conditional/model10/ref_img1/ref_image.png")
    plot_spatial_field((conditional_samples[0,:,:,:]).numpy().reshape((n,n)), -2, 4, "data/conditional/model10/ref_img1/conditional_sample_0.png")
    plot_masked_spatial_field(spatial_field = ref_img.detach().cpu().numpy().reshape((n,n)),
                   vmin = -2, vmax = 4, mask = mask.int().float().detach().cpu().numpy().reshape((n,n)), figname = "data/conditional/ref_img1/partially_observed_field.png")
"""
number_of_replicates = 2250
print(number_of_replicates)
seed_value = 23423
unconditional_true_samples = generate_true_unconditional_samples.generate_brown_resnick_process(range_value, smooth_value,
                                                                             seed_value, number_of_replicates,
                                                                             n)
np.save("data/unconditional/true/unconditional_model4_range_1.6_smooth_1.6_2250.npy", unconditional_true_samples)

"""


