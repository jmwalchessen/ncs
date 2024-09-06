import torch as th
import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
import time

home_folder = append_directory(6)
sde_folder = home_folder + "/sde_diffusion/masked/unparameterized_masked_score"
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
from models import ncsnpp
import sde_lib


#get trained score model
config = ncsnpp_config.get_config()
config.model.num_scales = 1000
config.model.beta_max = 20

score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/model6_beta_min_max_01_20_random02510_channel_mask.pth")))
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
variance = .4
lengthscale = 1.6

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

def time_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples):
    
    start = time.time()
    samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                           y, n, num_samples)
    end = time.time()
    difference = (end - start)
    return difference

def record_and_visualize_diffusion_sample_timing(vpsde, score_model, device, mask, y, n, samples_list,
                                                 recorded_times_file, figname):

    recorded_times = []

    for num_samples in samples_list:

        recorded_times.append(time_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples))

    np.save(recorded_times, np.array(recorded_times_file))
    fig, ax = plt.subplots(1)
    ax.plot(samples_list, recorded_times)
    ax.set_xtitle("Sample Size")
    ax.set_ytitle("Seconds")
    plt.savefig(figname)


device = "cuda:0"
diffusion_data_folder = (sde_folder + "/evaluation/diffusion_generation/data/model6/ref_image1")
mask = np.load((diffusion_data_folder + "/mask.npy"))
ref_image = np.load((diffusion_data_folder + "/ref_image.npy"))
samples_list = [1,5,10,20,25,50,100,200,250,500,1000]
mask = th.from_numpy(mask).to(device)
ref_image = th.from_numpy(ref_image).to(device)
y = ((th.mul(mask, ref_image)).to(device)).float()
recorded_times_file = "models/model6/recorded_times/ref_image1_025_recorded_times.npy"
figname = "models/model6/visualizations/ref_image1_025_1_1000_visualizaton.png"
record_and_visualize_diffusion_sample_timing(sdevp, score_model, device, mask, y, n, samples_list,
                                                 recorded_times_file, figname)
    