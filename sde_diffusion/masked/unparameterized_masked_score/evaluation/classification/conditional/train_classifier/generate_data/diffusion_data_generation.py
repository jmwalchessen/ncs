import torch as th
import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
import time
from true_unconditional_data_generation import *
home_folder = append_directory(6)
sde_folder = home_folder + "/unparameterized_masked_score"
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
print(sde_configs_vp_folder)
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

def mask_generation(p, nrep, n):

    masks = (torch.bernoulli(p*torch.ones((nrep,1,n,n)))).numpy()
    return masks

#y is observed part of field, modified to incorporate the mask as channel
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, y, t):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    masked_xt_and_mask = th.cat([masked_xt, masks], dim = 1)
    with th.no_grad():
        score_and_mask = score_model(masked_xt_and_mask, timestep)
    
    #first channel is score, second channel is mask
    score = score_and_mask[:,0:1,:,:]
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
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, mask, y, n,
                                                                                   num_samples_per_call)
        diffusion_samples = th.cat([current_diffusion_samples.detach().cpu(),
                                    diffusion_samples],
                                    dim = 0)
    return diffusion_samples

minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 1.6
p = .5
number_of_replicates = 5
seed_value = int(np.random.randint(0, 100000))

ys = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, seed_value)
masks = mask_generation(p, number_of_replicates, n)
device = "cuda:0"
p_mean_and_variance_from_score_via_mask(sdevp, score_model, device, masked_xt, masks, ys, t)