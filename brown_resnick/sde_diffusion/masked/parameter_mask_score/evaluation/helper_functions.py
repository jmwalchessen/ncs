import torch as th
import numpy as np
import subprocess
import os
from append_directories import *

#get trained score model
def load_score_model(process_type, model_name, mode):

    home_folder = append_directory(5)
    if "sde_diffusion" in home_folder:
        sde_folder = home_folder + "/masked/unparameterized_masked_score"
    else:
        sde_folder = home_folder + "/sde_diffusion/masked/unparameterized_masked_score"
    sde_configs_vp_folder = sde_folder + "/configs/vp"
    sys.path.append(sde_configs_vp_folder)
    import ncsnpp_config
    sys.path.append(sde_folder)
    from models import ncsnpp
    config = ncsnpp_config.get_config()

    score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
    score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/" + process_type + "/" + model_name)))
    if(mode == "train"):
        score_model.train()
    else:
        score_model.eval()
    return score_model

def load_sde(beta_min, beta_max, N):

    import sde_lib
    sdevp = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
    return sdevp

def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t,
                                            range_value, smooth_value):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    #need mask to be same size as masked_xt
    mask = mask.repeat((reps,1,1,1))
    parameter_mask = range_value*mask
    masked_xt_and_mask = th.cat([masked_xt, parameter_mask], dim = 1)
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

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, num_samples,
                                         range_value, smooth_value):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, 
                                                                 mask, y, t, range_value, smooth_value)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples, range_value, smooth_value):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + th.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples, range_value,
                                                         smooth_value)

    return masked_xt

def sample_unconditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
                                          num_samples_per_call, calls, range_value,
                                          smooth_value):
    
    diffusion_samples = th.zeros((0, 1, n, n))
    for call in range(0, calls):
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, mask, y, n,
                                                                                   num_samples_per_call,
                                                                                   range_value,
                                                                                   smooth_value)
        diffusion_samples = th.cat([current_diffusion_samples.detach().cpu(),
                                    diffusion_samples],
                                    dim = 0)
    return diffusion_samples

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    images = images.reshape((number_of_replicates,1,n,n))
    return images

def generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "schlather_data_generation.R", str(range_value), str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_schlather_samples.npy")
    os.remove("temporary_schlather_samples.npy")
    images = images.reshape((number_of_replicates,1,n,n))
    return images

def pwd_command():

    subprocess.run(["pwd"], shell = True)