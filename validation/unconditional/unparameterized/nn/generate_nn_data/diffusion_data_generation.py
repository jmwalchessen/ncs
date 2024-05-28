import torch as th
import numpy as np
import sys
import os
from append_directories import *

home_folder = append_directory(6)
twisted_diffusion_folder = home_folder + "/twisted_diffusion/image_exp/image_diffusion"
sde_folder = home_folder + "/sde_diffusion"
sys.path.append(twisted_diffusion_folder)
import my_smc_diffusion
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
from models import ncsnpp

n = 32
T = 250
number_of_replicates_per_call = 16
number_of_calls = 2


#get smc diffusion sampler
betas = my_smc_diffusion.get_named_beta_schedule("linear", T)
smc_diffusion_model = my_smc_diffusion.SMCDiffusion(use_timesteps = [i for i in range(0,T)],
                                                    original_num_steps = T,
                                                    particle_base_shape = (1,n,n), betas = betas)





#get trained score model
config = ncsnpp_config.get_config()
config.model.num_scales = 250
config.model.beta_max = 20

score_model = (ncsnpp.NCSNpp(config)).to("cuda:0")
score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/model24_large_ncsnpp_weighted_250_timesteps_beta_max_20_correct_images_20_epochs_batch_size_128_500000_lengthscale_1.6_variance_0.4_ncsnpp.pth")))
score_model.eval()



def generate_diffusion_samples_per_call(diffusion_sampler, trained_score_model, number_of_replicates):

    diffusion_samples = diffusion_sampler.posterior_sample_with_p_mean_variance(trained_score_model,
                                                                                number_of_replicates,
                                                                                clip_denoised=True,
                                                                                denoised_fn=None,
                                                                                model_kwargs=None)
    return diffusion_samples

def generate_diffusion_samples(diffusion_sampler, trained_score_model, number_of_replicates_per_call,
                               number_of_calls, n):
    
    diffusion_images = np.empty((0, n, n))

    for i in range(0, number_of_calls):

        current_call_diffusion_images = generate_diffusion_samples_per_call(diffusion_sampler, trained_score_model, number_of_replicates_per_call)
        diffusion_images = np.concatenate([diffusion_images,
                                           (current_call_diffusion_images.detach().cpu().numpy()).reshape((number_of_replicates_per_call,n,n))], axis = 0)

    return diffusion_images

diffusion_images = generate_diffusion_samples(smc_diffusion_model, score_model,
                                              number_of_replicates_per_call,
                                              number_of_calls, n)






    

