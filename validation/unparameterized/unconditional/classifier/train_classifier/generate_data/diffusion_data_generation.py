import torch as th
import numpy as np
import sys
import os
from append_directories import *

home_folder = append_directory(7)
twisted_diffusion_folder = home_folder + "/twisted_diffusion/image_exp/image_diffusion"
sde_folder = home_folder + "/sde_diffusion"
sys.path.append(twisted_diffusion_folder)
import my_smc_diffusion
import my_feynman_kac_image_ddpm
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
from models import ncsnpp
import time

n = 32
T = 250
device = "cuda:0"
#get trained score model
config = ncsnpp_config.get_config()
config.model.num_scales = 250
config.model.beta_max = 20

score_model = (ncsnpp.NCSNpp(config)).to("cuda:0")
score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/model24_large_ncsnpp_weighted_250_timesteps_beta_max_20_correct_images_20_epochs_batch_size_128_500000_lengthscale_1.6_variance_0.4_ncsnpp.pth")))
score_model.eval()

#get smc diffusion sampler
betas = my_smc_diffusion.get_named_beta_schedule("linear", T)
twisted_diffusion_model = my_feynman_kac_image_ddpm.TwistedDDPM(betas = betas, particle_base_shape = (1,32,32),
                        rescale_timesteps = False, conf=None,
                        probability_flow = False, device = device,
                        use_timesteps = [i for i in range(0,250)],
                        original_num_steps = 250)

def sample_unconditionally(diffusion_model, trained_score_model, replicates_per_call,
                           calls, n):
    
    unconditional_samples = th.zeros((0,n,n))
    for i in range(0, calls):
        current_samples = diffusion_model.posterior_sample_with_p_mean_variance(trained_score_model,
                                                                                replicates_per_call)
        unconditional_samples = th.cat([unconditional_samples,
                                        (current_samples.reshape((replicates_per_call,n,n))).detach().cpu()],
                                        dim = 0)
   
    return unconditional_samples

"""
replicates_per_call = 500
calls = 2
start = time.time()
unconditional_samples = sample_unconditionally(twisted_diffusion_model, score_model, replicates_per_call,
                           calls, n)
end = time.time()
print(print(end - start))

np.save("data/diffusion/eval_unconditional_lengthscale_1.6_variance_0.4_1000.npy", unconditional_samples.numpy())
"""