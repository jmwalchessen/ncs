import torch as th
import numpy as np
from append_directories import *
from functools import partial
import generate_true_conditional_samples
import matplotlib.pyplot as plt

home_folder = append_directory(6)
print(home_folder)
image_diffusion_folder = home_folder + "/twisted_diffusion/image_exp/image_diffusion"
sde_folder = home_folder + "/sde_diffusion/unparameterized"
sys.path.append(image_diffusion_folder)
print(image_diffusion_folder)
import my_smc_diffusion
import my_feynman_kac_image_ddpm
import operators
import image_util
import eval_util
twisted_diffusion_folder = home_folder + "/twisted_diffusion"
sys.path.append(twisted_diffusion_folder)
from smc_utils import feynman_kac_pf
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
print(sde_folder)
from models import ncsnpp

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


pred_xstart_var_type = 1
#systematic is recommended resampling strategry
resample_strategy = "systematic"
ess_threshold = 0
num_particles = 32
#how many particles per batch
batch_p = 32
# just a placeholder since we use unconditional diffusion models 
model_kwargs = {}
#get smc diffusion sampler
betas = my_smc_diffusion.get_named_beta_schedule("linear", T)
twisted_diffusion = my_feynman_kac_image_ddpm.TwistedDDPM(betas = betas, particle_base_shape = (1,32,32),
                        rescale_timesteps = False, conf=None,
                        probability_flow = False, device = device,
                        use_timesteps = [i for i in range(0,250)],
                        original_num_steps = 250)

#not sure what to do about t_truncate
twisted_diffusion.t_truncate = 0
#not sure about this configuration either
twisted_diffusion.use_mean_pred = True
#initialize cache
twisted_diffusion.clear_cache()

#M and G are from TwistedDDPM for both class and inpainting tasks
#Proposal distribution sort of is M
M = partial(twisted_diffusion.M, model=score_model, device=device, 
            pred_xstart_var_type=pred_xstart_var_type)
#Weight function, if debug_plot = True, return twisted_xpred_start in compute_twisted_helper_function
G = partial(twisted_diffusion.G, model=score_model, 
            debug_plot=False, debug_statistics=False, debug_info=False, 
            pred_xstart_var_type=pred_xstart_var_type)

def twisted_diffusion_samples_per_call(twisted_diffusion, replicates_per_call,
                                       mask, ref_img, n, ess_threshold):
    
    twisted_diffusion.task = "inpainting"
    operator = operators.get_operator(device=device, name=twisted_diffusion.task)
    recon_prob_fn = operators.ConditioningMethod(operator=operator).recon_prob
    if len(mask.shape) == 4:
        measurement_mask = mask[0] # first dimension is extra degree-of-freedom 
    else:
        measurement_mask = mask
    #operator = InpaintingOperator which masks out inputs so for a 
    #32 by 32 tensor
    measurement = operator(data = ref_img, mask=measurement_mask) # returns a one-dimensional tensor 

    assert measurement_mask.shape == ref_img.shape 

    recon_prob_fn = partial(recon_prob_fn, measurement=measurement, mask=mask)

    twisted_diffusion.mask = mask
    #ref_img is (1,28,28) array with mostly -1 entries but some non
    #diffusion.measurement = ref_img*measurement_mask
    twisted_diffusion.set_measurement(ref_img*measurement_mask) 
    # resetting 
    twisted_diffusion.recon_prob_fn = recon_prob_fn
    #recon_prob_fn is from operators.py, -.5*sum((measurement-mask*x0_hat)**2)
    #where measurement = mask*ref_image or mask*x0?
    partially_observed = (mask*ref_img).detach().cpu().numpy().reshape((32,32))
    fully_observed = ref_img.detach().cpu().numpy().reshape((32,32))
    m = mask.float().detach().cpu().numpy().reshape((32,32))
    # Sampling
    final_samples, log_w, normalized_w, resample_indices_trace, ess_trace,\
    log_w_trace, xt_trace, log_proposal_trace, log_potential_xt_trace,\
    log_potential_xtp1_trace, log_p_trans_untwisted_trace  = \
    feynman_kac_pf.smc_FK(M=M, G=G, resample_strategy=resample_strategy, 
                                     ess_threshold=ess_threshold, 
                                     T=twisted_diffusion.T, 
                                     P=replicates_per_call, 
                                     verbose=True, 
                                     log_xt_trace=False, 
                                     extra_vals={"model_kwargs": model_kwargs,
                                     "batch_p" : batch_p})
    
    final_samples.detach().cuda().reshape((replicates_per_call, n, n))
    return final_samples, partially_observed, fully_observed, m,\
          resample_indices_trace, ess_trace, log_w_trace, log_proposal_trace,\
              log_potential_xt_trace, log_potential_xtp1_trace, log_p_trans_untwisted_trace


def twisted_diffusion_samples_multiple_calls(twisted_diffusion, replicates_per_call,
                                             calls, mask, ref_img, n):
    
    conditional_samples = th.zeros((0,n,n))
    partialfield = th.zeros((n,n))
    fullfield = th.zeros((n,n))
    m = th.zeros((n,n))
    for i in range(0, calls):
        final_samples, partialfield, fullfield, m, resample_indices, ess_trace, log_w_trace, log_proposal_trace,\
            log_potential_xt_trace, log_potential_xtp1_trace, log_p_trans_untwisted_trace = twisted_diffusion_samples_per_call(twisted_diffusion,
                                                                                                                                 replicates_per_call,
                                                                                                                                 mask, ref_img, n)
        
        conditional_samples = np.concatenate([conditional_samples, final_samples.detach().cpu().numpy().reshape((replicates_per_call,n,n))], axis = 0)

    return conditional_samples, partialfield, fullfield, m, resample_indices, ess_trace,\
      log_w_trace, log_proposal_trace, log_potential_xt_trace, log_potential_xtp1_trace,\
      log_p_trans_untwisted_trace
                        