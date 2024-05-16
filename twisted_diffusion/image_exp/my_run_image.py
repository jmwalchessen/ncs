import random 
import torch as th
import torch.nn.functional as F
import numpy as np 
import time
from functools import partial
import sys 
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

sys.path.append("..")

from image_diffusion import dist_util
from smc_utils.feynman_kac_pf import smc_FK 
from smc_utils.smc_utils import compute_ess_from_log_w
from image_diffusion.operators import get_operator, ConditioningMethod
from image_diffusion.image_util import get_dataloader, gen_mask, toU8, imwrite 
from image_diffusion.eval_util import pred 
from image_diffusion.my_feynman_kac_image_ddpm import TwistedDDPM
from image_diffusion.my_smc_diffusion import *

#load some true unconditional samples to use as observed partial field
data_folder = '/home/julia/Dropbox/diffusion/sde/gaussian_process/unparameterized/sde_paper/classifier/generate_data/data/true'
true_images = th.from_numpy(np.load(data_folder + "/test_images_lengthscale_1.6_variance_0.4_32_by_32_1000.npy"))
sys.path.insert(1, '/home/julia/Dropbox/diffusion/sde/gaussian_process/unparameterized/sde_paper/configs/vp')
import ncsnpp_config
sys.path.insert(1, '/home/julia/Dropbox/diffusion/sde/gaussian_process/unparameterized/sde_paper')
from models import ncsnpp

#device = cuda
device = "cuda:0"
#load score model
home_folder = "/home/julia/Dropbox/diffusion/sde/gaussian_process/unparameterized/sde_paper"
config = ncsnpp_config.get_config()
config.model.num_scales = 250
config.model.beta_max = 20
score_model = (ncsnpp.NCSNpp(config)).to(device)
score_model.load_state_dict(th.load((home_folder + "/trained_score_models/model24_large_ncsnpp_weighted_250_timesteps_beta_max_20_correct_images_20_epochs_batch_size_128_500000_lengthscale_1.6_variance_0.4_ncsnpp.pth")))
score_model.eval()

score_model(th.ones((1,1,32,32)).to(device), th.tensor([5]).to(device))

betas = get_named_beta_schedule("linear", 250)

pred_xstart_var_type = 4
#systematic is recommended resampling strategry
resample_strategy = "systematic"
ess_threshold = 0
num_particles = 32
#how many particles per batch
batch_p = 32
# just a placeholder since we use unconditional diffusion models 
model_kwargs = {}

diffusion = TwistedDDPM(betas = betas, particle_base_shape = (1,32,32),
                        rescale_timesteps = False, conf=None,
                        probability_flow = False, device = device,
                        use_timesteps = [i for i in range(0,250)],
                        original_num_steps = 250)
diffusion.task = "inpainting"
operator = get_operator(device=device, name=diffusion.task)
recon_prob_fn = ConditioningMethod(operator=operator).recon_prob

#mask is a True/False (1,32,32) vector with .5 randomly missing pixels
#function gen_mask is in image_utils.py, 50 at end of random50 denotes
#50 percent missing
ref_img = (true_images[6,:,:]).to(device)
mask = gen_mask(mask_type = "half", ref_image = ref_img,
                ref_image_name = "zeroed spatial field")
if len(mask.shape) == 4:
    measurement_mask = mask[0] # first dimension is extra degree-of-freedom 
else:
    measurement_mask = mask
#operator = InpaintingOperator which masks out inputs so for a 
#32 by 32 tensor
measurement = operator(data = ref_img, mask=measurement_mask) # returns a one-dimensional tensor 

assert measurement_mask.shape == ref_img.shape 

recon_prob_fn = partial(recon_prob_fn, measurement=measurement, mask=mask)

diffusion.mask = mask
#ref_img is (1,28,28) array with mostly -1 entries but some non
#diffusion.measurement = ref_img*measurement_mask
diffusion.set_measurement(ref_img*measurement_mask) 
# resetting 
diffusion.recon_prob_fn = recon_prob_fn
#recon_prob_fn is from operators.py, -.5*sum((measurement-mask*x0_hat)**2)
#where measurement = mask*ref_image or mask*x0?

#not sure what to do about t_truncate
diffusion.t_truncate = 0
#not sure about this configuration either
diffusion.use_mean_pred = True
#initialize cache
diffusion.clear_cache()

#M and G are from TwistedDDPM for both class and inpainting tasks
#Proposal distribution sort of is M
M = partial(diffusion.M, model=score_model, device=device, 
            pred_xstart_var_type=pred_xstart_var_type)
#Weight function, if debug_plot = True, return twisted_xpred_start in compute_twisted_helper_function
G = partial(diffusion.G, model=score_model, 
            debug_plot=False, debug_statistics=False, debug_info=False, 
            pred_xstart_var_type=pred_xstart_var_type)



def visualize_sample(image, lower, upper, name):

    im = plt.imshow(image, interpolation='nearest',
                    origin='lower', vmin = lower, vmax = upper)
    plt.colorbar(im)
    plt.title(name)
    plt.show()

def compare_maps(m, partially_observed_field, fully_observed_field,
                conditional1, conditional1_partially_observed,
                lower, upper, figname):


    fig = plt.figure(figsize = (11,10))
    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,2),
                    axes_pad=0.55,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.3,
                    label_mode = "all"
                    )
    
    (grid[2]).imshow(conditional1, interpolation='nearest',
                    origin='lower', vmin = lower, vmax = upper)
    (grid[2]).set_title("Conditional TDS right min/max " + str(round(np.max(np.abs(conditional1[:,16:])),2)), fontsize = 14)
    im = (grid[3]).imshow(conditional1_partially_observed, interpolation='nearest',
                    origin='lower', alpha = m, vmin = lower, vmax = upper)
    (grid[3]).set_title(("Conditional TDS left min/max " + str(round(np.max(np.abs(conditional1[:,:16])),2))), fontsize = 14)
    im = (grid[0]).imshow(partially_observed_field, interpolation='nearest',
                    origin='lower', alpha = m, vmin = lower, vmax = upper)
    (grid[0]).set_title("True Partially Observed", fontsize = 20)
    im = (grid[1]).imshow(fully_observed_field,  interpolation='nearest',
                    origin='lower', vmin = lower, vmax = upper)
    (grid[1]).set_title("True Fully Observed", fontsize = 20)
    cbar = (grid[0]).cax.colorbar(im)
    plt.savefig(figname)
    plt.clf()
    plt.close()
j = 6

for i in range(0,1):
    partially_observed = (mask*ref_img).detach().cpu().numpy().reshape((32,32))
    fully_observed = ref_img.detach().cpu().numpy().reshape((32,32))
    m = mask.float().detach().cpu().numpy().reshape((32,32))
    # Sampling 
    final_sample, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace  = \
    smc_FK(M=M, G=G, 
            resample_strategy=resample_strategy, 
            ess_threshold=ess_threshold, 
            T=diffusion.T, 
            P=num_particles, 
            verbose=True, 
            log_xt_trace=False, 
            extra_vals={"model_kwargs": model_kwargs,
                        "batch_p" : batch_p})
    for k in range(0, 32):
        sample0 = final_sample[k,:,:,:].detach().cpu().numpy().reshape((32,32))
        #sample1 = final_sample[i+1,:,:,:].detach().cpu().numpy().reshape((32,32))
        partialsample0 = m*sample0
        compare_maps(m, partially_observed, fully_observed, sample0, partialsample0,
            -2, 2, ("visualizations/examples/ref_image1/model24_lengthscale_1.6_variance_0.4_particles_" +
                    str(num_particles) + "_ess_threshold_0_half_vartype_" + 
                    str(pred_xstart_var_type) + "_smc_" + str(i) + "_particle_" + str(k) + ".png"))