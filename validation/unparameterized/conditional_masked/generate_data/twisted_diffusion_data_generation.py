import torch as th
import numpy as np
from append_directories import *
from functools import partial
import generate_true_conditional_samples
import matplotlib.pyplot as plt

home_folder = append_directory(5)
sde_folder = home_folder + "/sde_diffusion"
print(sde_folder)
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
from models import ncsnpp

n = 32
T = 1000
device = "cuda:0"



#get trained score model
config = ncsnpp_config.get_config()
config.model.num_scales = 1000
config.model.beta_max = 20

score_model = (ncsnpp.NCSNpp(config)).to("cuda:0")
score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/model24_large_ncsnpp_weighted_250_timesteps_beta_max_20_correct_images_20_epochs_batch_size_128_500000_lengthscale_1.6_variance_0.4_ncsnpp.pth")))
score_model.eval()




def plot_spatial_field(spatial_field, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax)
    plt.savefig(figname)

def plot_masked_spatial_field(spatial_field, mask, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax, alpha = mask)
    plt.savefig(figname)


np.save("data/ref_image2/ref_image2.npy", ref_img.detach().cpu().numpy().reshape((n,n)))
np.save("data/ref_image2/diffusion/model24_particles_" + str(replicates_per_call) + "_var_type_1_ess_threshold_0_1000.npy", conditional_samples)
np.save("data/ref_image2/partially_observed_field.npy", partialfield.reshape((n,n)))
np.save("data/ref_image2/mask.npy", mask.int().detach().cpu().numpy().reshape((n,n)))
np.save("data/ref_image2/seed_value.npy", np.array([int(seed_value)]))

plot_spatial_field(ref_img.detach().cpu().numpy().reshape((n,n)), -2, 2, "data/ref_image1/ref_image.png")
plot_spatial_field(conditional_samples[0,:,:], -2, 2, "data/ref_image1/diffusion/visualizations/conditional_sample_0.png")
plot_masked_spatial_field(spatial_field = ref_img.detach().cpu().numpy().reshape((n,n)),
                   vmin = -2, vmax = 2, mask = mask.int().float().detach().cpu().numpy().reshape((n,n)), figname = "data/ref_image1/partially_observed_field.png")






        


    














    

