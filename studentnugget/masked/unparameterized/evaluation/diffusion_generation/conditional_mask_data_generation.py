import torch as th
import numpy as np
from append_directories import *
from functools import partial
import matplotlib.pyplot as plt

home_folder = append_directory(3)
#sde configs folder
sde_configs_vp_folder = home_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
print(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(home_folder)
from models import ncsnpp
import sde_lib
from student_t_true_conditional_data_generation import *

n = 32
T = 250
device = "cuda:0"



#get trained score model
config = ncsnpp_config.get_config()
config.model.num_scales = 250
config.model.beta_max = 25.

score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load((home_folder + "/trained_score_models/vpsde/model1_variance_10_lengthscale_1.6_df_1_beta_min_max_01_25_250_random050_masks.pth")))
score_model.eval()

sdevp = sde_lib.VPSDE(beta_min=0.1, beta_max=25, N=250)

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

def sample_unconditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
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
    


def plot_spatial_field(spatial_field, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax)
    plt.savefig(figname)

def plot_masked_spatial_field(spatial_field, mask, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax, alpha = mask)
    plt.savefig(figname)


replicates_per_call = 250
calls = 1
number_of_replicates = 1
seed_value = 43423
minX = minY = -10
maxX = maxY = 10
n = 32
variance = 10
lengthscale = 1.6
df = 1
number_of_replicates = 2
ref_vec, ref_img = generate_student_nugget(minX, maxX, minY, maxY, n, variance,
                                  lengthscale, df, number_of_replicates)
ref_img = ref_img[0,:,:]
ref_img = th.from_numpy(ref_img.reshape((1,n,n))).to(device)
p = .5
mask = (th.bernoulli(p*th.ones(1,1,n,n))).to(device)

for i in range(0, 4):
#mask = th.ones((1,n,n)).to(device)
#mask[:, int(n/4):int(n/4*3), int(n/4):int(n/4*3)] = 0
    y = ((th.mul(mask, ref_img)).to(device)).float()
    conditional_samples = sample_unconditionally_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls)

    partially_observed = (mask*ref_img).detach().cpu().numpy().reshape((n,n))
    np.save("data/model1/ref_image1/ref_image1.npy", ref_img.detach().cpu().numpy().reshape((n,n)))
    np.save("data/model1/ref_image1/diffusion/model1_beta_min_max_01_25_random50_250_" + str(i) + ".npy", conditional_samples)
    np.save("data/model1/ref_image1/partially_observed_field.npy", partially_observed.reshape((n,n)))
    np.save("data/model1/ref_image1/mask.npy", mask.int().detach().cpu().numpy().reshape((n,n)))
    np.save("data/model1/ref_image1/seed_value.npy", np.array([int(seed_value)]))

    plot_spatial_field(ref_img.detach().cpu().numpy().reshape((n,n)), -8, 8, "data/model1/ref_image1/ref_image.png")
    plot_spatial_field((conditional_samples[0,:,:,:]).numpy().reshape((n,n)), -8, 8, "data/model1/ref_image1/diffusion/visualizations/conditional_sample_0.png")
    plot_masked_spatial_field(spatial_field = ref_img.detach().cpu().numpy().reshape((n,n)),
                   vmin = -8, vmax = 8, mask = mask.int().float().detach().cpu().numpy().reshape((n,n)), figname = "data/model1/ref_image1/partially_observed_field.png")
                   