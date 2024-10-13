import torch as th
import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
import time

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
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t, variance, lengthscale):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    parameter_masks = lengthscale*masks
    masked_xt_and_mask = th.cat([masked_xt, parameter_masks], dim = 1)
    with th.no_grad():
        score_and_mask = score_model(masked_xt_and_mask, timestep)
    
    #first channel is score, second channel is mask
    score = score_and_mask[:,0:1,:,:]
    squared_sigmat = (th.square(th.tensor(vpsde.sigmas[t]))).to(device)
    sqrt_alphat = (th.sqrt(th.tensor(vpsde.alphas[t]))).to(device)
    unmasked_p_mean = (1/sqrt_alphat)*(masked_xt + squared_sigmat*score)
    masked_p_mean = th.mul((1-masks), unmasked_p_mean) + th.mul(masks, ys)
    unmasked_p_variance = squared_sigmat*th.ones_like(masked_xt)
    masked_p_variance = th.mul((1-masks), unmasked_p_variance)
    return masked_p_mean, masked_p_variance


def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t, variance, lengthscale):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t,
                                                                 variance, lengthscale)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-masks), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masks,
                                                   ys, n, variance, lengthscale):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + th.mul(mask, ys)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         masks, ys, t, variance, lengthscale)

    return masked_xt

def sample_conditionally_multiple_calls(vpsde, score_model, device, masks, ys, n,
                                          num_samples_per_call, calls, variance, lengthscale):
    
    diffusion_samples = th.zeros((0, 1, n, n))
    for call in range(0, calls):
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, masks, ys, n,
                                                                                   num_samples_per_call,
                                                                                   variance, lengthscale)
        diffusion_samples = th.cat([current_diffusion_samples.detach().cpu(),
                                    diffusion_samples],
                                    dim = 0)
    return diffusion_samples

def produce_parameters_via_uniform(number_of_parameters, boundary_start, boundary_end):

    uniform_generator = scipy.stats.uniform()
    parameters = ((boundary_end - boundary_start)*uniform_generator.rvs(number_of_parameters)) + boundary_start
    return parameters

def sample_conditionally_multipe_parameters(vpsde, score_model, device, masks, ys, n, num_samples_per_call, calls,
                                            number_of_parameters, boundary_start, bounday_end, variance, diffusion_file):

    lengthscales = produce_parameters_via_uniform(number_of_parameters, boundary_start, boundary_end)
    parameter_matrix = np.zeros((number_of_parameters,2))
    parameter_matrix[:,0] = variance*np.ones((number_of_parameters,1))
    parameter_matrix[:,1] = lengthscales.reshape((number_of_parameters,1))
    for i in range(parameter_matrix.shape[0]):

        variance = parameter_matrix[i,0]
        lengthscale = parameter_matrix[i,1]
        diffusion_images = sample_conditionally_multiple_calls(vpsde, score_model, device, masks, ys, n,
                                                               num_samples_per_call, calls, variance, lengthscale)
        np.save((diffusion_file + "_variance_" + str(round(variance,1) + "_lengthscale_"
                 + str(round(lengthscale,1)) + ".npy")), diffusion_images.numpy())
    pass


"unconditional case so ref_image doesn't matter"
for i in range(0, 1):
    num_samples_per_call = 500
    calls = 10
    device = "cuda:0"
    ref_image = np.load((sde_folder + "/evaluation/diffusion_generation/data/model6/ref_image3/ref_image.npy"))
    p = 0
    mask = (th.bernoulli(p*th.ones((1,1,n,n)))).to(device)
    ref_image = th.from_numpy(ref_image).to(device)
    #missing_indices = np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    #m = missing_indices.shape[0]
    y = (((th.mul(mask, ref_image)).to(device))).float()
    diffusion_samples = sample_conditionally_multiple_calls(sdevp, score_model, device, masks,
                                                            ys, n, num_samples_per_call, calls)
    np.save((sde_folder + "/evaluation/classification/train_classifier/generate_data/data/model6/unconditional/calibration_evaluation_unconditional_images_variance_.4_lengthscale_1.6_5000.npy"), diffusion_samples)