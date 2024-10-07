import torch as th
import numpy as np
from append_directories import *
from functools import partial
from brown_resnick_data_generation import *
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *


process_type = "brown"
model_name = "model2_beta_min_max_01_20_random025125_log_channel_mask.pth"
mode = "eval"

sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)
score_model = load_score_model(process_type, model_name, mode)

def produce_mask_generation(n, obsn):

    indices = [i for i in range(0, obsn)]
    mask = np.zeros((n**2))
    mask[indices] = 1
    mask = mask.reshape((n,n))
    return mask


#pixel_indices are flattened vector
def produce_pit_value_via_diffusion_multiple_pixels(pixel_indices, mask, range_value, smooth_value, n, vpsde, device, nrep):

    seed_value = int(np.random.randint(0, 100000))
    number_of_replicates = 1
    br_values = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))
    diffusion_images = posterior_sample_with_p_mean_variance_via_mask(vpsde = vpsde, score_model = score_model,
                                                                      device = device, mask = mask, y = br_values, n = n,
                                                                      num_samples = nrep)
    diffusion_images = diffusion_images.reshape((nrep,(n**2)))
    br_values = br_values.reshape((1,(n**2)))
    br_ref_pixels = br_values[pixel_indices]
    diffusion_ref_pixels = diffusion_images[:,pixel_indices]
    pit_values = np.mean((diffusion_ref_pixels <= br_ref_pixels), axis = 0)
    return pit_values

pixel_indices = [10,45, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
range_value = 1.6
smooth_value = 1.6
n = 32
device = "cuda:0"
nrep = 5
obsn = 10
mask = produce_mask_generation(n, obsn)
pit_values = produce_pit_value_via_diffusion_multiple_pixels(pixel_indices, mask, range_value, smooth_value,
                                                             n, sdevp, device, nrep)  
