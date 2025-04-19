import numpy as np
import matplotlib.pyplot as plt
from append_directories import *
import sys
import os
import torch as th
from brown_resnick_data_generation import *
evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *
import time

#assume we only want to sample once from diffusion model
def return_timing(smooth_value, range_value, n, number_of_locations, model_name, nrep):

    beta_min = .1
    beta_max = 20
    N = 1000
    device = "cuda:0"
    mask = th.zeros((n**2))
    observed_indices = th.from_numpy(np.random.randint(0, n**2, number_of_locations))
    mask[observed_indices] = 1
    mask = mask.reshape((1,1,n,n)).float().to(device)
    score_model = load_score_model("brown", model_name, "eval")
    vpsde = load_sde(beta_min, beta_max, N)
    seed_value = int(np.random.randint(0, 1000000, 1))
    number_of_replicates = 1
    time_elapsed = np.zeros((nrep))
    for irep in range(nrep):
        y = (th.from_numpy(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))).float().to(device)
        number_of_replicates = 1
        start = time.time()
        br_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                                    y, n, number_of_replicates)
        end = time.time()
        time_elapsed[irep] = end - start
    return time_elapsed

def return_timings(tnrep, smooth_value, range_value, n, number_of_locations_list, timing_file, model_name):
    
    timings = np.zeros((len(number_of_locations_list),(tnrep+1)))
    for i,nlocation in enumerate(number_of_locations_list):
        timings[i,0] = nlocation
        timings[i,1:(tnrep+1)] = return_timing(smooth_value, range_value, n, nlocation, model_name, tnrep)

    np.save(timing_file, timings)

def return_timings_multiple_ranges(tnrep):

    model_names = [
                   "model6/model6_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_1_channel_mask.pth",
                   "model7/model7_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_2_channel_mask.pth",
                   "model5/model5_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_3_channel_mask.pth",
                   "model8/model8_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_4_channel_mask.pth",
                   "model9/model9_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_5_channel_mask.pth"]
    model_versions = [6,7,5,8,9]
    range_values = [float(i) for i in range(1,6)]
    smooth_value = 1.5
    n = 32
    number_of_locations_list = [i for i in range(8,11)]
    for i,range_value in enumerate(range_values):
        timing_file = "data/" + str(model_versions[i]) + "_ncs_timing_array_azure_gpu_8_10_nrep_50.npy"
        return_timings(tnrep, smooth_value, range_value, n, number_of_locations_list,
                       timing_file, model_names[i])



return_timings_multiple_ranges(50)