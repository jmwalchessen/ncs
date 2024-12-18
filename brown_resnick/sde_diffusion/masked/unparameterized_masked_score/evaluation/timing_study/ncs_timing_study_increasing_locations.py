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
def return_timing(smooth_value, range_value, n, number_of_locations):

    beta_min = .1
    beta_max = 20
    N = 1000
    device = "cuda:0"
    mask = th.zeros((n**2))
    observed_indices = th.from_numpy(np.random.randint(0, n**2, number_of_locations))
    mask[observed_indices] = 1
    mask = mask.reshape((1,1,n,n)).float().to(device)
    score_model = load_score_model("brown", "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth", "eval")
    vpsde = load_sde(beta_min, beta_max, N)
    seed_value = int(np.random.randint(0, 1000000, 1))
    number_of_replicates = 1
    y = (th.from_numpy(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))).float().to(device)
    number_of_replicates = 1
    start = time.time()
    br_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                            y, n, number_of_replicates)
    end = time.time()
    time_elapsed = end - start
    return time_elapsed

def return_timings(tnrep, smooth_value, range_value, n, number_of_locations_list, timing_file):
    
    timings = np.zeros((len(number_of_locations_list),(tnrep+1)))
    for i,nlocation in enumerate(number_of_locations_list):
        timings[i,0] = nlocation
        for t in range(tnrep):
            timings[i,(t+1)] = return_timing(smooth_value, range_value, n, nlocation)


    np.save(timing_file, timings)

tnrep = 1
tnrep = 50
smooth_value = 1.5
range_value = 3.0
n = 32
number_of_locations_list = [1,2,3,4,5,6,7,8,9]
timing_file = "data/model4/increasing_number_of_observed_locations_timing_array_niasra_node_6_1_conditional_simulation_1_7_tnrep_50.npy"
return_timings(tnrep, smooth_value, range_value, n, number_of_locations_list, timing_file)