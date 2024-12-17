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


def return_timing(number_of_replicates, smooth_value, range_value, n, number_of_locations):

    beta_min = .1
    beta_max = 20
    N = 1000
    device = "cuda:0"
    mask = th.zeros((n**2))
    observed_indices = th.from_numpy(np.random.randint(0, n**2, number_of_locations))
    mask[observed_indices] = 1
    mask = mask.float().to(device)
    score_model = load_score_model("brown", "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth", "eval")
    vpsde = load_sde(beta_min, beta_max, N)
    seed_value = int(np.random.randint(0, 1000000, 1))
    y = (th.from_numpy(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))).to(device)
    start = time.time()
    br_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                            y, n, number_of_replicates)
    end = time.time()
    time_elapsed = end - start
    return time_elapsed

def return_timings(number_of_replicates, smooth_value, range_value, n, number_of_locations_list, timing_file):
    
    timings = np.zeros((2,len(number_of_locations_list)))
    for i,nlocation in enumerate(number_of_locations_list):

        timings[i,0] = nlocation
        timings[i,1] = return_timing(number_of_replicates, smooth_value, range_value, n, nlocation)


    np.save(timing_file, timings)

number_of_replicates = 1
smooth_value = 1.5
range_value = 3.0
n = 32
number_of_locations_list = [1,2,3,4,5] + [i*5 for i in range(1,204)]
timing_file = "data/model4/increasing_number_of_observed_locations_timing_array.npy"
return_timings(number_of_replicates, smooth_value, range_value, n, number_of_locations_list, timing_file)