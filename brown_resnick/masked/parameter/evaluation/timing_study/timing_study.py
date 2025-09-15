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


def return_timing(number_of_replicates, smooth_value, range_value, p, n):

    beta_min = .1
    beta_max = 20
    N = 1000
    device = "cuda:0"
    mask = (th.bernoulli(p*th.ones((1,1,n,n)))).float().to(device)
    score_model = load_score_model("brown", "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth", "eval")
    vpsde = load_sde(beta_min, beta_max, N)
    seed_value = int(np.random.randint(0, 1000000, 1))
    y = (th.from_numpy(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))).float().to(device)
    start = time.time()
    br_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                            y, n, number_of_replicates)
    end = time.time()
    time_elapsed = end - start
    return time_elapsed


def plot_sample_size_vs_time(replicates_list, smooth_value, range_value, p, n, figname, npfilename):


    fig, ax = plt.subplots()
    timings = np.zeros((len(replicates_list),2))
    for i in range(0, len(replicates_list)):

        timings[i,0] = replicates_list[i]
        timings[i,1] = return_timing(replicates_list[i], smooth_value, range_value, p, n)


    np.save(npfilename, timings)
    ax.plot(replicates_list, timings[:,1])
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Evaluation Time (Seconds)")
    plt.savefig(figname)

def plot_sample_size_vs_time_multiple_ranges(replicates_list, smooth_value, range_values, p, n, figname, npfilename):


    for range_value in range_values:
        current_figname = (figname + str(range_value) + ".png")
        current_npfilename = (npfilename + str(range_value) + ".npy")
        plot_sample_size_vs_time(replicates_list, smooth_value, range_value, p, n, current_figname, current_npfilename)

def plot_timing_vs_range_values(smooth_value, range_values, p, n, figname, npfilename):

    fig, ax = plt.subplots()
    timings = np.zeros((len(plist),2))
    for i in range(0, len(range_values)):

        timings[i,0] = plist[i]
        timings[i,1] = return_timing(1, smooth_value, range_values[i], p, n)


    np.save(npfilename, timings)
    ax.plot(plist, timings[:,1])
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Evaluation Time (Seconds)")
    plt.savefig(figname)


replicates_list = [1,2]
smooth_value = 1.5
range_value = 3.0
range_values = [1.0,2.0,3.0,4.0,5.0]
n = 32
p = .05
figname = "data/model4/timings_random05_smooth_1.5_range_"
npfilename = "data/model4/timing_array_random05_smooth_1.5_range_"
plot_sample_size_vs_time_multiple_percentages(replicates_list, smooth_value, range_values, p, n, figname, npfilename)
figname = "data/model4/timings_random05_smooth_1.5_range_1_5.png"
npfilename = "data/model4/timing_array_random05_smooth_1.5_ranges_1_5.npy"
plot_timing_vs_range_values(smooth_value, range_values, p, n, figname, npfilename)