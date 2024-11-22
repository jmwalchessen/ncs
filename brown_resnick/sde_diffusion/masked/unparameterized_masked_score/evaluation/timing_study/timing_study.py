import numpy as np
import matplotlib.pyplot as plt
from append_directories import *
import sys
import os
import torch as th
evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *
from brown_resnick_data_generation import *
import time


def return_timing(number_of_replicates, smooth_value, range_value, p, n):

    beta_min = .1
    beta_max = 20
    N = 1000
    device = "cuda:0"
    mask = (th.bernoulli(p*th.ones((1,1,n,n)))).to(device)
    score_model = load_score_model("brown", "model4_range_3.0_smooth_1.5_4000_random0.01.pth", "eval")
    vpsde = load_sde(beta_min, beta_max, N)
    seed_value = int(np.random.randint(0, 1000000, 1))
    y = generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n)
    start = time.time()
    br_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                            y, n, number_of_replicates)
    end = time.time()
    time_elapsed = end - start
    return time_elapsed


def plot_sample_size_vs_time(replicates_list, smooth_value, range_value, p, n, figname):


    fig, ax = plt.subplots()
    timings = []
    for i in range(0, len(replicates_list)):

        timings[i] = return_timing(replicates_list[i], smooth_value, range_value, p, n)


    ax.plot(replicates_list, timings)
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Evaluation Time (Seconds)")
    plt.savefig(figname)


replicates_list = [1,5,10,25,50,100,250,500,1000]
smooth_value = 1.5
range_value = 3.0
p = .05
n = 32
figname = "data/model4/timings_smooth_1.5_range_3.0_random05.png"
plot_sample_size_vs_time(replicates_list, smooth_value, range_value, p, n, figname)

    

