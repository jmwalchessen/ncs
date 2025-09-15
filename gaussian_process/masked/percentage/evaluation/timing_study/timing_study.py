import numpy as np
import matplotlib.pyplot as plt
from append_directories import *
import sys
import os
import torch as th
evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *
import time


def return_timing(number_of_replicates, variance, lengthscale, p, n):

    beta_min = .1
    beta_max = 20
    N = 1000
    device = "cuda:0"
    mask = (th.bernoulli(p*th.ones((1,1,n,n)))).float().to(device)
    score_model = load_score_model("model7_beta_min_max_01_20_random01525_variance_1.5_lengthscale_3_channel_mask.pth", "eval")
    vpsde = load_sde(beta_min, beta_max, N)
    seed_value = int(np.random.randint(0, 1000000, 1))
    y = (th.from_numpy(generate_gaussian_process(minX = -10, maxX = 10, minY = -10, maxY = 10, n = 32, variance = variance,
                                                 lengthscale = lengthscale, number_of_replicates = number_of_replicates,
                                                 seed_value = seed_value))).float().to(device)
    start = time.time()
    ncs_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                            y, n, number_of_replicates)
    end = time.time()
    time_elapsed = end - start
    return time_elapsed


def plot_sample_size_vs_time(replicates_list, variance, lengthscale, p, n, figname, npfilename):


    fig, ax = plt.subplots()
    timings = np.zeros((len(replicates_list),2))
    for i in range(0, len(replicates_list)):

        timings[i,0] = replicates_list[i]
        timings[i,1] = return_timing(replicates_list[i], variance, lengthscale, p, n)


    np.save(npfilename, timings)
    ax.plot(replicates_list, timings[:,1])
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Evaluation Time (Seconds)")
    plt.savefig(figname)

def plot_sample_size_vs_time_multiple_percentages(replicates_list, variance, lengthscale, plist, n, figname, npfilename):


    for p in plist:
        current_figname = (figname + str(p) + ".png")
        current_npfilename = (npfilename + str(p) + ".npy")
        plot_sample_size_vs_time(replicates_list, variance, lengthscale, p, n, current_figname, current_npfilename)

def plot_timing_vs_percentages(variance, lengthscale, plist, n, figname, npfilename):

    fig, ax = plt.subplots()
    timings = np.zeros((len(plist),2))
    for i in range(0, len(plist)):

        timings[i,0] = plist[i]
        timings[i,1] = return_timing(1, variance, lengthscale, plist[i], n)


    np.save(npfilename, timings)
    ax.plot(plist, timings[:,1])
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Evaluation Time (Seconds)")
    plt.savefig(figname)


replicates_list = [1,2]
variance = 1.5
lengthscale = 3.0
n = 32
plist = [.01,.05,.1,.25,.5]
figname = "data/model7/timings_random05_variance_1.5_lengthscale_"
npfilename = "data/model7/timing_array__variance_1.5_lengthscale_"
plot_sample_size_vs_time_multiple_percentages(replicates_list, variance, lengthscale, plist, n, figname, npfilename)
figname = "data/model7/timings_variance_1.5_lengthscale_3_varying_percentages_0105102550.png"
npfilename = "data/model7/timing_array_variance_1.5_lengthscale_3_varying_percentages_0105102550.npy"
plot_timing_vs_percentages(variance, lengthscale, plist, n, figname, npfilename)
    