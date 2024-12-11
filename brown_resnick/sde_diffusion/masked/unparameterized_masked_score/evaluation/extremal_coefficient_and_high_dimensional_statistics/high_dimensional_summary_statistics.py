import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_maximum_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n):

    ncs_images = np.load(ncs_images_file)
    true_images = np.log(np.load(true_images_file))
    ncs_maximums = np.max(ncs_images.reshape((nrep, n**2)), axis = 1)
    true_maximums = np.max(true_images.reshape((nrep, n**2)), axis = 1)

    fig, ax = plt.subplots()
    ncspdd = pd.DataFrame(ncs_maximums, columns = None)
    truepdd = pd.DataFrame(true_maximums, columns = None)
    sns.kdeplot(data = truepdd, palette=['blue'], ax = ax)
    sns.kdeplot(data = ncspdd, palette=['orange'], ax = ax)
    ax.legend(labels = ['true', 'NCS'])
    plt.savefig(figname)
    plt.clf()

def compute_minimum_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n):

    ncs_images = np.load(ncs_images_file)
    true_images = np.log(np.load(true_images_file))
    ncs_mins = np.min(ncs_images.reshape((nrep, n**2)), axis = 1)
    true_mins = np.min(true_images.reshape((nrep, n**2)), axis = 1)

    fig, ax = plt.subplots()
    ncspdd = pd.DataFrame(ncs_mins, columns = None)
    truepdd = pd.DataFrame(true_mins, columns = None)
    sns.kdeplot(data = truepdd, palette=['blue'], ax = ax)
    sns.kdeplot(data = ncspdd, palette=['orange'], ax = ax)
    ax.legend(labels = ['true', 'NCS'])
    plt.savefig(figname)
    plt.clf()


def compute_quantile_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n, q):

    ncs_images = np.load(ncs_images_file)
    true_images = np.log(np.load(true_images_file))
    ncs_quants = np.quantile(ncs_images.reshape((nrep, n**2)), q = q, axis = 1)
    true_quants = np.quantile(true_images.reshape((nrep, n**2)), q = q, axis = 1)

    fig, ax = plt.subplots()
    ncspdd = pd.DataFrame(ncs_quants, columns = None)
    truepdd = pd.DataFrame(true_quants, columns = None)
    sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
    sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
    ax.legend(labels = ['true', 'NCS'])
    plt.savefig(figname)
    plt.clf()

p = .01
figname = "high_dimensional_summary_statistics/ncs/model4/maximum_summary_statistics_ncs_vs_true_4000_random" + str(p) + ".png"
ncs_images_file = "data/ncs/model4/brown_resnick_ncs_images_range_3.0_smooth_1.5_4000_random" + str(p) + ".npy"
true_images_file = "data/true/brown_resnick_images_range_3.0_smooth_1.5_4000.npy"
nrep = 4000
n = 32
compute_maximum_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n)
figname = "high_dimensional_summary_statistics/ncs/model4/minimum_summary_statistics_ncs_vs_true_4000_random" + str(p) + ".png"
compute_minimum_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n)
q = .01
figname = "high_dimensional_summary_statistics/ncs/model4/quantile_" + str(q) + "_summary_statistics_ncs_vs_true_4000_random" + str(p) + ".png"
compute_quantile_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n, q)