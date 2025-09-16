import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_maximum_summary_statistic(approx_images_file, true_images_file, approx_label,
                                      figname, nrep, n):

    approx_images = np.load(approx_images_file)
    if(approx_label == "FCS"):
        approx_images = np.log(approx_images)
    true_images = np.log(np.load(true_images_file))
    approx_maximums = np.max(approx_images.reshape((nrep, n**2)), axis = 1)
    true_maximums = np.max(true_images.reshape((nrep, n**2)), axis = 1)

    fig, ax = plt.subplots()
    approxpdd = pd.DataFrame(approx_maximums, columns = None)
    truepdd = pd.DataFrame(true_maximums, columns = None)
    sns.kdeplot(data = truepdd, palette=['blue'], ax = ax)
    sns.kdeplot(data = approxpdd, palette=['orange'], ax = ax)
    ax.legend(labels = ['true', approx_label])
    plt.savefig(figname)
    plt.clf()

def compute_minimum_summary_statistic(approx_images_file, true_images_file, approx_label, figname, nrep, n):

    approx_images = np.load(approx_images_file)
    if(approx_label == "FCS"):
        approx_images = np.log(approx_images)
    true_images = np.log(np.load(true_images_file))
    approx_mins = np.min(approx_images.reshape((nrep, n**2)), axis = 1)
    true_mins = np.min(true_images.reshape((nrep, n**2)), axis = 1)
    fig, ax = plt.subplots()
    approxpdd = pd.DataFrame(approx_mins, columns = None)
    truepdd = pd.DataFrame(true_mins, columns = None)
    sns.kdeplot(data = truepdd, palette=['blue'], ax = ax)
    sns.kdeplot(data = approxpdd, palette=['orange'], ax = ax)
    ax.legend(labels = ['true', approx_label])
    plt.savefig(figname)
    plt.clf()


def compute_quantile_summary_statistic(approx_images_file, true_images_file, approx_label,
                                       figname, nrep, n, q):

    approx_images = np.load(approx_images_file)
    if(approx_label == "FCS"):
        approx_images = np.log(approx_images)
    true_images = np.log(np.load(true_images_file))
    approx_quants = np.quantile(approx_images.reshape((nrep, n**2)), q = q, axis = 1)
    true_quants = np.quantile(true_images.reshape((nrep, n**2)), q = q, axis = 1)
    fig, ax = plt.subplots()
    approxpdd = pd.DataFrame(approx_quants, columns = None)
    truepdd = pd.DataFrame(true_quants, columns = None)
    sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
    sns.kdeplot(data = approxpdd, palette =['orange'], ax = ax, legend = True)
    ax.legend(labels = ['true', approx_label])
    plt.savefig(figname)
    plt.clf()



def compute_high_dimensional_summary_metrics(qs, approx_images_file, true_images_file, approx_label,
                                             nrep, n, m):
    
    minimum_figname = ("high_dimensional_summary_metrics/fcs/fcs_vs_true_" + str(nrep)
                                 + "_obs_" + str(m) + "_minimum.png")
    compute_minimum_summary_statistic(approx_images_file, true_images_file, approx_label,
                                      minimum_figname, nrep, n)
    maximum_figname = ("high_dimensional_summary_metrics/fcs/fcs_vs_true_" + str(nrep)
                                 + "_obs_" + str(m) + "_maximum.png")
    compute_maximum_summary_statistic(approx_images_file, true_images_file, approx_label,
                                      maximum_figname, nrep, n)
    for q in qs:
        current_quant_figname = ("high_dimensional_summary_metrics/fcs/fcs_vs_true_" + str(nrep)
                                 + "_obs_" + str(m) + "_quant_" + str(q) + ".png")
        compute_quantile_summary_statistic(approx_images_file, true_images_file, approx_label,
                                           current_quant_figname, nrep, n, q)
    

def compute_high_dimensional_summary_metrics_with_variables():
    
    qs = [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
    n = 32
    m = 5
    nrep = 4000
    approx_label = "FCS"
    approx_images_file = ("data/fcs/processed_unconditional_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_4000.npy")
    true_images_file = ("data/true/brown_resnick_images_range_3.0_smooth_1.5_4000.npy")
    compute_high_dimensional_summary_metrics(qs, approx_images_file,
                                            true_images_file, approx_label,
                                            nrep, n, m)