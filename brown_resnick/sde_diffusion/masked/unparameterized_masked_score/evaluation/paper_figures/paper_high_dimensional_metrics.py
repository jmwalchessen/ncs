import numpy as np
import torch as th
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
import pandas as pd
from append_directories import *

def compute_maximum_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n, qs, range_value,
                                      smooth):

    ncs_images = np.zeros((len(qs),nrep,n,n))
    true_images = np.zeros((len(qs),nrep,n,n))
    ncs_maximums = np.zeros((len(qs),nrep))
    true_maximums = np.zeros((len(qs),nrep))

    for i in range(len(qs)):

        ncs_images[i,:,:,:] = np.load((ncs_images_file + str(qs[i]) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(true_images_file))).reshape((nrep,n,n))
        ncs_maximums[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maximums[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(qs)):
        ncspdd = pd.DataFrame(ncs_maximums[i,:], columns = None)
        truepdd = pd.DataFrame(true_maximums[i,:], columns = None)
        axes[i] = sns.kdeplot(data = truepdd, palette=['blue'], ax = axes[i])
        axes[i] = sns.kdeplot(data = ncspdd, palette=['orange'], ax = axes[i])
        axes[i].set_xlim((0,15))
        axes[i].legend(labels = ['true', 'NCS'])
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()

def compute_minimum_summary_statistic(ncs_images_file, true_images_file, figname, nrep,
                                      n, qs, range_value, smooth):

    ncs_images = np.zeros((len(qs),nrep,n,n))
    true_images = np.zeros((len(qs),nrep,n,n))
    ncs_mins = np.zeros((len(qs),nrep))
    true_mins = np.zeros((len(qs),nrep))

    for i in range(len(qs)):

        ncs_images[i,:,:,:] = np.load((ncs_images_file + str(qs[i]) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(true_images_file))).reshape((nrep,n,n))
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(qs)):

        ncspdd = pd.DataFrame(ncs_mins[i,:], columns = None)
        truepdd = pd.DataFrame(true_mins[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette=['blue'], ax = axes[i])
        sns.kdeplot(data = ncspdd, palette=['orange'], ax = axes[i])
        axes[i].legend(labels = ['true', 'NCS'])
        axes[i].set_xlim(-10,0)

    
    plt.savefig(figname)
    plt.clf()


def compute_quantile_summary_statistic(ncs_images_file, true_images_file, figname, nrep,
                                       n, qs, q, range_value, smooth):

    ncs_images = np.zeros((len(qs),nrep,n,n))
    true_images = np.zeros((len(qs),nrep,n,n))
    ncs_quants = np.zeros((len(qs),nrep))
    true_quants = np.zeros((len(qs),nrep))

    for i in range(len(qs)):

        ncs_images[i,:,:,:] = np.load((ncs_images_file + str(qs[i]) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(true_images_file))).reshape((nrep,n,n))
        ncs_quants[i,:] = np.quantile(ncs_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)
        true_quants[i,:] = np.quantile(true_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(qs)):
    
        ncspdd = pd.DataFrame(ncs_quants[i,:], columns = None)
        truepdd = pd.DataFrame(true_quants[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = ncspdd, palette =['orange'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'NCS'])
        axes[i].set_xlim((-2,2))
    
    plt.savefig(figname)
    plt.clf()

def compute_summation_statistic(ncs_images_file, true_images_file, figname, nrep,
                                n, qs, range_value, smooth):

    ncs_images = np.zeros((len(qs),nrep,n,n))
    true_images = np.zeros((len(qs),nrep,n,n))
    ncs_summation = np.zeros((len(qs),nrep))
    true_summation = np.zeros((len(qs),nrep))

    for i in range(len(qs)):

        ncs_images[i,:,:,:] = np.load((ncs_images_file + str(qs[i]) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(true_images_file))).reshape((nrep,n,n))
        ncs_summation[i,:] = np.sum(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_summation[i,:] = np.sum(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(qs)):
    
        ncspdd = pd.DataFrame(ncs_summation[i,:], columns = None)
        truepdd = pd.DataFrame(true_summation[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = ncspdd, palette =['orange'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'NCS'])
        axes[i].set_xlim((0,7500))
    
    plt.savefig(figname)
    plt.clf()

def compute_absolute_summation_statistic(ncs_images_file, true_images_file, figname, nrep,
                                         n, qs, range_value, smooth):

    ncs_images = np.zeros((len(qs),nrep,n,n))
    true_images = np.zeros((len(qs),nrep,n,n))
    ncs_abs_summation = np.zeros((len(qs),nrep))
    true_abs_summation = np.zeros((len(qs),nrep))

    for i in range(len(qs)):

        ncs_images[i,:,:,:] = np.load((ncs_images_file + str(qs[i]) + ".npy"))
        true_images[i,:,:,:] = np.log(np.load(true_images_file))
        ncs_abs_summation[i,:] = np.sum(np.abs(ncs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(qs)):
    
        ncspdd = pd.DataFrame(ncs_abs_summation[i,:], columns = None)
        truepdd = pd.DataFrame(true_abs_summation[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = ncspdd, palette =['orange'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'NCS'])
        axes[i].set_xlim((0,5000))
    
    plt.savefig(figname)
    plt.clf()


eval_folder = append_directory(2)
ncs_images_file = (eval_folder + "/extremal_coefficient_and_high_dimensional_statistics/data/ncs/model4/brown_resnick_ncs_images_range_3.0_smooth_1.5_4000_random")
true_images_file = (eval_folder + "/extremal_coefficient_and_high_dimensional_statistics/data/true/brown_resnick_images_range_3.0_smooth_1.5_4000.npy")
n = 32
nrep = 4000
smooth = 1.5
range_value = 3.0
q = .1
qs = [.01,.05,.1,.25,.5]
figname = "figures/br_percentage_model4_max_summary_statistic.png"
compute_maximum_summary_statistic(ncs_images_file, true_images_file, figname, nrep, n, qs,
                                  range_value, smooth)
figname = "figures/br_percentage_model4_min_summary_statistic.png"
compute_minimum_summary_statistic(ncs_images_file, true_images_file, figname, nrep,
                                      n, qs, range_value, smooth)
figname = "figures/br_percentage_model4_quantile_.1_summary_statistic.png"
compute_quantile_summary_statistic(ncs_images_file, true_images_file, figname, nrep,
                                       n, qs, q, range_value, smooth)
figname = "figures/br_percentage_model4_summation_statistic.png"
compute_summation_statistic(ncs_images_file, true_images_file, figname, nrep,
                                n, qs, range_value, smooth)
figname = "figures/br_percentage_model4_absolute_summation_statistic.png"
compute_absolute_summation_statistic(ncs_images_file, true_images_file, figname, nrep,
                                n, qs, range_value, smooth)