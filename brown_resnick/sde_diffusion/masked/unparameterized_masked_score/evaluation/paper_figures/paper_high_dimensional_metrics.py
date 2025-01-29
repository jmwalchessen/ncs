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
        axes[i].set_xlim(-5,0)

    
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
        axes[i].set_xlim((-.5,10))
    
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


def compute_fcs_vs_true_max_statistic(figname, obs):

    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    eval_folder = append_directory(2)
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_maxs = np.zeros((len(range_values),nrep))
    true_maxs = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        fcs_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/processed_unconditional_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_4000.npy")))
        true_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/true/brown_resnick_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy")))
        fcs_maxs[i,:] = np.max(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maxs[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        fcspdd = pd.DataFrame(fcs_maxs[i,:], columns = None)
        truepdd = pd.DataFrame(true_maxs[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'FCS'])
        axes[i].set_xlim((0,15))
    
    plt.savefig(figname)
    plt.clf()


def compute_fcs_vs_true_min_statistic(figname, obs):

    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    eval_folder = append_directory(2)
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_mins = np.zeros((len(range_values),nrep))
    true_mins = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        fcs_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/processed_unconditional_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_4000.npy")))
        true_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/true/brown_resnick_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy")))
        fcs_mins[i,:] = np.min(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        fcspdd = pd.DataFrame(fcs_mins[i,:], columns = None)
        truepdd = pd.DataFrame(true_mins[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'FCS'])
        axes[i].set_xlim((-35,2))
    
    plt.savefig(figname)
    plt.clf()


def compute_fcs_vs_true_quantile_statistic(figname, obs, q):

    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    eval_folder = append_directory(2)
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_quants = np.zeros((len(range_values),nrep))
    true_quants = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        fcs_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/processed_unconditional_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_4000.npy")))
        true_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/true/brown_resnick_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy")))
        fcs_quants[i,:] = np.quantile(fcs_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)
        true_quants[i,:] = np.quantile(true_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        fcspdd = pd.DataFrame(fcs_quants[i,:], columns = None)
        truepdd = pd.DataFrame(true_quants[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'FCS'])
        axes[i].set_xlim((-22,7))
    
    plt.savefig(figname)
    plt.clf()

def compute_fcs_vs_true_summation_statistic(figname, obs):

    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    eval_folder = append_directory(2)
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_summation = np.zeros((len(range_values),nrep))
    true_summation = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        fcs_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/processed_unconditional_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_4000.npy")))
        true_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/true/brown_resnick_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy")))
        fcs_summation[i,:] = np.sum((fcs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_summation[i,:] = np.sum((true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        fcspdd = pd.DataFrame(fcs_summation[i,:], columns = None)
        truepdd = pd.DataFrame(true_summation[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'FCS'])
        axes[i].set_xlim((-12000,5000))
    
    plt.savefig(figname)
    plt.clf()

def compute_fcs_vs_true_absolute_summation_statistic(figname, obs):

    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    eval_folder = append_directory(2)
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_abs_summation = np.zeros((len(range_values),nrep))
    true_abs_summation = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        fcs_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/processed_unconditional_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_4000.npy")))
        true_images[i,:,:,:] = np.log(np.load((eval_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/true/brown_resnick_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy")))
        fcs_abs_summation[i,:] = np.sum(np.abs(fcs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        fcspdd = pd.DataFrame(fcs_abs_summation[i,:], columns = None)
        truepdd = pd.DataFrame(true_abs_summation[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'FCS'])
        axes[i].set_xlim((0,10000))
    
    plt.savefig(figname)
    plt.clf()

obs = 5
figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_abs_summation.png"
compute_fcs_vs_true_absolute_summation_statistic(figname, obs)
figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_summation.png"
compute_fcs_vs_true_summation_statistic(figname, obs)
qs = [.1,.2,.8,.9]
for q in qs:
    figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_quant" + str(q) + ".png"
    compute_fcs_vs_true_quantile_statistic(figname, obs, q)
figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_min.png"
compute_fcs_vs_true_min_statistic(figname, obs)
figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_max.png"
compute_fcs_vs_true_max_statistic(figname, obs)