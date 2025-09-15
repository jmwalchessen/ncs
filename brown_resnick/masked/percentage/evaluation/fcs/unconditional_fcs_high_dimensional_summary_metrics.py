import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from append_directories import *

def compute_maximum_summary_statistic(figname, nrep, n, range_values,
                                      obs):
    
    ncs_images = np.zeros((len(range_values),nrep,n,n))
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    ncs_maximums = np.zeros((len(range_values),nrep))
    true_maximums = np.zeros((len(range_values),nrep))
    fcs_maximums = np.zeros((len(range_values),nrep))
    fcs_minimums = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):
        ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_values[i]-1))
        ncs_file_name = "diffusion/unconditional_fixed_ncs_images_range_" + str(range_values[i]) + "_smooth_1.5_" + str(nrep) + ".npy"
        ncs_images[i,:,:,:] = np.load((ref_folder + "/" + ncs_file_name))
        fcs_file_name = "processed_unconditional_fcs_fixed_mask_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"
        fcs_images[i,:,:,:] = np.log(np.load((ref_folder + "/" + fcs_file_name)))
        true_images_file = "true_brown_resnick_images_range_" + str(int(range_values[i])) + "_smooth_1.5_" + str(nrep) + ".npy"
        true_images[i,:,:,:] = (np.log(np.load((ref_folder + "/" + true_images_file)))).reshape((nrep,n,n))
        ncs_maximums[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maximums[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_maximums[i,:] = np.max(fcs_images[i,:,:,:].reshape((nrep,n**2)), axis = 1)
        fcs_minimums[i,:] = np.min(fcs_images[i,:,:,:].reshape((nrep,n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
        ncspdd = pd.DataFrame(ncs_maximums[i,:], columns = None)
        truepdd = pd.DataFrame(true_maximums[i,:], columns = None)
        fcspdd = pd.DataFrame(fcs_maximums[i,:], columns = None)
        axes[i] = sns.kdeplot(data = truepdd, palette=['blue'], ax = axes[i])
        axes[i] = sns.kdeplot(data = ncspdd, palette=['orange'], ax = axes[i])
        axes[i] = sns.kdeplot(data = fcspdd, palette=['purple'], ax = axes[i])
        axes[i].set_xlim((0,15))
        axes[i].legend(labels = ['true', 'NCS', 'FCS'])
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()

def compute_minimum_summary_statistic(figname, nrep,
                                      n, range_values, obs):

    ncs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    ncs_mins = np.zeros((len(range_values),nrep))
    true_mins = np.zeros((len(range_values),nrep))
    fcs_mins = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):
        ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_values[i]-1))
        ncs_file_name = "diffusion/unconditional_fixed_ncs_images_range_" + str(range_values[i]) + "_smooth_1.5_" + str(nrep) + ".npy"
        ncs_images[i,:,:,:] = np.load((ref_folder + "/" + ncs_file_name))
        fcs_file_name = "processed_unconditional_fcs_fixed_mask_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"
        fcs_images[i,:,:,:] = np.log(np.load((ref_folder + "/" + fcs_file_name)))
        true_images_file = "true_brown_resnick_images_range_" + str(int(range_values[i])) + "_smooth_1.5_" + str(nrep) + ".npy"
        true_images[i,:,:,:] = (np.log(np.load((ref_folder + "/" + true_images_file)))).reshape((nrep,n,n))
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_mins[i,:] = np.min(fcs_images[i,:,:,:].reshape((nrep,n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):

        ncspdd = pd.DataFrame(ncs_mins[i,:], columns = None)
        truepdd = pd.DataFrame(true_mins[i,:], columns = None)
        fcspdd = pd.DataFrame(fcs_mins[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette=['blue'], ax = axes[i])
        sns.kdeplot(data = ncspdd, palette=['orange'], ax = axes[i])
        sns.kdeplot(data = fcspdd, palette=['purple'], ax = axes[i])
        axes[i].legend(labels = ['true', 'NCS', 'FCS'])
        axes[i].set_xlim(-32,0)

    
    plt.savefig(figname)
    plt.clf()


def compute_quantile_summary_statistic(figname, nrep,
                                       n, q, range_values, obs, min_value, max_value):

    ncs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    ncs_quants = np.zeros((len(range_values),nrep))
    true_quants = np.zeros((len(range_values),nrep))
    fcs_quants = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)): 

        ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_values[i]-1))
        ncs_file_name = "diffusion/unconditional_fixed_ncs_images_range_" + str(range_values[i]) + "_smooth_1.5_" + str(nrep) + ".npy"
        ncs_images[i,:,:,:] = np.load((ref_folder + "/" + ncs_file_name))
        fcs_file_name = "processed_unconditional_fcs_fixed_mask_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"
        fcs_images[i,:,:,:] = np.log(np.load((ref_folder + "/" + fcs_file_name)))
        true_images_file = "true_brown_resnick_images_range_" + str(int(range_values[i])) + "_smooth_1.5_" + str(nrep) + ".npy"
        true_images[i,:,:,:] = (np.log(np.load((ref_folder + "/" + true_images_file)))).reshape((nrep,n,n))
        ncs_quants[i,:] = np.quantile(ncs_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)
        true_quants[i,:] = np.quantile(true_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)
        fcs_quants[i,:] = np.quantile(fcs_images[i,:,:,:].reshape((nrep, n**2)), q = q, axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        ncspdd = pd.DataFrame(ncs_quants[i,:], columns = None)
        truepdd = pd.DataFrame(true_quants[i,:], columns = None)
        fcspdd = pd.DataFrame(fcs_quants[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = ncspdd, palette =['orange'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].set_xlim(min_value,max_value)
        axes[i].legend(labels = ['true', 'NCS', 'FCS'])
    
    plt.savefig(figname)
    plt.clf()

def compute_summation_statistic(figname, nrep,
                                n, range_values, obs):

    ncs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    ncs_summation = np.zeros((len(range_values),nrep))
    true_summation = np.zeros((len(range_values),nrep))
    fcs_summation = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_values[i]-1))
        ncs_file_name = "diffusion/unconditional_fixed_ncs_images_range_" + str(range_values[i]) + "_smooth_1.5_" + str(nrep) + ".npy"
        ncs_images[i,:,:,:] = np.load((ref_folder + "/" + ncs_file_name))
        fcs_file_name = "processed_unconditional_fcs_fixed_mask_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"
        fcs_images[i,:,:,:] = np.log(np.load((ref_folder + "/" + fcs_file_name)))
        true_images_file = "true_brown_resnick_images_range_" + str(int(range_values[i])) + "_smooth_1.5_" + str(nrep) + ".npy"
        true_images[i,:,:,:] = (np.log(np.load((ref_folder + "/" + true_images_file)))).reshape((nrep,n,n))
        ncs_summation[i,:] = np.sum(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_summation[i,:] = np.sum(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_summation[i,:] = np.sum(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        ncspdd = pd.DataFrame(ncs_summation[i,:], columns = None)
        truepdd = pd.DataFrame(true_summation[i,:], columns = None)
        fcspdd = pd.DataFrame(fcs_summation[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = ncspdd, palette =['orange'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette = ['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'NCS', 'FCS'])
        axes[i].set_xlim((-10000,7500))
    
    plt.savefig(figname)
    plt.clf()

def compute_absolute_summation_statistic(figname, nrep,
                                         n, range_values, obs):

    ncs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    ncs_abs_summation = np.zeros((len(range_values),nrep))
    true_abs_summation = np.zeros((len(range_values),nrep))
    fcs_abs_summation = np.zeros((len(range_values),nrep))

    for i in range(len(range_values)):

        ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_values[i]-1))
        ncs_file_name = "diffusion/unconditional_fixed_ncs_images_range_" + str(range_values[i]) + "_smooth_1.5_" + str(nrep) + ".npy"
        ncs_images[i,:,:,:] = np.load((ref_folder + "/" + ncs_file_name))
        fcs_file_name = "processed_unconditional_fcs_fixed_mask_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"
        fcs_images[i,:,:,:] = np.log(np.load((ref_folder + "/" + fcs_file_name)))
        true_images_file = "true_brown_resnick_images_range_" + str(int(range_values[i])) + "_smooth_1.5_" + str(nrep) + ".npy"
        true_images[i,:,:,:] = (np.log(np.load((ref_folder + "/" + true_images_file)))).reshape((nrep,n,n))
        ncs_abs_summation[i,:] = np.sum(np.abs(ncs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        fcs_abs_summation[i,:] = np.sum(np.abs(fcs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)

    for i in range(len(range_values)):
    
        ncspdd = pd.DataFrame(ncs_abs_summation[i,:], columns = None)
        truepdd = pd.DataFrame(true_abs_summation[i,:], columns = None)
        fcspdd = pd.DataFrame(fcs_abs_summation[i,:], columns = None)
        sns.kdeplot(data = truepdd, palette = ['blue'], ax = axes[i], legend = True)
        sns.kdeplot(data = ncspdd, palette =['orange'], ax = axes[i], legend = True)
        sns.kdeplot(data = fcspdd, palette =['purple'], ax = axes[i], legend = True)
        axes[i].legend(labels = ['true', 'NCS', 'FCS'])
        axes[i].set_xlim((0,8000))
    
    plt.savefig(figname)
    plt.clf()

def visualzie_max_with_variables():

    obs_numbers = [i for i in range(1,8)]
    nrep = 4000
    n = 32
    range_values = [float(i) for i in range(1,6)]
    
    for obs in obs_numbers:
        obs_folder = "data/unconditional/fixed_locations/obs" + str(obs)
        figname = (obs_folder + "/unconditional_max_obs_" + str(obs) + ".png")
        compute_maximum_summary_statistic(figname, nrep, n, range_values,
                                      obs)

def visualzie_min_with_variables():

    obs_numbers = [i for i in range(1,8)]
    nrep = 4000
    n = 32
    range_values = [float(i) for i in range(1,6)]
    
    for obs in obs_numbers:
        obs_folder = "data/unconditional/fixed_locations/obs" + str(obs)
        figname = (obs_folder + "/unconditional_min_obs_" + str(obs) + ".png")
        compute_minimum_summary_statistic(figname, nrep, n, range_values,
                                      obs)

def visualzie_quantile_with_variables(q, min_value, max_value):

    obs_numbers = [i for i in range(1,8)]
    nrep = 4000
    n = 32
    range_values = [float(i) for i in range(1,6)]
    
    for obs in obs_numbers:
        print(obs)
        obs_folder = "data/unconditional/fixed_locations/obs" + str(obs)
        figname = (obs_folder + "/unconditional_quantile_" + str(q) + "_obs_" + str(obs) + ".png")
        compute_quantile_summary_statistic(figname, nrep, n, q, range_values,
                                      obs, min_value, max_value)


def visualzie_summation_with_variables():

    obs_numbers = [i for i in range(1,8)]
    nrep = 4000
    n = 32
    range_values = [float(i) for i in range(1,6)]
    
    for obs in obs_numbers:
        obs_folder = "data/unconditional/fixed_locations/obs" + str(obs)
        figname = (obs_folder + "/unconditional_summation_obs_" + str(obs) + ".png")
        compute_summation_statistic(figname, nrep, n, range_values,
                                      obs)
        
def visualzie_abs_summation_with_variables():

    obs_numbers = [i for i in range(1,8)]
    nrep = 4000
    n = 32
    range_values = [float(i) for i in range(1,6)]
    
    for obs in obs_numbers:
        obs_folder = "data/unconditional/fixed_locations/obs" + str(obs)
        figname = (obs_folder + "/unconditional_abs_summation_obs_" + str(obs) + ".png")
        compute_absolute_summation_statistic(figname, nrep, n, range_values,
                                      obs)



visualzie_quantile_with_variables(.05, -30, 0)
visualzie_quantile_with_variables(.1, -30, 0)
visualzie_quantile_with_variables(.2, -30, 0)
visualzie_quantile_with_variables(.3, -30, 0)
visualzie_quantile_with_variables(.4, -30, 2)
visualzie_quantile_with_variables(.5, -30, 2)
visualzie_quantile_with_variables(.6, -30, 4)
visualzie_quantile_with_variables(.7, -30, 4)
visualzie_quantile_with_variables(.8, -30, 6)
visualzie_quantile_with_variables(.9, -30, 6)
visualzie_quantile_with_variables(.95, -30, 8)
visualzie_quantile_with_variables(.99, -30, 8)
visualzie_summation_with_variables()
visualzie_abs_summation_with_variables()