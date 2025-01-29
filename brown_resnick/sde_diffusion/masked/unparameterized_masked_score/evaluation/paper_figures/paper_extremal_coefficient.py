import matplotlib.pyplot as plt
import torch as th
import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid

evaluation_folder = append_directory(2)

def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

def visualize_ncs_and_true_extremal_coefficient_multiple_percentages(range_value, smooth, ps, bins, figname, nrep):
    
    extremal_matrices = np.zeros((len(ps), (bins+1),3))
    ncs_extremal_matrices = np.zeros((len(ps), (bins+1),3))
    for i in range(len(ps)):

        extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_dimensional_statistics/data/true/extremal_coefficient_range_"
                                                    + str(range_value) + "_smooth_" + str(smooth) + "_bins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_dimensional_statistics/data/ncs/model4/brown_resnick_ncs_extremal_matrix_bins_"
                                            + str(bins) + "_range_" + str(range_value) + "_smooth_" + str(smooth) 
                                            + "_" + str(nrep) + "_random" + str(ps[i]) + ".npy"))

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)
    h = extremal_matrices[0,:,0]

    for i in range(len(ps)):

        ext_coeff = 2-extremal_matrices[i,:,2]
        ncs_ext_coeff = 2-ncs_extremal_matrices[i,:,2]
        axes[i].plot(h, ext_coeff, "blue")
        axes[i].plot(h, ncs_ext_coeff, "orange", linestyle = "dashed")
        axes[i].set_xlabel("Distance Lag (h)")
        axes[i].set_ylabel("2-Extremal Coefficient")
        axes[i].legend(labels = ['true', 'NCS'])
    
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)

def visualize_fcs_and_true_extremal_coefficient_multiple_ranges(range_values, smooth, bins, figname, nrep, obs):
    
    extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    fcs_extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    for i in range(len(range_values)):

        extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/true/extremal_coefficient_range_"
                                                    + str(range_values[i]) + "_smooth_" + str(smooth) + "_nbins_" + str(bins) + "_" + str(nrep) + ".npy"))
        fcs_extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/extremal_coefficient_fcs_range_" + str(range_values[i]) + "_smooth_" + str(smooth) 
                                            + "_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"))

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)
    h = extremal_matrices[0,:,0]

    for i in range(len(range_values)):

        ext_coeff = 2-extremal_matrices[i,:,2]
        fcs_ext_coeff = 2-fcs_extremal_matrices[i,:,2]
        axes[i].plot(h, ext_coeff, "blue")
        axes[i].plot(h, fcs_ext_coeff, "purple", linestyle = "dashed")
        axes[i].set_xlabel("Distance Lag (h)")
        axes[i].set_ylabel("2-Extremal Coefficient")
        axes[i].legend(labels = ['true', 'FCS'])
    
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


range_values = [float(i) for i in range(1,6)]
smooth = 1.5
bins = 100
nrep = 4000
obs = 5
figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_extremal_coefficient.png"
visualize_fcs_and_true_extremal_coefficient_multiple_ranges(range_values, smooth, bins, figname, nrep, obs)