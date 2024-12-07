import matplotlib.pyplot as plt
import torch as th
import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid

evaluation_folder = append_directory(2)

def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

def visualize_ncs_and_true_extremal_coefficient_multiple_ranges(range_values, smooth, bins, figname, nrep):
    
    extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    ncs_extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    for i in range(len(range_values)):

        extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_summary_statistics/data/true/extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(round(range_values[i])) + "_nbins_" + str(bins) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_summary_statistics/data/ncs/model4/extremal_coefficient_range_"
                                            + str(range_values[i]) + "_smooth_" + str(smooth) 
                                            + "_bins_" + str(bins) + "_" + str(nrep) + ".npy"))

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)
    h = extremal_matrices[0,:,0]

    for i in range(len(range_values)):

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


range_values = [1.0,2.0,3.0,4.0,5.0]
smooth = 1.5
bins = 100
nrep = 4000
figname = "figures/paper_ncs_vs_true_extremal_coefficient.png"
visualize_ncs_and_true_extremal_coefficient_multiple_ranges(range_values, smooth, bins, figname, nrep)