import matplotlib.pyplot as plt
import torch as th
import numpy as np


def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

def return_ref_folder(range_value, obs):

    ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_value-1))
    return ref_folder

def visualize_fcs_ncs_true_extremal_coefficient_multiple_ranges(range_values, smooth, bins, figname, nrep, obs):
    
    extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    fcs_extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    ncs_extremal_matrices = np.zeros((len(range_values), (bins+1),3))

    for i in range(len(range_values)):
        ref_folder = return_ref_folder(range_values[i],obs)
        extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/true_extremal_coefficient_range_"
                                                    + str(range_values[i]) + "_smooth_" + str(smooth) + "_nbins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/brown_resnick_ncs_extremal_matrix_bins_100_obs" + str(obs) + "_range_" + str(range_values[i]) + "_smooth_1.5_" + str(nrep) + ".npy"))
        fcs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/extremal_coefficient_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy"))

    fig, axes = plt.subplots(figsize=(10,2.5), nrows = 1, ncols = 5, sharey=True)
    h = extremal_matrices[0,:,0]

    for i in range(len(range_values)):

        ext_coeff = 2-extremal_matrices[i,:,2]
        fcs_ext_coeff = 2-fcs_extremal_matrices[i,:,2]
        ncs_ext_coeff = 2-ncs_extremal_matrices[i,:,2]
        axes[i].plot(h, ext_coeff, "blue")
        axes[i].plot(h, fcs_ext_coeff, "purple", linestyle = "dashed")
        axes[i].plot(h, ncs_ext_coeff, "orange", linestyle = "dashed")
        axes[i].set_xlabel("Distance Lag (h)")
        axes[i].set_ylabel("2-Extremal Coefficient")
        axes[i].legend(labels = ['true', 'FCS'])
    
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)
    


def visualize_fcs_ncs_true_extremal_coefficient_with_variables():

    obs_numbers = [i for i in range(1,8)]
    range_values = [float(i) for i in range(1,6)]
    smooth = 1.5
    bins = 100
    nrep = 4000
    for obs in obs_numbers:
        obs_folder = "data/unconditional/fixed_locations/obs" + str(obs)
        figname = (obs_folder + "/unconditional_fixed_locations_extremal_coefficient_nbins_" + str(bins) + "_visualization_obs_" + str(obs) + ".png")
        visualize_fcs_ncs_true_extremal_coefficient_multiple_ranges(range_values, smooth, bins, figname, nrep, obs)


visualize_fcs_ncs_true_extremal_coefficient_with_variables()