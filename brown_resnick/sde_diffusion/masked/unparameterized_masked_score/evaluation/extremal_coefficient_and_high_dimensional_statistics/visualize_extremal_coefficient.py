import matplotlib.pyplot as plt
import torch as th
import numpy as np


def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

#first entry is distance lag, second entry is madogram, third entry is extremal coefficient
def visualize_extremal_coefficient(extremal_matrix, range_value, smooth, bins, figname):

    h = extremal_matrix[:,0]
    ext_coeff = extremal_matrix[:,2]
    fig, ax = plt.subplots()
    ax.plot(h, ext_coeff)
    ax.set_xlabel("Distance Lag (h)")
    ax.set_ylabel("Extremal Coefficient")
    ax.set_title(("Extremal Coefficient (range = " + str(range_value) + ", smooth = "
                  + str(smooth) + ", bins = " + str(bins)))
    plt.savefig(figname)


def visualize_ncs_and_true_extremal_coefficient(extremal_matrix, ncs_extremal_matrix,
                                                range_value, smooth, bins, figname):

    h = extremal_matrix[:,0]
    ext_coeff = 2-extremal_matrix[:,2]
    ncs_ext_coeff = 2-ncs_extremal_matrix[:,2]
    fig, ax = plt.subplots()
    ax.plot(h, ext_coeff, "blue")
    ax.plot(h, ncs_ext_coeff, "orange")
    ax.set_xlabel("Distance Lag (h)")
    ax.set_ylabel("2-Extremal Coefficient")
    ax.set_title(("2-Extremal Coefficient (range = " + str(range_value) + ", smooth = "
                  + str(smooth) + ", bins = " + str(bins)))
    ax.legend(labels = ['true', 'NCS'])
    plt.savefig(figname)
    



smooth = 1.5
range_value = 3.0
bins = 100
nrep = 4000
p = .01

extremal_matrix = load_numpy_file(("data/true/extremal_coefficient_range_" + str(range_value) + "_smooth_" + 
                                  str(smooth) + "_bins_" + str(bins) + "_" + str(nrep) + ".npy"))
figname = ("extremal_coefficient/true_extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(range_value) + "_nbins_" + str(bins) + ".png")
#visualize_extremal_coefficient(extremal_matrix, range_value, smooth, bins, figname)
ncs_extremal_matrix = load_numpy_file(("data/ncs/model4/brown_resnick_ncs_extremal_matrix_bins_"
                                      + str(bins) + "_range_" + str(range_value) + "_smooth_" + str(smooth) 
                                      + "_" + str(nrep) + "_random" + str(p) + ".npy"))
figname = ("extremal_coefficient/ncs/model4/ncs_extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(range_value) + "_nbins_" + str(bins) + "_random" + str(p) + ".png")
visualize_ncs_and_true_extremal_coefficient(extremal_matrix, ncs_extremal_matrix, range_value,
                                            smooth, bins, figname)

    
