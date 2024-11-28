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

def visualize_ncs_extremal_coefficient(ref_folder, base_ncs_file, range_value, smooth_value, bins, nrep, figname):

    ext_ncs_images = np.load((ref_folder + "/" + base_ncs_file + "_" + str(range_value) +
                              "_smooth_" + str(smooth_value) + "_" + str(nrep) + ".npy"))
    



smooth = 1.5
range_value = 5
bins = 100
figname = ("data/true/extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(range_value) + "_nbins_" + str(bins) + ".png")
extremal_matrix = load_numpy_file(("data/true/extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(range_value) + "_nbins_" + str(bins) + ".npy"))
visualize_extremal_coefficient(extremal_matrix, range_value, smooth, bins, figname)

    
