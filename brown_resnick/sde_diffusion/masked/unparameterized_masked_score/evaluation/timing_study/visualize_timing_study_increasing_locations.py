import numpy as np
import matplotlib.pyplot as plt
import torch as th



def visualize_ncs_and_fcs_timing_array(ncs_timing_file, fcs_time_array, tnrep, figname):

    ncs_time_array = np.load("data/model4/" + ncs_timing_file)
    fcs_time_array = np.load("data/model4/" + fcs_time_array)
    ncs_spatial_location_numbers = ncs_time_array[:,0]
    fcs_spatial_location_numbers = np.array([i for i in range(1,8)])
    ncs_time_avg = np.mean(ncs_time_array[:,1:(tnrep+1)], axis = 1)
    fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].set_xlim((0,520))
    ax[1].set_xlim((0,520))
    ax[0].plot(fcs_spatial_location_numbers, fcs_time_avg, color = "purple")
    ax[1].plot(ncs_spatial_location_numbers, ncs_time_avg, color = 'orange')
    plt.savefig("visualizations/" + figname)
    plt.clf()


tnrep = 50
figname = "timing_plot_ncs_and_fcs_increasing_number_of_observed_locations_niasra_node_6_1_conditional_simulation_1_7_tnrep_50.png"
ncs_timing_array = "increasing_number_of_observed_locations_timing_array_niasra_node_6_1_conditional_simulation_1_7_tnrep_50.npy"

visualize_ncs_and_fcs_timing_array(ncs_timing_file, fcs_time_array, tnrep, figname)

