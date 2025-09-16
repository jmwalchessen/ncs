import numpy as np
import matplotlib.pyplot as plt
import torch as th



def visualize_ncs_and_fcs_timing_division_with_exponential_linear_extrapolation(fcs_type):

    figname = "ncs_fcs_" + str(fcs_type) + "_timing_division_with_exp_linear_extrapolation_1_7_50_azure_gpu"
    range_values = [i for i in range(1,6)]
    tnrep = 50
    obs = [i for i in range(1,8)]
    model_versions = [6,7,5,8,9]
    ncs_times = np.zeros((len(range_values),len(obs),tnrep))
    ncs_time_avg = np.zeros((len(range_values), len(obs)))
    ncs_time_div = np.zeros((len(range_values), len(obs)))
    ncs_time_extrp = np.ones((len(range_values), 4))
    fcs_times = np.zeros((len(range_values),len(obs),tnrep))
    fcs_time_avg = np.zeros((len(range_values), len(obs)))
    fcs_time_div = np.zeros((len(range_values), len(obs)))
    lma = [.25,.26,.23,.24,.23]
    lmb = [-.25,-.24,-.22,-.24,-.24]

    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_div[i,:] = (1/ncs_time_avg[i,0])*ncs_time_avg[i,:]
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_div[i,:] = (1/fcs_time_avg[i,0])*fcs_time_avg[i,:]

    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    extr = [i for i in range(7,11)]
    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.plot(obs, fcs_time_div[i,:], color = "purple", label = "FCS")
        x = [i for i in range(1,11)]
        y = [np.exp(lma[i]*x[j]+lmb[i]) for j in range(len(x))]
        ax.plot(x, y, color = "purple", linestyle = "dashed")
        ax.plot(obs, ncs_time_div[i,:], color = 'orange', label = "NCS")
        ax.legend(['FCS',"FCS (extrapolated)","NCS"], fontsize = 15)
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange')
        ax.set_ylim((0,10))
        ax.set_xticks([0,2,4,6,8,10],[0,2,4,6,8,10], fontsize = 15)
        ax.set_yticks([0,2,4,6,8,10],[0,2,4,6,8,10], fontsize = 15)
        ax.set_ylabel("Average Time Ratio", fontsize = 15)
        ax.set_xlabel("Number of Observations", fontsize = 15)
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()


