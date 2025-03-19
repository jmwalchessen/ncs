import numpy as np
import matplotlib.pyplot as plt
import torch as th



def visualize_ncs_and_fcs_timing_array(figname, fcs_type):

    range_values = [i for i in range(1,6)]
    tnrep = 50
    obs = [i for i in range(1,8)]
    model_versions = [6,7,5,8,9]
    ncs_times = np.zeros((len(range_values),len(obs),tnrep))
    ncs_time_avg = np.zeros((len(range_values), len(obs)))
    fcs_times = np.zeros((len(range_values),len(obs),tnrep))
    fcs_time_avg = np.zeros((len(range_values), len(obs)))
    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.set_ylim((0,30))
        ax.plot(obs, fcs_time_avg[i,:], color = "purple")
        ax.plot(obs, ncs_time_avg[i,:], color = 'orange')
        print(ncs_time_avg[i,:])
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()

def visualize_ncs_and_fcs_timing_difference(figname, fcs_type):

    range_values = [i for i in range(1,6)]
    tnrep = 50
    obs = [i for i in range(1,8)]
    model_versions = [6,7,5,8,9]
    ncs_times = np.zeros((len(range_values),len(obs),tnrep))
    ncs_time_avg = np.zeros((len(range_values), len(obs)))
    ncs_time_diff = np.zeros((len(range_values), len(obs)))
    fcs_times = np.zeros((len(range_values),len(obs),tnrep))
    fcs_time_avg = np.zeros((len(range_values), len(obs)))
    fcs_time_diff = np.zeros((len(range_values), len(obs)))
    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_diff[i,:] = np.concatenate([np.zeros((1)), np.subtract(ncs_time_avg[i,1:7], ncs_time_avg[i,0:6])], axis = 0)
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_diff[i,:] = np.concatenate([np.zeros((1)), np.subtract(fcs_time_avg[i,1:7], fcs_time_avg[i,0:6])], axis = 0)
    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.set_ylim((-2,20))
        ax.plot(obs, fcs_time_diff[i,:], color = "purple")
        ax.plot(obs, ncs_time_diff[i,:], color = 'orange')
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()

def visualize_ncs_and_fcs_timing_division(figname, fcs_type):

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
    fcs_time_extrp = np.zeros((len(range_values), 4))

    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_div[i,:] = (1/ncs_time_avg[i,0])*ncs_time_avg[i,:]
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_div[i,:] = (1/fcs_time_avg[i,0])*fcs_time_avg[i,:]
    #extrapolation
    fcs_time_extrp[:,0] = fcs_time_div[:,6]
    ncs_time_extrp[:,0] = ncs_time_div[:,6]
    fcs_time_extrp[:,1] = fcs_time_extrp[:,0]+.5*fcs_time_extrp[:,0]
    ncs_time_extrp[:,1] = ncs_time_div[:,6]
    fcs_time_extrp[:,2] = fcs_time_extrp[:,1]+.5*fcs_time_extrp[:,1]
    ncs_time_extrp[:,2] = ncs_time_div[:,6]
    fcs_time_extrp[:,3] = fcs_time_extrp[:,2]+.5*fcs_time_extrp[:,2]
    ncs_time_extrp[:,3] = ncs_time_div[:,6]
    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    extr = [i for i in range(7,11)]
    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.set_ylim((-2,20))
        ax.plot(obs, fcs_time_div[i,:], color = "purple")
        ax.plot(obs, ncs_time_div[i,:], color = 'orange')
        ax.plot(extr, fcs_time_extrp[i,:], color = "purple", linestyle = "dashed")
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange', linestyle = "dashed")
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()



fcs_type = "elapsed"
figname = "ncs_fcs_" + str(fcs_type) + "_timing_array_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_array(figname, fcs_type)
figname = "ncs_fcs_" + str(fcs_type) + "_timing_division_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_division(figname, fcs_type)

