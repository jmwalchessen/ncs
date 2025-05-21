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

    bell_numbers = np.array([1,1,1,1,2,5,15,52,203,877])

    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_div[i,:] = (1/ncs_time_avg[i,0])*ncs_time_avg[i,:]
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_div[i,:] = (1/fcs_time_avg[i,0])*fcs_time_avg[i,:]
        extrt = np.array([t for t in range(1,4)])
        fcs_time_extrp[i,1:4] = (np.exp(extrt+2))**(3/4)
    print(fcs_time_div)
    print(fcs_time_extrp)
    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    extr = [i for i in range(7,11)]
    fcs_time_extrp[:,0] = fcs_time_div[:,6]
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


def visualize_ncs_and_fcs_timing_division_with_exponential_linear_extrapolation(figname, fcs_type):

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
        ax.set_ylim((-2,20))
        ax.plot(obs, fcs_time_div[i,:], color = "purple")
        ax.plot(obs, ncs_time_div[i,:], color = 'orange')
        x = [i for i in range(1,11)]
        y = [np.exp(lma[i]*x[j]+lmb[i]) for j in range(len(x))]
        print(y)
        ax.plot(x, y, color = "purple", linestyle = "dashed")
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange', linestyle = "dashed")
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()



def visualize_ncs_and_fcs_timing_division_with_exponential_linear_extrapolation_integrated(figname, fcs_type):

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
        ax.set_ylim((-2,20))
        ax.plot(obs, fcs_time_div[i,:], color = "purple")
        ax.plot(obs, ncs_time_div[i,:], color = 'orange')
        x = [i for i in range(7,11)]
        y = [np.exp(lma[i]*x[j]+lmb[i]) for j in range(len(x))]
        y[0] = fcs_time_div[i,6]
        ax.plot(x, y, color = "purple", linestyle = "dashed")
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange', linestyle = "dashed")
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()



def visualize_ncs_and_fcs_timing_division_log(figname, fcs_type):

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

    bell_numbers = np.array([1,1,1,1,2,5,15,52,203,877])

    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_div[i,:] = (1/ncs_time_avg[i,0])*ncs_time_avg[i,:]
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_div[i,:] = (1/fcs_time_avg[i,0])*fcs_time_avg[i,:]
        extrt = np.array([t for t in range(1,4)])
        fcs_time_extrp[i,1:4] = (np.exp(extrt+2))**(3/4)
    print(fcs_time_div)
    print(fcs_time_extrp)
    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    extr = [i for i in range(7,11)]
    fcs_time_extrp[:,0] = fcs_time_div[:,6]
    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.set_ylim((-2,20))
        ax.plot(obs, np.log2(fcs_time_div[i,:]), color = "purple")
        ax.plot(obs, np.log2(ncs_time_div[i,:]), color = 'orange')
        ax.plot(extr, fcs_time_extrp[i,:], color = "purple", linestyle = "dashed")
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange', linestyle = "dashed")
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()


def visualize_ncs_and_fcs_timing_division_log_with_linear_extrapolation(figname, fcs_type):

    range_values = [i for i in range(1,6)]
    tnrep = 50
    obs = [i for i in range(1,8)]
    model_versions = [6,7,5,8,9]
    extr = [i for i in range(7,11)]
    ncs_times = np.zeros((len(range_values),len(obs),tnrep))
    ncs_time_avg = np.zeros((len(range_values), len(obs)))
    ncs_time_div = np.zeros((len(range_values), len(obs)))
    ncs_time_extrp = np.zeros((len(range_values), 4))
    fcs_times = np.zeros((len(range_values),len(obs),tnrep))
    fcs_time_avg = np.zeros((len(range_values), len(obs)))
    fcs_time_div = np.zeros((len(range_values), len(obs)))
    lma = .25
    lmb = -.25

    bell_numbers = np.array([1,1,1,1,2,5,15,52,203,877])

    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_div[i,:] = (1/ncs_time_avg[i,0])*ncs_time_avg[i,:]
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_div[i,:] = (1/fcs_time_avg[i,0])*fcs_time_avg[i,:]

    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.set_ylim((-2,5))
        ax.plot(obs, np.log(fcs_time_div[i,:]), color = "purple")
        ax.plot(obs, np.log(ncs_time_div[i,:]), color = 'orange')
        x = [i for i in range(1,11)]
        y = [lma*v+lmb for v in x]
        ax.plot(x, y, color = "purple", linestyle = "dashed")
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange', linestyle = "dashed")
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()

def visualize_ncs_and_fcs_timing_division_with_bell_number(figname, fcs_type):

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

    bell_numbers = np.array([5,15,52])

    for i in range(len(range_values)):
        ncs_times[i,:,:] = (np.load(("data/model" + str(model_versions[i]) + "_range_" + str(range_values[i]) + 
                                   "_ncs_timing_array_azure_gpu_1_7_nrep_50.npy")))[:,1:(tnrep+1)]
        fcs_times[i,:,:] = np.load(("data/range_" + str(range_values[i]) + "_fcs_" + fcs_type 
                                        + "_timing_azure_gpu_1_7_tnrep_50.npy"))
        ncs_time_avg[i,:] = np.mean(ncs_times[i,:,:], axis = 1)
        ncs_time_div[i,:] = (1/ncs_time_avg[i,0])*ncs_time_avg[i,:]
        fcs_time_avg[i,:] = np.mean(fcs_times[i,:,:], axis = 1)
        fcs_time_div[i,:] = (1/fcs_time_avg[i,0])*fcs_time_avg[i,:]
        extrt = np.array([t for t in range(1,4)])
        #fcs_time_extrp[i,1:4] = (np.exp(extrt+2))**(3/4)
        fcs_time_extrp[i,1:4] = np.array([fcs_time_div[i,6] + bell_numbers[j] for j in range(3)])
    print(fcs_time_div)
    print(fcs_time_extrp)
    #fcs_time_avg = np.mean(fcs_time_array, axis = 1)
    extr = [i for i in range(7,11)]
    fcs_time_extrp[:,0] = fcs_time_div[:,6]
    for i in range(len(range_values)):
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        #ax[0].set_xlim((0,520))
        ax.set_ylim((0,20))
        ax.plot(obs, fcs_time_div[i,:], color = "purple")
        ax.plot(extr, fcs_time_extrp[i,:], color = "purple", linestyle = "dashed")
        ax.plot(obs, ncs_time_div[i,:], color = 'orange')
        #ax.plot([i for i in range(1,11)], bell_numbers, color = "blue")
        ax.plot(extr, ncs_time_extrp[i,:], color = 'orange', linestyle = "dashed")
        ax.legend(["FCS", "FCS (extrapolated)", "NCS"])
        ax.set_xticks([1,2,4,6,8,10], [1,2,4,6,8,10])
        ax.set_yticks([1,2.5,5,7.5,10,12.5,15,17.5,20], [1,2.5,5,7.5,10,12.5,15,17.5,20])
        ax.set_xlabel("Number of Observations")
        ax.set_ylabel("Evaluation Ratio")
        plt.savefig("visualizations/" + figname + "_range_" + str(range_values[i]) + ".png")
        plt.clf()

fcs_type = "user"
figname = "ncs_fcs_" + str(fcs_type) + "_timing_array_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_array(figname, fcs_type)
figname = "ncs_fcs_" + str(fcs_type) + "_timing_division_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_division(figname, fcs_type)
figname = "ncs_fcs_" + str(fcs_type) + "_timing_division_bell_number_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_division_with_bell_number(figname, fcs_type)
figname = "ncs_fcs_" + str(fcs_type) + "_timing_division_with_exp_linear_extrapolation_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_division_with_exponential_linear_extrapolation(figname, fcs_type)
figname = "ncs_fcs_" + str(fcs_type) + "_timing_division_with_log_linear_extrapolation_1_7_50_azure_gpu"
visualize_ncs_and_fcs_timing_division_log_with_linear_extrapolation(figname, fcs_type)