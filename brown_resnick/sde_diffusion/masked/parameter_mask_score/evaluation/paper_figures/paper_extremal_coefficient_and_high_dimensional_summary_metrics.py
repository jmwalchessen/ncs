import matplotlib.pyplot as plt
import torch as th
import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import gridspec
import seaborn as sns
import pandas as pd

evaluation_folder = append_directory(2)
extr_folder = (evaluation_folder + "/extremal_coefficient_and_high_dimensional_summary_metrics")

def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

def visualize_ncs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_multiple_ranges(range_values, smooth,
                                                                                                          bins, figname, nrep):
    
    extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    ncs_extremal_matrices = np.zeros((len(range_values), (bins+1),3))
    ncs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    ncs_abs_summation = np.zeros((len(range_values),nrep))
    true_abs_summation = np.zeros((len(range_values),nrep))
    ncs_mins = np.zeros((len(range_values),nrep))
    ncs_maxs = np.zeros((len(range_values),nrep))
    true_mins = np.zeros((len(range_values),nrep))
    true_maxs = np.zeros((len(range_values),nrep))

    for i, range_value in enumerate(range_values):

        extremal_matrices[i,:,:] = load_numpy_file((extr_folder + "/data/true/extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(round(range_values[i])) + "_nbins_" + str(bins) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((extr_folder + "/data/ncs/model4/extremal_coefficient_range_"
                                            + str(range_values[i]) + "_smooth_" + str(smooth) 
                                            + "_bins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_images[i,:,:,:] = np.load((extr_folder + "/data/ncs/model4/brown_resnick_ncs_images_range_" + str(range_value) + "_smooth_1.5_4000.npy"))
        true_images[i,:,:,:] = (np.log(np.load(extr_folder + "/data/true/brown_resnick_images_random05_smooth_1.5_range_" + str(int(range_value)) + ".npy"))).reshape((nrep,n,n))
        ncs_abs_summation[i,:] = np.sum(np.abs(ncs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        ncs_maxs[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maxs[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(8)
 
    # set width of each subplot as 8
    fig.set_figwidth(11)
    spec = gridspec.GridSpec(ncols=5, nrows=4,
                         width_ratios=[1,1,1,1,1], wspace=0.1,
                         hspace=0.35, height_ratios=[1, 1, 1, 1])
    h = extremal_matrices[0,:,0]

    for i in range(20):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            ext_coeff = 2-extremal_matrices[i,:,2]
            ncs_ext_coeff = 2-ncs_extremal_matrices[i,:,2]
            ax.plot(h, ext_coeff, "blue")
            ax.plot(h, ncs_ext_coeff, "orange", linestyle = "dashed")
            if(i == 0):
                ax.set_xlabel("Distance Lag (h)", fontsize = 15)
                #ax.set_ylabel("2-Extremal Coeff.", fontsize = 15)
                ax.set_yticks(ticks = [0,.25,.5,.75], labels = np.array([0.,.25,.5,.75]), fontsize = 15)
                ax.set_xticks([])
                #ax.set_xticks(ticks = [0,10,20], labels = np.array([0,10,20]), fontsize = 15)
                ax.legend(labels = ['true', 'NCS'], fontsize = 13.5)
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
            if(i == 0):
                ax.set_xlabel("Distance Lag (h)", fontsize = 15)
                ax.set_yticks([0.,.25,.5,.75,1.], [0.,.25,.5,.75,1.], fontsize = 15)
            if((i != 0) & (i != 2)):
                ax.set_xticks(ticks = [0,10,20], labels = np.array([0,10,20]), fontsize = 15)
            else:
                ax.set_xticks([])
            if(i != 0):
                ax.set_yticks([])

        elif(i < 10):
            ncspdd = pd.DataFrame(ncs_mins[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_mins[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False)
            ax.set_xlim((-3,1))
            ax.set_ylim((0,1.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if((i != 5)):
                ax.set_yticks([])
            else:
                ax.set_yticks(ticks = [0.,.5,1.,1.5], labels = np.array([0.,.5,1.,1.5]), fontsize = 15)
            if((i != 7)):
                ax.set_xticks(ticks = [-2.,0.], labels = np.array([-2.,0.]), fontsize = 15)
            else:
                ax.set_xticks([])
        
        elif(i < 15):
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], linestyle = "dashed", ax = ax, legend = False)
            ax.set_xlim((0,15))
            ax.set_ylim((0,.45))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 10):
                ax.set_yticks([])
            else:
                ax.set_yticks(ticks = [0.,.2,.4], labels = np.array([0.,.2,.4]), fontsize = 15)
            if(i == 12):
                ax.set_xticks([])
            elif(i == 11):
                ax.set_xticks(ticks = [5,10], labels = np.array([5,10]), fontsize = 15)
            elif(i == 13):
                ax.set_xticks(ticks = [5,10], labels = np.array([5,10]), fontsize = 15)
            else:
                ax.set_xticks(ticks = [0,5,10,15], labels = np.array([0,5,10,15]), fontsize = 15)

        else:
            ncspdd = pd.DataFrame(ncs_abs_summation[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_abs_summation[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'],linestyle = "dashed", ax = ax, legend = False)
            ax.set_xlim((0,5000))
            ax.set_ylim((0,.0025))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 15):
                ax.set_yticks([])
                ax.set_xticks(ticks = [2500,5000], labels = np.array([2500,5000]), fontsize = 15)
            else:
                ax.set_yticks(ticks = [0.,.001,.002], labels = np.array([0,.001,.002]), fontsize = 15)
                ax.set_xticks(ticks = [0,2500,5000], labels = np.array([0,2500,5000]), fontsize = 15)
        
    fig.text(0.4, .89, "2-Extremal Coefficient", fontsize = 15)
    fig.text(0.47, .69, "Minimum", fontsize = 15)
    fig.text(0.47, .48, "Maximum", fontsize = 15)
    fig.text(0.41, .28, "Absolute Summation", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_ncs_and_true_min_max(range_values, figname, nrep, n):

    ncs_images = np.zeros((len(range_values),nrep,n,n))
    true_images = np.zeros((len(range_values),nrep,n,n))
    ncs_mins = np.zeros((len(range_values),nrep))
    ncs_maxs = np.zeros((len(range_values),nrep))
    true_mins = np.zeros((len(range_values),nrep))
    true_maxs = np.zeros((len(range_values),nrep))

    ncs_image_file = ""

    for i,range_value in enumerate(range_values):
        ncs_images[i,:,:,:] = np.load((extr_folder + "/data/ncs/model4/brown_resnick_ncs_images_range_" + str(range_value) + "_smooth_1.5_4000.npy"))
        true_images[i,:,:,:] = (np.log(np.load(extr_folder + "/data/true/brown_resnick_images_random05_smooth_1.5_range_" + str(int(range_value)) + ".npy"))).reshape((nrep,n,n))
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        ncs_maxs[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maxs[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(4)
 
    # set width of each subplot as 8
    fig.set_figwidth(11)
    spec = gridspec.GridSpec(ncols=5, nrows=2,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.35, height_ratios=[1, 1])
    
    for i in range(10):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            ncspdd = pd.DataFrame(ncs_mins[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_mins[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS'], fontsize = 7)
            ax.set_xlim((-3,1))
            ax.set_ylim((0,1.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS'], fontsize = 7)
            ax.set_xlim((0,15))
            ax.set_ylim((0,.45))
            ax.set_xlabel("")
            ax.set_ylabel("")

    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


range_value = 3.0
smooth = 1.5
range_values = [1.,2.,3.,4.,5.]
bins = 100
nrep = 4000
n = 32
figname = "figures/br_parameter_ncs_vs_true_extremal_coefficient_and_high_dimensional_summary_metrics.png"
visualize_ncs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_multiple_ranges(range_values, smooth, bins, figname, nrep)
figname = "figures/br_parameter_ncs_vs_true_min_max.png"
visualize_ncs_and_true_min_max(range_values, figname, nrep, n)