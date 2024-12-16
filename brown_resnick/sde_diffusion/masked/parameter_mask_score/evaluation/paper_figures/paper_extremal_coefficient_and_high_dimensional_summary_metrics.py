import matplotlib.pyplot as plt
import torch as th
import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
from paper_high_dimensional_metrics import *
from matplotlib import gridspec

evaluation_folder = append_directory(2)

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

    for i in range(len(range_values)):

        extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_summary_statistics/data/true/extremal_coefficient_smooth_" + str(smooth) + "_range_" + 
                                  str(round(range_values[i])) + "_nbins_" + str(bins) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((evaluation_folder + "/extremal_coefficient_and_high_summary_statistics/data/ncs/model4/extremal_coefficient_range_"
                                            + str(range_values[i]) + "_smooth_" + str(smooth) 
                                            + "_bins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_images[i,:,:,:] = np.load((ncs_images_file + str(range_values[i]) + "_smooth_"
                                       + str(smooth) + "_" + str(nrep) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(true_images_file + str(smooth) + "_range_" +
                                               str(round(range_values[i])) + ".npy"))).reshape((nrep,n,n))
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
                         width_ratios=[1,1,1,1,1], wspace=0.25,
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
                ax.set_xlabel("Distance Lag (h)")
                ax.set_ylabel("2-Extremal Coefficient")
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
            ax.legend(labels = ['true', 'NCS'], fontsize = 7)
        
        elif(i < 10):
            ncspdd = pd.DataFrame(ncs_mins[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_mins[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS'], fontsize = 7)
            ax.set_xlim((-3,1))
            ax.set_ylim((0,1.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 5):
                #ax.set_xticks([])
                ax.set_yticks([])
        
        elif(i < 15):
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS'], fontsize = 7)
            ax.set_xlim((0,15))
            ax.set_ylim((0,.45))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 10):
                #ax.set_xticks([])
                ax.set_yticks([])

        else:
            ncspdd = pd.DataFrame(ncs_abs_summation[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_abs_summation[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS'], fontsize = 7)
            ax.set_xlim((0,5000))
            ax.set_ylim((0,.0025))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 15):
                #ax.set_xticks([])
                ax.set_yticks([])
        
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


range_value = 3.0
smooth = 1.5
range_values = [1.,2.,3.,4.,5.]
bins = 100
nrep = 4000
figname = "figures/br_parameter_ncs_vs_true_extremal_coefficient_and_high_dimensional_summary_metrics.png"
visualize_ncs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_multiple_ranges(range_values, smooth, bins, figname, nrep)