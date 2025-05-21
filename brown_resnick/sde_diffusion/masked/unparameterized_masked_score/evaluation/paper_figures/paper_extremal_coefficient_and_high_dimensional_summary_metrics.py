import matplotlib.pyplot as plt
import torch as th
import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import gridspec
import pandas as pd
import seaborn as sns

evaluation_folder = append_directory(2)
extr_folder = (evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics")

def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

def visualize_ncs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_multiple_percentages(range_value, smooth,
                                                                                                          ps, bins, figname, nrep, n):
    
    extremal_matrices = np.zeros((len(ps), (bins+1),3))
    ncs_extremal_matrices = np.zeros((len(ps), (bins+1),3))
    ncs_images = np.zeros((len(ps),nrep,n,n))
    true_images = np.zeros((len(ps),nrep,n,n))
    ncs_abs_summation = np.zeros((len(ps),nrep))
    true_abs_summation = np.zeros((len(ps),nrep))
    ncs_mins = np.zeros((len(ps),nrep))
    ncs_maxs = np.zeros((len(ps),nrep))
    true_mins = np.zeros((len(ps),nrep))
    true_maxs = np.zeros((len(ps),nrep))

    for i in range(len(ps)):

        extremal_matrices[i,:,:] = load_numpy_file((extr_folder + "/data/true/extremal_coefficient_range_"
                                                    + str(range_value) + "_smooth_" + str(smooth) + "_nbins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((extr_folder + "/data/ncs/model4/brown_resnick_ncs_extremal_matrix_bins_"
                                            + str(bins) + "_range_" + str(range_value) + "_smooth_" + str(smooth) 
                                            + "_" + str(nrep) + "_random" + str(ps[i]) + ".npy"))
        ncs_images[i,:,:,:] = np.load((extr_folder + "/data/ncs/model4/brown_resnick_ncs_images_range_3.0_smooth_1.5_4000_random" + str(ps[i]) + ".npy"))
        true_images[i,:,:,:] = np.log(np.load(extr_folder + "/data/true/brown_resnick_images_range_3.0_smooth_1.5_4000.npy"))
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
            #ax.legend(labels = ['true', 'NCS'], fontsize = 15)
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
            #ax.legend(labels = ['true', 'NCS'], fontsize = 15)
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
            sns.kdeplot(data = ncspdd, palette =['orange'], linestyle = "dashed", ax = ax, legend = False)
            #ax.legend(labels = ['true', 'NCS'], fontsize = 15)
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

def visualize_ncs_and_true_min_max(ps, figname, nrep, n):

    ncs_images = np.zeros((len(ps),nrep,n,n))
    true_images = np.zeros((len(ps),nrep,n,n))
    ncs_mins = np.zeros((len(ps),nrep))
    ncs_maxs = np.zeros((len(ps),nrep))
    true_mins = np.zeros((len(ps),nrep))
    true_maxs = np.zeros((len(ps),nrep))

    ncs_image_file = ""

    for i,p in enumerate(ps):
        ncs_images[i,:,:,:] = np.load((extr_folder + "/data/ncs/model4/brown_resnick_ncs_images_range_3.0_smooth_1.5_4000_random" + str(ps[i]) + ".npy"))
        true_images[i,:,:,:] = np.log(np.load(extr_folder + "/data/true/brown_resnick_images_range_3.0_smooth_1.5_4000.npy"))
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
            ax.legend(labels = ['true', 'NCS'], fontsize = 15)
            ax.set_xlim((-3,1))
            ax.set_ylim((0,1.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS'], fontsize = 15)
            ax.set_xlim((0,15))
            ax.set_ylim((0,.45))
            ax.set_xlabel("")
            ax.set_ylabel("")

    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_ncs_fcs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_one_to_seven_range_5(range_value, smooth,
                                                                                                      bins, figname, nrep, n):
    
    obs_numbers = [1,2,3,5,7]
    extremal_matrices = np.zeros((len(obs_numbers), (bins+1),3))
    ncs_extremal_matrices = np.zeros((5, (bins+1),3))
    fcs_extremal_matrices = np.zeros((5, (bins+1),3))
    ncs_images = np.zeros((len(obs_numbers),nrep,n,n))
    fcs_images = np.zeros((len(obs_numbers),nrep,n,n))
    true_images = np.zeros((len(obs_numbers),nrep,n,n))
    ncs_abs_summation = np.zeros((len(obs_numbers),nrep))
    fcs_abs_summation = np.zeros((len(obs_numbers),nrep))
    true_abs_summation = np.zeros((len(obs_numbers),nrep))
    ncs_mins = np.zeros((len(obs_numbers),nrep))
    ncs_maxs = np.zeros((len(obs_numbers),nrep))
    fcs_mins = np.zeros((len(obs_numbers),nrep))
    fcs_maxs = np.zeros((len(obs_numbers),nrep))
    true_mins = np.zeros((len(obs_numbers),nrep))
    true_maxs = np.zeros((len(obs_numbers),nrep))

    model_versions = [5,5,5,5,5]

    for i in range(len(obs_numbers)):

        ref_folder = (evaluation_folder + "/fcs/data/unconditional/fixed_locations/obs" + 
                     str(obs_numbers[i]) + "/ref_image" + str(int(range_value-1)))
        extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/true_extremal_coefficient_range_"
                                                    + str(range_value) + "_smooth_" + str(smooth) + "_nbins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/brown_resnick_ncs_extremal_matrix_bins_" + str(bins) + "_obs" + str(obs_numbers[i])
                                                        + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_" + str(nrep) + ".npy"))
        fcs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/extremal_coefficient_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_"
                                                        + str(obs_numbers[i]) + "_" + str(nrep) + ".npy"))
        ncs_images[i,:,:,:] = np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + 
                                       str(range_value) + "_smooth_" + str(smooth) + "_model" + str(model_versions[i]) + "_" + str(nrep) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(ref_folder + "/true_brown_resnick_images_range_" + str(int(range_value)) + 
                                              "_smooth_" + str(smooth) + "_" + str(nrep) + ".npy"))).reshape((nrep,n,n))
        fcs_images[i,:,:,:] = np.log(np.load(ref_folder + "/processed_unconditional_fcs_fixed_mask_range_" + str(range_value) + 
                                              "_smooth_" + str(smooth) + "_nugget_1e5_obs_" + str(obs_numbers[i]) + "_" + str(nrep) + ".npy")).reshape((nrep,n,n))
        ncs_abs_summation[i,:] = np.sum(np.abs(ncs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        fcs_abs_summation[i,:] = np.sum(np.abs(fcs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_mins[i,:] = np.min(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        ncs_maxs[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maxs[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_maxs[i,:] = np.max(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(8)
 
    # set width of each subplot as 8
    fig.set_figwidth(11)
    spec = gridspec.GridSpec(ncols=5, nrows=4,
                         width_ratios=[1,1,1,1,1], wspace=0.13,
                         hspace=0.3, height_ratios=[1, 1, 1, 1])
    h = extremal_matrices[0,:,0]

    for i in range(20):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            ext_coeff = 2-extremal_matrices[i,:,2]
            ncs_ext_coeff = 2-ncs_extremal_matrices[i,:,2]
            fcs_ext_coeff = 2-fcs_extremal_matrices[i,:,2]
            ax.plot(h, ext_coeff, "blue")
            ax.plot(h, ncs_ext_coeff, "orange", linestyle = "dashed")
            ax.plot(h, fcs_ext_coeff, "purple", linestyle = "dashed")
            if(i == 0):
                ax.set_xlabel("Distance Lag (h)", fontsize = 15)
                ax.set_ylabel("2-Extremal Coeff.", fontsize = 15)
                ax.set_title("1 Obs.", fontsize = 15)
                ax.set_yticks([0., .25, .5, .75], [0., .25, .5, .75], fontsize = 15)
                ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 12)
                ax.set_xticks([])
        
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.set_title((str(obs_numbers[i]) + " Obs."), fontsize = 15)
                ax.set_xticks([0,10,20], [0,10,20], fontsize = 15)
        
        elif(i < 10):
            ncspdd = pd.DataFrame(ncs_mins[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_mins[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_mins[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False)
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = False)
            ax.set_xlim((-3,1))
            ax.set_ylim((0,1.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([-2,0], [-2,0], fontsize = 15)
            if(i != 5):
                #ax.set_xticks([])
                ax.set_yticks([])
            else:
                #ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 12)
                ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 15)
        
        elif(i < 15):
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False)
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = False, linestyle = "dashed")
            ax.set_xlim((0,15))
            ax.set_ylim((0,.45))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 10):
                ax.set_yticks([])
                ax.set_xticks([5,10,15], [5,10,15], fontsize = 15)
            else:
                ax.set_xticks([0,5,10,15], [0,5,10,15], fontsize = 15)
                #ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 12)
                ax.set_yticks([0.,.2,.4], [0.,.2,.4], fontsize = 15)

        else:
            ncspdd = pd.DataFrame(ncs_abs_summation[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_abs_summation[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_abs_summation[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False, linestyle = "dashed")
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = False, linestyle = "dashed")
            ax.set_xlim((0,5000))
            ax.set_ylim((0,.0025))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([0,2000,4000], [0,2000,4000], fontsize = 15)
            if(i != 15):
                #ax.set_xticks([])
                ax.set_yticks([])
            else:
                #ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 12)
                ax.set_yticks([0.,0.001,.002], [0.,0.001,.002], fontsize = 15)
        
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_ncs_fcs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_one_to_seven_range_3(range_value, smooth,
                                                                                                      bins, figname, nrep, n):
    
    obs_numbers = [1,2,3,5,7]
    extremal_matrices = np.zeros((len(obs_numbers), (bins+1),3))
    ncs_extremal_matrices = np.zeros((5, (bins+1),3))
    fcs_extremal_matrices = np.zeros((5, (bins+1),3))
    ncs_images = np.zeros((len(obs_numbers),nrep,n,n))
    fcs_images = np.zeros((len(obs_numbers),nrep,n,n))
    true_images = np.zeros((len(obs_numbers),nrep,n,n))
    ncs_abs_summation = np.zeros((len(obs_numbers),nrep))
    fcs_abs_summation = np.zeros((len(obs_numbers),nrep))
    true_abs_summation = np.zeros((len(obs_numbers),nrep))
    ncs_mins = np.zeros((len(obs_numbers),nrep))
    ncs_maxs = np.zeros((len(obs_numbers),nrep))
    fcs_mins = np.zeros((len(obs_numbers),nrep))
    fcs_maxs = np.zeros((len(obs_numbers),nrep))
    true_mins = np.zeros((len(obs_numbers),nrep))
    true_maxs = np.zeros((len(obs_numbers),nrep))

    model_versions = [5,5,5,5,5]

    for i in range(len(obs_numbers)):

        ref_folder = (evaluation_folder + "/fcs/data/unconditional/fixed_locations/obs" + 
                     str(obs_numbers[i]) + "/ref_image" + str(int(range_value-1)))
        extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/true_extremal_coefficient_range_"
                                                    + str(range_value) + "_smooth_" + str(smooth) + "_nbins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/brown_resnick_ncs_extremal_matrix_bins_" + str(bins) + "_obs" + str(obs_numbers[i])
                                                        + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_" + str(nrep) + ".npy"))
        fcs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/extremal_coefficient_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_"
                                                        + str(obs_numbers[i]) + "_" + str(nrep) + ".npy"))
        ncs_images[i,:,:,:] = np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + 
                                       str(range_value) + "_smooth_" + str(smooth) + "_model" + str(model_versions[i]) + "_" + str(nrep) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(ref_folder + "/true_brown_resnick_images_range_" + str(int(range_value)) + 
                                              "_smooth_" + str(smooth) + "_" + str(nrep) + ".npy"))).reshape((nrep,n,n))
        fcs_images[i,:,:,:] = np.log(np.load(ref_folder + "/processed_unconditional_fcs_fixed_mask_range_" + str(range_value) + 
                                              "_smooth_" + str(smooth) + "_nugget_1e5_obs_" + str(obs_numbers[i]) + "_" + str(nrep) + ".npy")).reshape((nrep,n,n))
        ncs_abs_summation[i,:] = np.sum(np.abs(ncs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        fcs_abs_summation[i,:] = np.sum(np.abs(fcs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_mins[i,:] = np.min(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        ncs_maxs[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maxs[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_maxs[i,:] = np.max(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(8)
 
    # set width of each subplot as 8
    fig.set_figwidth(11)
    spec = gridspec.GridSpec(ncols=5, nrows=4,
                         width_ratios=[1,1,1,1,1], wspace=0.1,
                         hspace=0.3, height_ratios=[1, 1, 1, 1])
    h = extremal_matrices[0,:,0]

    for i in range(20):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            ext_coeff = 2-extremal_matrices[i,:,2]
            ncs_ext_coeff = 2-ncs_extremal_matrices[i,:,2]
            fcs_ext_coeff = 2-fcs_extremal_matrices[i,:,2]
            ax.plot(h, ext_coeff, "blue")
            ax.plot(h, ncs_ext_coeff, "orange", linestyle = "dashed")
            ax.plot(h, fcs_ext_coeff, "purple", linestyle = "dashed")
            if(i == 0):
                ax.set_xlabel("Distance Lag (h)", fontsize = 7)
                ax.set_ylabel("2-Extremal Coefficient", fontsize = 7)
                ax.set_title("1 Obs. Location", fontsize = 9)
                ax.set_yticks([0., .25, .5, .75], [0., .25, .5, .75], fontsize = 7)
        
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.set_title((str(obs_numbers[i]) + " Obs. Locations"), fontsize = 9)
            ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 7)
            ax.set_xticks([0,10,20], [0,10,20], fontsize = 7)
        
        elif(i < 10):
            ncspdd = pd.DataFrame(ncs_mins[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_mins[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_mins[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True, linestyle = "dashed")
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 7)
            ax.set_xlim((-6,1))
            ax.set_ylim((0,1.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([-6,-4,-2,0], [-6,-4,-2,0], fontsize = 7)
            if(i != 5):
                #ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 7)
        
        elif(i < 15):
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True)
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 7)
            ax.set_xlim((0,15))
            ax.set_ylim((0,.45))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([0,5,10,15], [0,5,10,15], fontsize = 7)
            if(i != 10):
                ax.set_yticks([])
            else:
                ax.set_yticks([0.,.2,.4], [0.,.2,.4], fontsize = 7)

        else:
            ncspdd = pd.DataFrame(ncs_abs_summation[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_abs_summation[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_abs_summation[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = True)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = True, linestyle = "dashed")
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = True)
            ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 7)
            ax.set_xlim((0,5000))
            ax.set_ylim((0,.0025))
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([0,2500,5000], [0,2500,5000], fontsize = 7)
            if(i != 15):
                #ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_yticks([0.,0.001,.002], [0.,0.001,.002], fontsize = 7)
        
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


def visualize_ncs_fcs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_one_to_seven_range_1(range_value, smooth,
                                                                                                      bins, figname, nrep, n):
    
    obs_numbers = [1,2,3,5,7]
    extremal_matrices = np.zeros((len(obs_numbers), (bins+1),3))
    ncs_extremal_matrices = np.zeros((5, (bins+1),3))
    fcs_extremal_matrices = np.zeros((5, (bins+1),3))
    ncs_images = np.zeros((len(obs_numbers),nrep,n,n))
    fcs_images = np.zeros((len(obs_numbers),nrep,n,n))
    true_images = np.zeros((len(obs_numbers),nrep,n,n))
    ncs_abs_summation = np.zeros((len(obs_numbers),nrep))
    fcs_abs_summation = np.zeros((len(obs_numbers),nrep))
    true_abs_summation = np.zeros((len(obs_numbers),nrep))
    ncs_mins = np.zeros((len(obs_numbers),nrep))
    ncs_maxs = np.zeros((len(obs_numbers),nrep))
    fcs_mins = np.zeros((len(obs_numbers),nrep))
    fcs_maxs = np.zeros((len(obs_numbers),nrep))
    true_mins = np.zeros((len(obs_numbers),nrep))
    true_maxs = np.zeros((len(obs_numbers),nrep))

    model_versions = [5,5,5,5,5]

    for i in range(len(obs_numbers)):

        ref_folder = (evaluation_folder + "/fcs/data/unconditional/fixed_locations/obs" + 
                     str(obs_numbers[i]) + "/ref_image" + str(int(range_value-1)))
        extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/true_extremal_coefficient_range_"
                                                    + str(range_value) + "_smooth_" + str(smooth) + "_nbins_" + str(bins) + "_" + str(nrep) + ".npy"))
        ncs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/brown_resnick_ncs_extremal_matrix_bins_" + str(bins) + "_obs" + str(obs_numbers[i])
                                                        + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_" + str(nrep) + ".npy"))
        fcs_extremal_matrices[i,:,:] = load_numpy_file((ref_folder + "/extremal_coefficient_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_"
                                                        + str(obs_numbers[i]) + "_" + str(nrep) + ".npy"))
        ncs_images[i,:,:,:] = np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + 
                                       str(range_value) + "_smooth_" + str(smooth) + "_model" + str(model_versions[i]) + "_" + str(nrep) + ".npy"))
        true_images[i,:,:,:] = (np.log(np.load(ref_folder + "/true_brown_resnick_images_range_" + str(int(range_value)) + 
                                              "_smooth_" + str(smooth) + "_" + str(nrep) + ".npy"))).reshape((nrep,n,n))
        fcs_images[i,:,:,:] = np.log(np.load(ref_folder + "/processed_unconditional_fcs_fixed_mask_range_" + str(range_value) + 
                                              "_smooth_" + str(smooth) + "_nugget_1e5_obs_" + str(obs_numbers[i]) + "_" + str(nrep) + ".npy")).reshape((nrep,n,n))
        ncs_abs_summation[i,:] = np.sum(np.abs(ncs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        true_abs_summation[i,:] = np.sum(np.abs(true_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        fcs_abs_summation[i,:] = np.sum(np.abs(fcs_images[i,:,:,:]).reshape((nrep, n**2)), axis = 1)
        ncs_mins[i,:] = np.min(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_mins[i,:] = np.min(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_mins[i,:] = np.min(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        ncs_maxs[i,:] = np.max(ncs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        true_maxs[i,:] = np.max(true_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)
        fcs_maxs[i,:] = np.max(fcs_images[i,:,:,:].reshape((nrep, n**2)), axis = 1)

    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(8)
 
    # set width of each subplot as 8
    fig.set_figwidth(11)
    spec = gridspec.GridSpec(ncols=5, nrows=4,
                         width_ratios=[1,1,1,1,1], wspace=0.1,
                         hspace=0.3, height_ratios=[1, 1, 1, 1])
    h = extremal_matrices[0,:,0]

    for i in range(20):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            ext_coeff = 2-extremal_matrices[i,:,2]
            ncs_ext_coeff = 2-ncs_extremal_matrices[i,:,2]
            fcs_ext_coeff = 2-fcs_extremal_matrices[i,:,2]
            ax.plot(h, ext_coeff, "blue")
            ax.plot(h, ncs_ext_coeff, "orange", linestyle = "dashed")
            ax.plot(h, fcs_ext_coeff, "purple")
            if(i == 0):
                ax.set_xlabel("Distance Lag (h)", fontsize = 15)
                ax.set_ylabel("2-Extremal Coeff.", fontsize = 15)
                ax.set_title("1 Obs.", fontsize = 15)
                ax.set_yticks([0., .25, .5, .75], [0., .25, .5, .75], fontsize = 15)
                ax.set_xticks([])
                ax.legend(labels = ['true', 'NCS', 'FCS'], fontsize = 13)
        
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.set_title((str(obs_numbers[i]) + " Obs."), fontsize = 15)
                if((i != 2)):
                    ax.set_xticks([0,10,20], [0,10,20], fontsize = 15)
                else:
                    ax.set_xticks([])
        
        elif(i < 10):
            ncspdd = pd.DataFrame(ncs_mins[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_mins[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_mins[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False, linestyle = "dashed")
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = False)
            ax.set_xlim((-35,1))
            ax.set_ylim((0,2.25))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if((i != 5)):
                ax.set_yticks([])
            else:
                ax.set_yticks(ticks = [0.,1.,2.], labels = np.array([0.,]), fontsize = 15)
            if((i != 7)):
                ax.set_xticks(ticks = [-30,-20,-10,0], labels = np.array([-30,-20,-10,0]), fontsize = 15)
            else:
                ax.set_xticks([])
        
        elif(i < 15):
            ncspdd = pd.DataFrame(ncs_maxs[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_maxs[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_maxs[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False, linestyle = "dashed")
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = False)
            ax.set_xlim((0,15))
            ax.set_ylim((0,.5))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 10):
                ax.set_yticks([])
            else:
                ax.set_yticks(ticks = [0.,.2,.4], labels = np.array([0.,.2,.4]), fontsize = 15)
            if(i == 12):
                ax.set_xticks([])
            elif((i == 11) | (i == 14)):
                ax.set_xticks(ticks = [5,10], labels = np.array([5,10]), fontsize = 15)
            elif(i == 13):
                ax.set_xticks(ticks = [5,10,15], labels = np.array([5,10,15]), fontsize = 15)
            else:
                ax.set_xticks(ticks = [0,5,10,15], labels = np.array([0,5,10,15]), fontsize = 15)


        else:
            ncspdd = pd.DataFrame(ncs_abs_summation[(i%5),:], columns = None)
            fcspdd = pd.DataFrame(fcs_abs_summation[(i%5),:], columns = None)
            truepdd = pd.DataFrame(true_abs_summation[(i%5),:], columns = None)
            sns.kdeplot(data = truepdd, palette = ['blue'], ax = ax, legend = False)
            sns.kdeplot(data = ncspdd, palette =['orange'], ax = ax, legend = False, linestyle = "dashed")
            sns.kdeplot(data = fcspdd, palette =['purple'], ax = ax, legend = False)
            ax.set_xlim((0,12000))
            ax.set_ylim((0,.003))
            ax.set_xlabel("")
            ax.set_ylabel("")
            if(i != 15):
                ax.set_yticks([])
                ax.set_xticks(ticks = [5000,10000], labels = np.array([5000,10000]), fontsize = 15)
            else:
                ax.set_yticks(ticks = [0.,.001,.002], labels = np.array([0,.001,.002]), fontsize = 15)
                ax.set_xticks(ticks = [0,5000,10000], labels = np.array([0,5000,10000]), fontsize = 15)
        
    fig.text(0.47, .69, "Minimum", fontsize = 15)
    fig.text(0.47, .48, "Maximum", fontsize = 15)
    fig.text(0.41, .28, "Absolute Summation", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)


range_value = 3.0
smooth = 1.5
ps = [.01,.05,.1,.25,.5]
bins = 100
nrep = 4000
n = 32
figname = "figures/paper_ncs_vs_true_extremal_coefficient_and_high_dimensional_summary_metrics.png"
visualize_ncs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_multiple_percentages(range_value, smooth, ps, bins, figname, nrep, n)
#figname = "figures/presentation_ncs_vs_true_min_max.png"
#visualize_ncs_and_true_min_max(ps, figname, nrep, n)


range_value = 1.
smooth = 1.5
bins = 100
figname = "figures/paper_ncs_fcs_vs_true_extremal_coefficient_and_high_dimensional_summary_metrics_range_" + str(range_value) + ".png"
nrep = 4000
n = 32
visualize_ncs_fcs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_one_to_seven_range_1(range_value, smooth,
                                                                                                      bins, figname, nrep, n)
range_value = 5.
figname = "figures/paper_ncs_fcs_vs_true_extremal_coefficient_and_high_dimensional_summary_metrics_range_" + str(range_value) + ".png"
visualize_ncs_fcs_and_true_extremal_coefficient_and_high_dimensional_summary_metrics_one_to_seven_range_5(range_value, smooth,
                                                                                                      bins, figname, nrep, n)