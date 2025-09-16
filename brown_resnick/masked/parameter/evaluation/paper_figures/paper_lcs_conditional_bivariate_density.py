import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
from matplotlib import patches as mpatches
from matplotlib import gridspec
import pandas as pd
import seaborn as sns

evaluation_folder = append_directory(2)
data_generation_folder = (evaluation_folder + "/diffusion_generation")
sys.path.append(data_generation_folder)
sys.path.append(evaluation_folder)
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle



def produce_generated_and_bivariate_lcs_density_multiple_ranges(model_name, n, missing_indices1, missing_indices2,
                                                                range_values, bivariate_lcs_file,
                                                                figname, nrep):

    masks = np.zeros((len(range_values),n,n))
    ref_images = np.zeros((len(range_values),n,n))
    ncs_images = np.zeros((len(range_values),nrep, n,n))
    generated_bivariate_density = np.zeros((len(range_values),nrep,2))
    lcs_bivariate_density = np.zeros((len(range_values),nrep,2))

    for i in range(len(range_values)):

        missing_index1 = missing_indices1[i]
        missing_index2 = missing_indices2[i]
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        masks[i,:,:] = np.load((data_generation_folder + "/" + ref_image_folder + "/mask.npy"))
        ref_images[i,:,:] = np.load((data_generation_folder + "/" + ref_image_folder + "/ref_image.npy"))
        ncs_file_name = model_name + "_range_" + str(range_values[i]) + "_smooth_1.5_random0.05_4000.npy"
        ncs_images[i,:,:,:] = (np.load((data_generation_folder + "/" + ref_image_folder + "/diffusion/" + ncs_file_name))).reshape((nrep,n,n))
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                                                 bivariate_lcs_file + "_" + str(missing_index1) + "_" + str(missing_index2) + ".npy")))
        lcs_bivariate_density[i,:,:] = bilcs
        matrix_missing_index1 = index_to_matrix_index(missing_index1, n)
        matrix_missing_index2 = index_to_matrix_index(missing_index2, n)
        generated_bivariate_density[i,:,:] = np.concatenate([(ncs_images[i,:,int(matrix_missing_index1[0]),int(matrix_missing_index1[1])]).reshape((nrep,1)),
                                                           (ncs_images[i,:,int(matrix_missing_index2[0]),int(matrix_missing_index2[1])]).reshape((nrep,1))],
                                                           axis = 1)


    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(4)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=2,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.25, height_ratios=[1, 1])

    for i in range(0, 10):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            im = ax.imshow(ref_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "ro", markersize = 10, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "ro", markersize = 10, linewidth = 20)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', alpha = .7)
            sns.kdeplot(x = generated_bivariate_density[(i%5),:,0], y = generated_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .7)
            ax.axvline(ref_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(ref_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [-2,0,2,4,6], labels = np.array([-2,0,2,4,6]))
            purple_patch = mpatches.Patch(color='purple')
            orange_patch = mpatches.Patch(color='orange')
            ax.legend(handles = [purple_patch, orange_patch], labels = ['LCS', 'NCS'], fontsize = 7)
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)

    plt.savefig(figname)
    plt.clf()

def produce_generated_and_bivariate_lcs_density_multiple_ranges_with_variables():
    
    n = 32
    range_values = [1.,2.,3.,4.,5.]
    model_name = "model4"
    missing_indices1 = [401,500,934,200,822]
    missing_indices2 = [597,342,918,274,960]
    figname = "figures/br_parameter_lcs_vs_ncs_conditional_bivariate_density.png"
    nrep = 4000
    bivariate_lcs_file = "bivariate_lcs_4000_neighbors_7_nugget_1e5"
    produce_generated_and_bivariate_lcs_density_multiple_ranges(model_name, n, missing_indices1, missing_indices2,
                                                                range_values, bivariate_lcs_file,
                                                                figname, nrep)