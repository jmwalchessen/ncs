import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
from matplotlib import gridspec
import pandas as pd
from matplotlib import patches as mpatches



def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))

#visualize 7 observed fcs with ranges 1,2,3,4,5 and unconditional marginal and bivariate densities on border
def visualize_unconditional_fcs_vs_true_marginal_bivariate_density(nrep, missing_indices,
                                                                   missing_indices1, missing_indices2,
                                                                   n, figname, obs):
    
    evaluation_folder = append_directory(2)
    extr_folder = (evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics")
    range_values = [float(i) for i in range(1,6)]
    fcs_marginal_density = np.zeros((len(range_values),nrep))
    fcs_images = np.zeros((len(range_values),nrep,n,n))
    fcs_bivariate_density = np.zeros((5,nrep,2))
    true_marginal_density = np.zeros((len(range_values),nrep))
    true_images = np.zeros((len(range_values),nrep,n,n))
    true_bivariate_density = np.zeros((5,nrep,2))

    for i in range(0,5):

        missing_index = missing_indices[i]
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        missing_index1 = missing_indices1[i]
        matrix_missing_index1 = index_to_matrix_index(missing_index1, n)
        missing_index2 = missing_indices2[i]
        matrix_missing_index2 = index_to_matrix_index(missing_index2, n)
        fcs_file_name = (extr_folder + "/data/fcs/processed_unconditional_fcs_range_" + str(range_values[i]) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_4000.npy")
        fcs_images[i,:,:,:] = np.log(np.load(fcs_file_name))
        fcs_marginal_density[i,:] = fcs_images[i,:,matrix_missing_index[0],matrix_missing_index[1]]
        fcs_bivariate_density[i,:,:] = np.concatenate([fcs_images[i,:,matrix_missing_index1[0],matrix_missing_index1[1]].reshape((nrep,1)),
                                                    fcs_images[i,:,matrix_missing_index2[0],matrix_missing_index2[1]].reshape((nrep,1))],
                                                    axis = 1)
        true_file_name = (extr_folder + "/data/true/brown_resnick_images_range_" + str(range_values[i]) + "_smooth_1.5_4000.npy")
        true_images[i,:,:,:] = np.log(np.load(true_file_name))
        true_marginal_density[i,:] = true_images[i,:,matrix_missing_index[0],matrix_missing_index[1]]
        true_bivariate_density[i,:,:] = np.concatenate([true_images[i,:,matrix_missing_index1[0],matrix_missing_index1[1]].reshape((nrep,1)),
                                                    true_images[i,:,matrix_missing_index2[0],matrix_missing_index2[1]].reshape((nrep,1))],
                                                    axis = 1)

#fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=3,
                         width_ratios=[1,1,1,1,1], wspace=0.1,
                         hspace=0.25, height_ratios=[1, 1, 1])

    for i in range(0, 15):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 6, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 6, linewidth = 20)
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 6, linewidth = 20)
            if(i == 0):
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            else:
                ax.set_yticks([])
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        elif(i < 10):
            matrix_index = index_to_matrix_index(missing_indices[i%5], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i%5], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i%5], n)
            sns.kdeplot(fcs_marginal_density[(i % 5),:], ax = ax, color = 'purple')
            sns.kdeplot(true_marginal_density[(i % 5),:], ax = ax, color = 'blue')
            ax.set_xlim([-20,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            #if(i == 5):
                #ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5])
            #else:
                #ax.set_yticks([])
            #ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6])
            ax.legend(labels = ['FCS', 'true'], fontsize = 7)
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            print(fcs_bivariate_density[(i%5),:,0])
            sns.kdeplot(x = fcs_bivariate_density[(i%5),:,0], y = fcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            sns.kdeplot(x = true_bivariate_density[(i%5),:,0], y = true_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'blue', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            ax.set_xlim([-20,6])
            ax.set_ylim([-20,6])
            ax.set_ylabel("")
            #if(i == 10):
                #ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6])
            #else:
                #ax.set_yticks([])
            #ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6])
            purple_patch = mpatches.Patch(color='purple')
            orange_patch = mpatches.Patch(color='blue')
            ax.legend(handles = [purple_patch, orange_patch], labels = ['FCS', 'true'], fontsize = 4)

    plt.savefig(figname)
    plt.clf()


nrep = 4000
missing_indices = [100,100,100,100,100]
missing_indices1 = [900,900,900,900,900]
missing_indices2 = [500,500,500,500,500]
n = 32
obs = 5
figname = "figures/paper_fcs_vs_true_obs_" + str(obs) + "_unconditional_marginal_bivariate.png"
visualize_unconditional_fcs_vs_true_marginal_bivariate_density(nrep, missing_indices,
                                                                   missing_indices1, missing_indices2,
                                                                   n, figname, obs)