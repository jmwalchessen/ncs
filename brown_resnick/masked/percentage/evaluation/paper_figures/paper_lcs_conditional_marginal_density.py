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
from mcmc_interpolation_helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle



def produce_generated_and_univariate_lcs_marginal_density_multiple_percentages(n, missing_indices, ps,
                                                                               univariate_lcs_file,
                                                                               figname, nrep):

    masks = np.zeros((len(ps),n,n))
    ref_images = np.zeros((len(ps),n,n))
    ncs_images = np.zeros((len(ps),nrep, n,n))
    univariate_lcs_images = np.zeros((len(ps),nrep,n,n))
    generated_marginal_density = np.zeros((len(ps),nrep))
    lcs_marginal_density = np.zeros((len(ps),nrep))

    for i in range(len(ps)):

        missing_index = missing_indices[i]
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        masks[i,:,:] = np.load((data_generation_folder + "/" + ref_image_folder + "/mask.npy"))
        ref_images[i,:,:] = np.load((data_generation_folder + "/" + ref_image_folder + "/ref_image.npy"))
        ncs_file_name = model_name + "_range_3.0_smooth_1.5_4000_random" + str(ps[i]) + ".npy"
        ncs_images[i,:,:,:] = (np.load((data_generation_folder + "/" + ref_image_folder + "/diffusion/" + ncs_file_name))).reshape((nrep,n,n))
        univariate_lcs_images[i,:,:,:] = (np.load((data_generation_folder + ref_image_folder + "/lcs/univariate/" + univariate_lcs_file))).reshape((nrep,n,n))
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        generated_marginal_density[i,:] = ncs_images[i,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
        lcs_marginal_density[i,:] = univariate_lcs_images[i,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]


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
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            im = ax.imshow(ref_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "ro", markersize = 10, linewidth = 20)
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
            ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        else:
            sns.kdeplot(generated_marginal_density[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(lcs_marginal_density[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(ref_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks(ticks = [.5, 1, 1.5], labels = np.array([.5,1,1.5]))
            ax.tick_params(axis='both', which='major', labelsize=5, labelrotation=0)
            ax.legend(labels = ['NCS', 'LCS'], fontsize = 6)

    plt.savefig(figname)
    plt.clf()


n = 32
range_value = 3.0
model_name = "model4"
smooth_value = 1.5
missing_indices = [650,460,392,497,829]
figname = "figures/br_percentage_lcs_vs_ncs_conditional_marginal_density.png"
nrep = 4000
ps = [.01,.05,.1,.25,.5]
univariate_lcs_file = "univariate_lcs_4000_neighbors_7_nugget_1e5.npy"
produce_generated_and_univariate_lcs_marginal_density_multiple_percentages(n, missing_indices, ps,
                                                                           univariate_lcs_file,
                                                                           figname, nrep)