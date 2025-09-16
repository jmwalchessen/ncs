import numpy as np
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
from matplotlib import gridspec
import pandas as pd
from matplotlib import patches as mpatches

evaluation_folder = append_directory(2)
data_generation_folder = (evaluation_folder + "/diffusion_generation")

def index_to_matrix_index(index, n):
    return (int(index / n), int(index % n))

def produce_marginal_density(model_name, image_name, file_name, missing_index, n, nrep, variance, lengthscale):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    return diffusion_marginal_density


def produce_bivariate_densities(model_name, lengthscale, variance, image_name, nrep,
                                missing_index1, missing_index2, file_name):

    n = 32
    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    diffusion_images = diffusion_images.reshape((nrep,n**2))
    diffusion_bivariate_densities = np.concatenate([(diffusion_images[:,missing_index1]).reshape((nrep,1)),
                                          (diffusion_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return diffusion_bivariate_densities

def produce_bivariate_densities_in_fcs_folder(file_name, nrep,
                                              missing_index1, missing_index2,
                                              folder_name):

    n = 32
    eval_folder = append_directory(2)
    images = np.load(eval_folder + "/" + folder_name + "/" + file_name)
    images = images.reshape((nrep,n**2))
    bivariate_densities = np.concatenate([(images[:,missing_index1]).reshape((nrep,1)),
                                          (images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return bivariate_densities

def produce_marginal_densities_in_fcs_folder(file_name, nrep, missing_index, folder_name):

    n = 32
    eval_folder = append_directory(2)
    images = np.load(eval_folder + "/" + folder_name + "/" + file_name)
    images = images.reshape((nrep,n**2))
    marginal_densities = images[:,missing_index].reshape((nrep))
    return marginal_densities


def visualize_conditional_marginal_bivariate_density(model_name, range_value, smooth, nrep, missing_indices,
                                                     missing_indices1, missing_indices2, univariate_lcs_file,
                                                     bivariate_lcs_file, n, figname):
    
    ps = [.01,.05,.1,.25,.5]
    diffusion_bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    lcs_marginal_density = np.zeros((len(ps),nrep))
    univariate_lcs_images = np.zeros((5,nrep,n,n))
    lcs_bivariate_density = np.zeros((5,nrep,2))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        missing_index = missing_indices[i]
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_4000_random" + str(ps[i]))
        dbdensities = produce_bivariate_densities(model_name, range_value, smooth, 
                                                              image_name, nrep, missing_indices1[i],
                                                              missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_bivariate_densities[i,:,:] = dbdensities
        diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                              missing_indices[i], n, nrep, range_value,
                                                              smooth)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_images[i,:,:,:] = (np.load((data_generation_folder + ref_image_folder + "/lcs/univariate/" + 
                                                   univariate_lcs_file))).reshape((nrep,n,n))
        lcs_marginal_density[i,:] = univariate_lcs_images[i,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                               bivariate_lcs_file + "_random" + str(ps[i]) + "_" + str(missing_indices1[i])
                               + "_" + str(missing_indices2[i]) + ".npy")))
        lcs_bivariate_density[i,:,:] = bilcs


#fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=3,
                         width_ratios=[1,1,1,1,1], wspace=0.15,
                         hspace=0.25, height_ratios=[1, 1, 1])

    for i in range(0, 15):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            if(i == 0):
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_yticks([])
            if(((i == 0)) | (i == 4)):
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])

        elif(i < 10):
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(lcs_marginal_density[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            if(i == 5):
                ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 15)
                ol = mpatches.Patch(color='orange', lw = .2)
                pl = mpatches.Patch(color='purple', lw = .2)
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
            else:
                ax.set_yticks([])
            if((i == 5) | (i == 9)):
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            else:
                ax.set_xticks([])
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            ax.set_ylabel("")
            if(i == 10):
                ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
                a1 = sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                a2 = sns.kdeplot(x = diffusion_bivariate_densities[(i%5),:,0], y = diffusion_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                ol = mpatches.Patch(color='orange', lw = .2)
                pl = mpatches.Patch(color='purple', lw = .2)
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
                ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
                ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            else:
                sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                sns.kdeplot(x = diffusion_bivariate_densities[(i%5),:,0], y = diffusion_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
                ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
                ax.set_yticks([])
            if((i % 2) == 0):
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            else:
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)

    fig.text(x = .38, y = .9, s = "Partially Observed Field", fontsize = 15)
    fig.text(x = .35, y = .62, s = "Conditional Marginal Density", fontsize = 15)
    fig.text(x = .35, y = .34, s = "Conditional Bivariate Density", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()


def visualize_conditional_marginal_bivariate_density_other_grid(model_name, range_value, smooth, nrep, missing_indices,
                                                     missing_indices1, missing_indices2, univariate_lcs_file,
                                                     bivariate_lcs_file, n, figname):
    
    ps = [.01,.05,.1,.25,.5]
    diffusion_bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    lcs_marginal_density = np.zeros((len(ps),nrep))
    univariate_lcs_images = np.zeros((5,nrep,n,n))
    lcs_bivariate_density = np.zeros((5,nrep,2))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        missing_index = missing_indices[i]
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_4000_random" + str(ps[i]))
        dbdensities = produce_bivariate_densities(model_name, range_value, smooth, 
                                                              image_name, nrep, missing_indices1[i],
                                                              missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_bivariate_densities[i,:,:] = dbdensities
        diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                              missing_indices[i], n, nrep, range_value,
                                                              smooth)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_images[i,:,:,:] = (np.load((data_generation_folder + ref_image_folder + "/lcs/univariate/" + 
                                                   univariate_lcs_file))).reshape((nrep,n,n))
        lcs_marginal_density[i,:] = univariate_lcs_images[i,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                               bivariate_lcs_file + "_random" + str(ps[i]) + "_" + str(missing_indices1[i])
                               + "_" + str(missing_indices2[i]) + ".npy")))
        lcs_bivariate_density[i,:,:] = bilcs

    fig = plt.figure(figsize=(10,6))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode=None
                 )


    for i,ax in enumerate(grid):
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            if(i == 0):
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_yticks([])
            if((i%2) == 0):
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks(ticks = [0,8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)

        elif(i < 10):
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(lcs_marginal_density[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(reference_images[(i%5),matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            if(i == 5):
                ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 15)
                ol = mpatches.Patch(color='orange', lw = .2)
                pl = mpatches.Patch(color='purple', lw = .2)
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
            else:
                ax.set_yticks([])
            if((i % 2) == 1):
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            else:
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            ax.set_ylabel("")
            if(i == 10):
                ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
                a1 = sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                a2 = sns.kdeplot(x = diffusion_bivariate_densities[(i%5),:,0], y = diffusion_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                ol = mpatches.Patch(color='orange', lw = .2)
                pl = mpatches.Patch(color='purple', lw = .2)
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
                ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
                ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            else:
                sns.kdeplot(x = lcs_bivariate_density[(i%5),:,0], y = lcs_bivariate_density[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                sns.kdeplot(x = diffusion_bivariate_densities[(i%5),:,0], y = diffusion_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
                ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
                ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
                ax.set_yticks([])
            if((i % 2) == 0):
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            else:
                ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)

    fig.text(x = .4, y = .35, s = "Conditional Marginal Density", fontsize = 15)
    plt.savefig(figname)
    plt.clf()


def visualize_conditional_fcs_ncs_marginal_bivariate_density(model_version, range_value):
    
    missing_indices = [650,460,366,520,829]
    missing_indices1 = [390,586,144,203,333]
    missing_indices2 = [303,552,173,186,220]

    obs_numbers = [1,2,3,5,7]
    smooth = 1.5
    nrep = 4000
    n = 32

    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    ncs_marginal_densities = np.zeros((len(obs_numbers),nrep))
    ncs_bivariate_densities = np.zeros((len(obs_numbers),nrep,2))
    fcs_marginal_densities = np.zeros((len(obs_numbers),nrep))
    fcs_bivariate_densities = np.zeros((len(obs_numbers),nrep,2))
    fcs_images = np.zeros((5,nrep,n,n))
    ncs_images = np.zeros((5,nrep,n,n))

    figname = "figures/one_to_seven/fcs_vs_ncs_conditional_marginal_bivariate_density_range_" + str(range_value) + ".png"

    for i in range(0,5):

        obs = obs_numbers[i]
        image_name = "ref_image" + str(i)
        evaluation_folder = append_directory(2)
        ref_image_folder = ("fcs/data/conditional/obs" + str(obs) + "/ref_image" + str(int(range_value-1)))
        missing_index = missing_indices[i]
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        file_name = ("diffusion/model" + str(model_version) + "_range_" + str(range_value) + "_smooth_1.5_4000_random.npy")
        ncs_marginal_densities[i,:] = produce_marginal_densities_in_fcs_folder(file_name, nrep, missing_index, ref_image_folder)
        ncs_bivariate_densities[i,:,:] = produce_bivariate_densities_in_fcs_folder(file_name, nrep, missing_indices1[i],
                                                                                   missing_indices2[i], ref_image_folder)
        ncs_images[i,:,:,:] = (np.load((evaluation_folder + "/" + ref_image_folder + "/" + file_name))).reshape((nrep,n,n))
        masks[i,:,:] = np.load((evaluation_folder + "/" + ref_image_folder + "/mask.npy"))
        reference_images[i,:,:] = np.log(np.load((evaluation_folder + "/" + ref_image_folder + "/ref_image.npy")))
        file_name = ("processed_log_scale_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(obs) + "_" + str(nrep) + ".npy")
        fcs_marginal_densities[i,:] = produce_marginal_densities_in_fcs_folder(file_name, nrep, missing_index, ref_image_folder)
        fcs_bivariate_densities[i,:,:] = produce_bivariate_densities_in_fcs_folder(file_name, nrep, missing_indices1[i],
                                                                                   missing_indices2[i], ref_image_folder)
        fcs_images[i,:,:,:] = (np.load((evaluation_folder + "/" + ref_image_folder + "/" +  file_name)))


#fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=3,
                         width_ratios=[1,1,1,1,1], wspace=0.15,
                         hspace=0.25, height_ratios=[1, 1, 1])

    for i in range(0, 15):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize =15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "g*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index[1], matrix_index[0], "bP", markersize = 15, linewidth = 20)
            if(i == 0):
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_yticks([])
            ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)

        elif(i < 10):
            matrix_index = index_to_matrix_index(missing_indices[i%5], n)
            sns.kdeplot(ncs_marginal_densities[(i % 5),:], ax = ax, color = 'orange')
            sns.kdeplot(fcs_marginal_densities[(i % 5),:], ax = ax, color = 'purple')
            ax.axvline(reference_images[(i%5),matrix_index[0],matrix_index[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-3,8])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            if(i == 5):
                ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 15)
            else:
                ax.set_yticks([])
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6])
            ax.legend(labels = ['NCS', 'FCS'], fontsize = 12)
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            sns.kdeplot(x = fcs_bivariate_densities[(i%5),:,0], y = fcs_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            sns.kdeplot(x = ncs_bivariate_densities[(i%5),:,0], y = ncs_bivariate_densities[(i%5),:,1],
                    ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-3,8])
            ax.set_ylim([-3,8])
            ax.set_ylabel("")
            if(i == 10):
                ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            else:
                ax.set_yticks([])
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 15)
            purple_patch = mpatches.Patch(color='purple')
            orange_patch = mpatches.Patch(color='orange')
            ax.legend(handles = [purple_patch, orange_patch], labels = ['LCS', 'NCS'], fontsize = 15)

    fig.text(x = .4, y = .35, s = "Conditional Marginal Density", fontsize = 15)
    plt.savefig(figname)
    plt.clf()



def visualize_conditional_marginal_bivariate_density_transposed(model_name, range_value, smooth, nrep, missing_indices,
                                                     missing_indices1, missing_indices2, univariate_lcs_file,
                                                     bivariate_lcs_file, n, figname):
    
    ps = [.01,.05,.1,.25,.5]
    diffusion_bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    diffusion_marginal_densities = np.zeros((5,nrep))
    lcs_marginal_density = np.zeros((len(ps),nrep))
    univariate_lcs_images = np.zeros((5,nrep,n,n))
    lcs_bivariate_density = np.zeros((5,nrep,2))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        ref_image_folder = ("/data/model4/ref_image" + str(i))
        missing_index = missing_indices[i]
        matrix_missing_index = index_to_matrix_index(missing_index, n)
        file_name = (model_name + "_range_" + str(range_value) + "_smooth_" + str(smooth) + "_4000_random" + str(ps[i]))
        dbdensities = produce_bivariate_densities(model_name, range_value, smooth, 
                                                              image_name, nrep, missing_indices1[i],
                                                              missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        diffusion_bivariate_densities[i,:,:] = dbdensities
        diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                              missing_indices[i], n, nrep, range_value,
                                                              smooth)
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        univariate_lcs_images[i,:,:,:] = (np.load((data_generation_folder + ref_image_folder + "/lcs/univariate/" + 
                                                   univariate_lcs_file))).reshape((nrep,n,n))
        lcs_marginal_density[i,:] = univariate_lcs_images[i,:,int(matrix_missing_index[0]),int(matrix_missing_index[1])]
        bilcs = np.log(np.load((data_generation_folder + ref_image_folder + "/lcs/bivariate/" +
                               bivariate_lcs_file + "_random" + str(ps[i]) + "_" + str(missing_indices1[i])
                               + "_" + str(missing_indices2[i]) + ".npy")))
        lcs_bivariate_density[i,:,:] = bilcs


#fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(9)
 
    # set width of each subplot as 8
    fig.set_figwidth(5.5)
    spec = gridspec.GridSpec(ncols=3, nrows=5,
                         width_ratios=[1,1,1], wspace=0.3,
                         hspace=0.25, height_ratios=[1, 1, 1, 1, 1])

    counter = -1
    for i in range(0, 15):
        ax = fig.add_subplot(spec[i])
        if((i % 3) == 0):
            counter = counter + 1
            matrix_index = index_to_matrix_index(missing_indices[counter], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[counter], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[counter], n)
            im = ax.imshow(reference_images[counter,:,:], cmap = 'viridis', vmin = -2, vmax = 6, alpha = masks[counter,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 12)
            ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 12)

        elif((i % 3) == 1):
            sns.kdeplot(lcs_marginal_density[counter,:], ax = ax, color = 'purple')
            sns.kdeplot(diffusion_marginal_densities[counter,:], ax = ax, color = 'orange', linestyle = "dashed")
            ax.axvline(reference_images[counter,matrix_index[1],matrix_index[0]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([0,1.75])
            ax.set_ylabel("")
            ax.set_yticks([0.,.5,1.,1.5], [0.,.5,1.,1.5], fontsize = 12)
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 12)
            ol = mpatches.Patch(color='orange', lw = .2)
            pl = mpatches.Patch(color='purple', lw = .2)
            if(i == 1):
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
        else:
            matrix_index1 = index_to_matrix_index(missing_indices1[counter], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[counter], n)
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])
            ax.set_ylabel("")
            ax.set_yticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 12)
            a1 = sns.kdeplot(x = lcs_bivariate_density[counter,:,0], y = lcs_bivariate_density[counter,:,1],
                ax = ax, color = 'purple', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            a2 = sns.kdeplot(x = diffusion_bivariate_densities[counter,:,0], y = diffusion_bivariate_densities[counter,:,1],
                ax = ax, color = 'orange', levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5)
            ol = mpatches.Patch(color='orange', lw = .2)
            pl = mpatches.Patch(color='purple', lw = .2)
            if(i == 2):
                ax.legend(handles = [ol,pl],labels = ['NCS', 'LCS'], fontsize = 12)
            ax.axvline(reference_images[counter,matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[counter,matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 12)
            ax.set_xticks([-2,0,2,4,6], [-2,0,2,4,6], fontsize = 12)

    fig.text(x = .32, y = .89, s = "Proportion U-Net", fontsize = 18)
    plt.tight_layout()
    plt.savefig(figname, dpi = 500)
    plt.clf()



def visualize_conditional_marginal_bivariate_with_variables():

    model_name = "model4"
    nrep = 4000
    missing_indices = [100,245,874,568,398]
    #390,303
    #600,632
    missing_indices1 = [375,303,618,710,245]
    missing_indices2 = [439,326,689,675,268]
    n = 32
    bivariate_lcs_file = "bivariate_lcs_4000_neighbors_7_nugget_1e5"
    univariate_lcs_file = "univariate_lcs_4000_neighbors_7_nugget_1e5.npy"
    range_value = 3.
    smooth = 1.5
    figname = "figures/br_percentage_lcs_vs_ncs_conditional_marginal_bivariate_density.png"
    visualize_conditional_marginal_bivariate_density(model_name, range_value, smooth, nrep, missing_indices,
                                                        missing_indices1, missing_indices2, univariate_lcs_file,
                                                        bivariate_lcs_file, n, figname)
    

def visualize_conditional_marginal_bivariate_transposed_with_variables():

    model_name = "model4"
    nrep = 4000
    missing_indices = [100,245,874,568,398]
    #390,303
    #600,632
    missing_indices1 = [375,303,618,710,245]
    missing_indices2 = [439,326,689,675,268]
    n = 32
    bivariate_lcs_file = "bivariate_lcs_4000_neighbors_7_nugget_1e5"
    univariate_lcs_file = "univariate_lcs_4000_neighbors_7_nugget_1e5.npy"
    range_value = 3.
    smooth = 1.5
    figname = "figures/br_percentage_lcs_vs_ncs_conditional_marginal_bivariate_density_transposed.png"
    visualize_conditional_marginal_bivariate_density_transposed(model_name, range_value, smooth, nrep, missing_indices,
                                                     missing_indices1, missing_indices2, univariate_lcs_file,
                                                     bivariate_lcs_file, n, figname)
    
visualize_conditional_marginal_bivariate_transposed_with_variables()