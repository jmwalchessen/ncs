import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
from matplotlib import patches as mpatches
from matplotlib import gridspec
import pandas as pd
import seaborn as sns


def produce_bivariate_densities(model_name, image_name, nrep,missing_index1, missing_index2, file_name):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    diffusion_images = diffusion_images.reshape((nrep,n**2))
    diffusion_bivariate_densities = np.concatenate([(diffusion_images[:,missing_index1]).reshape((nrep,1)),
                                          (diffusion_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return diffusion_bivariate_densities

def visualize_bivariate_density(model_name, nrep, missing_indices1,
                                missing_indices2, n, figname, smooth):
    
    range_values = [1.0,2.0,3.0,4.0,5.0]
    bivariate_densities = np.zeros((5,nrep,2))
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))

    for i in range(0,5):

        image_name = "ref_image" + str(i)
        file_name = (model_name + "_range_" + str(range_values[i]) + "_smooth_" + str(smooth) + "_random0.5_4000")
        dbdensities = produce_bivariate_densities(model_name, image_name, nrep, missing_indices1[i], missing_indices2[i], file_name)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        bivariate_densities[i,:,:] = dbdensities
    
    #fig, axs = plt.subplots(ncols = 5, nrows = 2, figsize = (9,2.5))
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=2,
                         width_ratios=[1,1,1,1,1], wspace=0.25,
                         hspace=0.25, height_ratios=[1, 1])
    for i in range(0,10):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            missing_index1 = missing_indices1[i]
            missing_index2 = missing_indices2[i]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index1[1], matrix_index1[0], "r^", markersize = 10, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "k^", markersize = 10, linewidth = 20)
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_index1, n)
            matrix_index2 = index_to_matrix_index(missing_index2, n)
            pdd = pd.DataFrame(bivariate_densities[(i%5),:,:], columns = ['x', 'y'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64'})
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = ax, palette="orange",
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = 1)
            orange_patch = mpatches.Patch(color='orange')
            ax.legend(handles = [orange_patch], labels = ['diffusion'])
            ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-2,6])
            ax.set_ylim([-2,6])

    #fig.colorbar(im, ax=ax, shrink = .6)
    plt.savefig(figname)
    plt.clf()


model_name = "model3"
smooth = 1.5
nrep = 4000
missing_indices1 = [232,772,810,327,567]
missing_indices2 = [233,835,874,390,568]
figname = "figures/br_parameter_close_bivairate_density.png" 
n = 32
visualize_bivariate_density(model_name, nrep, missing_indices1,
                            missing_indices2, n, figname, smooth) 