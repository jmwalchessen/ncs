import numpy as np
from generate_true_conditional_samples import *
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from paper_figure_helper_functions import *
import seaborn as sns
from matplotlib import gridspec
from matplotlib import patches as mpatches

def produce_marginal_density(model_name, image_name, file_name, missing_index, n, nrep, variance, lengthscale):

    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_marginal_density = (diffusion_images.reshape((nrep,n**2)))[:,missing_index]
    minX = minY = -10
    maxX = maxY = 10
    true_images = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale, observations, nrep)
    true_images = concatenate_observed_and_kriging_samples(observations, true_images, mask, n)
    true_marginal_density = (true_images.reshape((nrep,n**2)))[:,missing_index]
    return true_marginal_density, diffusion_marginal_density

def produce_bivariate_densities(model_name, lengthscale, variance, image_name, nrep,
                                missing_index1, missing_index2, file_name):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    mask = load_mask(model_name, image_name)
    observations = load_observations(model_name, image_name, mask, n)
    diffusion_images = load_diffusion_images(model_name, image_name, file_name)
    conditional_unobserved_samples = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance,
                                                  lengthscale, observations, nrep)
    true_images = concatenate_observed_and_kriging_samples(observations, conditional_unobserved_samples, mask, n)
    true_images = true_images.reshape((nrep,n**2))
    diffusion_images = diffusion_images.reshape((nrep,n**2))
    bivariate_densities = np.concatenate([(true_images[:,missing_index1]).reshape((nrep,1)),
                                          (true_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    diffusion_bivariate_densities = np.concatenate([(diffusion_images[:,missing_index1]).reshape((nrep,1)),
                                          (diffusion_images[:,missing_index2]).reshape((nrep,1))], axis = 1)
    return bivariate_densities, diffusion_bivariate_densities

def visualize_marginal_and_bivariate_density(model_name, missing_indices, missing_indices1, missing_indices2,
                                             n, nrep, variance, figname):

    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    marginal_densities = np.zeros((5,nrep))
    bivariate_densities = np.zeros((5,2*nrep,2))
    diffusion_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) + "_beta_min_max_01_20_random05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        true_marginal_density, diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                                                     missing_indices[i], n, nrep, variance,
                                                                                     lengthscales[i])
        bdensities, dbdensities = produce_bivariate_densities(model_name, lengthscales[i], variance,
                                                                                         image_name, nrep, missing_indices1[i],
                                                                                         missing_indices2[i], file_name)
        marginal_densities[i,:] = true_marginal_density
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        bivariate_densities[i,:,:] = np.concatenate([bdensities,dbdensities], axis = 0)

    class_vector = np.concatenate([(np.ones((nrep))).reshape((nrep,1)),
                                   (np.zeros((nrep)).reshape((nrep,1)))], axis = 0)
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(6)
 
    # set width of each subplot as 8
    fig.set_figwidth(10)
    spec = gridspec.GridSpec(ncols=5, nrows=3,
                         width_ratios=[1,1,1,1,1], wspace=0.1,
                         hspace=0.25, height_ratios=[1, 1,1])
    for i in range(0,15):
        ax = fig.add_subplot(spec[i])
        if(i < 5):
            matrix_index = index_to_matrix_index(missing_indices[i], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[i], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[i], n)
            im = ax.imshow(reference_images[i,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[i,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
            if(i == 0):
                ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_yticks([])
            if((i == 0) | (i == 4)):
                ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])

        elif(i < 10):
            matrix_index = index_to_matrix_index(missing_indices[(i%5)], n)
            sns.kdeplot(marginal_densities[(i % 5),:], ax = ax, color = "blue")
            sns.kdeplot(diffusion_marginal_densities[(i % 5),:], ax = ax, color = 'orange', linestyle = '--')
            ax.axvline(reference_images[(i%5),matrix_index[0],matrix_index[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-4.5,4.5])
            ax.set_ylim([0,1])
            ax.set_ylabel("")
            if(i == 5):
                ax.set_yticks([0.,.5,1.], [0.,.5,1.], fontsize = 15)
                ax.legend(labels = ['true', 'NCS'], fontsize = 12)
            else:
                ax.set_yticks([])
            if((i == 5) | (i == 9)):
                ax.set_xticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)
            else:
                ax.set_xticks([])
        else:
            missing_index1 = missing_indices1[(i%5)]
            missing_index2 = missing_indices2[(i%5)]
            matrix_index1 = index_to_matrix_index(missing_indices1[(i%5)], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[(i%5)], n)
            pdd = pd.DataFrame(np.concatenate([bivariate_densities[(i%5),:,:], class_vector],axis = 1),
                                    columns = ['x', 'y', 'class'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64', 'class': 'float64'})
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = ax, hue = 'class',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5,
                               legend = False)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            ax.axvline(reference_images[(i%5),matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[(i%5),matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-4.5,4.5])
            ax.set_ylim([-4.5,4.5])
            ax.set_ylabel("")
            ax.set_xlabel("")
            if(i == 10):
                ax.set_yticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)
            else:
                ax.set_yticks([])
            if(i == 14):
                ax.legend(handles = [blue_patch, orange_patch], labels = ['true', 'NCS'], fontsize = 12)
            ax.set_xticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)

    #fig.colorbar(im, shrink = 1)
    fig.text(x = .38, y = .9, s = "Partially Observed Field", fontsize = 15)
    fig.text(x = .35, y = .62, s = "Conditional Marginal Density", fontsize = 15)
    fig.text(x = .35, y = .34, s = "Conditional Bivariate Density", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()


def visualize_marginal_and_bivariate_density_transposed(model_name, missing_indices, missing_indices1, missing_indices2,
                                             n, nrep, variance, figname):

    lengthscales = [1.0,2.0,3.0,4.0,5.0]
    masks = np.zeros((5,n,n))
    reference_images = np.zeros((5,n,n))
    marginal_densities = np.zeros((5,nrep))
    bivariate_densities = np.zeros((5,2*nrep,2))
    diffusion_marginal_densities = np.zeros((5,nrep))
    for i in range(0, 5):
        file_name = (model_name + "_variance_" + str(variance) + "_lengthscale_" + str(lengthscales[i]) + "_beta_min_max_01_20_random05_4000")
        image_name = "ref_image" + str(i)
        masks[i,:,:] = load_mask(model_name, image_name)
        reference_images[i,:,:] = load_reference_image(model_name, image_name)
        true_marginal_density, diffusion_marginal_density = produce_marginal_density(model_name, image_name, file_name,
                                                                                     missing_indices[i], n, nrep, variance,
                                                                                     lengthscales[i])
        bdensities, dbdensities = produce_bivariate_densities(model_name, lengthscales[i], variance,
                                                                                         image_name, nrep, missing_indices1[i],
                                                                                         missing_indices2[i], file_name)
        marginal_densities[i,:] = true_marginal_density
        diffusion_marginal_densities[i,:] = diffusion_marginal_density
        bivariate_densities[i,:,:] = np.concatenate([bdensities,dbdensities], axis = 0)

    class_vector = np.concatenate([(np.ones((nrep))).reshape((nrep,1)),
                                   (np.zeros((nrep)).reshape((nrep,1)))], axis = 0)
    fig = plt.figure()
    # set height of each subplot as 8
    fig.set_figheight(9)
 
    # set width of each subplot as 8
    fig.set_figwidth(6)
    spec = gridspec.GridSpec(ncols=3, nrows=5,
                         width_ratios=[1,1,1], wspace=0.1,
                         hspace=0.1, height_ratios=[1,1,1,1,1])
    counter = -1
    for i in range(0,15):
        ax = fig.add_subplot(spec[i])
        if((i%3)==0):
            counter = counter + 1
            matrix_index = index_to_matrix_index(missing_indices[counter], n)
            matrix_index1 = index_to_matrix_index(missing_indices1[counter], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[counter], n)
            im = ax.imshow(reference_images[counter,:,:], cmap = 'viridis', vmin = -4, vmax = 4, alpha = masks[counter,:,:].astype(float))
            ax.plot(matrix_index[1], matrix_index[0], "rP", markersize = 15, linewidth = 20)
            ax.plot(matrix_index1[1], matrix_index1[0], "r*", markersize = 15, linewidth = 20)
            ax.plot(matrix_index2[1], matrix_index2[0], "r*", markersize = 15, linewidth = 20)
            if((i%2)==0):
                ax.set_yticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_yticks(ticks = [7, 15, 23], labels = np.array([-5,0,5]), fontsize = 15)
            if(i == 12):
                ax.set_xticks(ticks = [0, 7, 15, 23, 31], labels = np.array([-10,-5,0,5,10]), fontsize = 15)
            else:
                ax.set_xticks([])

        elif((i % 3) == 1):
            matrix_index = index_to_matrix_index(missing_indices[counter], n)
            sns.kdeplot(marginal_densities[counter,:], ax = ax, color = "blue")
            sns.kdeplot(diffusion_marginal_densities[counter,:], ax = ax, color = 'orange', linestyle = '--')
            ax.axvline(reference_images[counter,matrix_index[0],matrix_index[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-4.5,4.5])
            ax.set_ylim([0,1])
            ax.set_ylabel("")
            ax.set_yticks([])
            if(i == 1):
                ax.legend(labels = ['true', 'NCS'], fontsize = 12)
            if(i == 13):
                ax.set_xticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)
            else:
                ax.set_xticks([])
            #if(i == 5):
                #ax.set_yticks([0.,.5,1.], [0.,.5,1.], fontsize = 15)
                #ax.legend(labels = ['true', 'NCS'], fontsize = 12)
            #else:
                #ax.set_yticks([])
            #if((i == 5) | (i == 9)):
                #ax.set_xticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)
            #else:
                #ax.set_xticks([])
        else:
            missing_index1 = missing_indices1[counter]
            missing_index2 = missing_indices2[counter]
            matrix_index1 = index_to_matrix_index(missing_indices1[counter], n)
            matrix_index2 = index_to_matrix_index(missing_indices2[counter], n)
            pdd = pd.DataFrame(np.concatenate([bivariate_densities[counter,:,:], class_vector],axis = 1),
                                    columns = ['x', 'y', 'class'])
            pdd = pdd.astype({'x': 'float64', 'y': 'float64', 'class': 'float64'})
            kde1 = sns.kdeplot(data = pdd, x = 'x', y = 'y', ax = ax, hue = 'class',
                               fill = False, levels = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99], alpha = .5,
                               legend = False)
            blue_patch = mpatches.Patch(color='blue')
            orange_patch = mpatches.Patch(color='orange')
            ax.axvline(reference_images[counter,matrix_index1[0],matrix_index1[1]], color='red', linestyle = 'dashed')
            ax.axhline(reference_images[counter,matrix_index2[0],matrix_index2[1]], color='red', linestyle = 'dashed')
            ax.set_xlim([-4.5,4.5])
            ax.set_ylim([-4.5,4.5])
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])
            #if(i == 10):
                #ax.set_yticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)
            #else:
                #ax.set_yticks([])
            if(i == 14):
                ax.set_xticks([-2.5,0.,2.5], [-2.5,0.,2.5], fontsize = 15)
            else:
                ax.set_xticks([])
            #ax.set_xticks([-4,-2,0,2,4], [-4,-2,0,2,4], fontsize = 15)

    #fig.colorbar(im, shrink = 1)
    fig.text(x = .13, y = .89, s = "Partially Obs.", fontsize = 15)
    fig.text(x = .38, y = .89, s = "Cond. Marginal", fontsize = 15)
    fig.text(x = .65, y = .89, s = "Cond. Bivariate", fontsize = 15)
    fig.text(x = .36, y = .93, s = "Parameter U-Net", fontsize = 15)
    plt.tight_layout()
    plt.savefig(figname, dpi = 500)
    plt.clf()

def visualize_marginal_and_bivariate_density_with_variables():
    model_name = "model7"
    missing_indices = [845, 700, 200, 301, 118]
    n = 32
    nrep = 4000
    variance = 1.5
    figname = "figures/gp_parameter_marginal_bivariate_density_model7_random05.png"
    missing_indices1 = [232,360,658,493,567]
    missing_indices2 = [233,362,654,505,568]
    visualize_marginal_and_bivariate_density(model_name, missing_indices, missing_indices1, missing_indices2,
                                                n, nrep, variance, figname)
    

def visualize_marginal_and_bivariate_density_transposed_with_variables():
    model_name = "model7"
    missing_indices = [845, 700, 200, 301, 118]
    n = 32
    nrep = 4000
    variance = 1.5
    figname = "figures/gp_parameter_marginal_bivariate_density_model7_random05_transposed.png"
    missing_indices1 = [232,360,658,493,567]
    missing_indices2 = [233,362,654,505,568]
    visualize_marginal_and_bivariate_density_transposed(model_name, missing_indices, missing_indices1, missing_indices2,
                                                n, nrep, variance, figname)
    
visualize_marginal_and_bivariate_density_with_variables()
visualize_marginal_and_bivariate_density_transposed_with_variables()