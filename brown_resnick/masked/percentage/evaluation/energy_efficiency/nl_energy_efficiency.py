import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import *
import torch_nn_architecture
import torch
import zeus
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices = [0,1,2,3])

#This is the transformation of the classifier into psi (which is proportional to the likelihood)
#function parameters:
    #images: numpy matrix of spatial field
    #parameters: numpy matrix of parameters
def multi_psi(images_and_masks, parameters_grid, parameter_classifier):

    classifier_outputs = parameter_classifier.forward(images_and_masks, parameters_grid)
    psi_values = np.zeros(shape = (classifier_outputs.shape[0], 1))
    for i in range(classifier_outputs.shape[0]):
        output = float(classifier_outputs[i,:])
        psi_value = (1-output)/output
        #psi_value = output[1]/(1-output[1])

        psi_values[i,:] = psi_value

    return psi_values

#Produce the neural likelihood surface for the parameter grid over the parameter space
#function parameters:
    #possible_ranges: range values on the parameter grid
    #possible_smooths: smooth values on the parameter grid
    #image: spatial field (numpy matrix)
    #n: the square root of the number of spatial observations
def produce_psi_field(possible_ranges, possible_smooths, image, mask, n, parameter_classifier):

    number_of_parameter_pairs = len(possible_ranges)*len(possible_smooths)
    image_and_mask = np.concatenate([image,mask], axis = 1)
    image_and_mask_matrix = (torch.from_numpy(np.repeat(image_and_mask, number_of_parameter_pairs, axis  = 0))).float()
    ranges = (np.repeat(np.asarray(possible_ranges), 
                               len(possible_smooths), axis = 0)).reshape((number_of_parameter_pairs, 1))
    smooths = []
    smooths = (np.array(sum([(smooths + possible_smooths) for i 
                             in range(0, len(possible_ranges))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = (torch.from_numpy(np.concatenate([ranges, smooths], axis = 1))).float()
    psi_field = (multi_psi(image_and_mask_matrix, parameter_matrix, parameter_classifier)).reshape((len(possible_ranges), len(possible_smooths)))

    return psi_field

def process_images(images):

    images = np.log(images)
    inf_tf_array = np.isinf(images)
    images[inf_tf_array] = 0
    return images


def visualize_image_and_nl_field(log_nl_field, image, filename, range_value, smooth_value, n, range_extent, range_number,
                                 mes):

    image = image.numpy().reshape((n,n))
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
    C = 10
    x = np.linspace(.05, 2, 40)
    y = np.linspace(.05, range_extent, range_number)
    X, Y = np.meshgrid(x, y)
    Z = log_nl_field
    Z[np.isinf(Z)] = np.min(Z[Z != -1*np.inf])
    im = ax[0].imshow(image, vmin = np.min(image), vmax = np.max(image))
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax[1].contourf(X, Y, Z, vmin = (np.amax(Z) - C), vmax = np.amax(Z), levels = 14000)
    #ax[1].scatter(smooth_value, range_value, color = "red")
    ax[1].set_xlabel("smooth")
    ax[1].set_ylabel("range")
    fig.text(.2, .95, ("Time " + str(round(mes.time,1)) + " second, Energy " + str(round(mes.total_energy,2)) + " J."), fontsize = 25)
    plt.savefig(filename)
    plt.clf()

def load_model():

    version_type = "model1"
    n = 20
    image_name = str(n) + "_by_" + str(n)
    classifier_name = "classifier_dropout_rebalanced_parameters_rebalanced_masked_not_boundary_log_lr_0005_param_3000_masks_50_nrep_10_range_2_epochs_10_draws_5"
    classifier = torch_nn_architecture.Net()
    classifier.load_state_dict(torch.load(("models/one_tenth/" + image_name + "/" +
                                           version_type + "/" + classifier_name + ".pth")))
    return classifier
    
def load_data():

    n = 20
    image_name = str(n) + "_by_" + str(n)
    local_folder = "data/one_tenth/" + image_name
    eval_parameters = np.load((local_folder + "/parameters/eval_parameters_range_2_smooth_2_.2_increment.npy"))
    eval_images = np.load((local_folder + "/images/eval_full_images_range_2_smooth_2_.2_increment_nrep_10.npy"))
    eval_masks = np.load((local_folder + "/masks/boundary_masks.npy"))
    return eval_parameters, eval_images, eval_masks

def evaluate_single_surface(iparam, imask):


    range_number = 40
    range_extent = 2.
    possible_ranges = [round(i*.05,2) for i in range(1,(range_number + 1))]
    possible_smooths = [round(i*.05,2) for i in range(1,41)]
    n = 20

    classifier = load_model()
    data_list = load_data()
    eval_parameters = data_list[0]
    eval_images = data_list[1]
    eval_masks = data_list[2]


    range_value = eval_parameters[iparam,0]
    smooth_value = eval_parameters[iparam,1]

    image = torch.from_numpy((process_images(eval_images[iparam,imask,:,:,:])).reshape((1,1,n,n))).float()
    mask = np.transpose(torch.from_numpy((eval_masks[imask,:,:])).reshape((n,n))).reshape((1,1,n,n)).float()
    monitor.begin_window("eval")
    field = produce_psi_field(possible_ranges, possible_smooths, image, mask, n, classifier)
    mes = monitor.end_window("eval")
    file_name = ("nl_image.png")
    visualize_image_and_nl_field(np.log(field), image, file_name, range_value, smooth_value, n, range_extent, range_number, mes)


iparam = 34
imask = 5
evaluate_single_surface(iparam, imask)