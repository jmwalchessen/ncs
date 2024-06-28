import numpy as np
import torch as th
import os
import sys
from append_directories import *
classifier_folder = append_directory(2)
train_nn_folder = (classifier_folder + "/train_classifier/train_nn")
sys.path.append(train_nn_folder)
from nn_architecture import *
unconditional_folder = append_directory(3)
data_generation_folder = (unconditional_folder + "/generate_data")
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *

def crop_image(images, n, crop_size):

    cropped_images = images[:,:,crop_size:(n-crop_size),crop_size:(n-crop_size)]
    return cropped_images

def load_images(path):

    evaluate_data_classifier_folder = (classifier_folder + "/evaluate_classifier/evaluation_data/data/diffusion/")
    images = th.from_numpy(np.load((evaluate_data_classifier_folder + path)))
    images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))
    return images

def load_classifier(device, model_name):

    classifier = (CNNClassifier()).to(device)
    classifier.load_state_dict(th.load((train_nn_folder + "/models/" + model_name)))
    return classifier

def load_smaller_classifier(device, model_name):

    smallclassifier = (SmallCNNClassifier()).to(device)
    smallclassifier.load_state_dict(th.load((train_nn_folder + "/models/small_classifier/"
                                             + model_name)))
    return smallclassifier

def generate_true_images(number_of_replicates, seed_value):

    true_images = generate_gaussian_process(minX = -10, maxX = 10, minY = -10, maxY = 10, n = 32,
                                         variance = .4, lengthscale = 1.6, number_of_replicates = number_of_replicates,
                                         seed_value = seed_value)[1]
    true_images = th.from_numpy(true_images)
    return true_images

def create_evaluation_images(number_of_replicates, seed_value, path, n, crop_size):

    true_images = generate_true_images(number_of_replicates, seed_value)
    diffusion_images = load_images(path)
    images = th.cat([diffusion_images, true_images], dim = 0)
    images = crop_image(images, n, crop_size)
    return images

def create_classes(number_of_replicates, device):

    true_classes = ((th.from_numpy((np.concatenate([np.ones(number_of_replicates), np.zeros(number_of_replicates)],
                                               axis = 0)))).to(device))
    return true_classes

