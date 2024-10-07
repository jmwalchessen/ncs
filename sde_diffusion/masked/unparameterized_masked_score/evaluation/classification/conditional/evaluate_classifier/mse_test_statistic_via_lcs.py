import numpy as np
import torch as th
import matplotlib.pyplot as plt
import os
import sys
from evaluation_helper_functions import *
from append_directories import *
import seaborn as sns
conditional_classifier_folder = append_directory(2)
train_nn_folder = (conditional_classifier_folder + "/train_classifier/train_nn")
sys.path.append(train_nn_folder)

def compute_mse_test_stastic(model_name, evaluation_file_name, n, crop_size,
                     classifier_name, classifier_file, calibrated_model_name,
                     calibrated_model_file):

    device = "cuda:0"
    eval_diffusion_images = load_images(model_name, evaluation_file_name)
    eval_diffusion_images = crop_image(eval_diffusion_images, n, crop_size)
    classifier = load_classifier(device, classifier_name, classifier_file)
    classifier_outputs = (classifier(eval_diffusion_images.float().to(device)))
    classifier_logits = logit_transformation_with_sigmoid(classifier_outputs)
    calibrated_probabilities = produce_calibrated_probabilities(classifier_logits, calibrated_model_name, calibrated_model_file)
    mse_test_statistic = np.mean(np.square((calibrated_probabilities - .5)))

    return mse_test_statistic


def compute_single_value_in_null_distribution(num_epochs, classifier, weight_decay, beta1, beta2, epsilon,
                                              loss_function, device, initial_learning_rate, diffusion_images_pathname, split,
                                              num_samples, batch_size, eval_batch_size, crop_size, calibrated_model_name,
                                              calibrated_model_file, model_name, evaluation_file_name, n):

    random_classifier, eval_losses, eval_train_losses = train_nn(num_epochs, classifier, weight_decay, beta1, beta2, epsilon,
                                                                 loss_function, device, initial_learning_rate, diffusion_images_pathname,
                                                                 split, num_samples, batch_size, eval_batch_size, crop_size,
                                                                 shuffle = False, masks_path_name = None)
    eval_diffusion_images = load_images(model_name, evaluation_file_name)
    eval_diffusion_images = crop_image(eval_diffusion_images, n, crop_size)
    random_classifier_outputs = (random_classifier(eval_diffusion_images.float().to(device)))
    random_classifier_logits = logit_transformation_with_sigmoid(random_classifier_outputs)
    calibrated_probabilities = produce_calibrated_probabilities(random_classifier_logits, calibrated_model_name, calibrated_model_file)
    mse_test_null = np.mean(np.square((calibrated_probabilities - .5)))
    return mse_test_null

def compute_null_distribution(nsim, num_epochs, classifier, weight_decay, beta1, beta2, epsilon,loss_function, device,
                              initial_learning_rate, diffusion_images_pathname, split, num_samples, batch_size,
                              eval_batch_size, crop_size, calibrated_model_name, calibrated_model_file, model_name,
                              evaluation_file_name, n):
    
    null_distribution = np.zeros((nsim))

    for i in range(0, nsim):
        null_distribution[i] = compute_single_value_in_null_distribution(num_epochs, classifier, weight_decay, beta1, beta2, epsilon,
                                                                  loss_function, device, initial_learning_rate, diffusion_images_pathname, split,
                                                                  num_samples, batch_size, eval_batch_size, crop_size, calibrated_model_name,
                                                                  calibrated_model_file, model_name, evaluation_file_name, n)
    return null_distribution

def compute_p_value(nsim, num_epochs, classifier, weight_decay, beta1, beta2, epsilon,loss_function, device,
                    initial_learning_rate, diffusion_images_pathname, split, num_samples, batch_size,
                    eval_batch_size, crop_size, calibrated_model_name, calibrated_model_file, model_name,
                    evaluation_file_name, n, classifier_name, classifier_file):


    mse_test_statistic = compute_mse_test_stastic(model_name, evaluation_file_name, n, crop_size, classifier_name,
                                                  classifier_file, calibrated_model_name, calibrated_model_file)
    
    mse_null_distribution = compute_null_distribution(nsim, num_epochs, classifier, weight_decay, beta1, beta2, epsilon,loss_function,
                                                      device, initial_learning_rate, diffusion_images_pathname, split, num_samples, batch_size,
                                                      eval_batch_size, crop_size, calibrated_model_name, calibrated_model_file, model_name,
                                                      evaluation_file_name, n)
    
    p_value = np.mean((mse_null_distribution >= mse_test_statistic))
    return p_value