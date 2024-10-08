import torch as th
import numpy as np
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import sys
from evaluation_helper_functions import *
from append_directories import *
import seaborn as sns


def permute_class_labels(nrep, device):

    class_labels = create_classes(nrep, device)
    permuted_class_labels = np.random.permutation(class_labels)
    return permuted_class_labels

def compute_auc_test_statistic(number_of_replicates, model_name, evaluation_file_name,
                               classifier, classifier_name, classifier_file, n, crop_size,
                               calibrated_model_name, calibrated_model_file):

    device = "cuda:0"
    true_classes = create_classes(number_of_replicates, device)
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    eval_images = create_evaluation_diffusion_images(number_of_replicates, model_name, evaluation_file_name, n, crop_size)
    classifier_outputs = (classifier(eval_images.float().to(device)))
    classifier_logits = logit_transformation_with_sigmoid(classifier_outputs)
    calibrated_probabilities = produce_calibrated_probabilities(classifier_logits, calibrated_model_name, calibrated_model_file)
    auc = roc_auc_score(true_classes, calibrated_probabilities)
    return auc

def compute_auc_null_distribution(number_of_simulations, number_of_replicates, model_name, evaluation_file_name,
                                  classifier, classifier_name, classifier_file, n, crop_size, calibrated_model_name,
                                  calibrated_model_file):

    device = "cuda:0"
    auc_distribution = np.zeros((number_of_simulations))
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    eval_images = create_evaluation_diffusion_images(number_of_replicates, model_name, evaluation_file_name, n, crop_size)
    classifier_outputs = (classifier(eval_images.float().to(device)))
    classifier_logits = logit_transformation_with_sigmoid(classifier_outputs)
    calibrated_probabilities = produce_calibrated_probabilities(classifier_logits, calibrated_model_name, calibrated_model_file)

    for i in range(0, number_of_simulations):

        permuted_class_labels = permute_class_labels(number_of_replicates, device)
        auc_distribution[i] = roc_auc_score(permuted_class_labels, calibrated_probabilities)

    return auc_distribution

def visualize_auc_test_statistic(number_of_simulations, number_of_replicates, model_name, evaluation_file_name,
                                classifier, classifier_name, classifier_file, n, crop_size, calibrated_model_name,
                                calibrated_model_file, figname):

    auc_test_statistic = compute_auc_test_statistic(number_of_replicates, model_name, evaluation_file_name,
                                                    classifier, classifier_name, classifier_file, n, crop_size,
                                                    calibrated_model_name, calibrated_model_file)
    auc_distribution = compute_auc_null_distribution(number_of_simulations, number_of_replicates,
                                                     model_name, evaluation_file_name, classifier,
                                                     classifier_name, classifier_file, n, crop_size,
                                                     calibrated_model_name, calibrated_model_file)
    fig, ax = plt.subplots(1)
    sns.kdeplot(auc_distribution, palette = ['blue'])
    plt.axvline(auc_test_statistic, color='red', linestyle = 'dashed')
    plt.legend(labels = ['AUC Null'])
    plt.savefig(figname)

number_of_simulations = 10000
number_of_replicates = 2000
n = 32
crop_size = 2
classifier_name = "classifier1"
evaluation_file_name = "evaluation_data_model6_variance_.4_lengthscale_1.6_2000.npy"
figname = ("classifiers/" + classifier_name + "/auc_test_statistic_classifier_" + str(classifier_name) + ".png")
model_name = "model2"
epochs = 500
calibrated_model_name = "calibrated_models/classifier1"
calibrated_model_file = "logistic_regression_model2_classifier1.pkl"
classifier_file = "model2_lengthscale_1.6_variance_0.4_epochs_" + str(epochs) + "_parameters.pth"
classifier = (CNNCLassifier().to("cuda:0"))
visualize_auc_test_statistic(number_of_simulations, number_of_replicates, model_name,
                             evaluation_file_name, classifier, classifier_name,
                             classifier_file, n, crop_size, calibrated_model_name,
                             calibrated_model_file, figname)


