import numpy as np
import torch as th
import matplotlib.pyplot as plt
import os
import sys
from evaluation_helper_functions import *
from append_directories import *
import pandas as pd
import seaborn as sns
classification_folder = append_directory(2)
sys.path.append((classification_folder + "/train_classifier/train_nn"))


def produce_weights(model_name, evaluation_file_name, n, crop_size, classifier, classifier_name, classifier_file):

    device = "cuda:0"
    eval_diffusion_images = load_images(model_name, evaluation_file_name)
    eval_diffusion_images = crop_image(eval_diffusion_images, n, crop_size)
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    sigmoid = th.nn.Sigmoid()
    class_probabilities = sigmoid(classifier(eval_diffusion_images.float().to(device)))
    print(th.sum(class_probabilities >= .5))
    weights = (1-class_probabilities)/class_probabilities
    weights = weights.detach().cpu().numpy()
    return weights

def produce_calibrated_weights(model_name, evaluation_file_name, n, crop_size, classifier, classifier_name, classifier_file,
                               calibrated_model_name, calibrated_model_file):

    device = "cuda:0"
    eval_diffusion_images = load_images(model_name, evaluation_file_name)
    eval_diffusion_images = crop_image(eval_diffusion_images, n, crop_size)
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    classifier_outputs = classifier(eval_diffusion_images.float().to(device))
    classifier_logits = logit_transformation_with_sigmoid(classifier_outputs)
    calibrated_probabilities = produce_calibrated_probabilities(classifier_logits, calibrated_model_name, calibrated_model_file)
    calibrated_weights = (1-calibrated_probabilities)/calibrated_probabilities
    return calibrated_weights


def visualize_weights(model_name, evaluation_file_name, n, crop_size, classifier_name, classifier_file, figname):

    weights = produce_weights(model_name, evaluation_file_name, n, crop_size, classifier_name, classifier_file)

    fig, ax = plt.subplots(1)
    pdd = pd.DataFrame(weights, columns = None)
    sns.kdeplot(data = pdd, palette = ['blue'])
    plt.axvline(1, color='red', linestyle = 'dashed')
    plt.legend(labels = ['Weights'])
    plt.savefig(("weights/" + classifier_name + "/" + figname))

def visualize_calibrated_weights(model_name, evaluation_file_name, n, crop_size, classifier, classifier_name, classifier_file,
                                 calibrated_model_name, calibrated_model_file, figname):

    weights = produce_calibrated_weights(model_name, evaluation_file_name, n, crop_size, classifier, classifier_name, classifier_file,
                                         calibrated_model_name, calibrated_model_file)

    fig, ax = plt.subplots(1)
    pdd = pd.DataFrame(weights, columns = None)
    sns.kdeplot(data = pdd, palette = ['blue'])
    plt.axvline(1, color='red', linestyle = 'dashed')
    plt.legend(labels = ['Calibrated Weights'])
    plt.savefig(("classifiers/" + classifier_name + "/" + figname))


n = 32
crop_size = 2
classifier_name = "classifier13"
evaluation_file_name = "evaluation_data_model6_variance_.4_lengthscale_1.6_4000.npy"
model_name = "model6"
epochs = 500
classifier_file = "smallercnnclassifier_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_500_parameters.pth"
calibrated_model_name = "classifiers/classifier13"
calibrated_model_file = "logistic_regression_model6_classifier13.pkl"

classifier = (SmallerCNNClassifier()).to("cuda:0")
classifier_name = "classifier13"
classifier_file = "smallercnnclassifier_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_500_parameters.pth"
figname = ("calibrated_weights_classifier_13_model6.png")
visualize_calibrated_weights(model_name, evaluation_file_name, n, crop_size, classifier, classifier_name, classifier_file,
                                 calibrated_model_name, calibrated_model_file, figname)