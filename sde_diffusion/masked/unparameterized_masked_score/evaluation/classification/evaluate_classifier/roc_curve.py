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



def create_roc_and_auc(number_of_replicates, model_name, evaluation_file_name,
                       classifier_name, classifier_file, n, crop_size):

    true_classes = create_classes(number_of_replicates, device)
    predicted_class_probabilities = classify_evaluation_data(number_of_replicates, model_name, evaluation_file_name,
                                                             classifier_name, classifier_file, n, crop_size)
    roc = roc_curve(true_classes, predicted_class_probabilities, pos_label = 1)
    auc = roc_auc_score(true_classes, predicted_class_probabilities)
    return roc, auc  

number_of_replicates = 4000
device = "cuda:1"
model_name = "model6"
evaluation_file_name = "evaluation_data_model6_variance_.4_lengthscale_1.6_4000.npy"
n = 32
crop_size = 4
classifier_name = "classifier1"
classifier_file = "model6_lengthscale_1.6_variance_0.4_epochs_80_parameters.pth"
roc, auc = create_roc_and_auc(number_of_replicates, model_name, evaluation_file_name,
                              classifier_name, classifier_file, n, crop_size)
loss = compute_evaluation_loss(number_of_replicates, model_name, evaluation_file_name,
                               classifier_name, classifier_file, n, crop_size)
print(loss)