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
                       classifier, classifier_name, classifier_file, n, crop_size):

    true_classes = create_classes(number_of_replicates, device)
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    seed_value = int(np.random.randint(0, 100000))
    eval_images = create_evaluation_images(number_of_replicates, seed_value, model_name, evaluation_file_name, n, crop_size)
    classifier_outputs = (classifier(eval_images.float().to(device)))
    sigmoid = th.nn.Sigmoid()
    predicted_class_probabilities = sigmoid(classifier_outputs)
    roc = roc_curve(true_classes, predicted_class_probabilities, pos_label = 1)
    auc = roc_auc_score(true_classes, predicted_class_probabilities)
    return roc, auc  

number_of_replicates = 4000
device = "cuda:1"
model_name = "model6"
evaluation_file_name = "evaluation_data_model6_variance_.4_lengthscale_1.6_4000.npy"
n = 32
crop_size = 2
classifier_name = "classifier11"
num_epochs = 500
classifier = Small1Classifier()
classifier_file = "small1_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_" + str(num_epochs) + "_parameters.pth"
roc, auc = create_roc_and_auc(number_of_replicates, model_name, evaluation_file_name,
                              classifier, classifier_name, classifier_file, n, crop_size)
#loss = compute_evaluation_loss(number_of_replicates, model_name, evaluation_file_name,
                               #classifier_name, classifier_file, n, crop_size)
print(auc)