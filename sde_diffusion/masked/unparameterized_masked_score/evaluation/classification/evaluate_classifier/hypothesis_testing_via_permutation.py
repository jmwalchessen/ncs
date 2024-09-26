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


def permute_evaluation_data(model_name, evaluation_file_name, nrep, device, crop_size, n):

    diffusion_images = load_images(model_name, evaluation_file_name)
    seed_value = int(np.random.randint(0, 1000000))
    true_images = generate_true_images(nrep, seed_value)
    class_labels = create_classes(nrep, device)
    permuted_class_labels = np.random.permutation(class_labels)
    images = th.cat((true_images, diffusion_images),dim = 0)
    images = crop_image(images, n, crop_size)
    return images, permuted_class_labels

def compute_auc(number_of_replicates, model_name, evaluation_file_name,
                       classifier_name, classifier_file, n, crop_size):

    true_classes = create_classes(number_of_replicates, device)
    classifier = load_classifier(device, classifier_name, classifier_file)
    class_probabilities = classifier(eval_images)
    auc = roc_auc_score(true_classes, predicted_class_probabilities)
    return auc  
