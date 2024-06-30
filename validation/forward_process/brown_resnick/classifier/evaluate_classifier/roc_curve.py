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
classifier_folder = append_directory(2)
train_nn_folder = (classifier_folder + "/train_classifier/train_nn")
sys.path.append(train_nn_folder)
from nn_architecture import *
evaluate_data_classifier_folder = (classifier_folder + "/evaluate_classifier/evaluation_data")
unconditional_folder = append_directory(3)
data_generation_folder = (unconditional_folder + "/generate_data")
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *


device = "cuda:0"
model_name = "model3_lengthscale_1.6_variance_0.4_epochs_240_parameters.pth"
classifier = load_smaller_classifier(device, model_name)

number_of_replicates = 2000
seed_value = 23424
diffusion_path = "unconditional_diffusion_lengthscale_1.6_variance_0.4_2000.npy"
n = 32
crop_size = 2

images = create_evaluation_images(number_of_replicates, seed_value, diffusion_path, n, crop_size)
images = images.float().to(device)

#class one is diffusion and class 0 is true
true_classes = create_classes(number_of_replicates, device)
true_classes = true_classes.detach().cpu().numpy()
predicted_probs = ((classifier(images))[:,0]).detach().cpu().numpy()
a = roc_curve(true_classes, predicted_probs, pos_label = 1)
auc = roc_auc_score(true_classes, predicted_probs)


def plot_roc_curve(fpr, tpr, figname):

    fig, ax = plt.subplots(figsize = (5,5))
    plt.plot(fpr,tpr, color = 'orange')
    ax.axline((0, 0), slope=1, color = 'blue')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(["AUC = " + str(round(auc, 2))])
    plt.savefig(figname)

rocfigname = "roc_curves/smallmodel3_lengthscale_1.6_variance_0.4_epochs_240_roc_curve_4000.png"
plot_roc_curve(a[0], a[1], rocfigname)