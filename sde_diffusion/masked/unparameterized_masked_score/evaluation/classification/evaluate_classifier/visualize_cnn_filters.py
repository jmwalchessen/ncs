import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluation_helper_functions import *
from append_directories import *
classifier_folder = append_directory(2)
train_nn_folder = (classifier_folder + "/train_classifier/train_nn")
sys.path.append(train_nn_folder)
from nn_architecture import *

classifier_name = "classifier11"
classifier_file = "small1_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_500_parameters.pth"
classifier = Small1CNNClassifier()
classifier.load_state_dict(torch.load((train_nn_folder + "/classifiers/" + classifier_name + "/" + classifier_file), map_location = torch.device("cpu")))




def visualize_filters(figname, classifier_folder, classifier, nconv):

    for idx, mod in enumerate(classifier.modules()):
        if(idx == 4):
            weights = mod.weight.data.numpy()
            n = weights.shape[2]
            for i in range(nconv):
                plt.imshow(weights[i,:,:,:].reshape((n,n)))
                plt.savefig((classifier_folder + "/conv_filters/" + figname + "_" + str(i) + ".png"))

figname = "conv_filter"
classifier_folder = "classifiers/classifier11"
nconv = 1
visualize_filters(figname, classifier_folder, classifier, nconv)

