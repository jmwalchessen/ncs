import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluation_helper_functions import *
from append_directories import *
classifier_folder = append_directory(2)
train_nn_folder = (classifier_folder + "/train_classifier/train_nn")
sys.path.append(train_nn_folder)
print(train_nn_folder)
from nn_architecture import *

classifier_name = "classifier9"
classifier_file = "small1_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_500_parameters.pth"
classifier = smallestCNNClassifier()
classifier.load_state_dict(torch.load((train_nn_folder + "/classifiers/" + classifier_name + "/" + classifier_file), map_location = torch.device("cpu")))


def convlayer(image):
    n = 4
    for idx, mod in enumerate(classifier.modules()):
        filtered_image = np.zeros((1,1,n,n))
        if(idx == 4):

            filtered_image = mod.forward(image)

    return filtered_image


def visualize_filters(figname):

    n = 5
    for idx, mod in enumerate(classifier.modules()):
        if(idx == 4):
            weights = mod.weight.data.numpy()
            for i in range(0,4):
                plt.imshow(weights[i,:,:,:].reshape((n,n)))
                plt.savefig((figname + "_" + str(i) + ".png"))

seed_value = int(np.random.randint(0, 100000))
number_of_replicates = 4000
model_name = "model6"
evaluation_file_name = "evaluation_data_model6_variance_.4_lengthscale_1.6_4000.npy"
n = 32
crop_size = 2
figname = "conv_filter"
evaluation_images = create_evaluation_images(number_of_replicates, seed_value, model_name, evaluation_file_name, n, crop_size)
visualize_filters(figname)

