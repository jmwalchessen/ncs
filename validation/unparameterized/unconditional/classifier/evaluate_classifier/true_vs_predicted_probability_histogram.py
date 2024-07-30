import torch as th
import numpy as np
import matplotlib.pyplot as plt
from evaluation_helper_functions import *

#number of replicates is number of either true or diffusion samples, the first number_of_replicates samples
#are truly diffusion and the second number_of_replicates samples are truly true
def plot_predicted_vs_true_probability_histogram(number_of_replicates, diffusion_probabilities, figname):

    fig, ax = plt.subplots(figsize = (5,5))
    #Blue corresponds to true which are the last half of diffusion_probabilities
    plt.hist(diffusion_probabilities[number_of_replicates:], bins = 20, alpha = 0.5, color = 'blue')
    plt.hist(diffusion_probabilities[0:number_of_replicates], bins = 20, alpha = 0.5, color = 'orange')
    plt.xlabel("Predicted Probability (Class 1 = diffusion)")
    plt.ylabel("Frequency")
    plt.title("Predicted Probability per Class Label Histogram")
    plt.legend(["True", "Generated"])
    plt.savefig(figname)


def plot_predicted_probabilitiy_per_class_label_histogram(number_of_replicates, diffusion_probabilities, figname):
    
        fig, ax = plt.subplots(nrows = 2, figsize = (5,10))
        ax[0].hist(diffusion_probabilities[number_of_replicates:], bins = 20, alpha = 0.5, color = 'blue')
        ax[1].hist(diffusion_probabilities[0:number_of_replicates], bins = 20, alpha = 0.5, color = 'orange')
        ax[0].set_xlabel("Predicted Probability (Class 1 = diffusion)")
        ax[0].set_ylabel("Frequency")
        ax[1].set_xlabel("Predicted Probability (Class 1 = diffusion)")
        ax[1].set_ylabel("Frequency")
        ax[0].set_title("Predicted Probability per Class Label Histogram")
        ax[1].set_title("Predicted Probability per Class Label Histogram")
        ax[0].legend(["True"])
        ax[1].legend(["Generated"])
        plt.savefig(figname)

number_of_replicates = 2000
diffusion_pathname = "model5_unconditional_lengthscale_1.6_variance_0.4_eval_2000.npy"
seed_value = 3203423
n = 32
crop_size = 4
device = "cuda:0"
model_name = "largemodel5_lengthscale_1.6_variance_0.4_epochs_30_parameters.pth"
#diffusion images are first then true images
evaluation_images = create_evaluation_images(number_of_replicates, seed_value, diffusion_pathname, n, crop_size)

classifier = load_classifier(device, model_name)
#class one is diffusion and class 0 is true, second column is the probability of being in class 1
classifier_output = classifier(evaluation_images.float().to(device))
#first column is probability of being in class 1
diffusion_probabilities =(classifier_output[:,0]).detach().cpu().numpy()
figname = "predicted_vs_true_probability_histograms/largemodel5_epochs_30_lengthscale_1.6_variance_0.4_epochs_240_predicted_vs_true_probability_histogram_4000.png"
plot_predicted_vs_true_probability_histogram(number_of_replicates, diffusion_probabilities, figname)
figname = "predicted_vs_true_probability_histograms/largemodel5_epochs_30_lengthscale_1.6_variance_0.4_epochs_240_predicted_probability_histogram_per_class_label_4000.png"
plot_predicted_probabilitiy_per_class_label_histogram(number_of_replicates, diffusion_probabilities, figname)