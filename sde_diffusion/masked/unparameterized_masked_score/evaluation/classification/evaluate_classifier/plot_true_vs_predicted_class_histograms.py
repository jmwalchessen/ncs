import matplotlib.pyplot as plt
import numpy as np
from evaluation_helper_functions import *

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


def classify_and_plot_predicted_vs_true_probability_histogram(number_of_replicates, figname,
                                                              model_name, evaluation_file_name, n, crop_size, classifier,
                                                              classifier_name, classifier_file, calibrated_model_name,
                                                              calibrated_model_file):

    device = "cuda:0"
    seed_value = int(np.random.randint(0, 100000))
    eval_images = create_evaluation_images(number_of_replicates, seed_value, model_name, evaluation_file_name, n, crop_size)
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    classifier_outputs = classifier(eval_images.float().to(device))
    classifier_logits = logit_transformation_with_sigmoid(classifier_outputs)
    diffusion_probabilities = produce_calibrated_probabilities(classifier_logits, calibrated_model_name, calibrated_model_file)
    plot_predicted_vs_true_probability_histogram(number_of_replicates, diffusion_probabilities, figname)


number_of_replicates = 4000
classifier_name = "classifier13"
figname = ("classifiers/" + classifier_name + "/true_vs_predicted_class_histogram_calibrated_4000.png")
model_name = "model6"
n = 32
crop_size = 2
device = "cuda:0"
classifier = (SmallerCNNClassifier()).to(device)
calibrated_model_name = "classifiers/classifier13"
calibrated_model_file = "logistic_regression_model6_classifier13.pkl"
epochs = 500
classifier_file = "smallercnnclassifier_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_500_parameters.pth"
evaluation_file_name = "evaluation_data_model6_variance_.4_lengthscale_1.6_4000.npy"
classify_and_plot_predicted_vs_true_probability_histogram(number_of_replicates, figname,
                                                              model_name, evaluation_file_name, n, crop_size, classifier,
                                                              classifier_name, classifier_file, calibrated_model_name,
                                                              calibrated_model_file)