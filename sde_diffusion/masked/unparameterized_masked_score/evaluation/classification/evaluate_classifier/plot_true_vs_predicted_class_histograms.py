import matplotlib.pyplot as plt
import numpy as np


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

