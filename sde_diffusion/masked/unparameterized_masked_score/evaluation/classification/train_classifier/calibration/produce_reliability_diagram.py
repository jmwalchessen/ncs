import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from calibration_helper_functions import *
from append_directories import *
train_classifier_folder = append_directory(2)
train_nn_folder = (train_classifier_folder + "/train_nn")
data_generation_folder = (train_classifier_folder + "/generate_data")
sys.path.append(data_generation_folder)

def compute_calibration(true_labels, pred_labels, predicted_probabilities, num_bins):
    """Collects predictions into bins used to draw a reliability diagram (predicted on x axis and empirical on y axis).
    Arguments:
        true_labels: the true labels (class) for the test examples
        pred_labels: the predicted labels (class) for the test examples
        predicted_probabilities: the predicted probability of the example belonging to class one (dependent class)
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    Returns a dictionary containing the following NumPy arrays:
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        empirical_probabilities: the empirical probability for class one (dependent class) for each bin
        predicted_probabilities: the predicted probability for class one (dependent class) for each bin
    """
    assert(len(predicted_probabilities) == len(pred_labels))
    assert(len(predicted_probabilities) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(predicted_probabilities, bins, right=True)

    bin_empirical_probabilities = np.zeros(num_bins, dtype=np.float64)
    bin_predicted_probabilities = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            print(np.mean(true_labels[selected]))
            bin_empirical_probabilities[b] = np.mean(1 == true_labels[selected])
            bin_predicted_probabilities[b] = np.mean(predicted_probabilities[selected])
            bin_counts[b] = len(selected)


    return { "empirical_probabilities": bin_empirical_probabilities, 
             "predicted_probabilities": bin_predicted_probabilities, 
             "counts": bin_counts, 
             "bins": bins}


#This helper function creates the matplotlib figure.

def reliability_diagram_subplot(ax, bin_data,
                                 draw_bin_importance=False,
                                 xlabel="Predicted Class Probability (C=1)", 
                                 ylabel="Empirical Class Probability"):
    """Draws a reliability diagram into a subplot."""
    empirical_probabilities = bin_data["empirical_probabilities"]
    predicted_probabilities = bin_data["predicted_probabilities"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*bin_size + 0.9*bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(empirical_probabilities - predicted_probabilities), 
                     bottom=np.minimum(empirical_probabilities, predicted_probabilities), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Miscalibration")

    #plot identify function in dashed gray
    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")

    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([.1*i for i in range(2,12,2)]) 
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)

    ax.legend(handles=[gap_plt], fontsize = 18)


def _reliability_diagram_combined(fig_name, bin_data, title, 
                                  draw_bin_importance, 
                                  figsize, dpi, return_fig):
    """Draws a reliability diagram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    plt.tight_layout()
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
    plt.title(title, fontsize = 30)

    reliability_diagram_subplot(ax, bin_data, draw_bin_importance,
                                xlabel="Predicted Probability", ylabel = "Empirical Probability")



    plt.savefig(fig_name, bbox_inches = "tight")

    if return_fig: return fig


def reliability_diagram(true_labels, pred_labels, predicted_probabilities, num_bins, fig_name, 
                        title, figsize=(10, 10), dpi=72, return_fig=False):
    """Draws a reliability diagram
    
    First, the model's predictions are divided up into bins based on the predicted
    probability for class one (dependent class). The independent class is class zero.
    The reliability diagram shows the gap between empirical and predicted probability
    in each bin. These are the red bars. Ideally, there is no gap.
    In that case, the model is properly calibrated.
    
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        predicted_probabilities: the predicted probability of the example belonging to class one (dependent class)
        num_bins: number of bins
        title: title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    bin_data = compute_calibration(true_labels, pred_labels, predicted_probabilities, num_bins)
    return _reliability_diagram_combined(fig_name, bin_data, title,
                                         figsize=figsize, 
                                         dpi=dpi, return_fig=return_fig, draw_bin_importance = False)

#Helper function to compute the predicted class for a pair of spatial field and paramter. The predicted label of a pair is class
#one (the dependent class) if the predicted probability from the classifier for the pair is greater than .5.
def compute_prediction_class(prediction):

    if(float(prediction) >= .5):
        return np.array([1])
    else:
        return np.array([0])

def compute_prediction_classes(probabilities):

    n = probabilities.shape[0]
    predicted_class_labels = (np.asarray([float((compute_prediction_class(probabilities[i]))[0])
                                          for i in range(0, n)]))
    predicted_class_labels = predicted_class_labels.reshape((n,1))
    return predicted_class_labels


def visualize_precalibrated_and_calibrated_reliability_diagrams(number_of_replicates, model_name, calibration_eval_file,
                                                n, crop_size, classifier_file, classifier_name, precalibrated_figname,
                                                calibrated_model_name, calibrated_model_file, calibrated_figname):

    seed_value = int(np.random.randint(0,100000))
    eval_classifier_probabilities = classify_precalibrated_data(number_of_replicates, seed_value, model_name,
                                                            calibration_eval_file, n, crop_size,
                                                            classifier_name, classifier_file)
    eval_calibrated_classifier_probabilities = classify_calibrated_data(number_of_replicates, seed_value, model_name,
                                                            calibration_eval_file, n, crop_size,
                                                            classifier_name, classifier_file,
                                                            calibrated_model_name, calibrated_model_file)
    eval_class_labels = create_classes(number_of_replicates)
    predicted_class_labels = compute_prediction_classes(eval_classifier_probabilities)
    #plot the diagram before calibration
    reliability_diagram(eval_class_labels, predicted_class_labels, eval_classifier_probabilities,
                        num_bins=50, figsize=(10, 10), dpi=72, return_fig=False, fig_name = precalibrated_figname,
                        title = "Before Calibration")
    reliability_diagram(eval_class_labels, predicted_class_labels, eval_calibrated_classifier_probabilities,
                        num_bins=50, figsize=(10, 10), dpi=72, return_fig=False, fig_name = calibrated_figname,
                        title = "Before Calibration")
    
number_of_replicates = 1000
model_name = "model6"
calibration_eval_file = "calibration_evaluation_unconditional_images_variance_.4_lengthscale_1.6_1000.npy"
n = 32
crop_size = 2
epochs = 60
classifier_file = "model6_lengthscale_1.6_variance_0.4_epochs_" + str(epochs) + "_parameters.pth"
classifier_name = "classifier6"
precalibrated_figname = "precalibrated_reliability_diagram_classifier6_model6.png"
calibrated_figname = "calibrated_reliability_diagram_calibrated_model2_classifier6_model6.png"
calibrated_model_name = "calibrated_model2"
calibrated_model_file = "logistic_regression_model6_classifier6.pkl"

visualize_precalibrated_and_calibrated_reliability_diagrams(number_of_replicates, model_name, calibration_eval_file,
                                                            n, crop_size, classifier_file, classifier_name, precalibrated_figname,
                                                            calibrated_model_name, calibrated_model_file, calibrated_figname)