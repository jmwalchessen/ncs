#This file is to produce the logistic regression model for calibrating the classifier outputs. We use a logistic regression
#model with a transformation of the classifier outputs via the logit transformation in order to transform the domain of the
#classifier probabilites from [0,1] to the real numbers (-inf,+inf)
import numpy as np
import torch as th
import pickle
from sklearn.linear_model import LogisticRegression
from append_directories import *
train_classifier_folder = append_directory(2)
train_nn_folder = (train_classifier_folder + "/train_nn")
data_generation_folder = (train_classifier_folder + "/generate_data")
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *
sys.path.append(train_nn_folder)
from nn_architecture import *
from calibration_helper_functions import * 


def produce_logistic_regression_model(model_name, n, crop_size, classifier_name, classifier_file,
                                      number_of_replicates, logistic_regression_filename, train_diffusion_file,
                                      train_true_file):

    device = "cuda:0"
    classifier = load_classifier(device, classifier_name, classifier_file)
    seed_value = int(np.random.randint(0, 100000))
    train_images = process_images(number_of_replicates, seed_value, model_name,
                                  train_diffusion_file, train_true_file, n, crop_size)
    classifier_outputs = (classifier(train_images.float().to(device)))
    train_logits = logit_transformation_with_sigmoid(classifier_outputs)
    train_true_labels = create_classes(number_of_replicates)
    #Train logistic regression model on the training data
    logistic_regression_model = LogisticRegression(random_state=0, 
                                               class_weight='balanced').fit(train_logits, train_true_labels)
    with open(logistic_regression_filename, "wb") as file:
        pickle.dump(logistic_regression_model, file)


n = 32
crop_size = 2
number_of_replicates = 5000
classifier_name = "classifier1"
calibration_diffusion_train_file = "calibration_conditional_diffusion_random50_variance_.4_lengthscale_1.6_model2_2500.npy"
calibration_true_train_file = "calibration_unconditional_true_variance_.4_lengthscale_1.6_2500.npy"
model_name = "model2"
epochs = 40
mask_name = "mask1"
classifier_file = "model2_random50_lengthscale_1.6_variance_.4_epochs_" + str(epochs) + "_parameters.pth"
logistic_regression_filename = ("calibrated_models/calibrated_model1/logistic_regression_model2_classifier1.pkl")
produce_logistic_regression_model(model_name, n, crop_size, classifier_name, classifier_file,
                                      number_of_replicates, logistic_regression_filename,
                                      calibration_diffusion_train_file, calibration_true_train_file)