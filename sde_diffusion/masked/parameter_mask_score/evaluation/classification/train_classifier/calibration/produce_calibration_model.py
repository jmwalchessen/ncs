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


def produce_logistic_regression_model(model_name, calibration_train_file, n, crop_size, classifier, classifier_name, classifier_file,
                                      number_of_replicates, logistic_regression_filename):

    device = "cuda:0"
    classifier = load_classifier_parameters(classifier, classifier_name, classifier_file)
    seed_value = int(np.random.randint(0, 100000))
    train_images = process_images(number_of_replicates, seed_value, model_name, calibration_train_file, n, crop_size)
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
classifier_name = "classifier13"
calibration_train_file = "calibration_unconditional_images_variance_.4_lengthscale_1.6_5000.npy"
model_name = "model6"
epochs = 60
device = "cuda:0"
classifier_file = "smallercnnclassifier_maxpool_classifier_model6_lengthscale_1.6_variance_0.4_epochs_500_parameters.pth"
logistic_regression_filename = ("calibrated_models/classifiers/classifier13/logistic_regression_model6_classifier13.pkl")
classifier = (SmallerCNNClassifier().to(device))
produce_logistic_regression_model(model_name, calibration_train_file, n, crop_size, classifier, classifier_name, classifier_file,
                                      number_of_replicates, logistic_regression_filename)