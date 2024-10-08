import numpy as np
import torch as th
import os
import sys
import pickle
from append_directories import *
classifier_folder = append_directory(2)
train_nn_folder = (classifier_folder + "/train_classifier/train_nn")
sys.path.append(train_nn_folder)
from nn_architecture import *
data_generation_folder = (classifier_folder + "/train_classifier/generate_data")
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *
calibration_folder = (classifier_folder + "/train_classifier/calibration")

def crop_image(images, n, crop_size):

    cropped_images = images[:,:,crop_size:(n-crop_size),crop_size:(n-crop_size)]
    return cropped_images

def load_images(model_name, evaluation_file_name):

    evaluate_data_classifier_folder = ("data/" + model_name + "/diffusion/" + evaluation_file_name)
    images = th.from_numpy(np.load((evaluate_data_classifier_folder)))
    return images

def load_classifier(device, classifier_name, classifier_file):

    classifier = (CNNClassifier()).to(device)
    classifier.load_state_dict(th.load((train_nn_folder + "/classifiers/" + classifier_name + "/" + classifier_file)))
    return classifier


def load_classifier_parameters(classifier, classifier_name, classifier_file)   
    classifier.load_state_dict(th.load((train_nn_folder + "/classifiers/" + classifier_name + "/" + classifier_file)))
    return classifier

def load_smaller_classifier(device, model_name):

    smallclassifier = (SmallCNNClassifier()).to(device)
    smallclassifier.load_state_dict(th.load((train_nn_folder + "/classifiers/small_classifier/"
                                             + model_name)))
    return smallclassifier

def generate_true_images(number_of_replicates, seed_value):

    true_images = generate_gaussian_process(minX = -10, maxX = 10, minY = -10, maxY = 10, n = 32,
                                         variance = .4, lengthscale = 1.6, number_of_replicates = number_of_replicates,
                                         seed_value = seed_value)[1]
    true_images = th.from_numpy(true_images)
    return true_images

def create_evaluation_diffusion_images(number_of_replicates, model_name, evaluation_file_name, n, crop_size):

    diffusion_images = load_images(model_name, evaluation_file_name)
    diffusion_images.reshape((number_of_replicates,1,n,n))
    diffusion_images = crop_image(diffusion_images, n, crop_size)
    return diffusion_images

def create_evaluation_images(number_of_replicates, model_name, evaluation_file_name, n, crop_size):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    seed_value = int(np.random.randint(0, 100000))
    true_images = (generate_true_images(number_of_replicates, seed_value)).reshape((number_of_replicates, 1, n, n))
    diffusion_images = load_images(model_name, evaluation_file_name)
    diffusion_images.reshape((number_of_replicates,1,n,n))
    images = np.concatenate([diffusion_images, true_images], axis = 0)
    images = th.from_numpy(images)
    images = crop_image(images, n, crop_size)

    return images

def create_classes(number_of_replicates, device):

    true_classes = (np.concatenate([np.ones(number_of_replicates), np.zeros(number_of_replicates)],
                                               axis = 0))
    true_classes = true_classes.reshape((2*number_of_replicates,1))
    return true_classes

def create_2dclasses(number_of_replicates, device):

    true_classes = create_classes(number_of_replicates, device)
    true_2dclasses = np.concatenate([true_classes.reshape((2*number_of_replicates,1)),
                                     (1-true_classes).reshape((2*number_of_replicates,1))], axis = 1)
    return true_2dclasses

def logit_transformation_with_sigmoid(classifier_outputs):

    sigmoid = th.nn.Sigmoid()
    classifier_probabilities = sigmoid(classifier_outputs)
    classifier_probabilities = classifier_probabilities.detach().cpu().numpy()
    classifier_logit = (np.log(classifier_probabilities/(1-classifier_probabilities)))
    classifier_logit = classifier_logit.reshape(-1,1)
    classifier_logit[classifier_logit == np.inf] = np.amax(classifier_logit[classifier_logit != np.inf])
    classifier_logit[classifier_logit == np.NaN] = np.amax(classifier_logit[classifier_logit != np.inf])
    classifier_logit[classifier_logit == -1*np.inf] = np.amin(classifier_logit[classifier_logit != -1*np.inf])

    return classifier_logit 

def produce_calibrated_probabilities(classifier_logits, calibrated_model_name, calibrated_model_file):

    calibrated_model_file = (calibration_folder + "/calibrated_models/" + str(calibrated_model_name) + "/" + calibrated_model_file)
    with open(calibrated_model_file, 'rb') as logregmodel_name:
        logregmodel = pickle.load(logregmodel_name)
    calibrated_probabilities = (logregmodel.predict_proba(classifier_logits))[:,1]
    return calibrated_probabilities


    

def classify_evaluation_data(number_of_replicates, model_name, evaluation_file_name,
                             classifier_name, classifier_file, n, crop_size):

    device = "cuda:1"
    seed_value = int(np.random.randint(0, 1000000, 1))
    eval_images = create_evaluation_images(number_of_replicates, seed_value, model_name, evaluation_file_name, n, crop_size)
    eval_images = eval_images.float().to(device)

    classifier = load_classifier(device, classifier_name, classifier_file)
    class_probabilities = classifier(eval_images)
    class_probabilities = class_probabilities.detach().cpu().numpy()
    class_probabilities = class_probabilities[:,0]
    return class_probabilities

def compute_evaluation_loss(number_of_replicates, model_name, evaluation_file_name,
                       classifier_name, classifier_file, n, crop_size):

    device = "cuda:1"
    seed_value = int(np.random.randint(0, 1000000, 1))
    evaluation_images = create_evaluation_images(number_of_replicates, seed_value, model_name,
                                                 evaluation_file_name, n, crop_size)
    evaluation_images = evaluation_images.float().to(device)
    classifier = load_classifier(device, classifier_name, classifier_file)
    classifier_probabilities = classifier(evaluation_images)
    true_classes = create_classes(number_of_replicates, device)
    true_classes = th.from_numpy(true_classes).to(device)
    loss_fn = th.nn.BCEWithLogitsLoss()
    print(true_classes[0:10,:])
    print(classifier_probabilities[0:10,:])
    print(true_classes[4000:4010,:])
    print(nn.Sigmoid(classifier_probabilities[4000:4010,:]))
    loss = loss_fn(classifier_probabilities, true_classes)
    return loss
