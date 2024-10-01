import numpy as numpy
import torch as th
import pickle
from append_directories import *
train_classifier_folder = append_directory(2)
data_generation_folder = (train_classifier_folder + "/generate_data")
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *
train_nn_folder = (train_classifier_folder + "/train_nn")
sys.path.append(train_nn_folder)
from nn_architecture import *

def crop_image(images, n, crop_size):

    cropped_images = images[:,:,crop_size:(n-crop_size),crop_size:(n-crop_size)]
    return cropped_images

def load_images(model_name, mask_name, file_name):

    images = th.from_numpy(np.load((data_generation_folder + "/data/fixed_mask/" + model_name + "/" + mask_name + "/" + file_name)))
    return images

def load_classifier(device, classifier_name, classifier_file):

    classifier = (CNNClassifier()).to(device)
    classifier.load_state_dict(th.load((train_nn_folder + "/classifiers/" + classifier_name + "/" + classifier_file)))
    return classifier

def generate_true_images(number_of_replicates, seed_value):

    true_images = generate_gaussian_process(minX = -10, maxX = 10, minY = -10, maxY = 10, n = 32,
                                         variance = .4, lengthscale = 1.6, number_of_replicates = number_of_replicates,
                                         seed_value = seed_value)[1]
    true_images = th.from_numpy(true_images)
    return true_images

def process_images(number_of_replicates, model_name, mask_name, true_file_name,
                   diffusion_file_name, n, crop_size):

    true_images = load_images(model_name, mask_name, true_file_name)
    diffusion_images = load_images(model_name, mask_name, diffusion_file_name)
    diffusion_images.reshape((number_of_replicates,1,n,n))
    images = th.cat([diffusion_images, true_images], dim = 0)
    images = crop_image(images, n, crop_size)
    return images

def create_classes(number_of_replicates):

    true_classes = (np.concatenate([np.ones(number_of_replicates), np.zeros(number_of_replicates)],
                                               axis = 0))
    true_classes = true_classes.reshape((2*number_of_replicates,1))
    return true_classes

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

def classify_precalibrated_data(number_of_replicates, seed_value, model_name, file_name, n, crop_size,
                              classifier_name, classifier_file):

    device = "cuda:0"
    images = process_images(number_of_replicates, seed_value, model_name, file_name, n, crop_size)
    classifier = load_classifier(device, classifier_name, classifier_file)
    classifier_outputs = classifier(images.float().to("cuda:0"))
    sigmoid = torch.nn.Sigmoid()
    classifier_probabilities = sigmoid(classifier_outputs)
    classifier_probabilities = classifier_probabilities.detach().cpu().numpy()
    return classifier_probabilities

def classify_calibrated_data(number_of_replicates, seed_value, model_name, file_name, n, crop_size,
                              classifier_name, classifier_file, calibrated_model_name, calibrated_model_file):

    calibrated_model_file = ("calibrated_models/" + str(calibrated_model_name) + "/" + calibrated_model_file)
    with open(calibrated_model_file, 'rb') as logregmodel_name:
        logregmodel = pickle.load(logregmodel_name)
    
    device = "cuda:0"
    images = process_images(number_of_replicates, seed_value, model_name, file_name, n, crop_size)
    classifier = load_classifier(device, classifier_name, classifier_file)
    classifier_outputs = classifier(images.float().to("cuda:0"))
    classifier_logits = logit_transformation_with_sigmoid(classifier_outputs)
    calibrated_probabilities = (logregmodel.predict_proba(classifier_logits))[:,1]

    return calibrated_probabilities


