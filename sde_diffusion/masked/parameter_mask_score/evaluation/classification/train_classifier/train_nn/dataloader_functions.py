import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from append_directories import *
classifier_folder = append_directory(2)
data_generation_folder = (classifier_folder + "/generate_data")
import os
import sys
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *



class CustomSpatialImageParameterAndClassDataset(Dataset):
    def __init__(self, images, lengthscales, classes):
        self.images = images
        self.classes = classes
        self.lengthscales = lengthscales

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        class_label = self.classes[idx,:]
        lengthscale = self.lengthscales[idx]
        return image, lengthscale, class_label
    

def create_dataloader(images, lengthscales, classes, batch_size):

    dataset = CustomSpatialImageParameterAndClassDataset(images, lengthscales, classes)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataloader

def prepare_images(generated_images, true_images):

    images = torch.from_numpy(np.concatenate([generated_images, true_images], axis = 0)).float()
    return images

def prepare_random_images(generated_images, true_images):

    images = prepare_images(generated_images, true_images)
    shuffled_indices = torch.randperm(images.shape[0])
    images = images[shuffled_indices]
    return images

def crop_images(images, n, crop_size):

    images = images[:,:,crop_size:(n-crop_size),crop_size:(n-crop_size)]
    return images

def prepare_classes(num_samples):

    classes = np.concatenate([np.ones((num_samples,1)), np.zeros((num_samples,1))], axis = 0)
    classes = torch.from_numpy(classes).float()
    return classes

def load_images(path):

    images = np.load(path)
    return images

def load_parameters(path):

    parameters = np.load(path)
    return parameters

def generate_gaussian_processes_multiple_lengthscales(minX, maxX, minY, maxY, n, variance, lengthscales, num_samples):

    gp_samples = np.zeros((0,1,n,n))
    for lengthscale in lengthscales:
        seed_value = int(np.random.randint(0, 100000))
        gp_samples = np.concatenate([gp_samples, (generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                                                lengthscale, num_samples, seed_value))[1]], axis = 0)
        
    return gp_samples

def prepare_and_create_dataloader(image_path, parameter_path, num_samples, minX, maxX, minY, maxY, n,
                                  variance, batch_size):

    diffusion_images = load_images(image_path)
    lengthscales = load_parameters(parameter_path)
    true_images = generate_gaussian_processes_multiple_lengthscales(minX, maxX, minY, maxY, n, variance, lengthscales, num_samples)
    diffusion_images = diffusion_images.reshape((num_samples,1,n,n))
    images = prepare_images(diffusion_images, true_images)
    classes = prepare_classes(num_samples)
    dataloader = create_dataloader(images, lengthscales, classes, batch_size)
    return dataloader

def prepare_crop_and_create_dataloaders(image_path, parameter_path, split, num_samples, minX, maxX, minY, maxY, n,
                                       variance, batch_size,
                                       eval_batch_size, crop_size, shuffle = False):

    diffusion_images = load_images(image_path)
    lengthscales = load_parameters(parameter_path)
    true_images = generate_gaussian_processes_multiple_lengthscales(minX, maxX, minY, maxY, n, variance, lengthscales, num_samples)
    diffusion_images = (diffusion_images.reshape((num_samples,1,n,n))+1)
    true_train_images = true_images[0:split,:,:,:]
    diffusion_train_images = diffusion_images[0:split,:,:,:]
    true_eval_images = true_images[split:,:,:,:]
    diffusion_eval_images = diffusion_images[split:,:,:,:]
    if(shuffle == True):
        train_images = prepare_random_images(diffusion_train_images, true_train_images)
        eval_images = prepare_random_images(diffusion_eval_images, true_eval_images)
    else:
        train_images = prepare_images(diffusion_train_images, true_train_images)
        eval_images = prepare_images(diffusion_eval_images, true_eval_images)
    train_images = crop_images(train_images, n, crop_size)
    eval_images = crop_images(eval_images, n, crop_size)
    train_classes = prepare_classes(split)
    eval_classes = prepare_classes((num_samples - split))
    train_dataloader = create_dataloader(train_images, train_classes, batch_size)
    eval_dataloader = create_dataloader(eval_images, eval_classes, eval_batch_size)
    eval_train_dataloader = create_dataloader(train_images, train_classes, eval_batch_size)
    return train_dataloader, eval_dataloader, eval_train_dataloader


