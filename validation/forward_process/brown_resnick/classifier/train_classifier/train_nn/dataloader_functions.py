import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from append_directories import *
unconditional_folder = append_directory(4)
data_generation_folder = (unconditional_folder + "/generate_data")
import os
import sys
sys.path.append(data_generation_folder)
from true_unconditional_data_generation import *



class CustomSpatialImageAndClassDataset(Dataset):
    def __init__(self, images, classes):
        self.images = images
        self.classes = classes

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        class_label = self.classes[idx,:]
        return image, class_label
    

def create_dataloader(images, classes, batch_size):

    dataset = CustomSpatialImageAndClassDataset(images, classes)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    return dataloader

def prepare_images(generated_images, true_images):

    images = torch.from_numpy(np.concatenate([generated_images, true_images], axis = 0)).float()
    return images

def crop_images(images, n, crop_size):

    images = images[:,:,crop_size:(n-crop_size),crop_size:(n-crop_size)]
    return images

def prepare_classes(num_samples):

    classes = np.concatenate([np.ones((num_samples,1)), np.zeros((num_samples,1))], axis = 1)
    classes = torch.from_numpy(np.concatenate((classes, 1-classes), axis = 0)).float()
    return classes

def load_images(path):

    images = np.load(path)
    return images

def prepare_and_create_dataloader(path, num_samples, minX, maxX, minY, maxY, n,
                                  variance, lengthscale, seed_value, batch_size):

    diffusion_images = load_images(path)
    true_images = (generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                            lengthscale, num_samples, seed_value))[1]
    diffusion_images = diffusion_images.reshape((num_samples,1,n,n))
    images = prepare_images(diffusion_images, true_images)
    classes = prepare_classes(num_samples)
    dataloader = create_dataloader(images, classes, batch_size)
    return dataloader

def prepare_crop_and_create_dataloader(path, num_samples, minX, maxX, minY, maxY, n,
                                       variance, lengthscale, seed_value, batch_size, crop_size):

    diffusion_images = load_images(path)
    true_images = generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                            lengthscale, num_samples, seed_value)[1]
    diffusion_images = diffusion_images.reshape((num_samples,1,n,n))
    images = prepare_images(diffusion_images, true_images)
    images = crop_images(images, n, crop_size)
    classes = prepare_classes(num_samples)
    dataloader = create_dataloader(images, classes, batch_size)
    return dataloader

