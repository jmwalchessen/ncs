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

def prepare_crop_and_create_dataloaders(diffusion_path, true_path, split, num_samples, minX, maxX, minY, maxY, n,
                                       variance, lengthscale, seed_value, batch_size,
                                       eval_batch_size, crop_size, shuffle = False):

    diffusion_images = load_images(diffusion_path)
    true_images = load_images(true_path)
    diffusion_images = diffusion_images.reshape((num_samples,1,n,n))
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


def prepare_crop_and_create_dataloaders_on_the_fly(path, train_start, train_end, split,
                                                   num_samples, minX, maxX, minY, maxY, n,
                                                   variance, lengthscale, seed_value, batch_size,
                                                   eval_batch_size, crop_size, shuffle = False):

    diffusion_images = load_images(path)
    true_images = generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                            lengthscale, num_samples, seed_value)[1]
    diffusion_images = diffusion_images.reshape((num_samples,1,n,n))
    true_train_images = true_images[train_start:train_end,:,:,:]
    diffusion_train_images = diffusion_images[train_start:train_end,:,:,:]
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


