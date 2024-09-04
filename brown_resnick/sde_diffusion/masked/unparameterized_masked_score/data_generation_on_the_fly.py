import numpy as np
import sys
from numpy import linalg
from block_mask_generation import *
import torch
from torch.utils.data import Dataset, DataLoader
import subprocess
import os

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    images = images.reshape((number_of_replicates,1,n,n))
    return images

def generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "schlather_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_schlather_samples.npy")
    os.remove("temporary_schlather_samples.npy")
    images = images.reshape((number_of_replicates,1,n,n))
    return images




def generate_random_masks_on_the_fly(n, number_of_random_replicates, random_missingness_percentages):

    mask_matrices = np.zeros((0,1,n,n))

    for idx in range(0, len(random_missingness_percentages)):
        missingness_percentage = random_missingness_percentages[idx]
        #larger p means more ones, and more ones means more missing values (unobserved values)
        current_mask_matrices = np.random.binomial(n = 1, p = missingness_percentage,
                                                   size = (number_of_random_replicates, 1, n, n))
        mask_matrices = np.concatenate([mask_matrices, current_mask_matrices])
    return mask_matrices

def generate_block_masks_on_the_fly(n, number_of_replicates_per_mask, weighted_lower_half_percentages, weighted_upper_half_percentages):

    block_masks = produce_nonrandom_block_masks(n, weighted_lower_half_percentages, weighted_upper_half_percentages)
    block_masks = block_masks.reshape((block_masks.shape[0],1,n,n))
    return np.repeat(block_masks, number_of_replicates_per_mask, axis = 0)


#create matrix with masks from random_masks_on_the_fly and block_masks
def generate_random_and_block_masks_on_the_fly(n, number_of_random_replicates_per_percentage, random_missingness_percentages, number_of_block_replicates_per_mask, weighted_lower_half_percentages, weighted_upper_half_percentages):

    random_masks = generate_random_masks_on_the_fly(n, number_of_random_replicates_per_percentage, random_missingness_percentages)
    block_masks = generate_block_masks_on_the_fly(n, number_of_block_replicates_per_mask, weighted_lower_half_percentages, weighted_upper_half_percentages)
    return np.concatenate([random_masks, block_masks], axis = 0)

def generate_train_and_evaluation_brown_resnick_process(range_value, smooth_value, seed_values,
                                                        number_of_replicates,
                                                        number_of_evaluation_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_values[0])],
                    check = True, capture_output = True, text = False)
    train_images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_evaluation_replicates), str(seed_values[1])],
                    check = True, capture_output = True, text = False)
    eval_images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    train_images = train_images.reshape((number_of_replicates,1,n,n))
    eval_images = eval_images.reshape((number_of_evaluation_replicates,1,n,n))
    return train_images, eval_images


def generate_train_and_evaluation_schlather_process(range_value, smooth_value, seed_values,
                                                    number_of_replicates, number_of_evaluation_replicates, n):

    subprocess.run(["Rscript", "schlather_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_values[0])],
                    check = True, capture_output = True, text = False)
    train_images = np.load("temporary_schlather_samples.npy")
    os.remove("temporary_schlather_samples.npy")
    subprocess.run(["Rscript", "schlather_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_evaluation_replicates), str(seed_values[1])],
                    check = True, capture_output = True, text = False)
    eval_images = np.load("temporary_schlather_samples.npy")
    os.remove("temporary_schlather_samples.npy")
    train_images = train_images.reshape((number_of_replicates,1,n,n))
    eval_images = eval_images.reshape((number_of_evaluation_replicates,1,n,n))
    return train_images, eval_images



class CustomSpatialImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        return image
    

class CustomSpatialImageandSingleMaskDataset(Dataset):

    def __init__(self, images, mask):
        self.images = images
        self.mask = mask

    def __len__(self):
        return ((self.images).shape[0])
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        mask = (self.mask)
        mask = mask.view(mask.shape[0], mask.shape[2], mask.shape[3])
        return image, mask
    
class CustomSpatialImageandMaskDataset(Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return ((self.images).shape[0])
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        mask = self.masks[idx,:,:,:]
        return image, mask
    
class CustomSpatialImageMaskDataset(Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return ((self.images).shape[0])
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        mask = self.masks[idx,:,:,:]
        image_and_mask = np.concatenate([image, mask], axis = 0)
        return image_and_mask
    
class CustomMaskDataset(Dataset):
    def __init__(self, masks):
        self.masks = masks

    def __len__(self):
        return ((self.masks).shape[0])

    def __getitem__(self, idx):
        mask = self.masks[idx,:,:,:]
        return mask

def get_next_batch(image_and_mask_iterator, config):

    images_and_masks = (next(image_and_mask_iterator))
    images_and_masks = images_and_masks.to(config.device).float()
    return images_and_masks

#seeds values list is a list of lists of tuples of length equal to number of missing percentages
def get_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     range_value, smooth_value, seed_values_list):
    
    n = 32
    train_images = np.zeros((0,1,n,n))
    eval_images = np.zeros((0,1,n,n))
    for i, p in enumerate(random_missingness_percentages):
        seed_values = seed_values_list[i]
        if(p == 0):
            train_images = np.concatenate([train_images, generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                                                        number_of_random_replicates*number_of_masks_per_image, n)])
            eval_images = np.concatenate([eval_images, generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                                                        number_of_evaluation_random_replicates*number_of_evaluation_masks_per_image, n)])
        else:
            timages = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                     number_of_random_replicates, n)
            eimages = generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                     number_of_evaluation_random_replicates, n)
            train_images = np.concatenate([train_images, np.repeat(timages, number_of_masks_per_image, axis = 0)])
            eval_images = np.concatenate([eval_images, np.repeat(eimages, number_of_evaluation_masks_per_image, axis = 0)])

    train_images = np.log(train_images)
    eval_images = np.log(eval_images)
    train_masks = generate_random_masks_on_the_fly(n, train_images.shape[0], random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, eval_images.shape[0], random_missingness_percentages)
    train_dataset = CustomSpatialImageMaskDataset(train_images, train_masks)
    eval_dataset = CustomSpatialImageMaskDataset(eval_images, eval_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

#seeds values list is a list of lists of tuples of length equal to number of missing percentages
def get_schlater_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     range_value, smooth_value, seed_values_list):
    
    n = 32
    train_images = np.zeros((0,1,n,n))
    eval_images = np.zeros((0,1,n,n))
    for i, p in enumerate(random_missingness_percentages):
        seed_values = seed_values_list[i]
        if(p == 0):
            train_images = np.concatenate([train_images, generate_schlather_process(range_value, smooth_value, seed_values[0],
                                                                                        number_of_random_replicates*number_of_masks_per_image, n)])
            eval_images = np.concatenate([eval_images, generate_schlather_process(range_value, smooth_value, seed_values[1],
                                                                                        number_of_evaluation_random_replicates*number_of_evaluation_masks_per_image, n)])
        else:
            timages = generate_schlather_process(range_value, smooth_value, seed_values[0],
                                                     number_of_random_replicates, n)
            eimages = generate_schlather_process(range_value, smooth_value, seed_values[1],
                                                     number_of_evaluation_random_replicates, n)
            train_images = np.concatenate([train_images, np.repeat(timages, number_of_masks_per_image, axis = 0)])
            eval_images = np.concatenate([eval_images, np.repeat(eimages, number_of_evaluation_masks_per_image, axis = 0)])

    train_images = np.log(train_images)
    eval_images = np.log(eval_images)
    train_masks = generate_random_masks_on_the_fly(n, train_images.shape[0], random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, eval_images.shape[0], random_missingness_percentages)
    train_dataset = CustomSpatialImageMaskDataset(train_images, train_masks)
    eval_dataset = CustomSpatialImageMaskDataset(eval_images, eval_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

        