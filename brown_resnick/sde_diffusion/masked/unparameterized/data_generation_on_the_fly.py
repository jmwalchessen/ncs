import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from block_mask_generation import *



def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images


def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    return images


def log_and_boundary_process(images):

    log_images = log_transformation(images)
    log01_images = (log_images - np.min(log_images))/(np.max(log_images) - np.min(log_images))
    centered_batch = log01_images - .5
    scaled_centered_batch = 6*centered_batch
    return scaled_centered_batch

def global_boundary_process(images, minvalue, maxvalue):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - .5
    log01cs = 6*log01c
    return log01cs

def log_and_normalize(images):

    images = np.log(images)
    images = (images - np.mean(images))/np.std(images)
    return images

def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs


def generate_train_and_evaluation_brown_resnick_process(range_value, smooth_value, seed_value,
                                                        number_of_replicates,
                                                        number_of_evaluation_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    train_images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_evaluation_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    eval_images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    train_images = train_images.reshape((number_of_replicates,1,n,n))
    eval_images = eval_images.reshape((number_of_evaluation_replicates,1,n,n))
    return train_images, eval_images



def generate_random_masks_on_the_fly(n, number_of_random_replicates, random_missingness_percentages):

    mask_matrices = np.zeros((0,1,n,n))

    for idx in range(0, len(random_missingness_percentages)):
        missingness_percentage = random_missingness_percentages[idx]
        current_mask_matrices = np.random.binomial(n = 1, p = missingness_percentage,
                                                   size = (number_of_random_replicates, 1, n, n))
        mask_matrices = np.concatenate([mask_matrices, current_mask_matrices])
    return mask_matrices


def generate_block_masks_on_the_fly(n, number_of_replicates_per_mask, weighted_lower_half_percentages, weighted_upper_half_percentages):

    block_masks = produce_nonrandom_block_masks(n, weighted_lower_half_percentages, weighted_upper_half_percentages)
    block_masks = block_masks.reshape((block_masks.shape[0],1,n,n))
    return np.repeat(block_masks, number_of_replicates_per_mask, axis = 0)


#create matrix with masks from random_masks_on_the_fly and block_masks
def generate_random_and_block_masks_on_the_fly(n, number_of_random_replicates_per_percentage, random_missingness_percentages,
                                               number_of_block_replicates_per_mask, weighted_lower_half_percentages,
                                               weighted_upper_half_percentages):

    random_masks = generate_random_masks_on_the_fly(n, number_of_random_replicates_per_percentage, random_missingness_percentages)
    block_masks = generate_block_masks_on_the_fly(n, number_of_block_replicates_per_mask, weighted_lower_half_percentages, weighted_upper_half_percentages)
    return np.concatenate([random_masks, block_masks], axis = 0)


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
    
class CustomMaskDataset(Dataset):
    def __init__(self, masks):
        self.masks = masks

    def __len__(self):
        return ((self.masks).shape[0])

    def __getitem__(self, idx):
        mask = self.masks[idx,:,:,:]
        return mask


    
def get_training_and_evaluation_dataset_per_mask(number_of_replicates, number_of_evaluation_replicates,
                                                 batch_size, eval_batch_size, seed_values, range_value,
                                                 smooth_value, mask, n):
    minX = minY = -10
    maxX = maxY = 10
    train_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                  number_of_replicates, n)

    train_dataset = CustomSpatialImageandSingleMaskDataset(train_images)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                 number_of_evaluation_replicates, n)
    
    eval_dataset = CustomSpatialImageandSingleMaskDataset(eval_images, mask)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_random_masking_training_and_evaluation_dataset(number_of_random_replicates,
                                                       random_missingness_percentages,
                                                       number_of_evaluation_random_replicates,
                                                       batch_size, eval_batch_size):
    minX = minY = -10
    maxX = maxY = 10
    n = 32
    train_masks = generate_random_masks_on_the_fly(n, number_of_random_replicates,
                                             random_missingness_percentages)

    train_dataset = CustomMaskDataset(train_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_masks = generate_random_masks_on_the_fly(n, number_of_evaluation_random_replicates,
                                                  random_missingness_percentages)
    
    eval_dataset = CustomMaskDataset(eval_masks)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_training_and_evaluation_image_datasets_per_mask(number_of_replicates_per_mask,
                                                        number_of_evaluation_replicates_per_mask,
                                                        total_masks, evaluation_total_masks,
                                                        range_value, smooth_value, seed_values,
                                                        image_batch_size, eval_batch_size):
    
    minX = minY = -10
    maxX = maxY = 10
    n = (32**2)
    train_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                  number_of_replicates_per_mask, n)
    train_dataset = CustomSpatialImageDataset(train_images)
    train_dataloader = DataLoader(train_dataset, batch_size = image_batch_size, shuffle = True)

    eval_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                  number_of_evaluation_replicates_per_mask, n)
    eval_dataset = CustomSpatialImageDataset(eval_images)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_training_and_evaluation_mask_and_image_datasets_per_mask(draw_number, number_of_random_replicates, 
                                                                 random_missingness_percentages, 
                                                                 number_of_evaluation_random_replicates,
                                                                 batch_size, eval_batch_size, range_value,
                                                                 smooth_value, seed_values, n,
                                                                 trainmaxminfile):
    
    minX = -10
    maxX = 10
    minY = -10
    maxX = 10
    maxY = 10
    
    train_masks = generate_random_masks_on_the_fly(n, number_of_random_replicates,
                                                   random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, number_of_evaluation_random_replicates,
                                                  random_missingness_percentages)
    train_images, eval_images = generate_train_and_evaluation_brown_resnick_process(range_value,
                                                                                    smooth_value,
                                                                                    seed_values[0],
                                                                                    number_of_random_replicates,
                                                                                    number_of_evaluation_random_replicates,
                                                                                    n)
    if(draw_number == 0):
        train_images = log_transformation(train_images)
        eval_images = log_transformation(eval_images)
        trainlogmin = float(np.min(train_images))
        trainlogmax = float(np.max(train_images))
        trainimages01 = (train_images - trainlogmin)/(trainlogmax - trainlogmin)
        trainquant01 = float(np.mean(trainimages01))
        print(trainquant01)
        print(float(np.quantile(trainimages01, [.5])))
        train_images = global_quantile_boundary_process(train_images, trainlogmin, trainlogmax, trainquant01)
        trainlogmaxmin = np.array([trainlogmin, trainlogmax, trainquant01])
        eval_images = global_quantile_boundary_process(eval_images, trainlogmin, trainlogmax, trainquant01)
        np.save(trainmaxminfile, trainlogmaxmin)

    else:
        trainlogmaxmin = np.load(trainmaxminfile)
        train_images = log_transformation(train_images)
        eval_images = log_transformation(eval_images)
        train_images = global_quantile_boundary_process(train_images, trainlogmaxmin[0], trainlogmaxmin[1], trainlogmaxmin[2])
        eval_images = global_quantile_boundary_process(eval_images, trainlogmaxmin[0], trainlogmaxmin[1], trainlogmaxmin[2])

    train_dataset = CustomSpatialImageandMaskDataset(train_images, train_masks)
    eval_dataset = (CustomSpatialImageandMaskDataset)(eval_images, eval_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_training_and_evaluation_random_and_block_mask_and_image_datasets_per_mask(number_of_random_replicates_per_percentage, 
                                                                                  random_missingness_percentages,
                                                                                  number_of_block_replicates_per_mask,
                                                                                  weighted_lower_half_percentages,
                                                                                  weighted_upper_half_percentages,
                                                                                  number_of_evaluation_random_replicates_per_percentage,
                                                                                  number_of_evaluation_block_replicates_per_mask,
                                                                                  batch_size, eval_batch_size, range_value,
                                                                                  smooth_value, seed_values):
    
    minX = -10
    maxX = 10
    minY = -10
    maxX = 10
    maxY = 10
    n = 32
    train_masks = generate_random_and_block_masks_on_the_fly(n, number_of_random_replicates_per_percentage, random_missingness_percentages,
                                                             number_of_block_replicates_per_mask, weighted_lower_half_percentages, weighted_upper_half_percentages)
    eval_masks = generate_random_and_block_masks_on_the_fly(n, number_of_evaluation_random_replicates_per_percentage, random_missingness_percentages,
                                                            number_of_evaluation_block_replicates_per_mask, weighted_lower_half_percentages,
                                                            weighted_upper_half_percentages)
    train_image_and_mask_number = train_masks.shape[0]
    eval_image_and_mask_number = eval_masks.shape[0]

    train_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                  train_image_and_mask_number, n)
    eval_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                  eval_image_and_mask_number, n)
    train_dataset = CustomSpatialImageandMaskDataset(train_images, train_masks)
    eval_dataset = CustomSpatialImageandMaskDataset(eval_images, eval_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader


def get_next_batch(image_and_mask_iterator, config):

    images, masks = (next(image_and_mask_iterator))
    images = images.to(config.device).float()
    masks = masks.to(config.device).float()
    return images, masks

def get_next_mask_batch(mask_iterator, config):

    masks = (next(mask_iterator))
    masks = masks.to(config.device).float()
    return masks

def get_next_image_batch(image_iterator, config):

    images = (next(image_iterator))
    images = images.to(config.device).float()
    return images


