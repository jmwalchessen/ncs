import numpy as np
import sys
from numpy import linalg
import torch
from torch.utils.data import Dataset, DataLoader
import subprocess
import os
import scipy

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(round(range_value, 2)),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
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



def generate_random_masks_via_observed_numbers_on_the_fly(n,number_of_random_replicates, observed_numbers):

    for i,m in enumerate(observed_numbers):
        masks_matrices = np.zeros((len(observed_numbers),number_of_random_replicates,n**2))
        for obs in range(m):
            obs_indices = (((n**2)*np.random.random(size = number_of_random_replicates)).astype(int)).reshape((number_of_random_replicates,1))
            obs_indices = np.concatenate([(np.arange(number_of_random_replicates)).reshape((number_of_random_replicates,1)),
                                          obs_indices], axis = 1)
            masks_matrices[i,obs_indices[:,0],obs_indices[:,1]] = 1

        masks_matrices = masks_matrices.reshape((len(observed_numbers)*number_of_random_replicates,1,n,n))
        return masks_matrices


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
            current_train_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                                                    number_of_random_replicates*number_of_masks_per_image, n)
            current_eval_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                                                    number_of_evaluation_random_replicates*number_of_evaluation_masks_per_image, n)
            train_images = np.concatenate([train_images, current_train_images])
            eval_images = np.concatenate([eval_images, current_eval_images])

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
def get_training_and_evaluation_data_per_observed_number(number_of_random_replicates, observed_numbers,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     range_value, smooth_value, seed_values_list):
    
    n = 32
    train_images = np.zeros((0,1,n,n))
    eval_images = np.zeros((0,1,n,n))

    for i, m in enumerate(observed_numbers):
        seed_values = seed_values_list[i]

        timages = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                number_of_random_replicates, n)
        eimages = generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                number_of_evaluation_random_replicates, n)

        train_images = np.concatenate([train_images, np.repeat(timages, number_of_masks_per_image, axis = 0)])
        eval_images = np.concatenate([eval_images, np.repeat(eimages, number_of_evaluation_masks_per_image, axis = 0)])

    train_images = np.log(train_images)
    eval_images = np.log(eval_images)
    nrep = int(train_images.shape[0]/len(observed_numbers))
    eval_nrep = int(eval_images.shape[0]/len(observed_numbers))
    train_masks = generate_random_masks_via_observed_numbers_on_the_fly(n, nrep, observed_numbers)
    eval_masks = generate_random_masks_via_observed_numbers_on_the_fly(n, eval_nrep, observed_numbers)
    train_dataset = CustomSpatialImageMaskDataset(train_images, train_masks)
    eval_dataset = CustomSpatialImageMaskDataset(eval_images, eval_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader


def produce_percentages_via_uniform(number_of_percentages, boundary_start, boundary_end):

    uniform_generator = scipy.stats.uniform()
    percentages = ((boundary_end - boundary_start)*uniform_generator.rvs(number_of_percentages)) + boundary_start
    return percentages

def get_training_and_evaluation_data_for_percentages(number_of_percentages, boundary_start, boundary_end, number_of_random_replicates, 
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size, range_value, smooth_value,
                                                     seed_values_list):
    
    random_missingness_percentages = produce_percentages_via_uniform(number_of_percentages, boundary_start, boundary_end)
    train_dataloader, eval_dataloader = get_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     range_value, smooth_value, seed_values_list)
    return train_dataloader, eval_dataloader


def get_training_and_evaluation_data_for_observed_numbers(observed_number_start, observed_number_end, number_of_random_replicates, 
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size, range_value, smooth_value,
                                                     seed_values_list):
    
    observed_numbers = [i for i in range(observed_number_start, observed_number_end, 1)]
    train_dataloader, eval_dataloader = get_training_and_evaluation_data_per_observed_number(number_of_random_replicates, observed_numbers,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     range_value, smooth_value, seed_values_list)
    return train_dataloader, eval_dataloader

   