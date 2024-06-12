import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def realization_pipeline(stdoutput, n, number_of_replicates):

    stdout_str = (stdoutput.stdout).decode()
    y_str_split = stdout_str.split()
    y_str = y_str_split[slice(2, (3*(n**number_of_replicates) + 2), 3)]
    y = np.asarray([float(y_str[i]) for i in range(0,n*number_of_replicates)])
    y = y.reshape((number_of_replicates,1,int(np.sqrt(n)),int(np.sqrt(n))))
    return y

def call_brown_resnick_script(range_value, smooth_value, seed_value, number_of_replicates, n):

    stdout = subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                         str(smooth_value), str(number_of_replicates), str(seed_value)],
                        check = True, capture_output = True, text = False)
    images = realization_pipeline(stdout, n, number_of_replicates)
    return images

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    calls = int(number_of_replicates/100)
    images = np.zeros((0,1,int(np.sqrt(n)),int(np.sqrt(n))))
    for i in range(0, calls):
        current_images = call_brown_resnick_script(range_value, smooth_value, seed_value,
                                                   100, n)
        images = np.concatenate([images, current_images], axis = 0)
    
    current_images = call_brown_resnick_script(range_value, smooth_value, seed_value,
                                                   (number_of_replicates % n), n)
    images = np.concatenate([images, current_images])
    return images



def generate_random_masks_on_the_fly(n, number_of_random_replicates, random_missingness_percentages):

    mask_matrices = np.zeros((0,1,n,n))

    for idx in range(0, len(random_missingness_percentages)):
        missingness_percentage = random_missingness_percentages[idx]
        current_mask_matrices = np.random.binomial(n = 1, p = missingness_percentage,
                                                   size = (number_of_random_replicates, 1, n, n))
        mask_matrices = np.concatenate([mask_matrices, current_mask_matrices])
    return mask_matrices


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
                                                 smooth_value, mask):
    minX = minY = -10
    maxX = maxY = 10
    n = (32**2)
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

def get_training_and_evaluation_mask_and_image_datasets_per_mask(number_of_random_replicates, 
                                                                 random_missingness_percentages, 
                                                                 number_of_evaluation_random_replicates,
                                                                 batch_size, eval_batch_size, range_value,
                                                                 smooth_value, seed_values):
    
    minX = -10
    maxX = 10
    minY = -10
    maxX = 10
    maxY = 10
    n = 32
    
    train_masks = generate_random_masks_on_the_fly(n, number_of_random_replicates,
                                                   random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, number_of_evaluation_random_replicates,
                                                  random_missingness_percentages)
    train_image_and_mask_number = len(random_missingness_percentages)*number_of_random_replicates
    eval_image_and_mask_number = len(random_missingness_percentages)*number_of_evaluation_random_replicates
    n = (31**2)
    train_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[0],
                                                  number_of_random_replicates, n)
    eval_images = generate_brown_resnick_process(range_value, smooth_value, seed_values[1],
                                                  number_of_random_replicates, n)
    train_images = np.pad(train_images, ((0,0), (0,0), (1,0), (1,0)))
    eval_images = np.pad(eval_images, ((0,0), (0,0), (1,0), (1,0)))
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


