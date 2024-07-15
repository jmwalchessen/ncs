import numpy as np
import sys
from numpy import linalg
from block_mask_generation import *
import torch
from torch.utils.data import Dataset, DataLoader
import scipy

def construct_norm_matrix(minX, maxX, minY, maxY, n):
    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    X_matrix = (np.repeat(X, n**2, axis = 0)).reshape((n**2, n**2))
    Y_matrix = (np.repeat(Y, n**2, axis = 0)).reshape((n**2, n**2))
    longitude_squared = np.square(np.subtract(X_matrix, np.transpose(X_matrix)))
    latitude_squared = np.square(np.subtract(Y_matrix, np.transpose(Y_matrix)))
    norm_matrix = np.sqrt(np.add(longitude_squared, latitude_squared))
    return norm_matrix

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel)

def construct_exp_kernel_without_variance_from_norm_matrix(norm_matrix, lengthscale):

    exp_kernel_without_variance = np.exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel_without_variance)

def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_replicates,
                              seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    L = np.linalg.cholesky(kernel)
    y_matrix = np.matmul(L, np.transpose(z_matrix))
    
    gp_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, y_matrix.shape[1]):
        gp_matrix[i,:,:,:] = y_matrix[:,i].reshape((1,n,n))
    return gp_matrix

def generate_student_nugget(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    studentgenerator = scipy.stats.multivariate_t(loc = np.zeros(n**2), shape = kernel, df = df, seed = 23423)
    #shape = (number_of_replicates, n**2)
    studentsamples = (studentgenerator.rvs(size = number_of_replicates))
    student_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, number_of_replicates):
        student_matrix[i,:,:,:] = studentsamples[i,:].reshape((1,n,n))
    return student_matrix


#first column of parameter_matrix is variance
def generate_data_on_the_fly(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates_per_mask):
    

    train_images = generate_student_nugget(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                           number_of_replicates_per_mask)
    return train_images



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
                                                 batch_size, eval_batch_size, variance,
                                                 lengthscale, mask, df):
    minX = minY = -10
    maxX = maxY = 10
    n = 32
    train_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale,
                                                              df, number_of_replicates)

    train_dataset = CustomSpatialImageandSingleMaskDataset(train_images)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n, variance, lengthscale,
                                           df, number_of_evaluation_replicates)
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
                                                        variance, lengthscale,
                                                        image_batch_size, eval_batch_size, df):
    
    minX = minY = -10
    maxX = maxY = 10
    n = 32
    train_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale, df,
                                                              total_masks*number_of_replicates_per_mask)
    train_dataset = CustomSpatialImageDataset(train_images)
    train_dataloader = DataLoader(train_dataset, batch_size = image_batch_size, shuffle = True)

    eval_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n, variance, lengthscale, df,
                                           total_masks*number_of_evaluation_replicates_per_mask)
    eval_dataset = CustomSpatialImageDataset(eval_images)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader


def get_training_and_evaluation_random_mask_and_image_datasets_per_mask(number_of_random_replicates, 
                                                                 random_missingness_percentages, 
                                                                 number_of_evaluation_random_replicates,
                                                                 batch_size, eval_batch_size, variance,
                                                                 lengthscale, df):
    
    minX = -10
    maxX = 10
    minY = -10
    maxX = 10
    maxY = 10
    n = 32
    
    train_masks = generate_random_masks_on_the_fly(n, number_of_random_replicates, random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, number_of_evaluation_random_replicates, random_missingness_percentages)
    train_image_and_mask_number = len(random_missingness_percentages)*number_of_random_replicates
    eval_image_and_mask_number = len(random_missingness_percentages)*number_of_evaluation_random_replicates

    train_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale, df,
                                                              train_image_and_mask_number)
    eval_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale, df,
                                                              eval_image_and_mask_number)
    train_dataset = CustomSpatialImageandMaskDataset(train_images, train_masks)
    eval_dataset = CustomSpatialImageandMaskDataset(eval_images, eval_masks)
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
                                                                                  batch_size, eval_batch_size, variance,
                                                                                  lengthscale, df):
    
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

    train_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale, df,
                                                              train_image_and_mask_number)
    eval_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale, df,
                                                              eval_image_and_mask_number)
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