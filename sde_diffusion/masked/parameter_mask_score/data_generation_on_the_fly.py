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

def generate_data_on_the_fly(minX, maxX, minY, maxY, n, parameter_matrix, number_of_replicates,
                             seed_values):
    
    x_train_images = np.empty((0, 1, n, n))
    x_train_parameters = np.empty((0, parameter_matrix.shape[1]))
    for i in range(0, parameter_matrix.shape[0]):
        current_images = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                                   parameter_matrix[i,0],
                                                   parameter_matrix[i,1],
                                                   number_of_replicates, seed_values[i])
        x_train_images = np.concatenate([x_train_images, current_images], axis = 0)
        x_train_parameters = np.concatenate([x_train_parameters,
                                             np.repeat((parameter_matrix[i,:]).reshape((1,2)),
                                             repeats = number_of_replicates, axis = 0)], axis = 0)

    return x_train_images, x_train_parameters



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




class CustomSpatialImageMaskandParameterDataset(Dataset):

    def __init__(self, images, masks, parameters):
        self.images = images
        self.masks = masks
        self.parameters = parameters

    def __len__(self):
        return ((self.images).shape[0])
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        mask = self.masks[idx,:,:,:]
        parameter = self.parameters[idx,:]
        image_and_mask = np.concatenate([image, mask], axis = 0)
        return image_and_mask, parameter


    
class CustomMaskDataset(Dataset):
    def __init__(self, masks):
        self.masks = masks

    def __len__(self):
        return ((self.masks).shape[0])

    def __getitem__(self, idx):
        mask = self.masks[idx,:,:,:]
        return mask
    
#seeds values list is a list of lists of tuples of length equal to number of missing percentages
def get_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     parameter_matrix, eval_parameter_matrix):
    
    diff = .6451612900000008
    minX = minY = -10-2*diff
    maxX = maxY = 10+2*diff
    n = 36
    train_images = np.zeros((0,1,n,n))
    eval_images = np.zeros((0,1,n,n))
    train_parameters = np.zeros((0,2))
    eval_parameters = np.zeros((0,2))

    for i, p in enumerate(random_missingness_percentages):
        print(i)
        if(p == 0):
            seed_values = np.random.randint(low = 0, high = 1000000, size = parameter_matrix.shape[0])
            timages, tparams = generate_data_on_the_fly(minX, maxX, minY, maxY, n, parameter_matrix,
                                                                                  number_of_random_replicates, seed_values)
            train_images = np.concatenate([train_images, timages], axis = 0)
            train_parameters = np.concatenate([train_parameters, tparams], axis = 0)

            eval_seed_values = np.random.randint(low = 0, high = 1000000, size = eval_parameter_matrix.shape[0])
            eimages, eparams = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              eval_parameter_matrix,
                                                              number_of_evaluation_random_replicates,
                                                              eval_seed_values)
            eval_images = np.concatenate([eval_images, eimages], axis = 0)
            eval_parameters = np.concatenate([eval_parameters, eparams], axis = 0)

        else:
            seed_values = np.random.randint(low = 0, high = 1000000, size = parameter_matrix.shape[0])
            eval_seed_values = np.random.randint(low = 0, high = 1000000, size = eval_parameter_matrix.shape[0])
            timages, tparams = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              parameter_matrix,
                                                              number_of_random_replicates,
                                                              seed_values)
            eimages, eparams = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              eval_parameter_matrix,
                                                              number_of_evaluation_random_replicates,
                                                              eval_seed_values)
            train_images = np.concatenate([train_images, np.repeat(timages, number_of_masks_per_image, axis = 0)])
            train_parameters = np.concatenate([train_parameters, np.repeat(tparams, number_of_masks_per_image, axis = 0)], axis =0)
            eval_images = np.concatenate([eval_images, np.repeat(eimages, number_of_evaluation_masks_per_image, axis = 0)])
            eval_parameters = np.concatenate([eval_parameters, np.repeat(eparams, number_of_evaluation_masks_per_image, axis = 0)], axis =0)

    n = 32
    train_masks = generate_random_masks_on_the_fly(n, train_images.shape[0], random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, eval_images.shape[0], random_missingness_percentages)
    train_images = train_images[:,:,2:34,2:34]
    eval_images = eval_images[:,:,2:34,2:34]
    train_dataset = CustomSpatialImageMaskandParameterDataset(train_images, train_masks, train_parameters)
    eval_dataset = CustomSpatialImageMaskandParameterDataset(eval_images, eval_masks, eval_parameters)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def produce_parameters_via_uniform(number_of_parameters, boundary_start, boundary_end):

    uniform_generator = scipy.stats.uniform()
    parameters = ((boundary_end - boundary_start)*uniform_generator.random(n = number_of_parameters)) + boundary_start
    return parameters


def get_training_and_evaluation_data_per_percentages_for_parameters(number_of_random_replicates, random_missingness_percentages,
                                                                    number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                                    number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                                    number_of_parameters, boundary_start, boundary_end, number_of_eval_parameters):
    
    parameter_matrix = produce_parameters_via_uniform(number_of_parameters, boundary_start, boundary_end)
    eval_parameter_matrix = produce_parameters_via_uniform(number_of_eval_parameters, boundary_start, boundary_end)
    train_dataloader, eval_dataloader = get_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                                                         number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                                                         number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                                                         parameter_matrix, eval_parameter_matrix)
    return train_dataloader, eval_dataloader




def get_training_and_evaluation_random_mask_and_image_datasets(number_of_random_replicates, 
                                                               random_missingness_percentages, 
                                                               number_of_evaluation_random_replicates,
                                                               number_of_masks_per_image,
                                                               number_of_evaluation_masks_per_image,
                                                               batch_size, eval_batch_size, variance,
                                                               lengthscale, seed_values):
    
    minX = -10
    maxX = 10
    minY = -10
    maxX = 10
    maxY = 10
    n = 32
    
    train_mask_number = len(random_missingness_percentages)*number_of_masks_per_image
    eval_mask_number = len(random_missingness_percentages)*number_of_evaluation_masks_per_image

    diff = .6451612900000008
    minX = minY = -10-2*diff
    maxX = maxY = 10+2*diff
    n = 36
    train_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale,
                                                              number_of_random_replicates,
                                                              seed_values[0])
    eval_images = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              variance, lengthscale,
                                                              number_of_evaluation_random_replicates,
                                                              seed_values[1])
    train_images = train_images[:,:,2:34,2:34]
    eval_images = eval_images[:,:,2:34,2:34]
    train_images = np.repeat(train_images, train_mask_number, axis = 0)
    eval_images = np.repeat(eval_images, eval_mask_number, axis = 0)
    n = 32
    train_masks = generate_random_masks_on_the_fly(n, train_images.shape[0], random_missingness_percentages)
    eval_masks = generate_random_masks_on_the_fly(n, eval_images.shape[0], random_missingness_percentages)
    train_dataset = CustomSpatialImageMaskDataset(train_images, train_masks)
    eval_dataset = CustomSpatialImageMaskDataset(eval_images, eval_masks)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_next_batch(image_mask_and_parameter_iterator, config):

    images_and_masks, parameters = (next(image_mask_and_parameter_iterator))
    images_and_masks= images_and_masks.to(config.device).float()
    parameters = parameters.to(config.device).float()
    return images_and_masks, parameters

