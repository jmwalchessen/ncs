import numpy as np
import sys
import scipy
from scipy import linalg
from numpy import linalg
from torch.utils.data import Dataset, DataLoader
from scipy.stats.qmc import LatinHypercube

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


#first column of parameter_matrix is variance
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


class CustomSpatialImageandParameterDataset(Dataset):
    def __init__(self, images, parameters):
        self.images = images
        self.parameters = parameters

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        parameter = self.parameters[idx,:]
        return image, parameter
    
def get_training_and_evaluation_dataset(number_of_replicates_per_parameter, number_of_evaluation_replicates_per_parameter,
                                        number_of_parameters, batch_size, eval_batch_size):
    minX = minY = -10
    maxX = maxY = 10
    n = 32
    parameter_sampler = LatinHypercube(d = 2)
    parameter_matrix = 2*parameter_sampler.random(n=number_of_parameters)
    seed_values = np.random.randint(0, 10000000, parameter_matrix.shape[0])
    train_images, train_parameters = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              parameter_matrix,
                                                              number_of_replicates_per_parameter,
                                                              seed_values)
    train_dataset = CustomSpatialImageandParameterDataset(train_images, train_parameters)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_images, eval_parameters = generate_data_on_the_fly(minX, maxX, minY, maxY, n,
                                                              parameter_matrix,
                                                              number_of_evaluation_replicates_per_parameter,
                                                              seed_values)
    eval_dataset = CustomSpatialImageandParameterDataset(eval_images, eval_parameters)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_next_batch(image_and_parameter_iterator, config):

    images, parameters = (next(image_and_parameter_iterator))
    images = images.to(config.device).float()
    parameters = parameters.to(config.device).float()
    return images, parameters