import numpy as np
import pandas as pd
import torch
import scipy
import seaborn as sns

def index_to_matrix_index(index,n):

    return (int(index / n), int(index % n))

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

#mask needs to be type numpy of 0s/1s, returns norm matrix with rows and columns associated with
#masked locations in original nxn matrix masked out in n**2xn**2 matrix (now (n**2-m)x(n**2-m))
def construct_masked_norm_matrix(mask, minX, maxX, minY, maxY, n):
    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    #X is a matrix of nxn which is latitudes of all nxn obs, same for Y
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    missing_indices = np.squeeze((np.argwhere(mask.reshape((n**2)))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    X = np.delete(X, missing_indices, axis = 0)
    Y = np.delete(Y, missing_indices, axis = 0)
    #reshape X and Y to (n**2-mx1) vectors of latitude and longitude respectively
    #repeat longitudes and latitudes such that you get n**2 x n**2 matrix
    X_matrix = (np.repeat(X, (n**2-m), axis = 0)).reshape((n**2-m), (n**2-m))
    Y_matrix = (np.repeat(Y, (n**2-m), axis = 0)).reshape((n**2-m), (n**2-m))

    longitude_squared = np.square(np.subtract(X_matrix, np.transpose(X_matrix)))
    latitude_squared = np.square(np.subtract(Y_matrix, np.transpose(Y_matrix)))
    masked_norm_matrix = np.sqrt(np.add(longitude_squared, latitude_squared))
    return masked_norm_matrix

def construct_masked_norm_matrix1(mask, minX, maxX, minY, maxY, n):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    missing_indices = np.squeeze((np.argwhere(mask.reshape((n**2)))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    masked_norm_matrix = np.delete(norm_matrix, missing_indices, axis = 0)
    masked_norm_matrix = np.delete(masked_norm_matrix, missing_indices, axis = 1)
    return masked_norm_matrix

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return exp_kernel

#sigma22 (# observed x # observed), can construct Sigma11 (# unobserved x unobserved) via (1-mask)
def construct_masked_exp_kernel(mask, minX, maxX, minY, maxY, n, variance, lengthscale):

    masked_norm_matrix = construct_masked_norm_matrix(mask, minX, maxX, minY, maxY, n)
    masked_exp_kernel = variance*np.exp((-1/lengthscale)*masked_norm_matrix)
    return masked_exp_kernel


def construct_masked_norm_vector(mask, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    missing_indices = np.squeeze(np.argwhere(mask.reshape((n**2))))
    missing_indices = missing_indices.reshape((missing_indices.shape[0]))
    m = missing_indices.shape[0]
    missing_xlocations = X[missing_indices]
    missing_ylocations = Y[missing_indices]
    missing_locations = np.zeros((m,2))
    missing_locations[:,0] = missing_xlocations.reshape((m))
    missing_locations[:,1] = missing_ylocations.reshape((m))
    X = np.delete(X, missing_indices)
    Y = np.delete(Y, missing_indices)
    masked_norm_vector = np.zeros(((n**2-m), m))

    for i in range(0, m):
        norm_vector = (np.sqrt(np.add(np.square(X-missing_locations[i,0]),
                                      np.square(Y-missing_locations[i,1]))))
        masked_norm_vector[:,i] = norm_vector.reshape((n**2-m))

    return masked_norm_vector


#returns matrix of size (# observed pixels x # unobserved) i.e. sigma_21
def construct_masked_exp_kernel_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale):

    masked_norm_vector = construct_masked_norm_vector(mask, minX, maxX, minY, maxY, n)
    masked_exp_vector = variance*np.exp((-1/lengthscale)*masked_norm_vector)
    return masked_exp_vector

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
    return y_matrix, gp_matrix


def generate_student_nugget(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates,
                            seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    studentgenerator = scipy.stats.multivariate_t(loc = np.zeros(n**2), shape = kernel, df = df, seed = seed_value)
    #shape = (number_of_replicates, n**2)
    studentsamples = (studentgenerator.rvs(size = number_of_replicates))
    student_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, number_of_replicates):
        student_matrix[i,:,:,:] = studentsamples[i,:].reshape((1,n,n))
    return studentsamples, student_matrix

#make sure observed_vector = ((n-m)x1) and unobserved_unconditional_mean = (m x 1)
# and observed unconditonal mean ((n-m)x1)
def construct_conditional_mean_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                      observed_vector, observed_unconditional_mean,
                                      unobserved_unconditional_mean):
    
    Sigma12 = construct_masked_exp_kernel_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale)
    Sigma22 = construct_masked_exp_kernel((1-mask), minX, maxX, minY, maxY, n, variance, lengthscale)
    conditional_mean = (unobserved_unconditional_mean + 
    np.matmul(np.matmul(Sigma12, np.linalg.inv(Sigma22)), observed_vector - observed_unconditional_mean))
    return conditional_mean

#construct matrix Sigma1|2, observed_vector = (n-m)x1, observed_unconditional_mean = (n-m)x1
#unobserved_unconditional_mean = ((n-m)x1)
def construct_conditional_covariance_matrix(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                      observed_vector, observed_unconditional_mean,
                                      unobserved_unconditional_mean, df):
    
    #dim of X2 (observed) = n-m
    obs_dim = observed_vector.shape[0]
    m = (n**2)-obs_dim
    conditional_df = obs_dim + df
    print("conditional df")
    print(conditional_df)
    #(n-m)x(n-m) matrix
    Sigma22 = construct_masked_exp_kernel((1-mask), minX, maxX, minY, maxY, n, variance, lengthscale)
    Sigma22inv =np.linalg.inv(Sigma22)
    #should be a scalar
    p1 = float(np.matmul(np.matmul(np.transpose((observed_vector-observed_unconditional_mean)), Sigma22inv),
              (observed_vector - observed_unconditional_mean)))
    scalar_modifier = (df+p1)/conditional_df
    print("scalar")
    print(scalar_modifier)
    Sigma12 = construct_masked_exp_kernel_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale)
    Sigma21 = np.transpose(Sigma12)
    #Sigma11 should be unobserved matrix part (m x m)
    Sigma11 = construct_masked_exp_kernel(mask, minX, maxX, minY, maxY, n, variance, lengthscale)
    #Sigma12 should be (m x (n-mx))
    p2matrix = np.subtract(Sigma11, np.matmul(np.matmul(Sigma12, Sigma22inv), Sigma21))
    conditional_covariance_matrix = scalar_modifier*p2matrix
    return conditional_covariance_matrix

def sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                    observed_vector, observed_unconditional_mean,
                                    unobserved_unconditional_mean, df,
                                    number_of_replicates, seed_value):

    conditional_mean =construct_conditional_mean_vector(mask, minX, maxX, minY, maxY, n,
                                                        variance, lengthscale, observed_vector,
                                                        observed_unconditional_mean,
                                                        unobserved_unconditional_mean)
    
    conditional_variance = construct_conditional_covariance_matrix(mask, minX, maxX, minY, maxY, n,
                                                                   variance, lengthscale, observed_vector,
                                                                   observed_unconditional_mean, 
                                                                   unobserved_unconditional_mean, df)
    

    nminusm = observed_vector.shape[0]
    m = (n**2) - nminusm
    conditional_df = df + nminusm
    print("conditional df")
    print(conditional_df)
    conditional_mean = conditional_mean.reshape((m))
    if(np.all(np.linalg.eigvals(conditional_variance) >= 0)):
        studentgenerator = scipy.stats.multivariate_t(loc = conditional_mean, shape = conditional_variance,
                                                  df = conditional_df, seed = seed_value)
        conditional_samples = (studentgenerator.rvs(size = number_of_replicates))
    else:
        return None
    
    return conditional_samples

def true_conditional_image_sampling(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                    observed_vector, observed_unconditional_mean,
                                    unobserved_unconditional_mean, df, number_of_replicates,
                                    seed_value, observed_matrix):
    
    cond_unobserved_samples = sample_conditional_distribution(mask, minX, maxX, minY, maxY, n,
                                                              variance, lengthscale,
                                                              observed_vector,
                                                              observed_unconditional_mean,
                                                              unobserved_unconditional_mean, df,
                                                              number_of_replicates, seed_value)
    m = missing_indices.shape[0]
    conditional_samples = np.zeros((number_of_replicates, n, n))
    for i in range(0,m):
        matrix_index = index_to_matrix_index(missing_indices[i], n)
        conditional_samples[:,matrix_index[0],matrix_index[1]] = cond_unobserved_samples[:,i]

    observed_matrix = np.repeat(observed_matrix.reshape((1,n,n)), repeats = number_of_replicates, axis = 0)
    conditional_samples = np.add(observed_matrix, conditional_samples)
    #conditional_samples = observed_matrix
    return conditional_samples

def visualize_conditional_density(mask, tsamplematrix, t, matrixindex):

    minX = minY = -10
    maxX = maxY = 10
    n = 32
    variance = .4
    lengthscale = 1.6
    true_conditional_image_sampling(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                    observed_vector, observed_unconditional_mean,
                                    unobserved_unconditional_mean, df, number_of_replicates,
                                    seed_value, observed_matrix)






    
    




#single
n = 32
p = .9
#mask = (torch.bernoulli(p*torch.ones((n,n)))).numpy()
mask = np.ones((n,n))
mask[8:24,8:24] = 0
number_of_replicates = 2
minX = minY = -10
maxX = maxY = 10
variance = .4
lengthscale = 1.6
df = 1
seed_value = 85634
student_vector, student_matrix = generate_student_nugget(minX, maxX, minY, maxY, n, variance,
                                                         lengthscale, df, seed_value,
                                                         number_of_replicates)
number_of_replicates = 1
student_vector = student_vector[0,:].reshape((1,n**2))
student_matrix = student_matrix[0,:,:].reshape((n,n))
observed_indices = (np.squeeze(np.argwhere((mask).reshape((n**2,)))))
missing_indices = (np.squeeze(np.argwhere((1-mask).reshape((n**2,)))))
nminusm = observed_indices.shape[0]
m = missing_indices.shape[0]
print("missing")
print(m)
observed_vector = ((student_vector[:,observed_indices]))
observed_vector = observed_vector.reshape((nminusm,number_of_replicates))
observed_unconditional_mean = np.zeros(((nminusm,number_of_replicates)))
unobserved_unconditional_mean = np.zeros(((m,number_of_replicates)))
cm = construct_conditional_mean_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                      observed_vector, observed_unconditional_mean,
                                      unobserved_unconditional_mean)
cm = construct_conditional_covariance_matrix(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                      observed_vector, observed_unconditional_mean,
                                      unobserved_unconditional_mean, df)

seed_value = 234324
number_of_replicates = 1000
observed_matrix = np.multiply(mask, student_matrix)
cs = true_conditional_image_sampling(mask, minX, maxX, minY, maxY, n, variance, lengthscale,
                                    observed_vector, observed_unconditional_mean,
                                    unobserved_unconditional_mean, df,
                                    number_of_replicates, seed_value, observed_matrix)

import matplotlib.pyplot as plt
plt.imshow(student_matrix, vmin = -2, vmax = 2)
plt.savefig("observed.png")
plt.clf()
plt.imshow(student_matrix, vmin = -2, vmax = 2, alpha = mask)
plt.savefig("partially_observed.png")
plt.clf()
plt.imshow(cs[0,:,:], vmin = -2, vmax = 2, alpha = mask)
plt.savefig("generated_observed.png")
plt.clf()
plt.imshow(cs[0,:,:], vmin = -2, vmax = 2, alpha = (1-mask))
plt.savefig("generated.png")
plt.clf()
plt.imshow(cs[0,:,:], vmin = -2, vmax = 2)
plt.savefig("generated_whole.png")
plt.clf()




                                      
                                      