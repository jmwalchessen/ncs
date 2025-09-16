import numpy as np

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
    missing_indices = (np.argwhere(mask.reshape((n**2))))
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
    missing_indices = (np.argwhere(mask.reshape((n**2))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    masked_norm_matrix = np.delete(norm_matrix, missing_indices, axis = 0)
    masked_norm_matrix = np.delete(masked_norm_matrix, missing_indices, axis = 1)
    return masked_norm_matrix

def construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale):

    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*np.exp((-1/lengthscale)*norm_matrix)
    return exp_kernel

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
    missing_indices = (np.argwhere(mask.reshape((n**2))))
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

def construct_masked_norm_vector1(mask, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    missing_indices = (np.argwhere(mask.reshape((n**2))))
    m = missing_indices.shape[0]
    missing_indices = missing_indices.reshape((m))
    missing_xlocations = X[missing_indices]
    missing_ylocations = Y[missing_indices]
    missing_locations = np.zeros((m,2))
    missing_locations[:,0] = missing_xlocations.reshape((m))
    missing_locations[:,1] = missing_ylocations.reshape((m))
    masked_norm_vector = np.zeros(((n**2), m))
    for i in range(0, m):
        norm_vector = (np.sqrt(np.add(np.square(X-missing_locations[i,0]),
                                      np.square(Y-missing_locations[i,1]))))
        masked_norm_vector[:,i] = norm_vector.reshape((n**2))
    
    masked_norm_vector = np.delete(masked_norm_vector, missing_indices, axis = 0)
    return masked_norm_vector


def construct_masked_exp_kernel_vector(mask, minX, maxX, minY, maxY, n, variance, lengthscale):

    masked_norm_vector = construct_masked_norm_vector(mask, minX, maxX, minY, maxY, n)
    masked_exp_vector = variance*np.exp((-1/lengthscale)*masked_norm_vector)
    return masked_exp_vector

def construct_empirical_mean_variance(mask, minX, maxX, minY, maxY, n, variance, lengthscale, y):

    #(n**2-m)x(n**2-m) matrix
    masked_exp_kernel = construct_masked_exp_kernel((1-mask), minX, maxX, minY, maxY, n, variance,
                                                    lengthscale)
    
    #(n**2-m)xm vector
    masked_exp_kernel_vector = construct_masked_exp_kernel_vector((1-mask), minX, maxX, minY, maxY,
                                                                  n, variance, lengthscale)
    kriging_matrix = np.matmul(np.transpose(masked_exp_kernel_vector),
                               np.linalg.inv(masked_exp_kernel))
    #construct a mx1 vector, m is the number of fixed locations vector
    conditional_mean =  np.matmul(kriging_matrix, y)
    variance_matrix = construct_masked_exp_kernel(mask, minX, maxX, minY, maxY, n, variance, lengthscale)
    cov_part = np.matmul(kriging_matrix, masked_exp_kernel_vector)
    conditional_variance = variance_matrix - cov_part
    return conditional_mean, conditional_variance

def sample_conditional_distribution(mask, minX, maxX, minY, maxY, n, variance, lengthscale, y,
                                    number_of_replicates):

    conditional_mean, conditional_variance = construct_empirical_mean_variance(mask, minX, maxX, minY,
                                                                             maxY, n, variance,
                                                                             lengthscale, y)
    if(np.all(np.linalg.eigvals(conditional_variance) >= 0)):
        conditional_samples = np.random.multivariate_normal(conditional_mean, conditional_variance,
                                                        number_of_replicates)
        return conditional_samples
    else:
        return None

def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    L = np.linalg.cholesky(kernel)
    y_matrix = (np.flip(np.matmul(L, np.transpose(z_matrix))))
    
    gp_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, y_matrix.shape[1]):
        gp_matrix[i,:,:,:] = y_matrix[:,i].reshape((1,n,n))
    return y_matrix, gp_matrix