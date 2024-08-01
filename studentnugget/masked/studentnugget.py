import numpy as np
import torch as th
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates, index):

    exp_kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    studentgenerator = scipy.stats.multivariate_t(loc = np.zeros(n**2), shape = exp_kernel, df = df, seed = 23423)
    studentsamples = (studentgenerator.rvs(size = number_of_replicates))
    marginal_density = studentsamples[:,index]
    fig, ax = plt.subplots(1)
    pdd = pd.DataFrame(marginal_density,
                                    columns = None)
    sns.kdeplot(data = pdd, ax = ax, palette=['blue'])
    plt.show()

def marginal_density_without_spatial_variation(n, df, number_of_replicates, index):

    tsample, tmatrix = generate_student_nugget_without_spatial_variation(n, df, number_of_replicates, 43025)
    marginal_density = tsample[:,index]
    fig, ax = plt.subplots(1)
    pdd = pd.DataFrame(marginal_density, columns = None)
    sns.kdeplot(data = pdd, ax = ax, palette=['blue'])
    plt.show()

def bivariate_density_without_spatial_variatoin(n, df, number_of_replicates, index1,
                                                index2):
    
    tsample, tmatrix = generate_student_nugget_without_spatial_variation(n, df, number_of_replicates, 43025)
    bivariate_density = tsample[:,np.array([index1,index2])]
    fig, ax = plt.subplots(1)
    sns.kdeplot(x = bivariate_density[:,0], y = bivariate_density[:,1], ax = ax)
    plt.show()


def generate_student_nugget_without_spatial_variation(n, df, number_of_replicates, seed_value):

    scale = np.identity(n**2)
    studentgenerator = scipy.stats.multivariate_t(loc = np.zeros(n**2), shape = scale, df = df, seed = seed_value)
    studentsamples = (studentgenerator.rvs(size = number_of_replicates))
    student_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, number_of_replicates):
        student_matrix[i,:,:,:] = studentsamples[i,:].reshape((1,n,n))
    return studentsamples, student_matrix

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



def visualize(image):

    plt.imshow(image, vmin = -2, vmax = 2)
    plt.show()

n = 32
minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
variance = .4
lengthscale = 10
df = 500
number_of_replicates = 10000
seed_value = 29389
tsample, tmatrix = generate_student_nugget(minX, maxX, minY, maxY, n, variance, lengthscale,
                                           df, number_of_replicates, seed_value)
visualize(tmatrix[0,:,:,:].reshape((n,n)))
tsample, tmatrix = generate_student_nugget_without_spatial_variation(n,
                                           df, number_of_replicates, seed_value)
#marginal_density_without_spatial_variation(n, df, number_of_replicates, 500)
index1 = 512
index2 = 513
#bivariate_density_without_spatial_variatoin(n, df, number_of_replicates, index1,
                                                #index2)