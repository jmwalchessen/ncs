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
lengthscale = 1.6
exp_kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
a = scipy.stats.multivariate_t(loc = np.zeros(n**2), shape = exp_kernel, df = 1, seed = 23423)
b = (a.rvs(size = 1))

number_of_replicates = 1000
index = 234
marginal_density(minX, maxX, minY, maxY, n, variance, lengthscale, df, number_of_replicates, index)