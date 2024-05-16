import torch as th
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys 

sys.path.append("..")

from image_diffusion import dist_util
from smc_utils.feynman_kac_pf import smc_FK 
from smc_utils.smc_utils import compute_ess_from_log_w
from image_diffusion.operators import get_operator, ConditioningMethod
from image_diffusion.image_util import get_dataloader, gen_mask, toU8, imwrite 
from image_diffusion.eval_util import pred 
from image_diffusion.my_feynman_kac_image_ddpm import TwistedDDPM
from image_diffusion.my_smc_diffusion import *



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

def generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                              number_of_replicates, seed_value):

    kernel = construct_exp_kernel(minX, maxX, minY, maxY, n, variance, lengthscale)
    np.random.seed(seed_value)
    z_matrix = np.random.multivariate_normal(np.zeros(n**2), np.identity(n**2), number_of_replicates)
    C = np.linalg.cholesky(kernel)
    y_matrix = (np.flip(np.matmul(np.transpose(C),
                                  np.transpose(z_matrix))))
    
    gp_matrix = np.zeros((number_of_replicates,1,n,n))
    for i in range(0, y_matrix.shape[1]):
        gp_matrix[i,:,:,:] = y_matrix[:,i].reshape((1,n,n))
    return y_matrix, gp_matrix

def construct_norm_vector(fixed_location, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    norm_vector = np.sqrt(np.add(np.square(X-fixed_location[0]), np.square(Y-fixed_location[1])))
    return norm_vector

#returns a (n**2)x1 vector
def construct_exp_kernel_vector(fixed_location, minX, maxX, minY,
                                maxY, n, variance, lengthscale):
    
    norm_vector = construct_norm_vector(fixed_location, minX, maxX, minY, maxY, n)
    exp_kernel_vector = variance*np.exp((-1/lengthscale)*norm_vector)
    return exp_kernel_vector

#construct a (n**2)xm vector, m is the number of fixed locations
def construct_norm_matrix_fixed_locations(fixed_locations, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),1))
    Y = Y.reshape((np.prod(Y.shape),1))
    #m is the number of fixed locations
    m = fixed_locations.shape[0]
    norm_matrix_fixed_locations = np.zeros((n**2, m))
    for i in range(0, m):
        norm_vector = (np.sqrt(np.add(np.square(X-fixed_locations[i,0]),
                                      np.square(Y-fixed_locations[i,1]))))
        norm_matrix_fixed_locations[:,i] = norm_vector.reshape((n**2))
    return norm_matrix_fixed_locations

#construct a (n**2)xm vector, m is the number of fixed locations
def construct_exp_kernel_matrix_fixed_locations(fixed_locations, minX, maxX, minY,
                                                maxY, n, variance, lengthscale):
    norm_matrix_fixed_locations = construct_norm_matrix_fixed_locations(fixed_locations,
                                                                        minX, maxX, minY,
                                                                        maxY, n)
    exp_matrix_fixed_locations = variance*np.exp((-1/lengthscale)*norm_matrix_fixed_locations)
    return exp_matrix_fixed_locations

def get_index(fixed_location, minX, maxX, minY, maxY, n):

    x = np.linspace(minX, maxX, n)
    y = np.linspace(minY, maxY, n)

    i = int((np.argwhere(abs(x - fixed_location[0]) < 0.1))[0])
    j = int((np.argwhere(abs(y - fixed_location[1]) < 0.1))[0])
    return (i,j), int(i*x.shape[0]+j)

def get_multiple_indices(fixed_locations, minX, maxX, minY, maxY, n):


    if(len(fixed_locations.shape)==1):
        index, combined_index = get_index(fixed_locations, minX, maxX, minY,
                                          maxY, n)
        m = 1
        index_matrix = np.zeros((m,2))
        index_vector = np.zeros((m,))
        index_matrix[0,:] = np.array([int(index[0]), int(index[1])])
        index_vector[0] = int(combined_index)
    else:
        m = fixed_locations.shape[0]
        index_matrix = np.zeros((m,2))
        index_vector = np.zeros((m,))
        for i in range(0,m):
            index, combined_index = get_index(fixed_locations[i,:], minX, maxX, minY,
                                                maxY, n)
            index_matrix[i,:] = np.array([int(index[0]), int(index[1])])
            index_vector[i] = int(combined_index)

    index_matrix = index_matrix.astype(np.int32)
    index_vector = index_vector.astype(np.int32)
    return index_matrix, index_vector

def remove_indices_matrix(fixed_locations, minX, maxX, minY, maxY, n, A):

    index_matrix, index_vector = get_multiple_indices(fixed_locations, minX, maxX,
                                                      minY, maxY, n)
    
    modified_A = np.delete(A, index_matrix[:,0], axis = 0)
    modified_A = np.delete(modified_A, index_matrix[:,1], axis = 1)
    return modified_A

def remove_indices_vector(fixed_locations, minX, maxX, minY, maxY, n, vector_A):

    index_matrix, index_vector = get_multiple_indices(fixed_locations, minX, maxX,
                                                      minY, maxY, n)
    modified_vectorA = np.delete(vector_A, index_vector, axis = 0)
    return modified_vectorA

#y should be n**2-1 vector
def construct_single_kriging_mean_variance(fixed_location, minX, maxX, minY,
                                maxY, n, variance, lengthscale, y):
    
    #n**2x1 vector
    exp_kernel_vector = construct_exp_kernel_vector(fixed_location, minX, maxX,
                                                    minY, maxY, n, variance,
                                                    lengthscale)
    #(n**2-1)x1 vector
    exp_kernel_vector = remove_indices_vector(fixed_location, minX, maxY,
                                              minX, maxX, n, exp_kernel_vector)
    #n**2 x n**2 matrix
    exp_kernel_matrix = construct_exp_kernel(minX, maxX, minY, maxY, n,
                                             variance, lengthscale)
    #(n**2-1)x(n**2-1) matrix
    exp_kernel_matrix = remove_indices_matrix(fixed_location, minX, maxX, minY,
                                              maxY, n, exp_kernel_matrix)
    
    #1x(n**2-1) vector
    kriging_vector = np.matmul(np.transpose(exp_kernel_vector),
                               np.linalg.inv(exp_kernel_matrix))
    
    #1x(n**2-1) vector
    conditional_mean = float(np.matmul(kriging_vector, y).reshape((1,)))
    conditional_variance = variance - float((np.matmul(kriging_vector, exp_kernel_vector)).reshape(1,))
    if (conditional_variance <= 0):
        conditional_variance = .001

    return conditional_mean, conditional_variance


#y should be (n**2-m)x1 matrix
def construct_kriging_mean_variance(fixed_locations, minX, maxX, minY,
                                maxY, n, variance, lengthscale, y):
    #n**2 x m matrix
    exp_kernel_fixed_locations_matrix = construct_exp_kernel_matrix_fixed_locations(fixed_locations,
                                                                    minX, maxX, minY,
                                                                    maxY, n, variance,
                                                                    lengthscale)
    exp_kernel_fixed_locations_matrix = remove_indices_vector(fixed_locations, minX,
                                                              maxX, minY, maxY, n,
                                                              exp_kernel_fixed_locations_matrix)
    #(n**2)x(n**2) matrix
    exp_kernel_matrix = construct_exp_kernel(minX, maxX, minY, maxY, n, variance,
                                             lengthscale)
    #(n**2-m)x(n**2-m) matrix
    exp_kernel_matrix = remove_indices_matrix(minX, maxX, minY, maxY, n,
                                              exp_kernel_matrix)
    
    #mx(n**2-m) vector
    kriging_matrix = np.matmul(np.transpose(exp_kernel_fixed_locations_matrix),
                               np.linalg.inv(exp_kernel_matrix))
    #construct a mx1 vector, m is the number of fixed locations vector
    conditional_mean =  np.matmul(kriging_matrix, y)
    conditional_variance = (variance - np.matmul(kriging_matrix,
                                                exp_kernel_fixed_locations_matrix))
    return conditional_mean, conditional_variance








def sample_single_conditional_distribution(fixed_location, minX, maxX, minY, maxY, n,
                                    variance, lengthscale, y, number_of_replicates):
    
    conditional_mean, conditional_variance = construct_single_kriging_mean_variance(fixed_location,
                                minX, maxX, minY, maxY, n, variance, lengthscale, y)
    conditional_samples = np.random.normal(conditional_mean, conditional_variance,
                                                        number_of_replicates)
    return conditional_samples

def sample_conditional_distribution(fixed_locations, minX, maxX, minY, maxY, n,
                                    variance, lengthscale, y, number_of_replicates):
    conditional_mean, conditional_variance = construct_kriging_mean_variance(fixed_locations,
                                minX, maxX, minY, maxY, n, variance, lengthscale, y)
    conditional_samples = np.random.multivariate_normal(conditional_mean, conditional_variance,
                                                        number_of_replicates)
    return conditional_samples

#fixed_locations is a mx2 matrix
def apply_mask(fixed_locations, minX, maxX, minY, maxY, n):

    location_indices_matrix, location_indices_vector = get_multiple_indices(fixed_locations,
                                            minX, maxX,
                                            minY, maxY, n)
    mask = (th.ones((n,n))).to(th.bool)
    m = fixed_locations.shape[0]
    location_indices_matrix = th.from_numpy(location_indices_matrix)
    for i in range(m):
        mask[location_indices_matrix[i,0],location_indices_matrix[i,1]] = False
    return mask




def sample_single_conditional_diffusion_distribution(diffusion, mask, resample_strategy,
                                                     ess_threshold, diffusion_timesteps,
                                                     particle_number,
                                                     model_kwargs, batch_particles, ref_image, 
                                                     pred_xstart_var_type, device):
    
    diffusion.task = "inpainting"
    operator = get_operator(device=device, name=diffusion.task)
    recon_prob_fn = ConditioningMethod(operator=operator).recon_prob
    diffusion.mask = mask
    measurement_mask = mask
    measurement = operator(data = ref_image, mask=measurement_mask)
    recon_prob_fn = partial(recon_prob_fn, measurement=measurement, mask=mask)
    #ref_img is (1,28,28) array with mostly -1 entries but some non
    #diffusion.measurement = ref_img*measurement_mask
    diffusion.set_measurement(ref_image*measurement_mask) 
    # resetting 
    diffusion.recon_prob_fn = recon_prob_fn
    #recon_prob_fn is from operators.py, -.5*sum((measurement-mask*x0_hat)**2)
    #where measurement = mask*ref_image or mask*x0?

    #not sure what to do about t_truncate
    diffusion.t_truncate = 0
    #not sure about this configuration either
    diffusion.use_mean_pred = True
    #initialize cache
    diffusion.clear_cache()

    #M and G are from TwistedDDPM for both class and inpainting tasks
    #Proposal distribution sort of is M
    M = partial(diffusion.M, model=score_model, device=device, 
            pred_xstart_var_type=pred_xstart_var_type)
#Weight function, if debug_plot = True, return twisted_xpred_start in compute_twisted_helper_function
    G = partial(diffusion.G, model=score_model, 
            debug_plot=False, debug_statistics=False, debug_info=False, 
            pred_xstart_var_type=pred_xstart_var_type)
    
    final_sample, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace  = \
    smc_FK(M=M, G=G, 
            resample_strategy=resample_strategy, 
            ess_threshold=ess_threshold, 
            T=diffusion_timesteps, 
            P=particle_number, 
            verbose=True, 
            log_xt_trace=False, 
            extra_vals={"model_kwargs": model_kwargs,
                        "batch_p" : batch_particles})
    
    return final_sample

def sample_multiple_single_conditional_diffusion_distribution(diffusion, mask, resample_strategy,
                                                     ess_threshold, diffusion_timesteps,
                                                     particle_number,
                                                     model_kwargs, batch_particles, ref_image, 
                                                     pred_xstart_var_type, device,
                                                     number_of_replicates):
    
    conditional_samples = th.zeros((number_of_replicates,n,n))
    
    for i in range(0, number_of_replicates):
        final_sample = sample_single_conditional_diffusion_distribution(diffusion, mask, resample_strategy,
                                                     ess_threshold, diffusion_timesteps,
                                                     particle_number,
                                                     model_kwargs, batch_particles, ref_image, 
                                                     pred_xstart_var_type, device)
        conditional_samples[i,:,:] = (final_sample[0,:,:,:]).reshape((n,n))

    
    return conditional_samples

def save_single_conditional_diffusion_samples(diffusion, mask, resample_strategy,
                                                 ess_threshold, diffusion_timesteps,
                                                 particle_number,
                                                 model_kwargs, batch_particles, ref_image, 
                                                 pred_xstart_var_type, device,
                                                 number_of_replicates, fixed_location):
    
    local_folder = "visualizations/evaluation/single_location_histograms/diffusion/ref_image1"
    np.save(local_folder + "/ref_image.npy", (ref_image.cpu().numpy()))
    conditional_samples = sample_multiple_single_conditional_diffusion_distribution(diffusion, mask, resample_strategy,
                                                     ess_threshold, diffusion_timesteps,
                                                     particle_number,
                                                     model_kwargs, batch_particles, ref_image, 
                                                     pred_xstart_var_type, device,
                                                     number_of_replicates)
    
    index_tuple, index_number = get_index(fixed_location, minX, maxX, minY, maxY, n)
    np.save((local_folder + "/conditional_samples_location_" 
             + str(index_tuple[0]) + "_" + 
             str((index_tuple[1])) +
             "_replicates_100_particles_4_ess_0_var_type_1.npy"),
             conditional_samples.cpu().numpy())
    
def save_multiple_conditional_diffusion_samples(diffusion, mask, resample_strategy,
                                                 ess_threshold, diffusion_timesteps,
                                                 particle_number,
                                                 model_kwargs, batch_particles, ref_image, 
                                                 pred_xstart_var_type, device,
                                                 number_of_replicates):
    
    local_folder = "visualizations/evaluation/multi_location_histograms/ref_image1/diffusion"
    np.save(local_folder + "/ref_image.npy", (ref_image.cpu().numpy()))
    conditional_samples = sample_multiple_single_conditional_diffusion_distribution(diffusion, mask, resample_strategy,
                                                     ess_threshold, diffusion_timesteps,
                                                     particle_number,
                                                     model_kwargs, batch_particles, ref_image, 
                                                     pred_xstart_var_type, device,
                                                     number_of_replicates)
    
    np.save((local_folder + "/conditional_samples_replicates_" + str(number_of_replicates) + 
             "_particles_4_ess_0_var_type_1.npy"),
             conditional_samples.cpu().numpy())
    
    
def plot_fixed_location_histogram(observations, fixed_index, figname, n):

    fig, ax = plt.subplots(1)
    ax.hist(observations, bins = 15)
    ax.set_title((str(int(fixed_index%n)) + " " 
                  + str(int(fixed_index/n))))
    plt.savefig(figname)

minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
m = 10
x = np.linspace(minX, maxX, n)
y = np.linspace(minY, maxY, n)
X, Y = np.meshgrid(x, y)
longitudes = X.reshape((np.prod(X.shape),1))
latitudes = Y.reshape((np.prod(Y.shape),1))
indices = np.random.randint(low = 0, high = n**2, size = m)
variance = .4
lengthscale = 1.6
observed_vector, observed_matrix = generate_gaussian_process(minX, maxX, minY, maxY, n, variance,
                                       lengthscale, 1, 342342)


device = "cuda:0"
diffusion = TwistedDDPM(betas = betas, particle_base_shape = (1,32,32),
                        rescale_timesteps = False, conf=None,
                        probability_flow = False, device = device,
                        use_timesteps = [i for i in range(0,250)],
                        original_num_steps = 250)
resample_strategy = "systematic"
ess_threshold = 0
diffusion_timesteps = 250
particle_number = 4
model_kwargs = {}
batch_particles = 4
ref_image = ((th.from_numpy(observed_matrix)).to(device)).reshape((1,n,n))
pred_xstart_var_type = 1

minX = -10
maxX = 10
minY = -10
maxY = 10
n = 32
m = 10
indices_tuple = [(7,7), (7,15), (15,7), (15,15), (15,23), (23,7), (23,15), (23,23)]
indices = [(7*n+7), (7*n+15), (16*n+7), (16*n+16), (16*n+24), (24*n+7), (24*n+16),
           (24*n+24)]
m = 8
fixed_locations = np.zeros((m,2))
fixed_locations[:,0] = longitudes[indices].reshape((m,))
fixed_locations[:,1] = latitudes[indices].reshape((m,))
number_of_replicates = 10

mask = th.ones((1,n,n))
mask[:,7:24,7:24] = 0
mask = mask.to(th.bool)

"""
save_multiple_conditional_diffusion_samples(diffusion, mask, resample_strategy,
                                            ess_threshold, diffusion_timesteps,
                                            particle_number,
                                            model_kwargs, batch_particles, ref_image, 
                                            pred_xstart_var_type, device,
                                            number_of_replicates)
"""
















