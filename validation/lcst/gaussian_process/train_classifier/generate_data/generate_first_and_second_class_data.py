import numpy as np
import torch
from append_directories import *
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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
def generate_first_class_data(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_replicates,
                             seed_value):
    

    train_images = generate_gaussian_process(minX, maxX, minY, maxY, n,
                                             variance, lengthscale,
                                             number_of_replicates, seed_value)
    return train_images

def load_score_function(model_name, device, num_scales, beta_min, beta_max):
    home_folder = append_directory(6)
    sde_folder = home_folder + "/sde_diffusion/masked/unparameterized"
    sde_configs_vp_folder = sde_folder + "/configs/vp"
    sys.path.append(sde_configs_vp_folder)
    import ncsnpp_config
    sys.path.append(sde_folder)
    from models import ncsnpp

    config = ncsnpp_config.get_config()
    config.model.num_scales = num_scales
    config.model.beta_max = beta_max
    config.model.beta_min = beta_min

    score_model = torch.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
    score_model.load_state_dict(torch.load((sde_folder + "/trained_score_models/vpsde/" + model_name)))
    score_model.eval()
    return score_model

def create_vpsde(beta_min, beta_max, N):
    home_folder = append_directory(5)
    sde_folder = home_folder + "/sde_diffusion/masked/unparameterized"
    sys.path.append(sde_folder)
    import sde_lib

    sdevp = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
    return sdevp

#y is observed part of field
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t):

    num_samples = masked_xt.shape[0]
    timestep = ((torch.tensor([t])).repeat(num_samples)).to(device)
    with torch.no_grad():
        score = score_model(masked_xt, timestep)
    unmasked_p_mean = (1/torch.sqrt(torch.tensor(vpsde.alphas[t])))*(masked_xt + torch.square(torch.tensor(vpsde.sigmas[t]))*score)
    masked_p_mean = torch.mul((1-mask), unmasked_p_mean) + torch.mul(mask, y)
    unmasked_p_variance = (torch.square(torch.tensor(vpsde.sigmas[t])))*torch.ones_like(masked_xt)
    masked_p_variance = torch.mul((1-mask), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, num_samples):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t)
    std = torch.exp(0.5 * torch.log(p_variance))
    noise = torch.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = torch.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample



def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples):

    unmasked_xT = torch.randn((num_samples, 1, n, n)).to(device)
    masked_xT = torch.mul((1-mask), unmasked_xT) + torch.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples)

    return masked_xt

def sample_conditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
                                          num_samples_per_call, calls):
    
    diffusion_samples = torch.zeros((0, 1, n, n))
    for call in range(0, calls):
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, mask, y, n,
                                                                                   num_samples_per_call)
        diffusion_samples = torch.cat([current_diffusion_samples.detach().cpu(),
                                    diffusion_samples],
                                    dim = 0)
    return diffusion_samples

def generate_masks(n, p, num_masks, seed_value):

    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed_value)
    masks = torch.bernoulli(p*torch.ones((num_masks, 1, n, n)), generator=g_cpu)
    return masks

#seed_values should be length of calls
def generate_second_class_from_first_class(first_class_images, score_model, sdevp, device, number_of_samples_per_call, calls, n, p, seed_values):

    diffusion_samples = torch.zeros((0, 1, n, n))
    masks = torch.zeros((0, 1, n, n))
    for i in range(calls):
        masks = torch.cat([masks, (generate_masks(n, p, number_of_samples_per_call, seed_values[i]).to(device)).detach().cpu()], dim = 0)
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model, device, masks,
                                                                                   first_class_images[calls*number_of_samples_per_call:(calls+1)*number_of_samples_per_call,:,:,:],
                                                                                   n, number_of_samples_per_call)
        diffusion_samples = torch.cat([current_diffusion_samples.detach().cpu(),diffusion_samples])
    return masks, diffusion_samples

def plot_spatial_field(image, min_value, max_value, figname):

    fig, ax = plt.subplots()
    plt.imshow(image, vmin = min_value, vmax = max_value)
    plt.savefig(figname)

def plot_masked_spatial_field(image, mask, min_value, max_value, figname):

    fig, ax = plt.subplots()
    plt.imshow(image, vmin = min_value, vmax = max_value, alpha = mask)
    plt.savefig(figname)

def generate_two_classes(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_replicates,
                         number_of_replicates_per_call, calls, first_class_seed_value,
                         second_class_seed_values, num_scales, beta_min, beta_max, model_name, p, device):

    first_class_images = (generate_first_class_data(minX, maxX, minY, maxY, n, variance,
                                               lengthscale, number_of_replicates, first_class_seed_value))
    score_model = load_score_function(model_name, device, num_scales, beta_min, beta_max)
    sdevp = create_vpsde(beta_min, beta_max, num_scales)
    first_class_images = (torch.from_numpy(first_class_images).to(device)).float()
    masks, second_class_images = generate_second_class_from_first_class(first_class_images, score_model,
                                                           sdevp, device, number_of_replicates_per_call,
                                                           calls, n, p, second_class_seed_values)
    return first_class_images, second_class_images, masks

class CustomSpatialImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        return image
    
def get_training_and_evaluation_datasets(variance, lengthscale, number_of_train_replicates, number_of_train_replicates_per_call, train_calls,
                                         number_of_eval_replicates, number_of_eval_replicates_per_call, eval_calls,
                                         train_first_class_seed_value, eval_first_class_seed_value, train_second_class_seed_values,
                                         eval_second_class_seed_values, num_scales, beta_min, beta_max, 
                                         model_name, p, device, batch_size, eval_batch_size):
    
    minX = -10
    maxX = 10
    minY = -10
    maxX = 10
    maxY = 10
    n = 32

    train_images = generate_two_classes(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_train_replicates,
                                        number_of_train_replicates_per_call, train_calls, train_first_class_seed_value,
                                        train_second_class_seed_values, num_scales, beta_min, beta_max, model_name, p, device)
    eval_images = generate_two_classes(minX, maxX, minY, maxY, n, variance, lengthscale, number_of_eval_replicates,
                                       number_of_eval_replicates_per_call, eval_calls, eval_first_class_seed_value,
                                        eval_second_class_seed_values, num_scales, beta_min, beta_max, model_name, p, device)
    train_dataset = CustomSpatialImageDataset(train_images)
    eval_dataset = CustomSpatialImageDataset(eval_images)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

variance = .4
lengthscale = 1.6
number_of_train_replicates = 10
number_of_train_replicates = 5
train_calls = 2
number_of_eval_replicates = 5
number_of_eval_replicates_per_call = 5
eval_calls = 1


get_training_and_evaluation_datasets(variance, lengthscale, number_of_train_replicates, number_of_train_replicates_per_call, train_calls,
                                         number_of_eval_replicates, number_of_eval_replicates_per_call, eval_calls,
                                         train_first_class_seed_value, eval_first_class_seed_value, train_second_class_seed_values,
                                         eval_second_class_seed_values, num_scales, beta_min, beta_max, 
                                         model_name, p, device, batch_size, eval_batch_size):
    







    


    