import numpy as np
from append_directories import *
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import subprocess
home_folder = append_directory(7)
sde_diffusion_folder = (home_folder + "/sde_diffusion/masked/unparameterized")
sys.path.append(sde_diffusion_folder)
from sde_lib import *

def log_transformation(images):

    images = np.log(np.where(images !=0, images, np.min(images[images != 0])))

    return images

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    return images


class CustomSpatialImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return ((self.images).shape[0])

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        return image

def generate_vpsde_forward_process_noise(total_number_of_samples, num_timesteps, betamin, betamax,
                                         range_value, smooth_value, seed_value, n):

    brimages = generate_brown_resnick_process(range_value, smooth_value,
                                              seed_value, total_number_of_samples, n)
    vpsde = VPSDE(beta_min = betamin, beta_max = betamax, N = total_number_of_samples)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
    #this is sigma_t i.e. std
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
    noise = torch.randn_like(brimages)
    perturbed_data = sqrt_alphas_cumprod[(num_timesteps-1), None, None, None] *  brimages +\
                     sqrt_1m_alphas_cumprod[(num_timesteps-1), None, None, None] * noise
    #generate brown resnick samples
    return perturbed_data

def generate_white_noise(n, total_number_of_samples):

    white_noise = np.random.normal(loc = 0, scale = 1, size = (total_number_of_samples, n, n))
    return white_noise

def generate_white_noise_and_forward_white_noise_data(total_number_of_samples, num_timesteps, betamin, betamax,
                                         range_value, smooth_value, seed_value, n):
    
    forward_white_noise = generate_vpsde_forward_process_noise(total_number_of_samples, num_timesteps,
                                                               betamin, betamax, range_value, smooth_value,
                                                               seed_value, n)
    white_noise = generate_white_noise(n, total_number_of_samples)
    return np.concatenate([white_noise, forward_white_noise], axis = 0)

def get_training_and_evaluation_image_datasets(total_number_of_samples, eval_total_number_of_samples,
                                               num_timesteps, betamin, betamax, seed_value, range_value,
                                               smooth_value, batch_size, eval_batch_size):
    
    n = (32**2)
    train_images = generate_white_noise_and_forward_white_noise_data(total_number_of_samples,
                                                                    num_timesteps, betamin, betamax,
                                                                    range_value, smooth_value, seed_value, n)
    train_dataset = CustomSpatialImageDataset(train_images)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    eval_images = generate_white_noise_and_forward_white_noise_data(eval_total_number_of_samples,
                                                                    num_timesteps, betamin, betamax,
                                                                    range_value, smooth_value, seed_value, n)
    eval_dataset = CustomSpatialImageDataset(eval_images)
    eval_dataloader = DataLoader(eval_dataset, batch_size = eval_batch_size, shuffle = True)
    return train_dataloader, eval_dataloader

def get_next_batch(image_iterator, config):

    images = (next(image_iterator))
    images = images.to(config.device).float()
    return images

    