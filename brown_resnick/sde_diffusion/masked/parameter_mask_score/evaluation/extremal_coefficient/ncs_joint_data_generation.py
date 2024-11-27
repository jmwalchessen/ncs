import torch as th
import numpy as np
from append_directories import *
from functools import partial
from brown_resnick_data_generation import *
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *

score_model = load_score_model("brown", "model4/model4_beta_min_max_01_20_range_.5_5.5_smooth_1.5_random05_log_parameterized_mask.pth", "eval")

sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)

def load_npfile(npfile):
    nparray = np.load(npfile)
    return nparray

def geneate_joint_ncs_images(masked_true_images_file, mask_file, vpsde, score_model, range_value, smooth_value, ncs_images_file):

    masks = load_npfile(mask_file)
    masked_true_images = load_npfile(masked_true_images_file)
    nimages = masked_true_images.shape[0]
    device = "cuda:0"
    num_samples = 1
    n = 32
    ncs_images = np.zeros((nimages,n,n))

    for i in range(nimages):

        current_masked_true_image = masked_true_images[i,:,:]
        current_mask = masks[i,:,:]
        ncs_images[i,:,:] = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, current_mask,
                                                       current_masked_true_image, n, num_samples,
                                                       range_value, smooth_value)
        
    
    np.load(ncs_images_file)


def generate_joint_ncs_images_multiple_ranges(masked_true_images_file, mask_file, vpsde, score_model,
                                              range_values, smooth_value, ncs_images_file, nrep):
    
    for range_value in range_values:
        current_masked_true_images_file = (masked_true_images_file + "_range_" + str(range_value) + "_smooth_" + str(smooth_value)
                                           + "_" + str(nrep) + ".npy")
        current_mask_file = (mask_file + "_range_" + str(range_value) + "_smooth_" + str(smooth_value)
                                           + "_" + str(nrep) + ".npy")
        current_ncs_images_file = (ncs_images_file + "_range_" + str(range_value) + "_smooth_" + str(smooth_value)
                                           + "_" + str(nrep) + ".npy")
        geneate_joint_ncs_images(current_masked_true_images_file, current_mask_file, vpsde, score_model, range_value, smooth_value,
                                 current_ncs_images_file)



masked_true_images_file = "data/ncs/model4/true_masked_brown_resnick"
mask_file = "data/ncs/model4/true_masks"
range_values = [1.0,2.0,3.0,4.0,5.0]
smooth_value = 1.5
ncs_images_file = "data/ncs/model4/brown_resnick_ncs_images"
generate_joint_ncs_images_multiple_ranges(masked_true_images_file, mask_file, sdevp, score_model, range_values, smooth_value,
                                          ncs_images_file)