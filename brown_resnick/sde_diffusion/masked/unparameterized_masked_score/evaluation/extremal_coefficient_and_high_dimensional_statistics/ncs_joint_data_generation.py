import torch as th
import numpy as np
from append_directories import *
from functools import partial
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *
score_model = load_score_model("brown", "model4/model4_beta_min_max_01_20_range_.5_5.5_smooth_1.5_random05_log_parameterized_mask.pth", "eval")
sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)

def load_npfile(npfile):
    nparray = np.load(npfile)
    return nparray

def generate_joint_ncs_images(masked_true_images_file, mask_file, vpsde, score_model, range_value, smooth_value,
                             ncs_images_file, batches_per_call, calls, n):

    masks = load_npfile(mask_file)
    masked_true_images = load_npfile(masked_true_images_file)
    masked_true_images[masked_true_images != 0] = np.log(masked_true_images[masked_true_images != 0])
    nimages = masked_true_images.shape[0]
    device = "cuda:0"
    num_samples = 1
    n = 32
    ncs_images = np.zeros((nimages,n,n))

    for i in range(calls):
        print(i)
        current_masked_true_images = ((th.from_numpy(masked_true_images[i*batches_per_call:(i+1)*batches_per_call,:,:])).float().to(device)).reshape((batches_per_call,1,n,n))
        print(current_masked_true_images.shape)
        current_masks = ((th.from_numpy(masks[i*batches_per_call:(i+1)*batches_per_call,:,:])).float().to(device)).reshape((batches_per_call,1,n,n))
        print(current_masks.shape)
        ncs_images[i*batches_per_call:(i+1)*batches_per_call,:,:] = ((multiple_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, current_masks,
                                                                            current_masked_true_images, n, range_value, smooth_value)).detach().cpu().numpy()).reshape((batches_per_call,n,n))
        
    
    np.save(ncs_images_file, ncs_images)


def generate_joint_ncs_images_multiple_percentages(masked_true_images_file, mask_file, vpsde, score_model, range_value, smooth_value,
                                                   ncs_images_file, batches_per_call, calls, n, ps):
    
    for p in ps:

        current_masked_true_images_file = (masked_true_images_file + "_random" + str(p) + ".npy")
        current_ncs_images_file = (ncs_images_file + "_random" + str(p) + ".npy")
        generate_joint_ncs_images(current_masked_true_images_file, mask_file, vpsde, score_model, range_value, smooth_value,
                                  current_ncs_images_file, batches_per_call, calls, n)


masked_true_images_file = "data/ncs/model4/true_masked_brown_resnick_range_3.0_smooth_1.5_4000"
mask_file = "data/ncs/model4/true_masks_range_3.0_smooth_1.5_4000"
range_value = 3.0
smooth_value = 1.5
ncs_images_file = "data/ncs/model4/brown_resnick_ncs_images_range_3.0_smooth_1.5_4000"
nrep = 4000
calls = 8
batches_per_call = 500
n = 32
ps = [.01,.05,.1,.25,.5]
generate_joint_ncs_images_multiple_percentages(masked_true_images_file, mask_file, sdevp, score_model, range_value, smooth_value,
                                               ncs_images_file, nrep, batches_per_call, calls, n, ps)