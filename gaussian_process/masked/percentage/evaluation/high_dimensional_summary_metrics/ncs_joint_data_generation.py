import torch as th
import numpy as np
from append_directories import *
from functools import partial
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *
score_model = load_score_model("model7_variance_1.5_lengthscale_.5_5.5_beta_min_max_01_20_random05_channel_mask.pth", "eval")
sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)

def load_npfile(npfile):
    nparray = np.load(npfile)
    return nparray

def generate_masks(n, nreps, p):

    masks = (th.bernoulli(p*th.ones((nreps,1,n,n)))).numpy()
    return masks

def generate_gaussian_process_images_and_masks(true_images_file, mask_file, lengthscale, variance, nreps, n, p):

    minX = minY = -10
    maxX = maxY = 10
    seed_value = int(np.random.randint(0, 10000000))
    gp_vec, gp_matrix = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                  nreps, seed_value)
    masks = generate_masks(n, nreps, p)
    np.save(mask_file, masks)
    np.save(true_images_file, gp_vec)

def generate_joint_ncs_images(true_images_file, mask_file, vpsde, score_model, lengthscale, variance,
                              ncs_images_file, batches_per_call, calls, n):

    masks = load_npfile(mask_file)
    true_images = load_npfile(true_images_file)
    masked_true_images = np.multiply(masks, true_images)
    nimages = masked_true_images.shape[0]
    device = "cuda:0"
    num_samples = 1
    n = 32
    ncs_images = np.zeros((nimages,n,n))

    for i in range(calls):
        print(i)
        current_masked_true_images = ((th.from_numpy(masked_true_images[i*batches_per_call:(i+1)*batches_per_call,:,:])).float().to(device)).reshape((batches_per_call,1,n,n))
        current_masks = ((th.from_numpy(masks[i*batches_per_call:(i+1)*batches_per_call,:,:])).float().to(device)).reshape((batches_per_call,1,n,n))
        ncs_images[i*batches_per_call:(i+1)*batches_per_call,:,:] = ((multiple_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, current_masks,
                                                                            current_masked_true_images, n, lengthscale, variance)).detach().cpu().numpy()).reshape((batches_per_call,n,n))
        
    
    np.save(ncs_images_file, ncs_images)


def generate_joint_ncs_images_multiple_percentages(masked_true_images_file, mask_file, vpsde, score_model,
                                                   lengthscale, variance, ncs_images_file, nrep, batches_per_call,
                                                   calls, n, ps):
    
    for p in ps:
        print(p)
        current_masked_true_images_file = (masked_true_images_file + "_lengthscale_" + str(lengthscale) + "_variance_" + str(variance)
                                           + "_" + str(nrep) + "_random" + str(p) + ".npy")
        current_mask_file = (mask_file + "_lengthscale_" + str(lengthscale) + "_variance_" + str(variance)
                                           + "_" + str(nrep) + "_random" + str(p) + ".npy")
        current_ncs_images_file = (ncs_images_file + "_lengthscale_" + str(lengthscale) + "_variance_" + str(variance)
                                           + "_" + str(nrep) + "_random" + str(p) + ".npy")
        generate_joint_ncs_images(current_masked_true_images_file, current_mask_file, vpsde, score_model, lengthscale, variance,
                                  current_ncs_images_file, batches_per_call, calls, n)



masked_true_images_file = "data/ncs/model7/true_masked_gp_"
mask_file = "data/ncs/model7/true_masks"
ps = [.01,.05,.1,.25,.5]
variance = 1.5
ncs_images_file = "data/ncs/model7/gp_ncs_images"
nrep = 4000
calls = 8
batches_per_call = 500
n = 32
lengthscale = 3.0
generate_joint_ncs_images_multiple_percentages(masked_true_images_file, mask_file, sdevp, score_model, ps,
                                               variance, lengthscale, ncs_images_file, nrep, batches_per_call, calls, n)