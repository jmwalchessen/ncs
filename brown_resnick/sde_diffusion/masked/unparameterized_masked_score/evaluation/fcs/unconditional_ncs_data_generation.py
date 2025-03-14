import torch as th
import numpy as np
from append_directories import *
from functools import partial
import matplotlib.pyplot as plt


from helper_functions import *
score_model = load_score_model("brown", "model11/model11_wo_l2_beta_min_max_01_20_obs_num_7_smooth_1.5_range_5_channel_mask.pth", "eval")
vpsde = load_sde(beta_min = .1, beta_max = 20, N = 1000)

def load_npfile(npfile):
    nparray = np.load(npfile)
    return nparray

def generate_ncs_images(ref_folder, vpsde, score_model, range_value,
                        smooth_value, ncs_images_file, batches_per_call, calls, n, device):

    mask = load_npfile((ref_folder + "/mask.npy"))
    ref_image = np.log(load_npfile((ref_folder + "/ref_image.npy")))
    masked_image = np.multiply(mask, ref_image)
    device = "cuda:0"
    ncs_images = np.zeros(((batches_per_call*calls),n,n))

    mask = (th.from_numpy((mask.reshape((1,1,n,n))))).float().to(device)
    masked_image = (th.from_numpy((masked_image.reshape((1,1,n,n))))).float().to(device)

    for i in range(calls):
        ncs_images[i*batches_per_call:(i+1)*batches_per_call,:,:] = ((posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                                                                                     masked_image, n, batches_per_call)).detach().cpu().numpy()).reshape((batches_per_call,n,n))
        
    
    np.save(ncs_images_file, ncs_images)


def generate_ncs_images_multiple_files(vpsde, score_model, range_values,
                                        smooth_value, batches_per_call, calls, n, device, ref_numbers):
    
    nrep = (batches_per_call * calls)
    for ref_number in range(ref_numbers):

        ref_folder = ("data/model4/ref_image" + str(ref_number))
        ncs_images_file = (ref_folder + "/ncs_images_range_" + str(range_value) +
                           "_smooth_" + str(smooth_value) + "_" + str(nrep) + ".npy")
        generate_ncs_images(ref_folder, vpsde, score_model, range_value,
                        smooth_value, ncs_images_file, batches_per_call, calls, n, device)

def return_observed_values(masks, brimages, nrep, n):

    observed_images = th.zeros((nrep,1,n,n))
    for irep in range(nrep):
        observed_images[irep,:,:,:] = (th.mul(masks[irep,:,:,:], brimages[irep,:,:,:]))

    return observed_images

def generate_unconditional_fixed_ncs_images(vpsde, score_model, n, range_value, smooth_value, obs,
                                            number_of_replicates):

    device = "cuda:0"
    seed_value = int(np.random.randint(0,10000))
    brimages = generate_brown_resnick_process(range_value, smooth_value, seed_value,
                                               number_of_replicates, n)
    mask = np.load(("data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image"
                    + str(int(range_value-1)) + "/mask.npy"))
    masks = np.tile(mask.reshape((1,1,n,n)), (number_of_replicates,1,1,1))
    masks = (th.from_numpy(masks)).float()
    brimages = (th.from_numpy(np.log(brimages))).float()
    ys = return_observed_values(masks, brimages, number_of_replicates, n)
    ys = ys.float().to(device)
    masks = masks.to(device)
    brimages = brimages.to(device)
    ncs_images = multiple_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device,
                                                                         masks, ys, n, range_value,
                                                                         smooth_value)
    return ncs_images


def generate_unconditional_fixed_ncs_images_multi(vpsde, score_model, n, range_value, smooth_value,
                                                  number_of_replicates, irep):

    obsn = [i for i in range(7,8)]
    for obs in obsn:
        ncs_file = ("data/unconditional/fixed_locations/obs" + str(obs) +
        "/ref_image" + str(int(range_value-1)) + "/diffusion/unconditional_fixed_model11_ncs_images_range_" +
        str(range_value) + "_smooth_1.5_" + str(number_of_replicates) + str(irep) + ".npy")
        ncs_images = generate_unconditional_fixed_ncs_images(vpsde, score_model, n, range_value, smooth_value, obs,
                                                             number_of_replicates)
        ncs_images = ncs_images.detach().cpu().numpy()
        np.save(ncs_file,ncs_images)


def generate_unconditional_fixed_ncs_images_multi_with_variables():

    range_values = [float(i) for i in range(5,6)]
    beta_min = .1
    beta_max = 20
    N = 1000
    smooth_value = 1.5
    number_of_replicates_per_call = 1000
    vpsde = load_sde(beta_min, beta_max, N)
    calls = 4
    n = 32
    for range_value in range_values:
        for irep in range(calls):
            generate_unconditional_fixed_ncs_images_multi(vpsde, score_model, n, range_value, smooth_value,
                                                      number_of_replicates_per_call, irep)


def split_extreme_unconditional_images():

    range_values = [float(i) for i in range(1,6)]
    obsn = [i for i in range(1,8)]
    n = 32
    nrep = 4000
    for range_value in range_values:
        for obs in obsn:
            ref_folder = "data/unconditional/fixed_locations/obs" + str(obs) + "/ref_image" + str(int(range_value-1))
            ncs_images = np.load((ref_folder + "/diffusion/unconditional_fixed_ncs_images_range_" + str(range_value) + "_smooth_1.5_4000.npy"))
            true_images = (np.log(np.load((ref_folder + "/true_brown_resnick_images_range_" + str(int(range_value)) + "_smooth_1.5_4000.npy")))).reshape((nrep,n,n))
            mask = np.load((ref_folder + "/mask.npy"))
            extreme_ncs_images = np.zeros((0,n,n))
            nonextreme_ncs_images = np.zeros((0,n,n))
            extreme_true_images = np.zeros((0,n,n))
            nonextreme_true_images = np.zeros((0,n,n))
            for irep in range(nrep):
                observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
                print(observed_indices)
                values = true_images[irep,observed_indices[:,0],observed_indices[:,1]]
                print(values)
                ncs_values = ncs_images[irep,observed_indices[:,0],observed_indices[:,1]]
                print(ncs_values)
                if(np.any(ncs_values) > 4.):
                    extreme_ncs_images = np.concatenate([extreme_ncs_images,ncs_images[irep:(irep+1),:,:]],axis = 0)
                    extreme_true_images = np.concatenate([extreme_true_images,true_images[irep:(irep+1),:,:]],axis = 0)
                else:
                    nonextreme_ncs_images = np.concatenate([extreme_ncs_images,ncs_images[irep:(irep+1),:,:]],axis = 0)
                    nonextreme_true_images = np.concatenate([extreme_true_images,true_images[irep:(irep+1),:,:]],axis = 0)

            print(extreme_ncs_images.shape)
            np.save((ref_folder + "/diffusion/extreme_unconditional_fixed_ncs_images_range_" + str(range_value) + "_smooth_1.5_4000.npy"), extreme_ncs_images)
            np.save((ref_folder + "/diffusion/nonextreme_unconditional_fixed_ncs_images_range_" + str(range_value) + "_smooth_1.5_4000.npy"), nonextreme_ncs_images)
            np.save((ref_folder + "/extreme_true_brown_resnick_images_range_" + str(int(range_value)) + "_smooth_1.5_4000.npy"), extreme_true_images)
            np.save((ref_folder + "/nonextreme_true_brown_resnick_images_range_" + str(int(range_value)) + "_smooth_1.5_4000.npy"), nonextreme_true_images)
            break


generate_unconditional_fixed_ncs_images_multi_with_variables()