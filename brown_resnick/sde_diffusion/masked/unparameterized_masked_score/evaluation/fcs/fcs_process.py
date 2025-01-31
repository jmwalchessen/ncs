import numpy as np
from append_directories import *


def load_ref_image(ref_folder):

    return np.load((ref_folder + "/ref_image.npy"))

def load_mask(ref_folder):

    return np.load((ref_folder + "/mask.npy"))

def process_fcs_file(ref_folder, mask_file, fcs_file, processed_fcs_file, nrep, n):

    mask = np.load((ref_folder + "/" + mask_file))
    ref_image = np.log(load_ref_image(ref_folder))
    observed_image = ((mask)*ref_image).reshape((1,n**2))
    print(observed_image)
    missing_fcs = np.log(np.load((ref_folder + "/" + fcs_file)))
    fcs = (np.tile(observed_image, reps = nrep)).reshape((nrep,n**2))
    fcs[:,mask.flatten() == 0] = missing_fcs
    print(fcs[0,mask.flatten() == 1])
    fcs = fcs.reshape((nrep,n,n))
    np.save((ref_folder + "/" + processed_fcs_file), fcs)

def process_unconditional_fcs_file(ref_folder, mask_file, obs_file, fcs_file,
                                   processesed_fcs_file, nrep, n, m):
    
    true_full_images = (np.load((ref_folder + "/" + obs_file))).reshape((nrep*(n**2)))
    missing_fcs = (np.load((ref_folder + "/" + fcs_file))).reshape((nrep*(n**2-m)))
    fcs_masks = np.load((ref_folder + "/" + mask_file)).reshape((nrep*(n**2)))
    fcs_images = np.zeros((nrep*(n**2)))
    fcs_images[fcs_masks == 0] = missing_fcs
    fcs_images[fcs_masks == 1] = true_full_images[fcs_masks == 1]
    fcs_images = fcs_images.reshape((nrep,n,n))
    np.save((ref_folder + "/" + processesed_fcs_file), fcs_images)

    
def process_unconditional_fcs_file_with_variables():
    
    evaluation_folder = append_directory(2)
    ref_folder = (evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs")
    ms = [i for i in range(1,8)]
    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    for m in ms:
        for range_value in range_values:
            fcs_file = "unconditional_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            processed_fcs_file = "processed_unconditional_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            mask_file = "unconditional_mask_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            obs_file = "unconditional_obs_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            process_unconditional_fcs_file(ref_folder, mask_file, obs_file, fcs_file,
                                        processed_fcs_file, nrep, n, m)


def process_conditional_fcs():          
    m = 1
    range_value = 5
    nrep = 4000
    ref_folder = "data/model4/obs" + str(m) + "/ref_image" + str(range_value-1)
    fcs_file = "fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
    processed_fcs_file = "processed_log_scale_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
    n = 32
    mask_file = "mask.npy"
    process_fcs_file(ref_folder, mask_file, fcs_file, processed_fcs_file, nrep, n)

process_conditional_fcs()