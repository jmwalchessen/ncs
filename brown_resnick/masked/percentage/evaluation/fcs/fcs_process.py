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
    missing_fcs = np.log(np.load((ref_folder + "/" + fcs_file)))
    fcs = (np.tile(observed_image, reps = nrep)).reshape((nrep,n**2))
    fcs[:,mask.flatten() == 0] = missing_fcs
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
            
def process_unconditional_fixed_location_fcs_file(ref_folder, mask_file, fcs_file, obs_file, processed_fcs_file, nrep, n, m):

    missing_fcs = (np.load((ref_folder + "/" + fcs_file))).reshape((nrep,(n**2-m)))
    true_images = (np.load((ref_folder + "/" + obs_file))).reshape((nrep,n**2))
    fcs_mask = np.load((ref_folder + "/" + mask_file)).reshape((n**2))
    fcs_images = np.zeros((nrep,(n**2)))
    fcs_images[:,fcs_mask == 0] = missing_fcs
    fcs_images[:,fcs_mask == 1] = true_images[:,fcs_mask == 1]
    fcs_images = fcs_images.reshape((nrep,n,n))
    np.save((ref_folder + "/" + processed_fcs_file), fcs_images)

            
def process_unconditional_fixed_location_fcs_file_with_variables():

    evaluation_folder = append_directory(2)
    ref_folder = (evaluation_folder + "/fcs/data/unconditional/fixed_locations")
    ms = [i for i in range(1,8)]
    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    for m in ms:
        for range_value in range_values:

            fcs_file = "unconditional_fcs_fixed_mask_obs" + str(m) + "_range_" + str(int(range_value)) + "_smooth_1.5_nugget_1e5_" + str(nrep) + ".npy"
            processed_fcs_file = "processed_unconditional_fcs_fixed_mask_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            mask_file = "mask.npy"
            obs_file = "true_brown_resnick_images_range_" + str(int(range_value)) + "_smooth_1.5_4000.npy"
            current_ref_folder = (ref_folder + "/obs" + str(m) + "/ref_image" + str(int(range_value-1)))
            process_unconditional_fixed_location_fcs_file(current_ref_folder, mask_file, fcs_file, obs_file,
                                        processed_fcs_file, nrep, n, m)
            
def collect_unconditional_ncs_file_with_variables():

    evaluation_folder = append_directory(2)
    ref_folder = (evaluation_folder + "/fcs/data/unconditional/fixed_locations")
    ms = [i for i in range(7,8)]
    n = 32
    nrep = 4000
    nrep_per_file = 1000
    nofile = 4
    range_values = [5.]
    model_name = "model11"
    for m in ms:
        for range_value in range_values:
            ncs_images = np.zeros((0,n,n))
            for i in range(nofile):
                ncs_file = "unconditional_fixed_" + str(model_name) + "_ncs_images_range_" + str(range_value) + "_smooth_1.5_" + str(nrep_per_file) + str(i) + ".npy"
                current_ref_folder = (ref_folder + "/obs" + str(m) + "/ref_image" + str(int(range_value-1)) +"/diffusion")
                current_images = np.load((current_ref_folder + "/" + ncs_file))
                ncs_images = np.concatenate([ncs_images, current_images.reshape((nrep_per_file,n,n))], axis = 0)
            ncs_file = "unconditional_fixed_ncs_images_range_" + str(range_value) + "_smooth_1.5_" + str(nrep) + ".npy"
            np.save((current_ref_folder + "/" + ncs_file), ncs_images)

    return ncs_images

def process_conditional_fcs():          
    ms = [i for i in range(1,8)]
    range_values = [float(i) for i in range(1,6)]
    nrep = 4000
    for m in ms:
        for range_value in range_values:
            ref_folder = "data/model4/obs" + str(m) + "/ref_image" + str(int(range_value-1))
            fcs_file = "fcs_range_" + str(int(range_value)) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            processed_fcs_file = "processed_log_scale_fcs_range_" + str(range_value) + "_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
            n = 32
            mask_file = "mask.npy"
            process_fcs_file(ref_folder, mask_file, fcs_file, processed_fcs_file, nrep, n)

collect_unconditional_ncs_file_with_variables()