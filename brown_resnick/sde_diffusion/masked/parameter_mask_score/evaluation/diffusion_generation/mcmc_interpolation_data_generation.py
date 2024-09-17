import numpy as np
import torch as th
import subprocess
import os

range_value = 1.6
smooth_value = 1.6
nugget = 0
ref_file_name = "data/model2/ref_image1/ref_image.npy"
mask_file_name = "data/model2/ref_image1/mask.npy"
condsim_file_name = "data/model2/ref_image1/mcmc_interpolation/mcmc_interpolation_missing_index"
cov_mod = "brown"
neighbors = 7
n = 32
nrep = 4000
missing_index_start = 1
missing_index_end = 1024

def produce_mcmc_interpolation_multiple_pixels(n, range_value, smooth_value, nugget, cov_mod,
                                               ref_file_name, mask_file_name, condsim_file_name,
                                               neighbors, nrep, missing_index_start):

    subprocess.run(["Rscript", "conditional_simulation_multiple_pixels.R", str(range_value),
                    str(smooth_value), str(nugget), str(ref_file_name), str(mask_file_name),
                    str(condsim_file_name), str(cov_mod), str(neighbors), str(n), str(nrep),
                    str(missing_index_start), str(missing_index_end)],
                    check = True, capture_output = True, text = False)
    #condsim = np.load(("conditional_brown_resnick_samples_" + str(missing_index) + ".npy"))
    #os.remove(("conditional_brown_resnick_samples_" + str(missing_index) + ".npy"))
    #return condsim

def extract_integer(filename):


    return int(filename.split('.')[0].split('_')[-1])

def concatenate_mcmc_interpolation_files(ref_image_folder, file_name, nrep):

    mcmc_folder = (ref_image_folder + "/mcmc_interpolation")
    ref_image = np.load((ref_image_folder + "/ref_image.npy"))
    mask = np.load((ref_image_folder + "/mask.npy"))
    missing_indices =  np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    filenames = [f for f in os.listdir(mcmc_folder) if os.path.isfile(os.path.join(mcmc_folder, f))]
    filenames = sorted(filenames, key = extract_integer)
    mcmc_samples = np.zeros((nrep,0))
    mcmc_mask = np.ones((n**2))

    for i,f in enumerate(filenames):

        missing_index = missing_indices[int(extract_integer(f))]
        current_mcmc_samples = np.load((mcmc_folder + "/" + f))
        if(current_mcmc_samples.size == nrep):
            current_mcmc_samples = np.log(current_mcmc_samples)
            mcmc_samples = np.concatenate([mcmc_samples, current_mcmc_samples.reshape((nrep,1))],
                                          axis = 1)
            mcmc_mask[int(missing_index)] = 0
        else:
            mcmc_mask[int(missing_index)] = -1

    mcmc_images = np.tile(ref_image.reshape(1,n**2), reps = (nrep,1))
    mcmc_images[:,mcmc_mask == 0] = mcmc_samples
    mcmc_images = mcmc_images.reshape((nrep,n,n))
    mcmc_mask = mcmc_mask.reshape((n,n))
    return mcmc_images, mcmc_mask
        

folder_name = "data/model2/ref_image2"
mcmc_folder_name = "data/model2/ref_image2/mcmc_interpolation"
file_name = "mcmc_interpolation_missing_index"
nrep = 4000
mcmc_images, mcmc_mask = concatenate_mcmc_interpolation_files(folder_name, file_name, nrep)
np.save((mcmc_folder_name + "/" + file_name + ".npy"), mcmc_images)
np.save((mcmc_folder_name + "/" + file_name + "_mask.npy"), mcmc_mask)