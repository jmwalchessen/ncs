import numpy as np
import torch as th
import subprocess
import os



def extract_integer(filename):


    return (int(filename.split('.')[0].split('_')[-1]))

def concatenate_lcs_files(ref_image_folder, nrep, n, unibi_type):

    lcs_folder = (ref_image_folder + "/lcs/" + unibi_type)
    ref_image = np.load((ref_image_folder + "/ref_image.npy"))
    mask = np.load((ref_image_folder + "/mask.npy"))
    missing_indices =  np.squeeze(np.argwhere((1-mask).reshape((n**2,))))
    filenames = [f for f in os.listdir(lcs_folder) if os.path.isfile(os.path.join(lcs_folder, f))]
    filenames = sorted(filenames, key = extract_integer)
    lcs_samples = np.zeros((nrep,0))
    lcs_mask = np.ones((n**2))

    print(len(filenames))
    for missing_index,f in enumerate(filenames):
        print(missing_index)

        current_lcs_samples = np.load((lcs_folder + "/" + f))
        if(current_lcs_samples.size == nrep):
            current_lcs_samples = np.log(current_lcs_samples)
            lcs_samples = np.concatenate([lcs_samples, current_lcs_samples.reshape((nrep,1))],
                                          axis = 1)
            if(missing_index < len(missing_indices)):
                lcs_mask[int(missing_indices[missing_index])] = 0
            else:
                lcs_mask[int(missing_indices[missing_index])] = -1

    lcs_images = np.tile(ref_image.reshape(1,n**2), reps = (nrep,1))
    lcs_images[:,lcs_mask == 0] = lcs_samples
    lcs_images = lcs_images.reshape((nrep,n,n))
    lcs_mask = lcs_mask.reshape((n,n))
    return lcs_images, lcs_mask
        

ref_image_folder = "data/model4/ref_image4"
nrep = 4000
n = 32
unibi_type = "univariate"
neighbors = 7
lcs_images, lcs_mask = concatenate_lcs_files(ref_image_folder, nrep, n, unibi_type)
lcs_file_name = "univariate_lcs_" + str(nrep) + "_neighbors_" + str(neighbors) + "_nugget_1e5"
lcs_mask_name = "univariate_lcs_" + str(nrep) + "_neighbors_" + str(neighbors) + "_nugget_1e5_mask"
np.save((ref_image_folder + "/lcs/univariate/" + lcs_file_name + ".npy"), lcs_images)
np.save((ref_image_folder + "/lcs/univariate/" + lcs_file_name + "_mask.npy"), lcs_mask)
