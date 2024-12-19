import numpy as np


def load_ref_image(ref_folder):

    return np.load((ref_folder + "/ref_image.npy"))

def load_mask(ref_folder):

    return np.load((ref_folder + "/mask.npy"))

def process_fcs_file(ref_folder, fcs_file, processed_fcs_file, nrep, n):

    mask = load_mask(ref_folder)
    ref_image = np.log(load_ref_image(ref_folder))
    observed_image = ((mask)*ref_image).reshape((1,n**2))
    print(observed_image)
    missing_fcs = np.log(np.load((ref_folder + "/" + fcs_file)))
    fcs = (np.tile(observed_image, reps = nrep)).reshape((nrep,n**2))
    fcs[:,mask.flatten() == 0] = missing_fcs
    print(fcs[0,mask.flatten() == 1])
    fcs = fcs.reshape((nrep,n,n))
    np.save((ref_folder + "/" + processed_fcs_file), fcs)

ref_folder = "data/model4/ref_image6"
m = 7
nrep = 4000
fcs_file = "fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
processed_fcs_file = "processed_fcs_range_3.0_smooth_1.5_nugget_1e5_obs_" + str(m) + "_" + str(nrep) + ".npy"
n = 32
process_fcs_file(ref_folder, fcs_file, processed_fcs_file, nrep, n)
