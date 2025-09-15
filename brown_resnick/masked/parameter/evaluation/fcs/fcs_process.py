import numpy as np


def load_ref_image(ref_folder):

    return np.load((ref_folder + "/ref_image.npy"))

def load_mask(ref_folder):

    return np.load((ref_folder + "/mask.npy"))

def process_fcs_file(ref_folder, fcs_file, processed_fcs_file, nrep, n):

    mask = load_mask(ref_folder)
    ref_image = load_ref_image(ref_folder)
    observed_image = (mask*ref_image).reshape((n**2,1))
    missing_fcs = np.load((ref_folder + "/" + fcs_file))
    fcs = (np.tile(observed_image, reps = nrep)).reshape((nrep,n**2))
    fcs[:,mask.flatten() == 0] = missing_fcs
    fcs = fcs.reshape((nrep,n,n))
    np.save((ref_folder + "/" + processed_fcs_file), fcs)

ref_folder = "data/model4/ref_image4"
fcs_file = "fcs_range_5.0_smooth_1.5_nugget_1e5_obs_7_10.npy"
processed_fcs_file = "processed_fcs_range_5.0_smooth_1.5_nugget_1e5_obs_7_10.npy"
nrep = 10
n = 32
process_fcs_file(ref_folder, fcs_file, processed_fcs_file, nrep, n)
