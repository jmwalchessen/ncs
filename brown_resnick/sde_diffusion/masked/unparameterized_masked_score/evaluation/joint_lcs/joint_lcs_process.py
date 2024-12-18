import numpy as np


def load_ref_image(ref_folder):

    return np.load((ref_folder + "/ref_image.npy"))

def load_mask(ref_folder):

    return np.load((ref_folder + "/mask.npy"))

def process_joint_lcs_file(ref_folder, joint_lcs_file, processed_joint_lcs_file, nrep, n):

    mask = load_mask(ref_folder)
    ref_image = np.log(load_ref_image(ref_folder))
    observed_image = ((mask)*ref_image).reshape((1,n**2))
    print(observed_image)
    missing_joint_lcs = np.log(np.load((ref_folder + "/" + joint_lcs_file)))
    joint_lcs = (np.tile(observed_image, reps = nrep)).reshape((nrep,n**2))
    joint_lcs[:,mask.flatten() == 0] = missing_joint_lcs
    print(joint_lcs[0,mask.flatten() == 1])
    joint_lcs = joint_lcs.reshape((nrep,n,n))
    np.save((ref_folder + "/" + processed_joint_lcs_file), joint_lcs)

ref_folder = "data/model4/ref_image6"
joint_lcs_file = "joint_lcs_range_3.0_smooth_1.5_nugget_1e5_obs_7_10.npy"
processed_joint_lcs_file = "processed_joint_lcs_range_3.0_smooth_1.5_nugget_1e5_obs_7_10.npy"
nrep = 10
n = 32
process_joint_lcs_file(ref_folder, joint_lcs_file, processed_joint_lcs_file, nrep, n)
