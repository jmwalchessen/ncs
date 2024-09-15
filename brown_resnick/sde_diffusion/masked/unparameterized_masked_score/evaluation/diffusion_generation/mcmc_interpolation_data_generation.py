import numpy as np
import torch as th
import subprocess

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


produce_mcmc_interpolation_multiple_pixels(n, range_value, smooth_value, nugget, cov_mod,
                                               ref_file_name, mask_file_name, condsim_file_name,
                                               neighbors, nrep, missing_index_start)