import subprocess
import numpy as np
import os


def produce_mcmc_interpolation_per_pixel(n, range_value, smooth_value, nugget, cov_mod, ref_file_name, mask_file_name,
                                        condsim_file_name, neighbors, nrep, missing_index):

    print(condsim_file_name)

    subprocess.run(["Rscript", "conditional_simulation_per_pixel.R", str(range_value),
                    str(smooth_value), str(nugget), str(ref_file_name), str(mask_file_name), str(condsim_file_name), str(cov_mod),
                    str(neighbors), str(n), str(nrep), str(missing_index)],
                    check = True, capture_output = True, text = False)
    #condsim = np.load(("conditional_brown_resnick_samples_" + str(missing_index) + ".npy"))
    #os.remove(("conditional_brown_resnick_samples_" + str(missing_index) + ".npy"))
    #return condsim

def produce_mcmc_interpolation(n, range_value, smooth_value, nugget, cov_mod,
                               ref_file_name, mask_file_name, neighbors, nrep, start, end):

    condsim = np.zeros(((end - start), nrep))
    condsim_file_name = ("data/25_by_25/condsim_range_1.6_smooth_1.6_missing_index_")
    for missing_index in range(start,end):
        print(missing_index)
        current_condsim_file_name = (condsim_file_name + str(missing_index) + ".npy")
        produce_mcmc_interpolation_per_pixel(n, range_value, smooth_value, nugget, cov_mod, ref_file_name, mask_file_name,
                                        current_condsim_file_name, neighbors, nrep, missing_index)
    


n = 25
range_value = 1.6
smooth_value = 1.6
nugget = 0
cov_mod = "brown"
mask_file_name = "data/25_by_25/mask.npy"
ref_file_name = "data/25_by_25/ref_image.npy"
neighbors = 7
nrep = 100
start = 1
end = 2
produce_mcmc_interpolation(n, range_value, smooth_value, nugget, cov_mod, ref_file_name,
                                     mask_file_name, neighbors, nrep, start, end)