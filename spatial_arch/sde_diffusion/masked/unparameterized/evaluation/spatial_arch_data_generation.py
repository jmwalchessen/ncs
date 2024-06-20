import subprocess
import numpy as np
import os



def generate_spatial_arch_process(rho, alpha, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "spatial_arch_data_generation.R", str(rho),
                    str(alpha), str(n), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_spatial_arch_samples.npy")
    os.remove("temporary_spatial_arch_samples.npy")
    return images
