import subprocess
import numpy as np



def call_cauchy_script(range_value, smooth_value, seed_value, number_of_replicates, n):

    stdout = subprocess.run(["Rscript", "cauchy_process_data_generation.R", str(range_value),
                         str(smooth_value), str(number_of_replicates), str(seed_value)],
                        check = True, capture_output = True, text = False)
    images = realization_pipeline(stdout, n, number_of_replicates)
    return images

def generate_cauchy_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    images = call_cauchy_script(range_value, smooth_value, seed_value, number_of_replicates, n)
    return images