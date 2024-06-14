import subprocess
import numpy as np


def realization_pipeline(stdoutput, n, number_of_replicates):

    stdout_str = (stdoutput.stdout).decode()
    y_str_split = stdout_str.split()
    y_str = y_str_split[slice(2, (3*(n**number_of_replicates) + 2), 3)]
    y = np.asarray([float(y_str[i]) for i in range(0,n*number_of_replicates)])
    y = y.reshape((number_of_replicates,1,int(np.sqrt(n)),int(np.sqrt(n))))
    return y

def call_brown_resnick_script(range_value, smooth_value, seed_value, number_of_replicates, n):

    stdout = subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                         str(smooth_value), str(number_of_replicates), str(seed_value)],
                        check = True, capture_output = True, text = False)
    images = realization_pipeline(stdout, n, number_of_replicates)
    return images

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    calls = int(number_of_replicates/100)
    images = np.zeros((0,1,int(np.sqrt(n)),int(np.sqrt(n))))
    for i in range(0, calls):
        current_images = call_brown_resnick_script(range_value, smooth_value, seed_value,
                                                   100, n)
        images = np.concatenate([images, current_images], axis = 0)
    
    current_images = call_brown_resnick_script(range_value, smooth_value, seed_value,
                                                   (number_of_replicates % 100), n)
    images = np.concatenate([images, current_images])
    return images