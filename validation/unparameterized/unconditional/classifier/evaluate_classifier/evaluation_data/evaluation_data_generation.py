import torch as th
import numpy as np
import os
import sys
from append_directories import *
classifier_folder = append_directory(3)
data_generation_folder = (classifier_folder + "/train_classifier/generate_data")
sys.path.append(data_generation_folder)
from diffusion_data_generation import *

replicates_per_call = 500
calls = 4
n = 32
unconditional_samples = sample_unconditionally(twisted_diffusion_model, score_model,
                                               replicates_per_call, calls, n)

uncond_files = (append_directory(1) + "/data/diffusion/unconditional_diffusion_lengthscale_1.6_variance_0.4_2000.npy")
np.save(uncond_files, unconditional_samples)