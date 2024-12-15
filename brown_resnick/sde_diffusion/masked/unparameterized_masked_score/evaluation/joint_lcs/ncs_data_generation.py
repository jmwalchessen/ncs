import torch as th
import numpy as np
from append_directories import *
from functools import partial
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *
score_model = load_score_model("brown", "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth", "eval")
vpsde = load_sde(beta_min = .1, beta_max = 20, N = 1000)

def load_npfile(npfile):
    nparray = np.load(npfile)
    return nparray

def generate_ncs_images(ref_folder, vpsde, score_model, range_value,
                        smooth_value, ncs_images_file, batches_per_call, calls, n, device):

    mask = load_npfile((ref_folder + "/mask.npy"))
    ref_image = np.log(load_npfile((ref_folder + "/ref_image.npy")))
    masked_image = np.multiply(mask, ref_image)
    device = "cuda:0"
    ncs_images = np.zeros(((batches_per_call*calls),n,n))

    mask = (th.from_numpy((mask.reshape((1,1,n,n))))).float().to(device)
    masked_image = (th.from_numpy((masked_image.reshape((1,1,n,n))))).float().to(device)

    for i in range(calls):
        ncs_images[i*batches_per_call:(i+1)*batches_per_call,:,:] = ((posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                                                                                     masked_image, n, batches_per_call, range_value, smooth_value)).detach().cpu().numpy()).reshape((batches_per_call,n,n))
        
    
    np.save(ncs_images_file, ncs_images)


def generate_ncs_images_multiple_files(vpsde, score_model, range_values,
                                        smooth_value, batches_per_call, calls, n, device, m):
    
    nrep = (batches_per_call * calls)
    for i in range(m):

        ref_folder = ("data/model4/ref_image" + str(i))
        ncs_images_file = (ref_folder + "/ncs_images_range_" + str(range_values[i]) +
                           "_smooth_" + str(smooth_value) + "_" + str(nrep) + ".npy")
        generate_ncs_images(ref_folder, vpsde, score_model, range_value,
                        smooth_value, ncs_images_file, batches_per_call, calls, n, device)
        


range_value = 3.0
smooth_value = 1.5
batches_per_call = 5
calls = 2
n = 32
device = "cuda:0"
m = 5
generate_ncs_images_multiple_files(vpsde, score_model, range_value,
                                    smooth_value,
                                    batches_per_call, calls, n, device, 5) 

    
