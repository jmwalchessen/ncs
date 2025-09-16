import torch as th
import numpy as np
from append_directories import *
from functools import partial
from brown_resnick_data_generation import *
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *

def load_score_model_with_variables():

    model_name = "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth"
    mode = "eval"
    score_model = load_score_model(model_name, mode)
    return score_model

def load_sde_with_variables():

    sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)
    return sdevp

score_model = load_score_model_with_variables()
sdevp = load_sde_with_variables()

def random_mask_generation(n, p):

    mask = th.bernoulli(p*th.ones((1,1,n,n)))
    return mask

def produce_pit_ncs_data_per_reference_image(range_value, smooth_value, p, n, vpsde, score_model,
                                             device, nrep_per_reference, ncs_image_file,
                                             ref_image_file):
    
    mask = random_mask_generation(n, p)
    seed_value = int(np.random.randint(0,10000,1))
    mask = mask.float().to(device)
    ref_image =  (np.log(generate_brown_resnick_process(range_value, smooth_value,
                                                        seed_value, 1, n))).reshape((1,1,n,n))
    ref_image = ref_image.float().to(device)
    ncs_images = posterior_sample_with_p_mean_variance_via_mask(vpsde = vpsde, score_model = score_model,
                                                                device = device, mask = mask, y = ref_image,
                                                                n = n, num_samples = nrep_per_reference)
    np.save(ncs_image_file, ncs_images)
    np.save(ref_image_file, ref_image)

def produce_pit_ncs_data(range_value, smooth_value, p, n, vpsde, score_model,
                         device, nrep_per_reference, nrep, pit_ncs_folder):
    
    basic_ncs_file = (pit_ncs_folder + "/ncs_images_range_" + str(range_value) + "_smooth_"
                      + str(smooth_value) + "_" + str(nrep_per_reference) + "_random" + str(p))
    basic_ref_file = (pit_ncs_folder + "/ref_image")
    for i in range(nrep):
        ncs_image_file = (basic_ncs_file + "_" + str(i) + ".npy")
        ref_image_file = (basic_ref_file + str(i) + ".npy")
        produce_pit_ncs_data_per_reference_image(range_value, smooth_value, p, n, vpsde, score_model,
                                             device, nrep_per_reference, ncs_image_file,
                                             ref_image_file)

        
    



  
