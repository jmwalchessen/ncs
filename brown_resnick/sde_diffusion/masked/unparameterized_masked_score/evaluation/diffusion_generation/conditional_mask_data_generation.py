import torch as th
import numpy as np
from append_directories import *
from functools import partial
from brown_resnick_data_generation import *
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *

score_model = load_score_model("schlather", "model4_beta_min_max_01_20_range_2.2_smooth_1.9_random025_log_parameterized_mask.pth", "eval")

sdevp = load_sde(beta_min = .1, beta_max = 20, N = 1000)

#mask is a True/False (1,32,32) vector with .5 randomly missing pixels
#function gen_mask is in image_utils.py, 50 at end of random50 denotes
#50 percent missing
    
def plot_spatial_field(spatial_field, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax)
    plt.savefig(figname)

def plot_masked_spatial_field(spatial_field, mask, vmin, vmax, figname):

    fig, ax = plt.subplots()
    ax.imshow(spatial_field, vmin = vmin, vmax = vmax, alpha = mask)
    plt.savefig(figname)

def generate_validation_data(process_type, folder_name, n, range_value, smooth_value, replicates_per_call, calls, p, validation_data_name):

    seed_value = int(np.random.randint(0, 100000))
    number_of_replicates = 1

    if(os.path.exists(os.path.join(os.getcwd(), folder_name)) == False):
        os.makedirs(folder_name)

    if(os.path.exists(os.path.join(os.getcwd(), folder_name, "diffusion")) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name, "diffusion"))

    if(process_type == "schlather"):

        ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))
    else:
        ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))

    device = "cuda:0"
    mask = (th.bernoulli(p*th.ones(n,n))).numpy()
    partially_observed = (mask*ref_img)
    np.save((folder_name + "/ref_image.npy"), ref_img.reshape((n,n)))

    conditional_samples = np.zeros((0,1,n,n))
    np.save((folder_name + "/partially_observed_field.npy"), partially_observed.reshape((n,n)))
    np.save((folder_name + "/mask.npy"), mask)
    np.save((folder_name + "seed_value.npy"), np.array([int(seed_value)]))

    mask = th.from_numpy(mask.reshape((1,1,n,n)))
    ref_img = th.from_numpy(ref_img.reshape((1,1,n,n)))
    y = ((th.mul(mask, ref_img)).to(device)).float()
    mask = mask.to(device)

    for i in range(0, calls):

        conditional_samples = np.concatenate([conditional_samples, sample_unconditionally_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls)], axis = 0)

    

    np.save((folder_name + "/diffusion/" + validation_data_name), conditional_samples)

    plot_spatial_field(ref_img.detach().cpu().numpy().reshape((n,n)), -2, 6, (folder_name + "/ref_image.png"))
    plot_spatial_field((conditional_samples[0,:,:,:]).numpy().reshape((n,n)), -2, 6, (folder_name + "/diffusion_sample.png"))
    plot_masked_spatial_field(spatial_field = ref_img.reshape((n,n)),
                   vmin = -2, vmax = 6, mask = mask.int().float().detach().cpu().numpy().reshape((n,n)), figname = (folder_name + "/partially_observed_field.png"))
    


process_type = "schlather"
folder_name = "data/schlather/model4/ref_image1"
n = 32
range_value = 2.2
smooth_value = 1.9
replicates_per_call = 250
calls = 4
p = 0
validation_data_name = "model4_random" + str(p) + "4000.npy"
generate_validation_data(process_type, folder_name, n, range_value, smooth_value, replicates_per_call, calls, p, validation_data_name)



