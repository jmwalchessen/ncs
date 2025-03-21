import torch as th
import numpy as np
from append_directories import *
from functools import partial
from brown_resnick_data_generation import *
import matplotlib.pyplot as plt

evaluation_folder = append_directory(2)
sys.path.append(evaluation_folder)
from helper_functions import *

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

def generate_validation_data_with_reference_image(process_type, folder_name, n, range_value, smooth_value,
                                                  replicates_per_call, calls, validation_data_name, score_model):

    seed_value = int(np.random.randint(0, 100000))
    number_of_replicates = 1

    if(os.path.exists(os.path.join(os.getcwd(), folder_name, "diffusion")) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name, "diffusion"))


    device = "cuda:0"
    mask = np.load((folder_name + "/mask.npy"))
    ref_img = np.log(np.load((folder_name + "/ref_image.npy")))
    partially_observed = (mask*ref_img)

    conditional_samples = np.zeros((0,1,n,n))
    np.save((folder_name + "/partially_observed_field.npy"), partially_observed.reshape((n,n)))

    mask = th.from_numpy(mask.reshape((1,1,n,n)))
    ref_img = th.from_numpy(ref_img.reshape((1,1,n,n)))
    y = ((th.mul(mask, ref_img)).to(device)).float()
    mask = mask.float().to(device)
    print(y[y!=0])

    for i in range(0, calls):
        print(i)

        conditional_samples = np.concatenate([conditional_samples, sample_unconditionally_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls)], axis = 0)

    
    print(conditional_samples.shape)
    print((folder_name + "/diffusion/" + validation_data_name))
    np.save((folder_name + "/diffusion/" + validation_data_name), conditional_samples)

    ref_img = ref_img.detach().cpu().numpy().reshape((n,n))
    mask = mask.float().detach().cpu().numpy().reshape((n,n))
    plot_spatial_field(ref_img, -2, 6, (folder_name + "/ref_image.png"))
    plot_spatial_field((conditional_samples[0,:,:,:]).reshape((n,n)), -2, 6, (folder_name + "/diffusion_sample.png"))
    plot_masked_spatial_field(spatial_field = ref_img,
                   vmin = -2, vmax = 6, mask = mask, figname = (folder_name + "/partially_observed_field.png"))


def generate_parameter_validation_data_with_reference_image(process_type, folder_name, n, range_value, smooth_value,
                                                  replicates_per_call, calls, validation_data_name, score_model):

    seed_value = int(np.random.randint(0, 100000))
    number_of_replicates = 1

    if(os.path.exists(os.path.join(os.getcwd(), folder_name, "diffusion")) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name, "diffusion"))


    device = "cuda:0"
    mask = np.load((folder_name + "/mask.npy"))
    ref_img = np.load((folder_name + "/ref_image.npy"))
    partially_observed = (mask*ref_img)

    conditional_samples = np.zeros((0,1,n,n))
    np.save((folder_name + "/partially_observed_field.npy"), partially_observed.reshape((n,n)))

    mask = th.from_numpy(mask.reshape((1,1,n,n)))
    ref_img = th.from_numpy(ref_img.reshape((1,1,n,n)))
    y = ((th.mul(mask, ref_img)).to(device)).float()
    mask = mask.float().to(device)


    for i in range(0, calls):
        print(i)

        conditional_samples = np.concatenate([conditional_samples, sample_unconditionally_parameter_multiple_calls(sdevp, score_model, device, mask, y, n,
                                          replicates_per_call, calls, range_value, smooth_value)], axis = 0)

    
    print(conditional_samples.shape)
    print((folder_name + "/diffusion/" + validation_data_name))
    np.save((folder_name + "/diffusion/" + validation_data_name), conditional_samples)

    ref_img = ref_img.detach().cpu().numpy().reshape((n,n))
    mask = mask.float().detach().cpu().numpy().reshape((n,n))
    plot_spatial_field(ref_img, -2, 6, (folder_name + "/ref_image.png"))
    plot_spatial_field((conditional_samples[0,:,:,:]).reshape((n,n)), -2, 6, (folder_name + "/diffusion_sample.png"))
    plot_masked_spatial_field(spatial_field = ref_img,
                   vmin = -2, vmax = 6, mask = mask, figname = (folder_name + "/partially_observed_field.png"))


def generate_validation_data_multiple_percentages(process_type, model_folder_name, n, range_value, smooth_value,
                                                  replicates_per_call, calls, ps, validation_data_name):

    for i,p in enumerate(ps):

        current_validation_data_name = (validation_data_name + str(p) + ".npy")
        current_folder_name = (model_folder_name + "/ref_image" + str(i))
        generate_validation_data(process_type, current_folder_name, n, range_value, smooth_value,
                                 replicates_per_call, calls, p, current_validation_data_name)


    
def generate_validation_data_with_multiple_reference_images(model_name, model_number, range_value):
    
    process_type = "brown"
    n = 32
    replicates_per_call = 250
    calls = 4
    smooth_value = 1.5
    score_model = load_score_model("brown", model_name, "eval")
    folder_name = (evaluation_folder + "/fcs/data/conditional/")
    validation_data_name = ("model" + str(model_number) + "_range_" + str(range_value) + "_smooth_1.5_4000_random")
    for obs in range(1,7):
        current_folder = (folder_name + "obs" + str(obs) + "/ref_image" + str(int(range_value-1)))
        generate_validation_data_with_reference_image(process_type, current_folder, n, range_value, smooth_value,
                                                                    replicates_per_call, calls, validation_data_name,
                                                                    score_model)


def generate_validation_data_with_multiple_reference_images_with_variables():

    model_names = ["model6/model6_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_1_channel_mask.pth",
                   "model7/model7_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_2_channel_mask.pth",
                   "model5/model5_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_3_channel_mask.pth",
                   "model8/model8_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_4_channel_mask.pth",
                   "model9/model9_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_5_channel_mask.pth"]
    range_values = [float(i) for i in range(1,6)]
    model_numbers = [6,7,5,8,9]
    for i in range(0,5):
        generate_validation_data_with_multiple_reference_images(model_names[i], model_numbers[i], range_values[i])

generate_validation_data_with_multiple_reference_images_with_variables()