import numpy as np
import torch as th
import sys
from append_directories import *
import matplotlib.pyplot as plt
import subprocess
from matplotlib.patches import Rectangle


#get trained score model
def load_score_model(process_type, model_name, mode):

    home_folder = append_directory(6)
    if "sde_diffusion" in home_folder:
        sde_folder = home_folder + "/masked/unparameterized_masked_score"
    else:
        sde_folder = home_folder + "/sde_diffusion/masked/unparameterized_masked_score"
    sde_configs_vp_folder = sde_folder + "/configs/vp"
    sys.path.append(sde_configs_vp_folder)
    import ncsnpp_config
    sys.path.append(sde_folder)
    from models import ncsnpp
    config = ncsnpp_config.get_config()

    score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
    if(process_type == "schlather"):
        score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/" + process_type + "/" + model_name)))
    else:
        score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/" + model_name)))
    if(mode == "train"):
        score_model.train()
    else:
        score_model.eval()
    return score_model

def load_sde(beta_min, beta_max, N):
    sys.path.append(append_directory(3))
    import sde_lib
    sdevp = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
    return sdevp

def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    #need mask to be same size as masked_xt
    mask = mask.repeat((reps,1,1,1))
    masked_xt_and_mask = th.cat([masked_xt, mask], dim = 1)
    with th.no_grad():
        score_and_mask = score_model(masked_xt_and_mask, timestep)
    
    #first channel is score, second channel is mask
    score = score_and_mask[:,0:1,:,:]
    #reduce dimension of mask
    mask = mask[0:1,:,:,:]
    unmasked_p_mean = (1/th.sqrt(th.tensor(vpsde.alphas[t])))*(masked_xt + th.square(th.tensor(vpsde.sigmas[t]))*score)
    masked_p_mean = th.mul((1-mask), unmasked_p_mean) + th.mul(mask, y)
    unmasked_p_variance = (th.square(th.tensor(vpsde.sigmas[t])))*th.ones_like(masked_xt)
    masked_p_variance = th.mul((1-mask), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def multiple_p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t,
                                                     range_value, smooth_value):

    num_samples = masked_xt.shape[0]
    timestep = ((th.tensor([t])).repeat(num_samples)).to(device)
    reps = masked_xt.shape[0]
    masked_xt_and_mask = th.cat([masked_xt, masks], dim = 1)
    with th.no_grad():
        score_and_mask = score_model(masked_xt_and_mask, timestep)
    
    #first channel is score, second channel is mask
    score = score_and_mask[:,0:1,:,:]
    squared_sigmat = (th.square(th.tensor(vpsde.sigmas[t]))).to(device)
    sqrt_alphat = (th.sqrt(th.tensor(vpsde.alphas[t]))).to(device)
    unmasked_p_mean = (1/sqrt_alphat)*(masked_xt + squared_sigmat*score)
    masked_p_mean = th.mul((1-masks), unmasked_p_mean) + th.mul(masks, ys)
    unmasked_p_variance = squared_sigmat*th.ones_like(masked_xt)
    masked_p_variance = th.mul((1-masks), unmasked_p_variance)
    return masked_p_mean, masked_p_variance

def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, mask, y, t, num_samples):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, mask, y, t)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-mask), noise)
    sample = p_mean + std*masked_noise
    return sample


def multiple_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t, range_value, smooth_value):

    p_mean, p_variance = multiple_p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t, range_value, smooth_value)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-masks), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                   y, n, num_samples):

    unmasked_xT = th.randn((num_samples, 1, n, n)).to(device)
    masked_xT = th.mul((1-mask), unmasked_xT) + th.mul(mask, y)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         mask, y, t, num_samples)

    return masked_xt

def multiple_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masks,
                                                   ys, n, range_value, smooth_value):

    nrep = masks.shape[0]
    unmasked_xT = th.randn((nrep, 1, n, n)).to(device)
    masked_xT = th.mul((1-masks), unmasked_xT) + th.mul(masks, ys)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = multiple_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         masks, ys, t, range_value, smooth_value)

    return masked_xt

def sample_unconditionally_multiple_calls(vpsde, score_model, device, mask, y, n,
                                          num_samples_per_call, calls):
    
    diffusion_samples = th.zeros((0, 1, n, n))
    for call in range(0, calls):
        current_diffusion_samples = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model,
                                                                                   device, mask, y, n,
                                                                                   num_samples_per_call)
        diffusion_samples = th.cat([current_diffusion_samples.detach().cpu(),
                                    diffusion_samples],
                                    dim = 0)
    return diffusion_samples

def generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n):

    subprocess.run(["Rscript", "brown_resnick_data_generation.R", str(range_value),
                    str(smooth_value), str(number_of_replicates), str(seed_value)],
                    check = True, capture_output = True, text = False)
    images = np.load("temporary_brown_resnick_samples.npy")
    os.remove("temporary_brown_resnick_samples.npy")
    images = images.reshape((number_of_replicates,1,n,n))
    return images


def generate_masks(nrep,n,m):

    mask_matrices = np.zeros((nrep,n**2))
    for irep in range(nrep):
        obs_indices = [int((n**2)*np.random.random(size = 1)) for i in range(0,m)]
        mask_matrices = mask_matrices.astype('float')
        mask_matrices[irep,obs_indices] = 1
    return th.from_numpy(mask_matrices.reshape((nrep,n,n)))

def generate_ncs_images():

    range_value = 5.
    nrep = 10
    number_of_replicates = 10
    smooth_value = 1.5
    seed_value = np.random.randint(0,10000)
    n = 32
    beta_min = .1
    beta_max = 20
    N = 1000
    m = 1
    vpsde = load_sde(beta_min, beta_max, N)
    device = "cuda:0"
    masks = generate_masks(nrep,n,m)
    score_model = load_score_model("brown", "model9/model9_wo_l2_beta_min_max_01_20_obs_num_1_10_smooth_1.5_range_5_channel_mask.pth", "eval")
    ys = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))
    ys = ((th.from_numpy(ys)).float().reshape((nrep,1,n,n))).to(device)
    masks = masks.reshape((nrep,1,n,n)).float().to(device)
    images = multiple_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masks,
                                                                    ys, n, range_value, smooth_value)
    images = images.detach().cpu().numpy()
    return images, masks

def visualize_images_from_ncs_unconditional():

    m = 7
    nrep = 200
    n = 32
    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image4/mask.npy"))
    images = np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image4/diffusion/unconditional_fixed_ncs_images_range_5.0_smooth_1.5_4000.npy"))
    for irep in range(nrep):
        print(irep)
        fig,ax = plt.subplots()
        observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
        print(images[irep,observed_indices[:,0],observed_indices[:,1]])
        im = ax.imshow(images[irep,:,:].reshape((n,n)), vmin = -2, vmax = 6)
        for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
        plt.colorbar(im)
        figname = "figures/ncs_image_model11_obs_" + str(m) + "_range_5_" + str(irep) + ".png"
        plt.savefig(figname)
        plt.clf()

def visualize_images_from_ncs_unconditional_extreme():

    m = 7
    nrep = 4000
    n = 32
    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs7/ref_image4/mask.npy"))
    images = np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs7/ref_image4/diffusion/unconditional_fixed_model11_ncs_images_range_5.0_smooth_1.5_4000.npy"))
    for irep in range(0,nrep):
        fig,ax = plt.subplots()
        observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
        values = images[irep,observed_indices[:,0],observed_indices[:,1]]
        if(np.any(values > 4.)):
            im = ax.imshow(images[irep,:,:].reshape((n,n)), vmin = -2, vmax = 6)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            plt.colorbar(im)
            figname = "figures/extremed_ncs_image_obs_" + str(m) + "_range_5_" + str(irep) + ".png"
            plt.savefig(figname)
            plt.clf()

def visualize_true_images_extreme():
    m = 1
    nrep = 4000
    n = 32
    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs1/ref_image4/mask.npy"))
    images = np.log(np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs1/ref_image4/true_brown_resnick_images_range_5_smooth_1.5_4000.npy")))
    images = images.reshape((nrep,n,n))
    for irep in range(0,nrep):
        fig,ax = plt.subplots()
        observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
        values = images[irep,observed_indices[:,0],observed_indices[:,1]]
        ncs_values = images[irep,observed_indices[:,0],observed_indices[:,1]]
        if(np.any(values > 4.)):
            im = ax.imshow(images[irep,:,:].reshape((n,n)), vmin = -2, vmax = 6)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            plt.colorbar(im)
            figname = "figures/extremed_true_image_obs_" + str(m) + "_range_5_" + str(irep) + ".png"
            plt.savefig(figname)
            plt.clf()

def visualize_true_images():
    m = 7
    nrep = 4000
    n = 32
    eval_folder = append_directory(2)
    mask = np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image4/mask.npy"))
    images = np.log(np.load((eval_folder + "/fcs/data/unconditional/fixed_locations/obs" + str(m) + "/ref_image4/true_brown_resnick_images_range_5_smooth_1.5_4000.npy")))
    images = images.reshape((nrep,n,n))
    for irep in range(0,nrep):
        fig,ax = plt.subplots()
        observed_indices = np.argwhere(mask.reshape((n,n)) > 0)
        values = images[irep,observed_indices[:,0],observed_indices[:,1]]
        ncs_values = images[irep,observed_indices[:,0],observed_indices[:,1]]
        if(np.any(values > 4.)):
            im = ax.imshow(images[irep,:,:].reshape((n,n)), vmin = -2, vmax = 6)
            for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            plt.colorbar(im)
            figname = "figures/true_image_obs_" + str(m) + "_range_5_" + str(irep) + ".png"
            plt.savefig(figname)
            plt.clf()



def visualize_images():

    nrep = 10
    n = 32
    m = 1
    imagemask = generate_ncs_images()
    images = imagemask[0]
    masks = (imagemask[1]).detach().cpu().numpy()
    for irep in range(nrep):
        fig,ax = plt.subplots()
        observed_indices = np.argwhere(masks[irep,:,:,:].reshape((n,n)) > 0)
        im = ax.imshow(images[irep,:,:,:].reshape((n,n)), vmin = -2, vmax = 6)
        for j in range(observed_indices.shape[0]):
                rect = Rectangle(((observed_indices[j,1]-.55), (observed_indices[j,0]-.55)), width=1, height=1, facecolor='none', edgecolor='r')
                ax.add_patch(rect)
        plt.colorbar(im)
        figname = "figures/ncs_image_model9_obs_" + str(m) + "_range_5_" + str(irep+10) + ".png"
        plt.savefig(figname)


visualize_true_images()