import torch as th
import numpy as np
from append_directories import *
import matplotlib.pyplot as plt
import time
from true_unconditional_data_generation import *
sde_folder = append_directory(6)
#sde configs folder
sde_configs_vp_folder = sde_folder + "/configs/vp"
sys.path.append(sde_configs_vp_folder)
import ncsnpp_config
sys.path.append(sde_folder)
from models import ncsnpp
import sde_lib




def mask_generation(p, nrep, n):

    masks = (th.bernoulli(p*th.ones((nrep,1,n,n))))
    return masks

def load_score_model(model_name):

    config = ncsnpp_config.get_config()
    score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
    score_model.load_state_dict(th.load((sde_folder + "/trained_score_models/vpsde/" + model_name)))
    score_model.eval()
    return score_model


def load_sde(beta_min, beta_max, N):

    sdevp = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
    return sdevp


#y is observed part of field, modified to incorporate the mask as channel
def p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t):

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


def sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t):

    p_mean, p_variance = p_mean_and_variance_from_score_via_mask(vpsde, score_model, device, masked_xt, masks, ys, t)
    std = th.exp(0.5 * th.log(p_variance))
    noise = th.randn_like(masked_xt)
    #just to make sure that the masked values aren't perturbed by the noise, the variance should already be masked though
    masked_noise = th.mul((1-masks), noise)
    sample = p_mean + std*masked_noise
    return sample


def posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masks,
                                                   ys, n):

    nrep = masks.shape[0]
    unmasked_xT = th.randn((nrep, 1, n, n)).to(device)
    masked_xT = th.mul((1-masks), unmasked_xT) + th.mul(masks, ys)
    masked_xt = masked_xT
    for t in range((vpsde.N-1), 0, -1):
        masked_xt = sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masked_xt,
                                                         masks, ys, t)

    return masked_xt



def fixed_mask_data_generation(p, nrep, number_of_calls, variance, lengthscale, model_name,
                               folder_name, diffusion_file_name, true_file_name):

    n = 32
    mask = mask_generation(p, 1, n)
    repeated_mask = mask.repeat((nrep,1,1,1))
    seed_value = int(np.random.randint(0, 100000))
    minX = -10
    maxX = 10
    minY = -10
    maxY = 10
    beta_min = .1
    beta_max = 20
    N = 1000
    score_model = load_score_model(model_name)
    sdevp = load_sde(beta_min, beta_max, N)
    device = "cuda:0"
    repeated_mask = repeated_mask.float().to(device)
    diffusion_images = np.zeros((0,1,n,n))
    true_images = np.zeros((0,1,n,n))
    for i in range(0, number_of_calls):
        print(i)

        gp_vec, ys = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale, nrep, seed_value)
        ys = (th.from_numpy(ys)).float().to(device)
        current_diffusion_images = posterior_sample_with_p_mean_variance_via_mask(sdevp, score_model, device,
                                                                                  repeated_mask, ys, n)
        diffusion_images = np.concatenate([diffusion_images, current_diffusion_images.detach().cpu().numpy()],
                                           axis = 0)
        true_images = np.concatenate([true_images, ys.detach().cpu().numpy()], axis = 0)

    if(os.path.exists(os.path.join(os.getcwd(), folder_name)) == False):
        os.makedirs(os.path.join(os.getcwd(), folder_name))
       
    np.save((folder_name + "/" + diffusion_file_name), diffusion_images)
    np.save((folder_name + "/" + true_file_name), true_images)
    np.save((folder_name + "/mask.npy"), mask.numpy())


