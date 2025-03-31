from append_directories import *
eval_folder = append_directory(2)
import sys
sys.path.append(eval_folder)
from helper_functions import *
import zeus
from zeus.monitor import ZeusMonitor
import torch
import numpy as np
import matplotlib.pyplot as plt

monitor = ZeusMonitor(gpu_indices = [0,1,2,3])



def visualie_ncs(ref_image, mask, ncs_image, mes, n):

    fig,ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10,5))

    ax[0].imshow(ref_image.reshape((n,n)), alpha = mask.reshape((n,n)), vmin = -2, vmax = 6)
    ax[1].imshow(ref_image.reshape((n,n)), vmin = -2, vmax = 6)
    ax[2].imshow(ncs_image.reshape((n,n)), vmin = -2, vmax = 6)
    fig.text(.3, .95, "Time " + str(round(mes.time)) + " seconds, Energy " + str(round(mes.total_energy)) + " J.")
    plt.savefig("ncs_image.png")


def ncs_sample_with_monitoring(vpsde, score_model, device, mask, y, n, num_samples, ref_img):

    monitor.begin_window("eval")
    ncs_image = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                               y, n, num_samples)
    mes = monitor.end_window("eval")
    visualie_ncs(ref_img.detach().cpu().numpy(), mask.detach().cpu().numpy(),
                 ncs_image.detach().cpu().numpy(), mes, n)


def produce_ncs_realization_with_variables():

    beta_min = .01
    beta_max = 20
    N = 1000
    number_of_replicates = 1
    n = 32
    process_type = "brown"
    device = "cuda:0"
    range_value = 3.
    smooth_value = 1.5
    seed_value = int(np.random.randint(0,100000,1))
    p = .05
    mask = torch.bernoulli(p*torch.ones((1,1,n,n)))
    ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))
    ref_img = th.from_numpy(ref_img.reshape((1,1,n,n)))
    y = ((th.mul(mask, ref_img)).to(device)).float()
    mask = mask.to(device)
    vpsde = load_sde(beta_min, beta_max, N)
    mode = "eval"
    model_name = "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth"
    score_model = load_score_model(process_type, model_name, mode)
    num_samples = 1
    ncs_sample_with_monitoring(vpsde, score_model, device, mask, y, n, num_samples, ref_img)

produce_ncs_realization_with_variables()