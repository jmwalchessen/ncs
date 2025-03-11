import numpy as np
import sys
from append_directories import *
sys.path.append(append_directory(2))
from helper_functions import *
import matplotlib.pyplot as plt


def generate_masks(nrep,n):

    mask_matrices = np.zeros((nrep,n,n))
    for irep in range(nrep):
        obs_indices = [int((n**2)*np.random.random(size = 1)) for i in range(0,m)]
        mask_matrices = mask_matrices.astype('float')
        mask_matrices[irep,obs_indices] = 1
    return th.from_numpy(mask_matrices.reshape((nrep,n,n)))

def generate_ncs_images():

    range_value = 5.
    nrep = 10
    smooth_value = 1.5
    seed_value = np.random.randint(0,10000)
    number_of_replicates = 10
    n = 32
    beta_min = .1
    beta_max = 20
    N = 1000
    vpsde = load_sde(beta_min, beta_max, N)
    device = "cuda:0"
    masks = generate_masks(nrep,n)
    score_model = load_score_model("brown", "model9/model9_wo_l2_beta_min_max_01_20_obs_num_1_7_smooth_1.5_range_5_channel_mask.pth", "eval")
    ys = generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n)
    images = multiple_posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masks,
                                                                    ys, n, range_value, smooth_value)
    return images

def visualize_images():

    nrep = 10
    n = 32
    images = generate_ncs_images()
    for irep in range(nrep):
        fig,ax = plt.subplots()
        ax.imshow(images[irep,:,:,:].reshape((n,n)), vmin = -2, vmax = 6)
        figname = "visualizations/figures/ncs_image_range_5_" + str(irep) + ".png"
        plt.savefig(figname)


visualize_images()