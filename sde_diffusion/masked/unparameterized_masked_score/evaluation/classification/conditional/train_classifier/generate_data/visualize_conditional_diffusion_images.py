import numpy as np
import torch as th
from diffusion_data_generation import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def visualize_conditional_diffusion_image(beta_min, beta_max, N, model_name, p, n, variance, lengthscale, figname):

    fig = plt.figure(figsize=(15,10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 3),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )
    device = "cuda:0"
    nrep = 2
    masks = (mask_generation(p, nrep, n)).float().to(device)
    vpsde = load_sde(beta_min, beta_max, N)
    score_model = load_score_model(model_name)
    minX = minY = -10
    maxX = maxY = 10
    seed_value = int(np.random.randint(0, 100000))
    ref_vectors, ref_images = generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                                        nrep, seed_value)
    ref_images = th.from_numpy(ref_images).float().to(device)
    diffusion_images = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, masks,
                                                                      ref_images, n)
    diffusion_images = diffusion_images.detach().cpu().numpy()
    ref_images = ref_images.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    for i, ax in enumerate(grid):

        if(i == 0):
            im = ax.imshow(ref_images[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
        
        if(i == 1):
            ax.imshow(ref_images[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2, alpha = (masks[0,:,:,:]).reshape(n,n))
        
        if(i == 2):
            ax.imshow(diffusion_images[0,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)

        if(i == 3):
            ax.imshow(ref_images[1,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)
        
        if(i == 4):
            ax.imshow(ref_images[1,:,:,:].reshape((n,n)), vmin = -2, vmax = 2, alpha = (masks[1,:,:,:]).reshape((n,n)))
        
        if(i == 5):
            ax.imshow(diffusion_images[1,:,:,:].reshape((n,n)), vmin = -2, vmax = 2)

    grid[0].cax.colorbar(im)
    plt.savefig(figname)


p = .5
n = 32
variance = .4
lengthscale = 1.6
figname = "conditional_diffusion.png"
beta_min = .1
beta_max = 20
N = 1000
model_name = "model6_beta_min_max_01_20_random02510_channel_mask.pth"
visualize_conditional_diffusion_image(beta_min, beta_max, N, model_name,
                                      p, n, variance, lengthscale, figname)      

    
    
