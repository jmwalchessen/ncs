import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_generation_on_the_fly import *
from models.ema import ExponentialMovingAverage
from models.ncsnpp import *
from evaluation.helper_functions import *
import losses
import sde_lib
from configs.vp import ncsnpp_config as vp_ncsnpp_config
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def visualize_loss(epochs_and_draws, train_losses, eval_losses, figname):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.plot(epochs_and_draws, train_losses, label = 'Train Loss')
    ax.plot(epochs_and_draws, eval_losses, label = 'Eval Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(figname)
def plot_original_and_diffusion_images(ref_image, mask, diffusion_images, vmin, vmax, figname, n):

    fig = plt.figure(figsize=(10, 10))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(2,2),
                    axes_pad=0.35,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    label_mode = "L"
                    )

    for i,ax in enumerate(grid):

        if(i == 0):
            im = ax.imshow(ref_image.reshape((n,n)), vmin = vmin, vmax = vmax)
        if(i == 1):
            ax.imshow(ref_image.reshape((n,n)), alpha = mask.float().detach().cpu().numpy().reshape((n,n)),
             vmin = vmin, vmax = vmax)
        else:
            ax.imshow(diffusion_images[(i-2),:,:,:].detach().cpu().numpy().reshape((n,n)), vmin = vmin, vmax = vmax)

    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)

    

def evaluate_diffusion(score_model, sde, process_type, range_value, smooth_value, p, folder_name,
                       vmin, vmax, figname):

    n = 32
    num_samples = 2
    device = "cuda:0"
    mask = (torch.bernoulli(p*torch.ones((1,1,n,n)))).to(device)

    if(process_type == "schlather"):

        number_of_replicates = 1
        ref_img = np.log(generate_schlather_process(range_value, smooth_value, seed_value, number_of_replicates, n))
    else:
        number_of_replicates = 50
        seed_value = int(np.random.randint(0, 1000000))
        ref_img = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))
        ref_img = ref_img[0:1,:,:,:]
    
    score_model.eval()
    y = ((torch.mul(mask, (torch.from_numpy(ref_img)).to(device))).to(device)).float()
    diffusion_images = posterior_sample_with_p_mean_variance_via_mask(sde, score_model, device, mask,
                                                   y, n, num_samples, range_value, smooth_value)

    if(os.path.exists(os.path.join(os.getcwd(), folder_name)) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name))

    figname = folder_name + "/" + figname
    plot_original_and_diffusion_images(ref_img, mask, diffusion_images, vmin, vmax, figname, n)




def train(config, data_draws, epochs_per_drawn_data,
          random_missingness_percentages, number_of_random_replicates,
          number_of_evaluation_random_replicates, number_of_masks_per_image,
          number_of_evaluation_masks_per_image, seed_values_list, eval_seed_values_list,
          train_parameter_matrix, eval_parameter_matrix, batch_size, eval_batch_size,
          score_model_path, loss_path, spatial_process_type, folder_name, vmin, vmax,
          eval_range_value, smooth_eval_value):
    
    # Initialize model.
    #score_model = mutils.create_model(config)
    score_model = nn.DataParallel((NCSNpp(config)).to(config.device))
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    initial_step = int(state['step'])
    eval_losses = []
    train_losses = []

    if(os.path.exists(os.path.join(os.getcwd(), folder_name)) == False):
        os.mkdir(os.path.join(os.getcwd(), folder_name))
    
    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=config.model.num_scales)
    #vesde 
    else:
        sde = sde_lib.VESDE(sigma_min=0.01, sigma_max=50, N = config.model.num_scales)
        sampling_eps = 1e-3

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                        reduce_mean=reduce_mean, continuous=continuous,
                                        likelihood_weighting=likelihood_weighting,
                                        masked = True)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    masked = True)
    
    num_train_steps = config.training.n_iters
    for data_draw in range(0, data_draws):
        print(data_draw)

        train_dataloader, eval_dataloader = get_training_and_evaluation_data_per_percentages(number_of_random_replicates, random_missingness_percentages,
                                                     number_of_evaluation_random_replicates, number_of_masks_per_image,
                                                     number_of_evaluation_masks_per_image, batch_size, eval_batch_size,
                                                     train_parameter_matrix, eval_parameter_matrix, seed_values_list[data_draw],
                                                     eval_seed_values_list[data_draw], spatial_process_type)   
        
        
        for epoch in range(0, epochs_per_drawn_data):
            #want to iterate over the same masks and images for each epoch (taking epectation with respect to p(X,M)=p(X)p(M))
            train_losses_per_epoch = []
            eval_losses_per_epoch = []
            train_iterator = iter(train_dataloader)
            eval_iterator = iter(eval_dataloader)
            #train for this epoch, then do eval
            while True:
                try:
                    batch = get_next_batch(train_iterator, config)
                    loss = train_step_fn(state, batch)
                    train_losses_per_epoch.append(float(loss))
                except StopIteration:
                    train_losses.append((sum(train_losses_per_epoch)/len(train_losses_per_epoch)))
                    break

            while True:
                try:
                    batch = get_next_batch(eval_iterator, config)
                    eval_loss = eval_step_fn(state, batch)
                    print(eval_loss)
                    eval_losses_per_epoch.append(float(eval_loss))
                except StopIteration:
                    eval_losses.append((sum(eval_losses_per_epoch)/len(eval_losses_per_epoch)))
                    break


            figname = ("diffusion_images_data_draw_" + str(data_draw) + "_epoch_" + str(epoch) + ".png")
            for p in random_missingness_percentages:
                n = 32
                evaluate_diffusion(score_model, sde, spatial_process_type, eval_range_value, eval_smooth_value, p, folder_name,
                                   vmin, vmax, figname)

            

    torch.save(score_model.state_dict(), score_model_path)
    epochs_and_draws = [i for i in range(0, len(train_losses))]
    visualize_loss(epochs_and_draws, train_losses, eval_losses, loss_path)




vp_ncsnpp_configuration = vp_ncsnpp_config.get_config()
vpconfig = vp_ncsnpp_configuration

data_draws = 5
epochs_per_data_draws = 10
number_of_random_replicates = 5000
number_of_evaluation_random_replicates = 50
number_of_masks_per_image = 100
number_of_evaluation_masks_per_image = 5
#smaller p means less ones which means more observed values
random_missingness_percentages = [.5]
batch_size = 512
eval_batch_size = 32
train_parameter_matrix = np.matrix([[10,1],[11,1],[12,1],[13,1],[14,1],[15,1]])
eval_parameter_matrix = np.matrix([[11,1],[14,1]])
tnparam = train_parameter_matrix.shape[0]
enparam = eval_parameter_matrix.shape[0]
eval_smooth_value = 1
eval_range_value = 12
seed_values_list = [[[int(np.random.randint(0, 100000)) for k in range(0, tnparam)] for j in range(0, len(random_missingness_percentages))] for i in range(0, data_draws)]
eval_seed_values_list = [[[int(np.random.randint(0, 100000)) for k in range(0, enparam)] for j in range(0, len(random_missingness_percentages))] for i in range(0, data_draws)]
score_model_path = "trained_score_models/vpsde/model1/model1_beta_min_max_01_20_range_10_20_smooth_1_random50_log_parameterized_mask.pth"
loss_path = "trained_score_models/vpsde/model1/model1_beta_min_max_01_20_range_10_20_smooth_1_random50_log_parameterized_mask_loss.png"
torch.cuda.empty_cache()
spatial_process_type = "brown"
folder_name = "trained_score_models/vpsde/model1"
vmin = -2
vmax = 6
eval_seed_values_list = [[(int(np.random.randint(0, 100000)), int(np.random.randint(0, 100000))) for j in range(0, len(random_missingness_percentages))] for i in range(0, data_draws)]


train(vpconfig, data_draws, epochs_per_data_draws,
      random_missingness_percentages, number_of_random_replicates,
      number_of_evaluation_random_replicates, number_of_masks_per_image,
      number_of_evaluation_masks_per_image, seed_values_list, eval_seed_values_list,
      train_parameter_matrix, eval_parameter_matrix, batch_size, eval_batch_size,
      score_model_path, loss_path, spatial_process_type, folder_name, vmin, vmax,
      eval_range_value, eval_smooth_value)