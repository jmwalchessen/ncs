import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_generation_on_the_fly import *
from models.ema import ExponentialMovingAverage
from models import ncsnpp
import losses
import sde_lib
from configs.vp import ncsnpp_config as vp_ncsnpp_config
from configs.ve import ncsnpp_config as ve_ncsnpp_config
import matplotlib.pyplot as plt

vp_ncsnpp_configuration = vp_ncsnpp_config.get_config()
config = vp_ncsnpp_configuration

score_model = th.nn.DataParallel((ncsnpp.NCSNpp(config)).to("cuda:0"))
score_model.load_state_dict(th.load(("trained_score_models/vpsde/model5_beta_min_max_01_20_1000_1.6_1.6_random50_bounded_masks.pth")))
score_model.eval()
optimize_fn=None
reduce_mean=False
continuous=True
likelihood_weighting=False
masked = True

optimizer = losses.get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=config.model.num_scales)

def get_masked_loss_fn(vpsde, train, reduce_mean = True):
  """DDPM loss modified to incorporate a mask (fixed or not)"""
  assert isinstance(vpsde, sde_lib.VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch, labels):

    batch_images = batch[0]
    batch_masks = batch[1]
    model_fn = model
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch_images.device)
    #this is sigma_t i.e. std
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch_images.device)
    noise = torch.randn_like(batch_images)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch_images + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    #this is based on equation 3 in paper All in One Simulation Based Inference
    masked_perturbed_data = (torch.mul((1-batch_masks), perturbed_data) + torch.mul(batch_masks, batch_images))
    #not necessarily needed to add mask as input to score model
    with th.no_grad():
        score = model_fn(masked_perturbed_data, labels)
    stds = sqrt_1m_alphas_cumprod[labels]
    #stds_inverse = sqrt_1m_alphas_cumprod[labels].pow(-1)
    #weighted losses
    stds = stds.view((batch_images.shape[0],1,1,1))
    losses = torch.square(torch.mul(stds, (torch.mul(stds, score) + noise)))
    masked_losses = torch.mul((1-batch_masks), losses)
    masked_losses = reduce_op(masked_losses.reshape(masked_losses.shape[0], -1), dim=-1)
    return masked_losses
  
  return loss_fn

masked_loss_fn = get_masked_loss_fn(sde, train = False, reduce_mean = True)

number_of_random_replicates = 1000
random_missingness_percentages = [.5]
number_of_eval_random_replicates = 1000
batch_size = 10
eval_batch_size = 10
range_value = 1.6
smooth_value = 1.6
seed_values = (234234, 234)
n = 32
train_dataloader, eval_dataloader = get_training_and_evaluation_mask_and_image_datasets_per_mask(number_of_random_replicates,
                                                                                                         random_missingness_percentages,
                                                                                                         number_of_eval_random_replicates,
                                                                                                         batch_size, eval_batch_size, range_value,
                                                                                                         smooth_value, seed_values, n)
eval_iterator = iter(eval_dataloader)
labels = torch.arange(0, 1000, 10)

def plot_eval_loss(labels, repeats, figname):
  
  eval_losses = []
  for label in labels:
    label_repeats = torch.tensor([label]).repeat(repeats)
    print("a")
    eval_batch = get_next_batch(eval_iterator, config)
    print("b")
    eval_loss = ((masked_loss_fn(score_model, eval_batch, label_repeats)).detach().cpu().numpy())[0]
    print("c")
    eval_losses.append(float(eval_loss))
  
  fig, ax = plt.subplots(1)
  plt.plot(labels.detach().cpu().numpy(), eval_losses)
  plt.xlabel("timestep")
  plt.ylabel("loss")
  plt.savefig(figname)

repeats = 10
plot_eval_loss(labels, repeats, "evaluation/visualizations/models/model5/loss_vs_timestep_1.png")