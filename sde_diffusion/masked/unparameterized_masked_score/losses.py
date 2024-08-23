
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import model_utils as mutils
from sde_lib import VPSDE, VESDE
from utils import *


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        #g['lr'] = lr * np.minimum(step / warmup, 1.0)
        number_of_steps_per_epoch = int(config.training.data_size/config.training.batch_size) 
        epoch = int(step/number_of_steps_per_epoch)
        #g['lr'] = lr * (.99**epoch)
        g['lr'] = lr * (.9**(int(step/250)))
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    batch_images = batch[0]
    batch_masks = batch[1]
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch_images, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, batch_masks, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch_images), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn



def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    batch_images = batch[0]
    batch_masks = batch[1]
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch_images.shape[0],), device=batch_images[0].device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch_images.device)
    #this is sigma_t i.e. std
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch_images.device)
    noise = torch.randn_like(batch_images)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch_images + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, batch_masks, labels)
    #score=-(1/sigma_t)*noise
    stds = sqrt_1m_alphas_cumprod[labels]
    #stds_inverse = sqrt_1m_alphas_cumprod[labels].pow(-1)
    #weighted losses
    stds = stds.view((batch_images.shape[0],1,1,1))
    losses = torch.square(torch.mul(stds, (torch.mul(stds, score) + noise)))
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_masked_ddpm_loss_fn(vpsde, train, reduce_mean = True):
  """DDPM loss modified to incorporate a mask (fixed or not)"""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):

    batch_images = batch[0]
    batch_masks = batch[1]
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch_images.shape[0],), device=batch_images[0].device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch_images.device)
    #this is sigma_t i.e. std
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch_images.device)
    noise = torch.randn_like(batch_images)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch_images + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    #this is based on equation 3 in paper All in One Simulation Based Inference
    masked_perturbed_data = (torch.mul((1-batch_masks), perturbed_data) + torch.mul(batch_masks, batch_images))
    #not necessarily needed to add mask as input to score model
    score = model_fn(masked_perturbed_data, batch_masks, labels)
    stds = sqrt_1m_alphas_cumprod[labels]
    #stds_inverse = sqrt_1m_alphas_cumprod[labels].pow(-1)
    #weighted losses
    stds = stds.view((batch_images.shape[0],1,1,1))
    losses = torch.square(torch.mul(stds, (torch.mul(stds, score) + noise)))
    masked_losses = torch.mul((1-batch_masks), losses)
    masked_losses = reduce_op(masked_losses.reshape(masked_losses.shape[0], -1), dim=-1)
    mask_loss = torch.mean(masked_losses)
    return mask_loss
  
  return loss_fn

def get_masked_score_ddpm_loss_fn(vpsde, train, reduce_mean = True):
  """DDPM loss modified to incorporate a mask (fixed or not)"""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  #batch is one element (1,2,32,32) 1rst channel is image, 2nd channel is mask
  def loss_fn(model, batch):

    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch[0].device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    #this is sigma_t i.e. std
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    batch_images = batch[:,0:1,:,:]
    batch_masks = batch[:,1:2,:,:]
    noise = torch.randn_like(batch_images)
    perturbed_image = sqrt_alphas_cumprod[labels, None, None, None] * batch_images + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    #this is based on equation 3 in paper All in One Simulation Based Inference
    masked_perturbed_image = (torch.mul((1-batch_masks), perturbed_image) + torch.mul(batch_masks, batch_images))
    #not necessarily needed to add mask as input to score model
    masked_perturbed_batch = torch.cat((masked_perturbed_image, batch_masks), dim=1)
    score = model_fn(masked_perturbed_batch, labels)
    image_score = score[:,0:1,:,:]
    mask_score = score[:,1:2,:,:]
    stds = sqrt_1m_alphas_cumprod[labels]
    #stds_inverse = sqrt_1m_alphas_cumprod[labels].pow(-1)
    #weighted losses
    stds = stds.view((batch_images.shape[0],1,1,1))
    image_losses = torch.square(torch.mul(stds, (torch.mul(stds, image_score) + noise)))
    masked_image_losses = torch.mul((1-batch_masks), image_losses)
    masked_image_losses = reduce_op(masked_image_losses.reshape(masked_image_losses.shape[0], -1), dim=-1)
    ml = torch.mean(torch.square(torch.subtract(batch_masks,mask_score)))
    print(ml)
    mask_loss = torch.mean(masked_image_losses)+torch.mean(torch.square(torch.subtract(batch_masks,mask_score)))
    return mask_loss
  
  return loss_fn


def get_step_fn(sde, train, optimize_fn=None,
                reduce_mean=False, continuous=True,
                likelihood_weighting=False, masked = True):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if (isinstance(sde, VPSDE)):
      if masked:
        loss_fn = get_masked_score_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean) 
      else:
        loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn