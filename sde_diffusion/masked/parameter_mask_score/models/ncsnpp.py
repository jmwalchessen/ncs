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

# pylint: skip-file
import sys
import os
sys.path.append(os.path.dirname(__file__))
from model_utils import *
print("a")
from layers import *
from layerspp import *
print("b")
from normalization import *
import torch.nn as nn
import functools
import torch
import numpy as np
sde_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append((sde_folder + "/configs/vp"))
from ncsnpp_config import *



ResnetBlockDDPM = ResnetBlockDDPMpp
ResnetBlockBigGAN = ResnetBlockBigGANpp
Combine = Combine
conv3x3 = conv3x3
conv1x1 = conv1x1
get_act = get_act
get_normalization = get_normalization
default_initializer = default_init
#from append_directories import *
#import sys
#home_folder = append_directory(1)
#sys.path.append((home_folder))
#from configs.vp import ncsnpp_config

@register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    #get activation function which is SiLU in this case i.e. swish
    self.act = act = get_act(config)
    #I'm not sure if sigmas are necessary (I think only for SMLD models and VESDEs?)
    self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))
    #nf is for embedding I think (128)
    self.nf = nf = config.model.nf
    #(1,2,2,2)
    ch_mult = config.model.ch_mult
    #4 residual blocks
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    #(16,), not sure what this does yet
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    #.1
    dropout = config.model.dropout
    #True
    resamp_with_conv = config.model.resamp_with_conv
    #4 bc ch_mult = (1,2,2,2)?
    self.num_resolutions = num_resolutions = len(ch_mult)
    #image_size for cifar10 is 32 so prob need to change this for img_size = 25 bc result is [32,16,8,4] or
    #something around there
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
    #true
    self.conditional = conditional = config.model.conditional  # noise-conditional
    #True, filter, (upfirdn = upsample filter and downsample)
    fir = config.model.fir
    #[1,3,3,1]
    fir_kernel = config.model.fir_kernel
    #True (when doing adding identity i.e. residual part, scale by sqrt 2)
    self.skip_rescale = skip_rescale = config.model.skip_rescale
    #use biggan i.e. ResnetBlockBigGANpp from layerspp I think
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    #none (not sure what this is for)
    self.progressive = progressive = config.model.progressive.lower()
    #residual
    self.progressive_input = progressive_input = config.model.progressive_input.lower()
    #positional (not fourier)
    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    #0
    init_scale = config.model.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    #sum instead of concatenating
    combine_method = config.model.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      assert config.training.continuous, "Fourier features are only used for continuous training."

      modules.append(GaussianFourierProjection(
        embedding_size=nf, scale=config.model.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      #so embed_dim = 128
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      #linear layer (128, 4*128)
      modules.append(nn.Linear(embed_dim, nf * 4))
      #might use variance scaling I'm not sure, function is in layers.py
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      #linear layer (4*128, 4*128)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      #bias is initially zero
      nn.init.zeros_(modules[-1].bias)

      #so far in modules list, is two linear layers with certain initialized weights and biases 

    #self attention block
    AttnBlock = functools.partial(AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)
    #Upsample block
    UpsampleBlock = functools.partial(Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    #Not applicable
    if progressive == 'output_skip':
      self.pyramid_upsample = UpsampleBlock(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(UpsampleBlock,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)
    #downsample block
    DownsampleBlock = functools.partial(Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
    #progressive_input = residual
    if progressive_input == 'input_skip':
      self.pyramid_downsample = DownsampleBlock(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(DownsampleBlock,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)


    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)
      CondResnetBlock = functools.partial(CondResnetBlockBigGANpp,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    #num_channels = 3
    channels = config.data.num_channels
    if progressive_input != 'none':
      input_pyramid_ch = channels

    #modules = [linear(nf, 4*nf), linear(4*nf, 4*nf), conv3x3(channels = 3, nf,
    # kernel = 3, stride = 1, padding = 1)]
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    #num_resolutions = 4
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        #ch_mult = [1,2,2,2] and out_ch = [nf,2*nf,2*nf,2*nf]
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        #all_resolutions = [32,16,8,4] or something like that, attn_resolutions = (16,)
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(DownsampleBlock(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)
      
    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))
    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(CondResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(UpsampleBlock(in_ch=in_ch))
        else:
          modules.append(CondResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = get_timestep_embedding(timesteps, self.nf)

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if not self.config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h





  
