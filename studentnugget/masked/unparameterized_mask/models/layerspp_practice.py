# coding=utf-8
# Copyright 2020 The Google Research Authors
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
"""Layers for defining NCSN++.
"""
from layers import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


#My addition for embedding parameter
#embedding for class, text etc (is a single layer nn), embed_dim is output dim
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    #tensor to embed must be same size as input_dim or a multiple of input_dim
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
    
#added helper functions to help with maskembedfc
class ConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: tuple) -> None:
        super().__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = 1, padding = 0)
        self.conv1 = nn.Sequential(
            conv1,
            #try with and without batch normalization
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)
        
class MaxPool2dlayer(nn.Module):

    #kernel size is int i.e. assumed to be square (int, int) or tupe (int, int)
    #padding is int or tuple of int
    def __init__(self, kernel: tuple, padding: tuple):
        super(MaxPool2dlayer, self).__init__()
        self.MaxPoolLayer = nn.MaxPool2d(dilation = 1, kernel_size = kernel,
                                         stride = kernel, padding = padding)

    def forward(self, x):
        return self.MaxPoolLayer(x)
    
class AvgPool2dlayer(nn.Module):

    #kernel size is int i.e. assumed to be square (int, int) or tupe (int, int)
    #padding is int or tuple of int
    def __init__(self, kernel: tuple, padding: tuple):
        super(AvgPool2dlayer, self).__init__()
        self.AvgPoolLayer = nn.AvgPool2d(kernel_size = kernel,
                                         stride = kernel, padding = padding)

    def forward(self, x):
        return self.AvgPoolLayer(x)
    
class MaskEmbedFC(nn.Module):
  "Embed mask to vector"

  def __init__(self):
        super(MaskEmbedFC, self).__init__()

        layers = [#ConvLayer(in_channels = 1, out_channels = 4, kernel = (3,3)),
                  AvgPool2dlayer(kernel = (2,2), stride = 1, padding = (0,0)),
                  #ConvLayer(in_channels = 4, out_channels = 8, kernel = (3,3)),
                  #AvgPool2dlayer(kernel = (2,2), padding = (1,1)),
                  #ConvLayer(in_channels = 8, out_channels =8, kernel = (3,3)),
                  #AvgPool2dlayer(kernel = (2,2), padding = (1,1)),
                  nn.Flatten()]
        self.layers = layers
        self.model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)
  
