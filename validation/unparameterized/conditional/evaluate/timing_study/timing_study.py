import time
import torch as th
import matplotlib.pyplot as plt
import os
import sys
from append_directories import *
data_generation_folder = (append_directory(3) + "/generate_data")
sys.path.append(data_generation_folder)
from twisted_diffusion_data_generation_functions import *
from generate_true_conditional_samples import *

n = 32
device = "cuda:0"
mask = (th.bernoulli(input = .5*th.ones((1,n,n)), out = th.ones((1,n,n)))).to(device)
mask = mask.to(bool)

    
