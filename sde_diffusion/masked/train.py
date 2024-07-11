import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from data_generation_on_the_fly import *
from models.ema import ExponentialMovingAverage
from models.ncsnpp import *
import losses
import sde_lib
from configs.vp import ncsnpp_config as vp_ncsnpp_config
from configs.ve import ncsnpp_config as ve_ncsnpp_config
import matplotlib.pyplot as plt

vp_ncsnpp_configuration = vp_ncsnpp_config.get_config()
ve_ncsnpp_configuration = ve_ncsnpp_config.get_config()
vpconfig = vp_ncsnpp_configuration
veconfig = ve_ncsnpp_configuration

ncsn = NCSNpp(vpconfig)