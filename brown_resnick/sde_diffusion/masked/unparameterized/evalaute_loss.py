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
score_model.load_state_dict(th.load(("trained_score_models/vpsde/model4_beta_min_max_01_20_random50_masks.pth")))
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
eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    masked = True)