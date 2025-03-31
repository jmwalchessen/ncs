from append_directories import *
eval_folder = append_directory(2)
import sys
sys.path.append(eval_folder)
from helper_functions import *
import zeus
from zeus.monitor import ZeusMonitor
import torch
import numpy as np
import matplotlib.pyplot as plt



def count_parameters():

    mode = "eval"
    process_type = "brown"
    model_name = "model2/model2_beta_min_max_01_20_range_1_2_smooth_1.6_random50_log_parameterized_mask.pth"
    score_model = load_score_model(process_type, model_name, mode)
    pytorch_total_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    print(pytorch_total_params)


count_parameters()