import torch as th
import numpy as np
from data_generation_on_the_fly import *
import matplotlib.pyplot as plt
import seaborn as sns
from configs.vp import ncsnpp_config as vp_ncsnpp_config



vp_ncsnpp_configuration = vp_ncsnpp_config.get_config()
vpconfig = vp_ncsnpp_configuration
data_draw = 0
number_of_random_replicates = 512
random_missingness_percentages = [0,.5]
number_of_eval_random_replicates = 250
batch_size = 512
eval_batch_size = 250
range_value = 1.6
smooth_value = 1.6
seed_values = (234234, 3234)
n = 32
"""
trainmaxminfile = "trained_score_models/vpsde/model8_train_logminmax.npy"
train_dataloader, eval_dataloader = get_training_and_evaluation_mask_and_image_datasets_per_mask(data_draw, number_of_random_replicates,
                                                                                                 random_missingness_percentages,
                                                                                                 number_of_eval_random_replicates,
                                                                                                 batch_size, eval_batch_size, range_value,
                                                                                                 smooth_value, seed_values,
                                                                                                 n, trainmaxminfile)

train_iterator = iter(train_dataloader)
batch = get_next_batch(train_iterator, vpconfig)
"""

def global_boundary_process(images, minvalue, maxvalue):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - .5
    log01cs = 6*log01c
    return log01cs

def global_quantile_boundary_process(images, minvalue, maxvalue, quantvalue01):

    log01 = (images-minvalue)/(maxvalue-minvalue)
    log01c = log01 - quantvalue01
    log01cs = 6*log01c
    return log01cs

def batch_density(batch_images, matrixindex, figname):

    marginal_density = batch_images[:,0,matrixindex[0],matrixindex[1]]
    fig, ax = plt.subplots(1)
    sns.kdeplot(marginal_density)
    plt.savefig(figname)


matrixindex = (8,8)
figname = "loglocalboundcenterdensity_batch_512.png"
batch = log_transformation((np.load("brown_resnick_samples_512.npy")).reshape((510,1,n,n)))
minvalue = float(np.min(batch))
maxvalue = float(np.max(batch))
quantvalue01 = float(np.quantile((batch-minvalue/(maxvalue-minvalue)), [.4]))
print(quantvalue01)
batch = global_quantile_boundary_process(batch, minvalue, maxvalue, quantvalue01)
#batch = log_and_boundary_process(np.load("brown_resnick_samples_512.npy")).reshape((510,1,n,n))

for i in range(0, 500):

    print(np.max(batch[i,:,:,:]))
    print(np.min(batch[i,:,:,:]))