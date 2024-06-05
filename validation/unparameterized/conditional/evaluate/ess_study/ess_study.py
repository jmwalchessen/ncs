import time
import torch as th
import numpy as np
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

def get_xstart_var(twisted_diffusion_model, t):
            
    #sigmasq_ = (1-alphas_cumprod_t) / alphas_cumprod_t
    #sigmasq_= 2*self.sigmas_cumprod[t-1]
    alphas_cumprod_t = twisted_diffusion_model.alphas_cumprod[t]
    sigmasq_ = twisted_diffusion_model.sigmas[t-1]
    return sigmasq_

def plot_unconditional_sigma(twisted_diffusion_model, figname):

    sigmas = [twisted_diffusion_model.sigmas[i] for i in range((twisted_diffusion_model.T-1), 0, -1)]
    fig, ax = plt.subplots(figsize = (10,10))
    plt.plot([i for i in range((twisted_diffusion_model.T-1),0,-1)], sigmas)
    plt.title("Unconditional Sigma")
    ax.xaxis.set_inverted(True)
    plt.savefig(figname)

def plot_unconditional_sigma_prod(twisted_diffusion_model, figname):

    sigmas = [twisted_diffusion_model.sigmas[i] for i in range((twisted_diffusion_model.T-1), 0, -1)]
    sigma_cumprod = [twisted_diffusion_model.sigmas_cumprod[i] for i in range((twisted_diffusion_model.T-1), 0, -1)]
    fig, ax = plt.subplots(figsize = (10,10))
    plt.plot([i for i in range((twisted_diffusion_model.T-1),0,-1)], sigma_cumprod)
    plt.title("Unconditional Sigma Cumulative Product")
    ax.xaxis.set_inverted(True)
    plt.savefig(figname)

def plot_transformed_sigma(twisted_diffusion_model, figname):

    sigmas = [twisted_diffusion_model.sigmas[i] for i in range((twisted_diffusion_model.T-1), 0, -1)]
    sigma_trans = [get_xstart_var(twisted_diffusion_model,t) for t in range((twisted_diffusion_model.T-1), 0, -1)]
    fig, ax = plt.subplots(figsize = (10,10))
    plt.plot([i for i in range((twisted_diffusion_model.T-1),0,-1)], sigma_trans)
    plt.title("Sigmasq_ 8")
    ax.xaxis.set_inverted(True)
    plt.savefig(figname)

plot_transformed_sigma(twisted_diffusion, ("resampled_indices/sigmasq_8.png"))


def max_median_min_weight(log_w_trace, T):

    w_trace = th.exp(log_w_trace)
    sum_w_trace = th.sum(w_trace, dim = 1)
    sum_w_trace = th.transpose(sum_w_trace.repeat(particles,1), dim0 = 0, dim1 = 1)
    normalized_w = th.div(input = w_trace, other = sum_w_trace)
    normalized_w = th.nan_to_num(normalized_w, nan=0.0)
    w_min_trace = [th.min(normalized_w[i,:]) for i in range(0, (T+1))]
    w_median_trace = [th.median(normalized_w[i,:]) for i in range(0, (T+1))]
    w_max_trace = [th.max(normalized_w[i,:]) for i in range(0, (T+1))]
    return w_min_trace, w_median_trace, w_max_trace

def max_median_min_prob(log_prob_trace, T):
    prob_min_trace = [float(th.min(log_prob_trace[i])) for i in range(0, (T+1))]
    prob_median_trace = [float(th.median(log_prob_trace[i])) for i in range(0, (T+1))]
    prob_max_trace = [float(th.max(log_prob_trace[i])) for i in range(0, (T+1))]
    return prob_min_trace, prob_median_trace, prob_max_trace

def probabilities_from_diffusion_call(twisted_diffusion_model, mask, ref_image, n, particles, ess_threshold,
                             figname):

    finalsamples, partobs, fullobs, m, reindices_trace, ess_trace, log_w_trace,\
    log_proposal_trace, log_potential_xt_trace, log_potential_xtp1_trace,\
    log_p_trans_untwisted_trace = twisted_diffusion_samples_per_call(twisted_diffusion_model,particles,
                                                                     mask, ref_image, n, ess_threshold)
    return log_proposal_trace, log_potential_xt_trace, log_p_trans_untwisted_trace

def get_sigmasq_trace(twisted_diffusion_model):

    sigmasq_trace = [get_xstart_var(twisted_diffusion_model, t) for t in range((twisted_diffusion_model.T-1), 0, -1)]
    return sigmasq_trace

def plot_unconditional_ess_plot(twisted_diffusion_model, score_model, n, particles, figname):

    uncondsamples, ptrans_untwisted_trace = twisted_diffusion_model.posterior_sample_with_p_mean_variance(score_model, particles)
    print(ptrans_untwisted_trace)
    prob_min_trace, prob_median_trace, prob_max_trace = max_median_min_prob(ptrans_untwisted_trace, (twisted_diffusion_model.T-2))
    fig, ax = plt.subplots(figsize = (5,5))
    plt.plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], prob_min_trace, "blue")
    plt.plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], prob_median_trace, "green")
    plt.plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], prob_max_trace, "purple")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Unconditional Prob p(xtm1 | xt)")
    ax.set_title("Unconditional Prob Trace per timestep")
    ax.xaxis.set_inverted(True)
    plt.legend(['min', 'max', 'median'])
    plt.savefig(figname)
    plt.clf()


def plot_resampling_ess_w_trace(twisted_diffusion_model, mask, ref_image, n, particles, ess_threshold,
                             figname, smallfigname):

    finalsamples, partobs, fullobs, m, reindices_trace, ess_trace, log_w_trace, log_proposal_trace, log_potential_xt_trace, log_potential_xtp1_trace, log_p_trans_untwisted_trace = twisted_diffusion_samples_per_call(twisted_diffusion_model,particles,
                                                                                            mask, ref_image, n, ess_threshold)
    sampled_particles = [th.unique(reindices_trace[i,:]).shape[0] for i in range(0,twisted_diffusion_model.T)]
    fig, ax = plt.subplots(ncols = 3, nrows = 3, figsize = (15,15))
    ax[0,0].plot([i for i in range(twisted_diffusion_model.T, 0, -1)], sampled_particles)
    ax[0,0].set_xlabel("Timestep")
    ax[0,0].set_ylabel("Unique Resampled Particles")
    ax[0,0].set_title("Resampled Particles per timestep (ess threshold = " + str(ess_threshold) +")")
    ax[0,0].xaxis.set_inverted(True)
    ax[0,0].set_ylim(0, (particles + 1))
    ax[0,1].plot([i for i in range((twisted_diffusion_model.T+1),0,-1)], ess_trace)
    ax[0,1].set_xlabel("Timestep")
    ax[0,1].set_ylabel("ESS")
    ax[0,1].set_title("ESS per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[0,1].set_ylim(0, (particles+1))
    ax[0,1].xaxis.set_inverted(True)
    min_w_trace, median_w_trace, max_w_trace = max_median_min_weight(log_w_trace, twisted_diffusion_model.T)
    ax[0,2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], min_w_trace, 'blue')
    ax[0,2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], max_w_trace, 'green')
    ax[0,2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], median_w_trace, 'purple')
    ax[0,2].set_xlabel("Timestep")
    ax[0,2].set_ylabel("Weight")
    ax[0,2].set_title("Weight Trace per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[0,2].set_ylim(0, 1)
    ax[0,2].xaxis.set_inverted(True)
    ax[0,2].legend(['min', 'max', 'median'])
    log_proposal_min_trace, log_proposal_median_trace, log_proposal_max_trace = max_median_min_prob(log_proposal_trace,
                                                                                                    twisted_diffusion_model.T)
    log_potential_min_trace, log_potential_median_trace, log_potential_max_trace = max_median_min_prob(log_potential_xt_trace,
                                                                                                    (twisted_diffusion_model.T-1))
    log_p_trans_untwisted_min_trace, log_p_trans_untwisted_median_trace, log_p_trans_untwisted_max_trace = max_median_min_prob(log_p_trans_untwisted_trace,
                                                                                                                             twisted_diffusion_model.T)
    log_potential_difference_trace = [log_potential_xt_trace[i-1]-log_potential_xt_trace[i] for i in range(1, 250)]
    log_potential_difference_min_trace, log_potential_difference_median_trace, log_potential_difference_max_trace = max_median_min_prob(log_potential_difference_trace,
                                                                                                                                         (twisted_diffusion_model.T-2))
    log_potential_difference_minus_proposal_trace = [log_potential_difference_trace[i] - log_proposal_trace[i] for i in range(0,249)]
    log_potential_difference_minus_proposal_min_trace, log_potential_difference_minus_proposal_median_trace, log_potential_difference_minus_proposal_max_trace = max_median_min_prob(log_potential_difference_minus_proposal_trace,
                                                                                                                                                                                    (twisted_diffusion_model.T-2))
    sigmasq_trace = get_sigmasq_trace(twisted_diffusion_model)
    ax[1,0].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], log_proposal_min_trace, 'blue')
    ax[1,0].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], log_proposal_max_trace, 'green')
    ax[1,0].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], log_proposal_median_trace, 'purple')
    ax[1,0].set_xlabel("Timestep")
    ax[1,0].set_ylabel("Log proposal (p(xtm1 | xt,y)), denom")
    ax[1,0].legend(['min', 'max', 'median'])
    ax[1,0].set_title("Log Proposal Trace")
    ax[1,0].xaxis.set_inverted(True)
    ax[1,1].plot([i for i in range((twisted_diffusion_model.T), 0, -1)], log_potential_min_trace, 'blue')
    ax[1,1].plot([i for i in range((twisted_diffusion_model.T), 0, -1)], log_potential_max_trace, 'green')
    ax[1,1].plot([i for i in range((twisted_diffusion_model.T), 0, -1)], log_potential_median_trace, 'purple')
    ax[1,1].set_xlabel("Timestep")
    ax[1,1].set_ylabel("Log potential (twisted_p(y|xt)), nom")
    ax[1,1].set_title("Log Potential Trace")
    ax[1,1].legend(['min', 'max', 'median'])
    ax[1,1].xaxis.set_inverted(True)
    ax[1,2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], log_p_trans_untwisted_min_trace, 'blue')
    ax[1,2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], log_p_trans_untwisted_max_trace, 'green')
    ax[1,2].plot([i for i in range((twisted_diffusion_model.T+1), 0, -1)], log_p_trans_untwisted_median_trace, 'purple')
    ax[1,2].set_xlabel("Timestep")
    ax[1,2].set_ylabel("Log Trans Untwisted (p(xtm1 | xt)) nom")
    ax[1,2].set_ylim(-100,2500)
    ax[1,2].xaxis.set_inverted(True)
    ax[1,2].set_title("Log Trans Untwisted Trace per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[1,2].legend(['min', 'max', 'median'])

    #plot sigmasq_ and log_potential_xtm1-log_potential_xt
    ax[2,0].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], log_potential_difference_min_trace, 'blue')
    ax[2,0].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], log_potential_difference_max_trace, 'green')
    ax[2,0].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], log_potential_difference_median_trace, 'purple')
    ax[2,0].set_xlabel("Timestep")
    ax[2,0].set_ylabel("Log Potential Difference")
    ax[2,0].set_ylim(-100,100)
    ax[2,0].legend(['min', 'max', 'median'])
    ax[2,0].set_title("Log Potential Difference  Trace")
    ax[2,0].xaxis.set_inverted(True)

    ax[2,1].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], sigmasq_trace, 'blue')
    ax[2,1].set_xlabel("Timestep")
    ax[2,1].set_ylabel("Sigmasq (sigma tilde in twistedp(y|xt))")
    ax[2,1].set_title("Sigmasq Trace")
    ax[2,1].xaxis.set_inverted(True)

    ax[2,2].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], log_potential_difference_minus_proposal_min_trace, 'blue')
    ax[2,2].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], log_potential_difference_minus_proposal_max_trace, 'green')
    ax[2,2].plot([i for i in range((twisted_diffusion_model.T-1), 0, -1)], log_potential_difference_minus_proposal_median_trace, 'purple')
    ax[2,2].set_xlabel("Timestep")
    ax[2,2].set_ylabel("Log Potential Difference Minus Proposal")
    ax[2,2].set_ylim(-2000,2000)
    ax[2,2].legend(['min', 'max', 'median'])
    ax[2,2].set_title("Log Potential Difference Minus Proposal  Trace")
    ax[2,2].xaxis.set_inverted(True)

    plt.savefig(figname)
    plt.clf()

    fig, ax = plt.subplots(ncols = 3, nrows = 3, figsize = (15,15))
    ax[0,0].plot([i for i in range(50, 0, -1)], sampled_particles[200:250])
    ax[0,0].set_xlabel("Timestep")
    ax[0,0].set_ylabel("Unique Resampled Particles")
    ax[0,0].set_title("Resampled Particles per timestep (ess threshold = " + str(ess_threshold) +")")
    ax[0,0].xaxis.set_inverted(True)
    ax[0,0].set_ylim(0, (particles + 1))
    ax[0,1].plot([i for i in range(50,0,-1)], ess_trace[200:250])
    ax[0,1].set_xlabel("Timestep")
    ax[0,1].set_ylabel("ESS")
    ax[0,1].set_title("ESS per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[0,1].set_ylim(0, (particles+1))
    ax[0,1].xaxis.set_inverted(True)
    min_w_trace, median_w_trace, max_w_trace = max_median_min_weight(log_w_trace, twisted_diffusion_model.T)
    ax[0,2].plot([i for i in range(50, 0, -1)], min_w_trace[200:250], 'blue')
    ax[0,2].plot([i for i in range(50, 0, -1)], max_w_trace[200:250], 'green')
    ax[0,2].plot([i for i in range(50, 0, -1)], median_w_trace[200:250], 'purple')
    ax[0,2].set_xlabel("Timestep")
    ax[0,2].set_ylabel("Weight")
    ax[0,2].set_title("Weight Trace per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[0,2].set_ylim(0, 1)
    ax[0,2].xaxis.set_inverted(True)
    ax[0,2].legend(['min', 'max', 'median'])
    log_proposal_min_trace, log_proposal_median_trace, log_proposal_max_trace = max_median_min_prob(log_proposal_trace,
                                                                                                    twisted_diffusion_model.T)
    log_potential_min_trace, log_potential_median_trace, log_potential_max_trace = max_median_min_prob(log_potential_xt_trace,
                                                                                                    (twisted_diffusion_model.T-1))
    log_p_trans_untwisted_min_trace, log_p_trans_untwisted_median_trace, log_p_trans_untwisted_max_trace = max_median_min_prob(log_p_trans_untwisted_trace,
                                                                                                                             twisted_diffusion_model.T) 
    ax[1,0].plot([i for i in range(51, 0, -1)], log_proposal_min_trace[200:251], 'blue')
    ax[1,0].plot([i for i in range(51, 0, -1)], log_proposal_max_trace[200:251], 'green')
    ax[1,0].plot([i for i in range(51, 0, -1)], log_proposal_median_trace[200:251], 'purple')
    ax[1,0].set_xlabel("Timestep")
    ax[1,0].set_ylabel("Log proposal (p(xtm1 | xt,y)), denom")
    ax[1,0].legend(['min', 'max', 'median'])
    ax[1,0].set_title("Log Proposal Trace")
    ax[1,0].xaxis.set_inverted(True)
    ax[1,1].plot([i for i in range(50, 0, -1)], log_potential_min_trace[200:250], 'blue')
    ax[1,1].plot([i for i in range(50, 0, -1)], log_potential_max_trace[200:250], 'green')
    ax[1,1].plot([i for i in range(50, 0, -1)], log_potential_median_trace[200:250], 'purple')
    ax[1,1].set_xlabel("Timestep")
    ax[1,1].set_ylabel("Log potential (twisted_p(y|xt)), nom")
    ax[1,1].set_title("Log Potential Trace")
    ax[1,1].legend(['min', 'max', 'median'])
    ax[1,1].xaxis.set_inverted(True)
    ax[1,2].plot([i for i in range(51, 0, -1)], log_p_trans_untwisted_min_trace[200:251], 'blue')
    ax[1,2].plot([i for i in range(51, 0, -1)], log_p_trans_untwisted_max_trace[200:251], 'green')
    ax[1,2].plot([i for i in range(51, 0, -1)], log_p_trans_untwisted_median_trace[200:251], 'purple')
    ax[1,2].set_xlabel("Timestep")
    ax[1,2].set_ylabel("Log Trans Untwisted (p(xtm1 | xt)) nom")
    ax[1,2].set_ylim(-100,2500)
    ax[1,2].xaxis.set_inverted(True)
    ax[1,2].set_title("Log Trans Untwisted Trace per timestep (ess threshold = " + str(ess_threshold) + ")")
    ax[1,2].legend(['min', 'max', 'median'])

    #plot sigmasq_ and log_potential_xtm1-log_potential_xt
    ax[2,0].plot([i for i in range(51, 0, -1)], log_potential_difference_min_trace[200:251], 'blue')
    ax[2,0].plot([i for i in range(51, 0, -1)], log_potential_difference_max_trace[200:251], 'green')
    ax[2,0].plot([i for i in range(51, 0, -1)], log_potential_difference_median_trace[200:251], 'purple')
    ax[2,0].set_xlabel("Timestep")
    ax[2,0].set_ylabel("Log Potential Difference")
    ax[2,0].set_ylim(-100,100)
    ax[2,0].legend(['min', 'max', 'median'])
    ax[2,0].set_title("Log Potential Difference  Trace")
    ax[2,0].xaxis.set_inverted(True)

    ax[2,1].plot([i for i in range(51, 0, -1)], sigmasq_trace[200:250], 'blue')
    ax[2,1].set_xlabel("Timestep")
    ax[2,1].set_ylabel("Sigmasq (sigma tilde in twistedp(y|xt))")
    ax[2,1].set_title("Sigmasq Trace")
    ax[2,1].xaxis.set_inverted(True)

    ax[2,2].plot([i for i in range(51, 0, -1)], log_potential_difference_minus_proposal_min_trace[200:251], 'blue')
    ax[2,2].plot([i for i in range(51, 0, -1)], log_potential_difference_minus_proposal_max_trace[200:251], 'green')
    ax[2,2].plot([i for i in range(51, 0, -1)], log_potential_difference_minus_proposal_median_trace[200:251], 'purple')
    ax[2,2].set_xlabel("Timestep")
    ax[2,2].set_ylabel("Log Potential Difference Minus Proposal")
    ax[2,2].set_ylim(-2000,2000)
    ax[2,2].legend(['min', 'max', 'median'])
    ax[2,2].set_title("Log Potential Difference Minus Proposal  Trace")
    ax[2,2].xaxis.set_inverted(True)

    plt.savefig(smallfigname)
    plt.clf()



minX = -10
maxX = 10
minY = -10
maxY = 10
variance = .4
lengthscale = 1.6
number_of_replicates = 1
seed_value = 43234
particles = 500
ess_threshold = .5
ref_image = ((generate_gaussian_process(minX, maxX, minY, maxY, n, variance, lengthscale,
                                          number_of_replicates, seed_value))[1]).reshape((1,n,n))
ref_image = th.from_numpy(ref_image).to(device)

plot_unconditional_sigma(twisted_diffusion, ("resampled_indices/unconditional_sigma.png"))
plot_resampling_ess_w_trace(twisted_diffusion, mask, ref_image, n, particles, ess_threshold,
                            ("resampled_indices/ess_particles_" + str(particles) + "_var_type_sigmasq8_ess_" + 
                             str(ess_threshold) + ".png"),
                             ("resampled_indices/ess_particles_" + str(particles) + "_var_type_sigmasq8_ess_" + 
                             str(ess_threshold) + "_50_0.png"))


    
