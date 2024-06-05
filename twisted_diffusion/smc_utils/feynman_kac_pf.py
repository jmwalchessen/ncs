import torch
import numpy as np 
from .smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights
#M is Markov Chain and G is potential function, see chap 5 in Chopin
#In words, sample xt from M and proposal weight for resampling from G, if variation of weights
# are below an ess_threshold, then resample xt, then set xtp1 = xt and repeat for next time step
def smc_FK(M, G, resample_strategy, T, P,  ess_threshold, 
           verbose=False, verbose_notebook=False, log_xt_trace=False, extra_vals=None):
    """smc_FK is sequential Monte Carlo in the Feynman-Kac formulation.

    This is mean to exactly mirror the formulation in Chopin's book, except stated in reverse time.

    M & G have an extra input & output, "extra_vals".  This allows the possibility of saving redundant
    computation across M and G at each step.

    Args:
        M: intial/transition distribution.
            x_T, extra_vals = M(T, None, extra_vals, P=P) # Initial sample x_T ~ M_0(dt)
            x_t, extra_vals = M(t, x_{t+1}, extra_vals) if t<T
        G: potential function
            w_T, extra_vals = G(T, None, x_T, extra_vals) # Initial potential
            w_t, extra_vals = G(t, x_{t+1}, x_t, extra_vals) # subsequent potentials
        resample: resampling function
            log_w_t, resample_indices = resample(log_w_t) # performs resampling step (or not), resets weights
            to 0 after resamplings.
        T: final time step
        P: number of particles

    Returns:
        x_ts (shape [T, P, ...]), log_w, resample_indices_trace, ess_trace
    """
    #ess_threshold if you want to do adaptive resampling strategy
    #this means that if the weights have sufficient variation, then the ess is below the
    #ess_threshold and you resample the weights, otherwise you keep the weights as is
    #This allows you to resample weights with higher weights
    resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold, verbose=False)
    
    if log_xt_trace:
        xt_trace = [] 
    else: xt_trace = None 
    resample_indices_trace = []
    ess_trace = []
    log_w_trace = []
    log_proposal_trace = []
    log_potential_xt_trace = []
    log_potential_xtp1_trace = []
    log_p_trans_untwisted_trace = []
    xtp1 = None
    extra_vals = extra_vals if extra_vals else {} 
    log_wtp1 = 0 
    
    if verbose_notebook:
        from tqdm.notebook import trange
        time_range = trange(T, -1, -1)
    else:
        time_range = range(T, -1, -1)
        if verbose:
            from tqdm.auto import tqdm
            time_range = tqdm(time_range)

    for t in time_range:
        # Transition kernel and reweighting
        #also if t>1, then xt should be samples from normal with mean: xtm1_mean from TDS, 
        #the mean of the posterior q(x_t-1 | x_t, xhat(x_t,y,t))  and 
        #the variance (untwisted so variance of p_theta(x_tm1 | xt) I think which is sigma_t)
        xt, extra_vals = M(t=t, xtp1=xtp1, extra_vals=extra_vals, P=P)
        log_wt, log_proposal, log_potential_xt, log_potential_xtp1, log_p_trans_untwisted = G(t=t, xtp1=xtp1, xt=xt, extra_vals=extra_vals)
        log_proposal_trace.append(log_proposal)
        log_potential_xt_trace.append(log_potential_xt)
        log_potential_xtp1_trace.append(log_potential_xtp1)
        log_p_trans_untwisted_trace.append(log_p_trans_untwisted)
        log_w = log_wtp1 + log_wt        # unnormalized 

        # if xt and log_w are tensors detach them.  Otherwise convert to tensors
        if isinstance(xt, torch.Tensor):
            xt = xt.detach()
        else:
            xt = torch.tensor(xt)

        if isinstance(log_w, torch.Tensor):
            log_w = log_w.detach()
        else:
            log_w = torch.tensor(log_w)

        if log_xt_trace:
            xt_trace.append(xt.cpu())
        ess_trace.append(compute_ess_from_log_w(log_w).cpu())
        log_w_trace.append(log_w.cpu())

        assert log_w.shape == (P,), log_w.shape
        
        if t > 0:
            #don't do resampling of particles
            if ess_threshold == 0:
                resample_indices = torch.arange(P)
                log_wtp1 = log_w
            else:
                # resample (using something such as a multinomial)
                resample_indices, is_resampled = resample_fn(log_w)
                xt = xt[resample_indices]
                if is_resampled:
                    log_wtp1 = 0 # not accumulating 
                else: 
                    log_wtp1_normalized = normalize_log_weights(log_w, dim=0)
                    log_wtp1 = log_wtp1_normalized + np.log(P) 

            extra_vals[('resample_idcs', t)] = resample_indices
            
            resample_indices_trace.append(resample_indices)

        xtp1 = xt

    # Stack before returning
    if log_xt_trace:
        xt_trace = torch.stack(xt_trace, dim=0)
    ess_trace = torch.stack(ess_trace, dim=0)
    log_w_trace = torch.stack(log_w_trace, dim=0)
    if len(resample_indices_trace) > 0:
        resample_indices_trace = torch.stack(resample_indices_trace, dim=0)

    normalized_w = normalize_weights(log_w, dim=0) 

    return xt, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace, log_proposal_trace, log_potential_xt_trace, log_potential_xtp1_trace, log_p_trans_untwisted_trace
