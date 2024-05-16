# Adapted from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/ 

import enum
import math 
import numpy as np
import torch as th
from torch.distributions import Normal 
import sys
import os
image_exp_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append((image_exp_path + "/smc_utils"))
import dist_util
import unet
import time

#betas are from .0001 to .02 by increments of x for length 1000
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    



def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


def _gaussian_sample(mean, variance=None, scale=None, sample_shape=th.Size([]), return_logprob=False):
    if scale is None:
        assert variance is not None 
        scale = th.sqrt(variance)

    normal_dist = Normal(loc=mean, scale=scale)
    samples = normal_dist.sample(sample_shape)
    if return_logprob:
        log_prob = normal_dist.log_prob(samples)
        return samples, log_prob 
    return samples

class VPSDEDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    #this is probably from get_named_beta_schedule
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        particle_base_shape,
        rescale_timesteps=False,
        conf=None,
        probability_flow = False,
        device = "cuda:0"
    ):
        
        #my additions
        self.probability_flow = probability_flow
        self.device = "cuda:0"
        self.particle_base_shape = particle_base_shape # (C, H, W) for image 

        #original additions minus a few irrelevant ones
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.beta_0 = betas[0]
        self.beta_1 = betas[(self.num_timesteps-1)]
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas = alphas 
        #sigma_t = sqrt(1-alpha_t)=sqrt(betas)
        self.sigmas = np.sqrt(betas)
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        #(σ̅ₜ)² = 1−α̅ₜ
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        #σ̅ₜ = √{1−α̅ₜ}
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sigmas_cumprod = self.sqrt_one_minus_alphas_cumprod
        #(√{α̅ₜ})⁻¹
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0/self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        #for posterior_q i.e. q(xtm1 | xt, x0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
    #equation from sde (originally called marginal_prob), compute mean and variance of q(x_{t} | x_{0},t)
    def q_mean_variance(self, x_start, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta[1] - self.beta[0]) - 0.5 * t * self.beta[0]
        mean = th.exp(log_mean_coeff[:, None, None, None]) * x_start
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
        variance = th.square(std)
        log_variance = th.log(variance)
        return mean, variance, log_variance
    
    #sample from q(x_t | x_0)=N(sqrt alphabart*x_0, (1-alphabart)*I) by number of particles also give log prob
    #of these samples, replace with q_mean_variance which computed mean and variance of q(x_{t}|x_{0}) via sde_lib
    #way
    def q_sample(self, x_start, t, noise=None, num_particles=16, return_logprob=False):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        assert noise is None 
        #mean = self.sqrt_alphas_cumprod[t] * x_start 
        #std deviation is scale
        #scale = self.sqrt_one_minus_alphas_cumprod[t] * x_start 
        mean, scale = self.q_mean_variance(x_start, t)
        # return samples, logprob (if required)
        return _gaussian_sample(mean, scale=scale,
                                sample_shape=th.Size([num_particles]), return_logprob=return_logprob)
    
    #q(x_{t+1} | x_{t})=N(sqrt(alpha_tp1)*x_t, (1-alpha_tp1)) see p 2 of ddpm Ho (this can stay fixed because 
    #q(x_{t+1} | x_{t})=N(sqrt(1-beta_{t+1})*x_{t}, beta_{t+1}I) from sde paper p. 5 and beta_{t+1}=1-alpha_{t+1}
    def q_sample_tp1_given_t(self, x_t, t):
        """
        sample xtp1 given xt for t in [0, T-1] 
        """
        alphas_tp1 = self.alphas[t]  # due to python indexing
        mean = np.sqrt(alphas_tp1) * x_t
        scale = np.sqrt(1-alphas_tp1)
        #scale is std dev (doesn't include x_t)
        return mean + scale * th.randn_like(x_t)
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        #see p 3 of DDPM Ho paper
        posterior_mean = self.posterior_mean_coef1[t] * x_start \
            + self.posterior_mean_coef2[t] * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

        #sample from q(x_{t+1}|x_t,x_0) given q_posterior_mean_variance
    def q_posterior_sample(self, x_start, x_t, t, return_logprob=False, t_init=False):

        posterior_mean, posterior_variance, posterior_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_t, t)
        if t_init: 
            out = (posterior_mean, th.zeros_like(posterior_mean))
        else:
            out =  _gaussian_sample(mean=posterior_mean, variance=posterior_variance, return_logprob=return_logprob)
        return out 

    def sde(self, x, t):
        #drift = f, diffusion = g
        beta_t = (th.tensor([self.beta_0 + t * (self.beta_1 - self.beta_0)])).to(self.device)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = th.sqrt(beta_t)
        return drift, diffusion
    
    #get forward process f and g                                                 
    def forward_discretize(self, x_t, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.num_timesteps
        drift, diffusion = self.sde(x_t, t)
        f = drift * dt
        G = diffusion * th.sqrt(th.tensor(dt, device=self.device))
        return f, G
    
    def forward_discretize_step(self, x_t, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
            f, G
        """
        f,G = self.forward_discretize(x_t,t)
        z = th.randn_like(x_t)
        x_tp1 = x_t + f + G*z
        return x_tp1
    
    def forward_process(self, x0):

        xt = x0
        for t in range(0, self.num_timesteps, 1):
            xt = self.forward_discretize_step(xt, t)
        xT = xt
        return xT

    
    def reverse_discretize_step(self, score_model, x_t, t):

        f, G = self.forward_discretize(x_t, t)
        with th.no_grad():
            score = score_model(x_t, t)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = th.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G
    #scaling timesteps to something in [0,1000]
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t 
    
    #mean and variance of p(x_{t-1} | x_t), not sure if predict_x0 is necessary 
    def p_mean_variance(self, score_model, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):

        rev_f, rev_G = self.reverse_discretize_step(score_model, x_t, t)
        p_mean = x_t-rev_f
        p_std = rev_G
        if (len(rev_G.shape) > 1):
            p_variance = th.matmul(th.transpose(rev_G, 0, 1), rev_G)
        else:
            p_variance = rev_G**2
        p_log_variance = th.log(p_variance)
        #not sure if this is correct way to predict epsilon since nn outputs
        #score not eps
        pred_eps = _gaussian_sample(mean = p_mean, variance = p_variance)
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=pred_eps)
        return {
            "mean": p_mean,
            "variance": p_variance,
            "log_variance": p_log_variance,
            "pred_xstart": pred_xstart
        }
    
    def p_mean_variance_from_score(self, score_model, x_t, t, clip_denoised = True,
                                   denoised_fn = None, model_kwargs = None):
        
        with th.no_grad():
            score = score_model(x_t, th.tensor([t]).to(self.device))
        #from eq in appendix a of Trippe paper or my spring 2024 notes eq 16
        #p_mean = ((th.sqrt(th.tensor(self.alphas[t]))*x_t) + 
        #(th.square(th.tensor(self.sigmas[t]))*score))
        #p_variance = th.square(th.tensor(self.sigmas[t]))
        #p_log_variance = th.log(p_variance)
        #see my spring 2024 notes eq 7 (Brian and Mikael meeting 3/29) 
        #for how to get DDPM prediction from score for VPSDE


        scaling_term = (th.sqrt(th.tensor(self.alphas_cumprod[t])))**(-1)
        pred_xstart = scaling_term*(((th.tensor(self.sigmas_cumprod[t]))**2)*score+x_t)
        #from eqs in section 2.2 of Song's score-based generation paper, the eqs
        # in Trippe's appendix did not work as expected
        p_mean = (1/th.sqrt(th.tensor(self.alphas[t])))*(x_t +
                                      th.square(th.tensor(self.sigmas[t]))*score)
        p_variance = (th.square(th.tensor(self.sigmas[t])))*th.ones_like(x_t)
        p_log_variance = th.log(p_variance)
        return {
            "mean": p_mean,
            "variance": p_variance,
            "log_variance": p_log_variance,
            "pred_xstart": pred_xstart
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * eps
        )

    def _predict_xstart_from_score(self, score_model, x_t, t):

        with th.no_grad():
            score = score_model(x_t, t)
        #(σ̅ₜ)² = 1−α̅ₜ
        squared_prod_sigma_t = self.one_minus_alphas_cumprod[t]
        #(√{α̅ₜ})⁻¹
        inner_term = (squared_prod_sigma_t*score+x_t)
        x0_hat = squared_prod_sigma_t*inner_term
        return x0_hat


        
    #scaling timesteps to something in [0,1000]
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        score_model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = th.randn_like(x)

        #p_mean_variance returns model_mean ()
        out = self.p_mean_variance_from_score(
            score_model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x[0].shape) - 1)))
        )  # no noise when t == 0

        #not sure this is necessary to take log and exp but this is how the original code
        #does it
        std = th.exp(0.5 * out["log_variance"])
        #std = th.sqrt(out['variance'])
        sample = out["mean"] + nonzero_mask * \
            std * noise
        #no pred_x_start as there is in original code
        return {"sample": sample, 
                "mean": out['mean'], 
                "std": std}
    
    def posterior_sample_with_p_mean_variance(self, score_model, P, clip_denoised=True, denoised_fn=None,
                                              model_kwargs=None):
        
        xT = (th.randn(P, *self.particle_base_shape)).to(self.device)
        xt = xT
        for t in range((self.num_timesteps-1),0,-1):
            timestep = (th.tensor([t])).to(self.device)
            xt = (self.p_sample(score_model, xt, timestep, clip_denoised=True,
                                denoised_fn=None, model_kwargs=None))['sample']
        x0 = xt
        return x0

    


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, int):
        section_counts = [section_counts]
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(VPSDEDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, original_num_steps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = original_num_steps

        base_diffusion = VPSDEDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t

class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
    

class SMCDiffusion(SpacedDiffusion):
    """A diffusion model that supports FK_SMC sampling"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.T = len(self.use_timesteps)
        #particle_base_shape # (C, H, W) for image
        self.particle_base_dims = [-(i+1) for i in range(len(self.particle_base_shape))] 
        self.clear_cache()

    def clear_cache(self):
        self.cache = {} 

    def ref_sample(self, P, device):
        return th.randn(P, *self.particle_base_shape).to(device)
    
    def ref_log_density(self, samples):
        return Normal(loc=0, scale=1.0).log_prob(samples)
    
    def M(self, t, xtp1, extra_vals, P, model, device, **kwargs):
        raise NotImplementedError
    
    def G(self, t, xtp1, xt, extra_vals, model, debug=False, debug_info=None, **kwargs):
        raise NotImplementedError 

    def p_trans_model(self, xtp1, t, model, clip_denoised, model_kwargs):
        # compute mean and variance of p(x_t|x_{t+1}) under the model
        out = self.p_mean_variance_from_score(model, x_t=xtp1, t=t, clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs)
        
        return {
            "mean_untwisted": out['mean'], 
            "var_untwisted": out['variance'], 
            "pred_xstart": out['pred_xstart']
        }
    
    def set_measurement_cond_fn(self, measurement_cond_fn):
        self.measurement_cond_fn = measurement_cond_fn 

    def set_measurement(self, measurement):
        self.measurement = measurement 

    def _sample_forward(self, x0, batch_size=None):
        # generate sample trajectory [x0, x1, .., xt]
        if batch_size is not None:
            new_shape = (batch_size,) + x0.shape  # create new shape tuple
            x0 = x0.expand(*new_shape)

        xts_forward = th.empty(self.T+1, *x0.shape) # (T+1, *img_shape) or (T+1, bsz, *img_shape)
        xts_forward[0] = x0.cpu() 

        xt = x0 
        for t in range(self.T):
            xt = self.q_sample_tp1_given_t(xt, t) 
            xts_forward[t+1] = xt.cpu() 

        return xts_forward





