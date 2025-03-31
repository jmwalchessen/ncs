from helper_functions import *
import zeus
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices = [0,1,2,3])


def produce_ncs_realization_with_variables():

    beta_min = .01
    beta_max = 20
    N = 1000
    number_of_replicates = 1
    n = 32
    process_type = "brown"
    device = "cuda:0"
    range_value = 3.
    smooth_value = 1.5
    seed_value = int(torch.random.randint(0,100000,1))
    p = .05
    mask = (torch.bernoulli(p*torch.ones((1,1,n,n)))).to(device)
    ref_image = np.log(generate_brown_resnick_process(range_value, smooth_value, seed_value, number_of_replicates, n))
    ref_img = th.from_numpy(ref_img.reshape((1,1,n,n)))
    y = ((th.mul(mask, ref_img)).to(device)).float()
    vpsde = load_sde(beta_min, beta_max, N)
    mode = "eval"
    model_name = "model4_beta_min_max_01_20_random01525_smooth_1.5_range_3_channel_mask.pth"
    score_model = load_score_model(process_type, model_name, mode)
    monitor.begin_window("eval")
    x = posterior_sample_with_p_mean_variance_via_mask(vpsde, score_model, device, mask,
                                                       y, n, num_samples)
    mes = monitor.end_window("eval")
    print(mes.time)
    print(mes.total_energy)
