import matplotlib.pyplot as plt
import numpy as np

def load_mcmc_data(range_value, smooth, m, nrep, irep):

    ref_folder = "data/mcmc_convergence"
    condsims = np.load((ref_folder + "/fcs_images_burnin_1000_center_mask_obs_" +
                        str(m) + "_range_" + str(range_value) + "_smooth_"
                        + str(smooth) + "_nugget_1e5_" + str(nrep) + "_"
                        + str(irep) + ".npy"))
    
    return condsims


def plot_mcmc_convergence(range_value, smooth, m, nrep, missing_index, irep, figname):

    condsims = load_mcmc_data(range_value, smooth, m, nrep, irep)
    mcmc_chain = np.log(condsims[:,missing_index])
    plt.plot(range(0,nrep), mcmc_chain)
    plt.savefig(figname)
    plt.clf()

def plot_mcmc_convergences(range_value, smooth, m, nrep, missing_indices, irep, basic_figname):

    for missing_index in missing_indices:
        figname = (basic_figname + "_" + str(missing_index) + ".png")
        plot_mcmc_convergence(range_value, smooth, m, nrep, missing_index, irep, figname)





range_value = 3.
smooth = 1.5
m = 1
nrep = 500
n = 32
missing_indices = [i for i in range(0,((n**2)-m))]
irep = 1
basic_figname = "data/mcmc_convergence/visualizations/obs1/mcmc_chain_burnin_1000_center_mask_obs_" + str(m) + "_range_3.0_smooth_1.5_nugget_1e5_500_1_missing_index"
plot_mcmc_convergences(range_value, smooth, m, nrep, missing_indices, irep, basic_figname)