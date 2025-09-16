import numpy as np
import matplotlib.pyplot as plt
from append_directories import *
from mpl_toolkits.axes_grid1 import ImageGrid
import torch as th

evaluation_folder = append_directory(2)


def visualize_ks_statistic_multiple_ranges(obs):

    figname = "figures/paper_fcs_vs_true_ks_obs_"+str(obs) + "_range_1_5_smooth_1.5_nugget_1e5.png"
    n = 32
    nrep = 4000
    range_values = [float(i) for i in range(1,6)]
    ks_statistics = np.load((evaluation_folder + "/extremal_coefficient_and_high_dimensional_metrics/data/fcs/ks_gof_statistic_obs_1_7_range_1_5_smooth_1.5_nugget_1e5.npy"))
    fig = plt.figure(figsize=(10,2))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 5),  # creates 2x2 grid of Axes
                 axes_pad=0.1,  # pad between Axes in inch.
                 cbar_mode="single"
                 )

    for i, ax in enumerate(grid):

        im = ax.imshow(ks_statistics[i,(obs-1),:,:], vmin = 0, vmax = 1)
        ax.set_xticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
        ax.set_yticks(ticks = [0, 8, 16, 24, 31], labels = np.array([-10,-5,0,5,10]))
    
    #fig.text(0.3, .9, "Extremal Coefficient", fontsize = 15)
    ax.cax.colorbar(im)
    plt.tight_layout()
    plt.savefig(figname)
