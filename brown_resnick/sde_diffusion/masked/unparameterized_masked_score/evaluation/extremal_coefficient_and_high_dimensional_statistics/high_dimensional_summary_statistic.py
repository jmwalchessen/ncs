import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_maximum_summary_statistic(ncs_images_file, true_images_file, figname):

    ncs_images = np.load(ncs_images_file)
    true_images = np.log(np.load(true_images_file))
    ncs_maximums = np.maximum(ncs_images, axis = 0)
    true_maximums = np.maximum(true_images, axis = 0)

    fig, ax = plt.subplots()
    ncspdd = pd.DataFrame(ncs_maximums, columns = None)
    truepdd = pd.DataFrame(true_maximums, columns = None)
    sns.kdeplot(data = ncspdd, palette=['orange'], ax = ax)
    sns.kdeplot(data = truepdd, palette=['blue'], ax = ax)
    plt.savefig(figname)
    plt.clf()

def compute_maximum_summary_statistic_multiple_percentages(ncs_images_file,
                                                           true_images_file, figname, ps):
    
    for p in ps:

        current_ncs_images_file = (ncs_images_file + "_random" + str(p) + ".npy")
        current_true_images_file = (true_images_file + "_random" + str(p) + ".npy")
        current_ncs_images = np.load(current_ncs_images_file)
        current_true_images = np.load(current_true_images_file)
        current_figname = (figname + "_random" + str(p) + ".npy")
        compute_maximum_summary_statistic(current_ncs_images_file, current_true_images_file, current_figname)



