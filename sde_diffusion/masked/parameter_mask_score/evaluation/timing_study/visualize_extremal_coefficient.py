import matplotlib.pyplot as plt
import torch as th
import numpy as np


def load_numpy_file(npfile):

    nparr = np.load(npfile)
    return nparr

#first entry is lag, second entry is madogram, third entry is extremal coefficient
def visualize_extremal_coefficient(extremal_matrix, range, smooth, bins):

    
