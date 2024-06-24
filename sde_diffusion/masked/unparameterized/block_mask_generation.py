import numpy as np
import matplotlib.pyplot as plt

#return upper, lower, right and left masks
def half_mask(n):

    maskpattern = np.zeros((4,n,n))
    maskpattern[0,:int(n/2),:] = 1
    maskpattern[1,:,:int(n/2)] = 1
    maskpattern[2,int(n/2):,:] = 1
    maskpattern[3,:,int(n/2):] = 1
    return maskpattern

#return quarter and thre quarters masks for all positions
def produce_quarter_mask(n):

    quarter_masks = np.zeros((8,n,n))
    quarter_masks[0,:int(n/2),:int(n/2)] = 1
    quarter_masks[1,:int(n/2),int(n/2):] = 1
    quarter_masks[2,int(n/2):,int(n/2):] = 1
    quarter_masks[3,:int(n/2),:int(n/2)] = 1
    quarter_masks[4:8,:,:] = 1-quarter_masks[0:3,:,:]
    return quarter_masks

def visualize_mask(mask):

    plt.imshow(mask)
    plt.show()

n = 32
quarter_mask = produce_quarter_mask(n)
visualize_mask(quarter_mask[0,:,:])
