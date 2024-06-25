import numpy as np
import matplotlib.pyplot as plt

#return upper, lower, right and left masks
def produce_half_mask(n):

    maskpattern = np.zeros((4,n,n))
    maskpattern[0,:int(n/2),:] = 1
    maskpattern[1,:,:int(n/2)] = 1
    maskpattern[2,int(n/2):,:] = 1
    maskpattern[3,:,int(n/2):] = 1
    return maskpattern
#produce mask of 0s and 1s with dimension (nxn) in which
#there is a probability of p in the first half and probability q in the second half
def produce_weighted_half_mask(n, p, q):

    maskpattern = np.zeros((2,n,n))
    n_size = int(int(n/2)*n)
    n_shape = (int(n/2),n)
    maskpattern[0,:int(n/2),:] = np.random.binomial(n = n_size, p = p, size = n_shape)
    maskpattern[1,int(n/2):,:] = np.random.binomial(n = n_size, p = q, size = n_shape)
    return maskpattern

def produce_weighted_half_masks(n, ps, qs):

    nonzero_masks = np.zeros((len(ps)*len(qs), n, n))
    for p in ps:
        for q in qs:
            mask = produce_weighted_half_mask(n, p, q)
            nonzero_masks[(ps.index(p)*len(qs) + qs.index(q)),:,:] = mask[0,:,:]
    return nonzero_masks

#return quarter and thre quarters masks for all positions
def produce_quarter_mask(n):

    quarter_masks = np.zeros((8,n,n))
    quarter_masks[0,:int(n/2),:int(n/2)] = 1
    quarter_masks[1,:int(n/2),int(n/2):] = 1
    quarter_masks[2,int(n/2):,int(n/2):] = 1
    quarter_masks[3,:int(n/2),:int(n/2)] = 1
    quarter_masks[4:8,:,:] = 1-quarter_masks[0:4,:,:]
    return quarter_masks

def produce_triangular_mask(n):

    triangular_mask = np.zeros((4,n,n))
    triangular_mask[0,np.triu_indices(n)[0],np.triu_indices(n)[1]] = 1
    triangular_mask[1,np.tril_indices(n)[0],np.tril_indices(n)[1]] = 1
    triangular_mask[2:4,:,:] = np.flip((1-triangular_mask[0:2,:,:]), axis = 1)
    return triangular_mask

def produce_single_block_mask(n):

    blocksizelist = [2*i for i in range(1,16)]
    block_masks = np.zeros((28, n, n))
    for i in range(len(blocksizelist)):
        block_size = int(blocksizelist[i]/2)
        block_masks[i,(int(n/2)-block_size):(int(n/2)+block_size),
                    (int(n/2)-block_size):(int(n/2)+block_size)] = 1 
    
    block_masks[14:28,:,:] = 1 - block_masks[0:14,:,:]
    return block_masks

def produce_checkered_mask(n):

    checkered_masks = np.zeros((6,n,n))
    checkersizelist = [2,4,8]
    for i in range(len(checkersizelist)):
        checkersize = checkersizelist[i]
        tophalf = np.concatenate((np.zeros((checkersize,checkersize)),
                       np.ones((checkersize, checkersize))), axis = 0)
        bottomhalf = np.concatenate((np.ones((checkersize,checkersize)),
                       np.zeros((checkersize, checkersize))), axis = 0)
        checker_pattern = np.concatenate([tophalf, bottomhalf], axis = 1)
        checkered_masks[i,:,:] = np.tile(checker_pattern, ((n//(2*checkersize)),
                                         (n//(2*checkersize))))
    
    checkered_masks[3:6,:,:] = (1-checkered_masks[0:3,:,:])
    return checkered_masks

#create a matrix of masks from the previous mask patterns given in the above functions
def produce_nonrandom_block_masks(n, ps, qs):

    half_masks = produce_half_mask(n)
    weighted_half_masks = produce_weighted_half_masks(n, ps, qs)
    quarter_masks = produce_quarter_mask(n)
    triangular_masks = produce_triangular_mask(n)
    single_block_masks = produce_single_block_mask(n)
    checkered_masks = produce_checkered_mask(n)
    all_masks = np.concatenate([half_masks, weighted_half_masks, quarter_masks,
                                triangular_masks, single_block_masks, checkered_masks], axis = 0)
    return all_masks

    

def visualize_mask(mask):

    plt.imshow(mask)
    plt.show()

