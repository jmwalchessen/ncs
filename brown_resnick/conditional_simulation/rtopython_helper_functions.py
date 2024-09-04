import numpy as np

#once you've loaded seed value, mask, observed and conditional simulations from ref image folder,
# process them, (mask and observed will be shape (1024)) and conditional simulations will be (nrep, m)
def process_rproducts(mask, observation, conditional_simulations, nrep, n):

    condsim = np.zeros((nrep, (n**2)))
    condsim[:,mask == 0] = conditional_simulations
    obs = observation.reshape((1,n**2))
    condsim[:,mask == 1] = (np.repeat(obs[:,mask ==1], repeats = nrep, axis = 0))
    
    #shape to matrices
    condsim = condsim.reshape((nrep,n,n))
    observation = observation.reshape((n,n))
    mask = mask.reshape((n,n))
    return condsim

def load_files(ref_image_folder):

    conditional_simulations = np.load(("data/mwe/" + ref_image_folder + "/preprocessed_conditional_simulations_powexp_range_3_smooth_1.6.npy"))
    observation = np.load("data/mwe/" + ref_image_folder + "/observed_simulation_powexp_range_3_smooth_1.6.npy")
    mask = np.load("data/mwe/" + ref_image_folder + "/mask.npy")
    return mask, observation, conditional_simulations







