import numpy as np

#once you've loaded seed value, mask, observed and conditional simulations from ref image folder,
# process them, (mask and observed will be shape (1024)) and conditional simulations will be (nrep, m)
def process_rproducts(mask, observation, conditional_simulations, nrep, nobs, n):

    condbrsim = np.zeros((nrep, (n**2)))
    condbrsim[:,mask == 0] = conditional_simulations
    condbrsim[:,mask == 1] = (np.tile(observation[mask ==1], reps = nrep)).reshape((nrep, nobs))
    
    #shape to matrices
    condbrsim = condbrsim.reshape((nrep,n,n))
    observation = observation.reshape((n,n))
    mask = mask.reshape((n,n))
    return mask, observation, condbrsim

def load_files(ref_image_folder):

    conditional_simulations = np.load(("data/mwe/" + ref_image_folder + "/preprocessed_conditional_simulations_powexp_range_1.6_smooth_1.6.npy"))
    observation = np.load("data/mwe/" + ref_image_folder + "/observed_simulation_powexp_range_1.6_smooth_1.6.npy")
    mask = np.load("data/mwe/" + ref_image_folder + "/mask.npy")
    return mask, observation, conditional_simulations


nrep = 50
nobs = 5
n = 32
refimage_folder = "ref_image1"
mask, obs, condsim = load_files(refimage_folder)
mask, obs, condbrsim = process_rproducts(mask, obs, condsim, nrep, nobs, n)
np.save("data/mwe/" + refimage_folder + "/conditional_simulations_powexp_range_1.6_smooth_1.6.npy", condbrsim)




