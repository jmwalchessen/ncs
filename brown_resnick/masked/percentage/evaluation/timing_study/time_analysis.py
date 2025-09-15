import numpy as np

def load_fcs_elapsed_time_array():

    elapsed_time = np.load(("data/model4/fcs_increasing_number_of_observed_locations_elapsed" +
                            "_timing_array_niasra_node_6_1_conditional_simulation_1_7_tnrep_50.npy"))
    return elapsed_time

def load_ncs_time_array():

    ncs_time = np.load(("data/model4/increasing_number_of_observed_locations_timing_array" +
                        "_niasra_node_6_1_conditional_simulation.npy"))
    return ncs_time

def avg_std_fcs():

    fcs_time = load_fcs_elapsed_time_array()
    fcs_avg = np.average(fcs_time, axis = 1)
    fcs_std = np.std(fcs_time, axis = 1)
    return fcs_avg, fcs_std

def avg_std_ncs():

    ncs_time = load_ncs_time_array()
    print(ncs_time.shape)
    ncs_avg = np.average(ncs_time, axis = 1)
    ncs_std = np.std(ncs_time, axis = 1)
    return ncs_avg, ncs_std

fcs_avg, fcs_std = avg_std_fcs()
print(fcs_avg)
print(fcs_std)
ncs_avg, ncs_std = avg_std_ncs()
print(ncs_avg)
print(ncs_std)