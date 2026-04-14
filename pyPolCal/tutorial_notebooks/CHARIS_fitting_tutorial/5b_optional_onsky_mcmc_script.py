# Running a longer mcmc that will take a while via screen session

# IMPORTING DATA

from pyPolCal.csv_tools import read_csv_physical_model_all_bins
from importlib.resources import files
from pathlib import Path
import multiprocessing as mp
import os
# mp.set_start_method("spawn", force=True) # Jax was slowing down from os.fork() and this fixed it
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
def main():
    # Defining Path to my CSVs
    csvdir = Path('/home/thomasmc/pyPolCal/pyPolCal/CHARIS/datacsvs/onsky_nbs/HD293396')

    # Reading in data
    interleaved_values_all, interleaved_stds_all, configuration_list_all = read_csv_physical_model_all_bins(csvdir, m3=True)
    # GENERATE MODEL
    from pyPolCal.utils import generate_system_mueller_matrix

    system_dict = {
    "components" : {
    "wollaston" : {
    "type" : "fitted_wollaston_function_12_28_2025",
    "properties" : {"beam": 'o'}, 
    "tag": "internal",
    },

    "nbs_rot": {
        "type": "rotator_function",
        "properties": {"pa": 90},
        "tag": "internal",
    },
    "image_rotator" : {
    "type" : "fitted_derotator_function_12_28_2025",
    "properties" : {"delta_theta":1.384e-02}, 
    "tag": "internal",
    },

    "hwp" : {
        "type" : "two_layer_HWP_function", # Joost 't Hart 2021 HWP model
        "properties" : {"w_SiO2":1.638, "w_MgF2":1.28, "delta_theta": -3.168e-02},
        "tag": "internal",
    },

    "altitude_rot" : {
        "type" : "rotator_function",
        "properties" : {"pa":0},
        "tag":"internal",
    },
    "M3" : {
        "type" : "SUBARU_M3_function",
        "properties" : {"delta_theta":0},
        "tag": "internal",
    },

    "parang_rot" : {
        "type" : "rotator_function",
        "properties" : {"pa":0},
        "tag":"internal",
    },
    },
    }
    system_mm = generate_system_mueller_matrix(system_dict)

    # Define starting guesses

    m1, b1, m2, b2 = (1.781,12.47,2.264,14.67) # from minimize
    p0_dict = {
    "M3":{"m1": m1,
    "b1": b1,
    "m2": m2,
    "b2": b2
    }
    }
    m1_bounds = (0.5*m1, 2*m1)
    m2_bounds = (0.5*m2, 2*m2)
    b1_bounds = (0.5*b1, 2*b1)
    b2_bounds = (0.5*b2, 2*b2)
    boundslist = [m1_bounds, b1_bounds, m2_bounds, b2_bounds]

    bounds_dict = {

    "M3" : {
    "m1": m1_bounds,
    "b1": b1_bounds,
    "m2": m2_bounds,
    "b2": b2_bounds
    }
    }

    # Define priors
    prior_dict = {

    "M3": {
    "m1": {"type": "uniform", "kwargs": {"low":0.5*m1, "high": 2*m1}},
    "b1": {"type": "uniform", "kwargs": {"low":0.5*b1, "high": 2*b1}},
    "m2": {"type": "uniform", "kwargs": {"low":0.5*m2, "high": 2*m2}},
    "b2": {"type": "uniform", "kwargs": {"low":0.5*b2, "high": 2*b2}},
    }}
    from pyPolCal.instruments_jax import run_mcmc, process_model, process_dataset, process_errors

    # Path for the h5 emcee output file
    output_h5 = 'mcmc_tutorial_output_onsky.h5'

    ndim = 6  # Number of parameters to fit
    pool_processes = 12 # Number of CPU cores to use
    nwalkers = max(2 * ndim, pool_processes * 2) # Number of walkers at least twice the number of dimensions
    if nwalkers % pool_processes != 0:
        nwalkers += pool_processes - (nwalkers % pool_processes)

    print(f"{nwalkers} walkers for {ndim} parameters")
    sampler, p_keys = run_mcmc(p0_dict, system_mm, interleaved_values_all,configuration_list_all,prior_dict,bounds_dict,output_h5,errors=interleaved_stds_all,nwalkers=nwalkers,pool_processes=pool_processes,process_model=process_model, process_dataset=process_dataset,process_errors=process_errors,nsteps=100000, include_sums=False)
if __name__ == "__main__":
    main()