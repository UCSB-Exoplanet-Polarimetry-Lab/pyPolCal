# Running a longer mcmc that will take a while via screen session

# IMPORTING DATA

from pyPolCal.csv_tools import read_csv_physical_model_all_bins
from pathlib import Path
import multiprocessing as mp
import os
import numpy as np
from pyPolCal.constants import wavelength_bins
from importlib.resources import files
# mp.set_start_method("spawn", force=True) # Jax was slowing down from os.fork() and this fixed it
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
def main():
    # Defining Path to my CSVs
    # Defining Path to my CSVs

    csvdir = files('pyPolCal.CHARIS.datacsvs').joinpath('nbs_in_internalcal_csvs')

    # Reading in data
    interleaved_values_all, interleaved_stds_all, configuration_list_all = read_csv_physical_model_all_bins(csvdir)
    # GENERATE MODEL
    from pyPolCal.utils import generate_system_mueller_matrix



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
            "properties" : {"delta_theta":0}, 
            "tag": "internal",
            },
            
            "hwp" : {
                "type" : "two_layer_HWP_function", # Joost 't Hart 2021 HWP model
                "properties" : {"delta_theta": 0},
                "tag": "internal",
            },


            "lp" :{
                "type" : "diattenuator_retarder_function",
                "properties" : {"epsilon": -1, "delta_theta":0}, 
                "tag": "internal",
            },
    }
    }
    system_mm = generate_system_mueller_matrix(system_dict)
    # Define starting guesses

    
# Define starting guesses
    wsio2 = 1.636
    wmgf2 = 1.28

    p0_dict = {
        "image_rotator" : 
            {"delta_theta": 0},
        "hwp" :  
            {"w_SiO2": wsio2, "w_MgF2": wmgf2,"delta_theta": 0},
        "lp" : 
            {"delta_theta": 0}
    }

    # Define bounds
    offset_bounds = (-5,5)
    wsio2_bounds = (0.5*wsio2, 1.5*wsio2)
    wmgf2_bounds = (0.5*wmgf2, 1.5*wmgf2)

    bounds_dict = {
        "image_rotator" : {
            "delta_theta": offset_bounds
        },
        "hwp" : {
            "w_SiO2": wsio2_bounds, "w_MgF2": wmgf2_bounds,"delta_theta": offset_bounds
        },
        "lp" : {
            "delta_theta": offset_bounds
        }
    }

    # Define priors
    prior_dict = {
        "image_rotator": {
            "delta_theta": {"type": "gaussian", "kwargs": {"mu":0, "sigma": 1}},
        },
        "hwp": {
            "w_SiO2": {"type": "uniform", "kwargs": {"low": 0.5*wsio2, "high": 1.5*wsio2}},
            "w_MgF2":{"type": "uniform", "kwargs": {"low": 0.5*wmgf2, "high": 1.5*wmgf2}},
            "delta_theta": {"type": "gaussian", "kwargs": {"mu":0, "sigma": 1}},
        },
        "lp": {
            "delta_theta": {"type": "gaussian", "kwargs": {"mu":0, "sigma": 1}},
    }}
    from pyPolCal.instruments_jax import run_mcmc, process_model, process_dataset 

    # Path for the h5 emcee output file
    output_h5 = 'mcmc_tutorial_output.h5'

    ndim = 6  # Number of parameters to fit
    pool_processes = 12 # Number of CPU cores to use
    nwalkers = max(2 * ndim, pool_processes * 2) # Number of walkers at least twice the number of dimensions
    if nwalkers % pool_processes != 0:
        nwalkers += pool_processes - (nwalkers % pool_processes)

    print(f"{nwalkers} walkers for {ndim} parameters")
    sampler, p_keys = run_mcmc(p0_dict, system_mm, interleaved_values_all,configuration_list_all,prior_dict,bounds_dict,output_h5,nwalkers=nwalkers,pool_processes=pool_processes,process_model=process_model,process_dataset=process_dataset,nsteps=1000, include_sums=False)
if __name__ == "__main__":
    main()