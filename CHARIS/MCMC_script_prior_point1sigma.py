import multiprocessing as mp
import os
mp.set_start_method("spawn", force=True) # Jax was slowing down from os.fork() and this fixed it
os.environ["JAX_PLATFORM_NAME"] = "cpu" # Jax wasn't working with our GPU for unknown reasons
import sys
import numpy as np
from pathlib import Path
parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))
import re
import instruments as inst
from instruments_jax import *
from physical_models import *
from scipy import stats as scipy_stats
import h5py
import corner
import shutil
import tqdm
# test new read func
def main():
    csv_dir = Path('datacsvs/csvs_nopickoff')
    interleaved_values_all, interleaved_stds_all, configuration_list_all = inst.read_csv_physical_model_all_bins(csv_dir)
    # Getting the system dictionary setup and defining starting guesses values
    wavelength_bins = np.array([1159.5614, 1199.6971, 1241.2219, 1284.184 , 1328.6331, 1374.6208,
    1422.2002, 1471.4264, 1522.3565, 1575.0495, 1629.5663, 1685.9701,
    1744.3261, 1804.7021, 1867.1678, 1931.7956, 1998.6603, 2067.8395,
    2139.4131, 2213.4641, 2290.0781, 2369.3441])
    wavelength_bin = 15 # placeholder
    epsilon_cal = 1 # defining as perfect, reasoning in Joost t Hart 2021
    offset_imr = -0.01062 
    offset_hwp = -0.0022 
    offset_cal = -0.0315 
    imr_theta = 0 # placeholder 
    hwp_theta = 0 # placeholder
    # Past fits from scipy minimize on the naive fits
    d = 259.7 
    wsio2 = 1.617
    wmgf2 = 1.264

    # Define instrument configuration as system dictionary
    # Wollaston beam, imr theta/phi, and hwp theta/phi will all be updated within functions, so don't worry about their values here

    system_dict = {
    "components" : {
        "wollaston" : {
            "type" : "wollaston_prism_function",
            "properties" : {"beam": 'o'}, 
            "tag": "internal",
        },
        "image_rotator" : {
            "type" : "SCExAO_IMR_function",
            "properties" : {"wavelength":wavelength_bins[wavelength_bin], "d": d, "theta": imr_theta, "delta_theta": offset_imr},
            "tag": "internal",
        },
        "hwp" : {
            "type" : "two_layer_HWP_function",
            "properties" : {"wavelength": wavelength_bins[wavelength_bin], "w_SiO2": wsio2, "w_MgF2": wmgf2, "theta":hwp_theta, "delta_theta": offset_hwp},
            "tag": "internal",
        },
        "lprot": { # changed from delta_theta to match Joost t Hart
            "type": "rotator_function",
            "properties" : {'pa':offset_cal},
            "tag": "internal",
        },
        "lp" : {  # calibration polarizer for internal calibration source
            "type": "diattenuator_retarder_function",
            "properties": {"epsilon": epsilon_cal},
            "tag": "internal",
        }}
    }
        
    # Starting guesses

    p0_dict = {
        "image_rotator" : 
            {"d": d, "delta_theta": offset_imr},
        "hwp" :  
            {"w_SiO2": wsio2, "w_MgF2": wmgf2, "delta_theta": offset_hwp},
        "lprot" : 
            {"pa": offset_cal},
    }

    system_mm = inst.generate_system_mueller_matrix(system_dict) # Generating pyMuellerMat system MM

    p0 = [1.623, 1.268, 262.56] # Starting guesses from Joost t Hart 2021 
    offset_bounds = (-5.0,5.0) 
    d_bounds = (0.8*p0[2], 1.2*p0[2]) # Physical parameters shouldn't have changed much
    imr_offset_bounds = offset_bounds
    wsio2_bounds = (0.8*p0[0], 1.2*p0[0])
    wmgf2_bounds = (0.8*p0[1], 1.2*p0[1])
    hwp_offset_bounds = offset_bounds
    cal_offset_bounds = offset_bounds

    bounds = {
        "image_rotator" : 
            {"d": d_bounds, "delta_theta": imr_offset_bounds},
        "hwp" :  
            {"w_SiO2": wsio2_bounds, "w_MgF2": wmgf2_bounds, "delta_theta": hwp_offset_bounds},
        "lprot" : 
            {"pa": cal_offset_bounds},
    }

    # Defining uniform priors

    offset_prior = partial(mcmc.uniform_prior, low=-5.0, high=5.0)
    d_prior = partial(mcmc.uniform_prior, low=0.7*p0[2], high=1.3 * p0[2])
    imr_offset_prior = offset_prior
    wsio2_prior = partial(mcmc.uniform_prior, low=0.7*p0[0], high=1.3 * p0[0])
    wmgf2_prior = partial(mcmc.uniform_prior, low=0.7*p0[1], high=1.3 * p0[1])
    hwp_offset_prior = offset_prior
    cal_offset_prior = offset_prior


    prior_dict = {
        "image_rotator": {
            "d": {"type": "uniform", "kwargs": {"low":0.8*p0[2], "high": 1.2 * p0[2]}},
            "delta_theta": {"type": "gaussian", "kwargs": {"mu": 0, "sigma": 0.1}},
        },
        "hwp": {
            "w_SiO2": {"type": "uniform", "kwargs": {"low": 0.8*p0[0], "high": 1.2 * p0[0]}},
            "w_MgF2":{"type": "uniform", "kwargs": {"low": 0.8*p0[1], "high": 1.2 * p0[1]}},
            "delta_theta": {"type": "gaussian", "kwargs": {"mu": 0, "sigma": 0.1}},
        },
        "lprot": {
            "pa": {"type": "gaussian", "kwargs": {"mu": 0, "sigma": 0.1}},
        },
    }

    # fix errors

    # Storing samples in an h5

    output_h5 = Path('mcmc_output_larger_bounds_fixed_gaussian_smaller_dds.h5')

    # Interactive plotting

    ndim = 6  # Number of parameters to fit
    pool_processes = max(1, 12) 
    nwalkers = max(2 * ndim, pool_processes * 2)
    if nwalkers % pool_processes != 0:
        nwalkers += pool_processes - (nwalkers % pool_processes)
    sampler, p_keys = run_mcmc(p0_dict, system_mm, interleaved_values_all, interleaved_stds_all,configuration_list_all,prior_dict,bounds,logl_with_logf, output_h5,nwalkers=nwalkers, include_log_f=True, log_f=-3.0, pool_processes=pool_processes,process_model=process_model, process_errors=process_errors,process_dataset=process_dataset,nsteps=40000,plot=True, include_sums=False)

if __name__ == "__main__":
    main()