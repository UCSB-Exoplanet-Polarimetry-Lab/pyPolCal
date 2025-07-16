import sys
import os

# Add the directory containing instruments.py to the Python path
custom_module_path = "/home/rebeccaz/Github/vampires_calibration"
sys.path.append(custom_module_path)
mcmc_helper_funcs_path = ""

import numpy as np
import json
import instruments_jax as inst
# from instruments_jax import logl_with_logf, process_dataset, process_errors, process_model
import mcmc_helper_funcs_jax as mcmc
from functools import partial

# Example file path and configuration
filter_wavelength = 610
wavelength_index = 0
nsteps = 1200000
obs_mode = "MBI"
txt_file_folder = "/home/rebeccaz/Github/vampires_calibration/mcmc/start_txt_files/"
csv_path = "/home/rebeccaz/Github/vampires_calibration/data/20230914_processed_table.csv"
output_h5_file = "/home/rebeccaz/Github/vampires_calibration/mcmc/results/" + str(filter_wavelength) + "nm_Adjusted_EM_Gain_" + str(nsteps) + "_steps.h5"

IPOL_em_gains = [1.14, 1.18, 1.18, 1.18]
MBI_em_gains = [1.23, 1.19, 1.2, 1.08]
if obs_mode == "IPOL":
    em_gain = IPOL_em_gains[wavelength_index]
elif obs_mode == "MBI":
    em_gain = MBI_em_gains[wavelength_index]

# Load dataset (replace with your file path)
interleaved_values, interleaved_stds, configuration_list = \
    inst.read_csv(csv_path, obs_mode=obs_mode, obs_filter=filter_wavelength)

# Define ideal system configuration (this should reflect the setup of your optical train)
system_dict = {
    "components": {
        "wollaston": {
            "type": "wollaston_prism_function",
            "properties": {"beam": "o", "transmission_ratio": em_gain},
        },
        "dichroic": {
            "type": "diattenuator_retarder_function",
            "properties": {"phi": 0, "epsilon": 0, "theta" : 0},
        },
        "flc": {
            "type": "general_retarder_function",
            "properties": {"phi": 0.5 * 2 * np.pi, "theta": 0, "delta_theta": 0},
        },
        "optics": {
            "type": "diattenuator_retarder_function",
            "properties": {"phi": 0, "epsilon": 0, "theta": 0},
        },
        "image_rotator": {
            "type": "general_retarder_function",
            "properties": {"phi": 0.5 * 2 * np.pi, "theta": 0, "delta_theta": 0},
        },
        "hwp": {
            "type": "general_retarder_function",
            "properties": {"phi": 0.5, "theta": 0, "delta_theta": 0},
        },
        "lp": {
            "type": "general_linear_polarizer_function_with_theta",
            "properties": {"theta": 0},
        },
    }
}

# Build system Mueller Matrix
system_mm = inst.generate_system_mueller_matrix(system_dict)

# Load p0 dictionary from .txt file
with open(txt_file_folder + str(filter_wavelength) + "nm.txt", "r") as f:
    p0 = json.load(f)

# Parse p0 to get keywords and initial values
p0_values, p0_keys = inst.parse_configuration(p0)
ndim = len(p0_values)

# Auto-detect computing resources
# pool_processes = max(1, os.cpu_count() - 1) # Leaving one free
pool_processes = 7 # Leaving four CPUs free and allowing for four MBI processes to work at the same time
nwalkers = max(2 * ndim, pool_processes * 2)
if nwalkers % pool_processes != 0:
    nwalkers += pool_processes - (nwalkers % pool_processes)

print(f"Auto-detected: {pool_processes} processes, {nwalkers} walkers for {ndim} parameters")

# Define bounds and priors for each parameter (customize as needed)
bounds = {
    "dichroic": {
        "phi": (-2 * np.pi, 2 * np.pi),
        "epsilon": (0, 1),
        "theta": (-90, 90)
    },
    "flc": {
        "phi": (-2 * np.pi, 2 * np.pi),
        "delta_theta": (-5, 5)
    },
    "optics": {
        "phi": (-2 * np.pi, 2 * np.pi),
        "epsilon": (0, 1),
        "theta": (-90, 90)
    },
    "image_rotator": {
        "phi": (-2 * np.pi, 2 * np.pi)
    },
    "hwp": {
        "phi": (-2 * np.pi, 2 * np.pi),
        "delta_theta": (-5, 5)
    },
    "lp": {
        "theta": (-5, 5)
    }
}

# Setting all uniform priors for now
prior_dict = {
    "dichroic": {
        "phi": {"type": "uniform", "kwargs": {"low": -2 * np.pi, "high": 2 * np.pi}},
        "epsilon": {"type": "uniform", "kwargs": {"low": 0, "high": 1}},
        "theta": {"type": "uniform", "kwargs": {"low": -90, "high": 90}},
    },
    "flc": {
        "phi": {"type": "uniform", "kwargs": {"low": -2 * np.pi, "high": 2 * np.pi}},
        "delta_theta": {"type": "gaussian", "kwargs": {"mu": 0.0, "sigma": 1.0}},
    },
    "optics": {
        "phi": {"type": "uniform", "kwargs": {"low": -2 * np.pi, "high": 2 * np.pi}},
        "epsilon": {"type": "uniform", "kwargs": {"low": 0, "high": 1}},
        "theta": {"type": "uniform", "kwargs": {"low": -90, "high": 90}},
    },
    "image_rotator": {
        "phi": {"type": "uniform", "kwargs": {"low": -2 * np.pi, "high": 2 * np.pi}},
    },
    "hwp": {
        "phi": {"type": "uniform", "kwargs": {"low": -2 * np.pi, "high": 2 * np.pi}},
        "delta_theta": {"type": "gaussian", "kwargs": {"mu": 0.0, "sigma": 1.0}},
    },
    "lp": {
        "theta": {"type": "gaussian", "kwargs": {"mu": 0.0, "sigma": 1.0}},
    },
}

# Saving parameters
s_in = np.array([1, 0, 0, 0])

# Run MCMC with emcee and include log_f
sampler, fitted_keys = inst.run_mcmc(
    p0_dict=p0,
    system_mm=system_mm,
    dataset=interleaved_values,
    errors=interleaved_stds,
    configuration_list=configuration_list,
    priors=prior_dict,
    bounds=bounds,
    logl_function=inst.logl_with_logf,
    output_h5_file=output_h5_file,
    nwalkers=nwalkers,
    nsteps=nsteps,
    pool_processes=pool_processes,
    s_in=s_in,
    process_dataset=inst.process_dataset,
    process_errors=inst.process_errors,
    process_model=inst.process_model,
    include_log_f=True,
    log_f=-3.0
)

# Access chain or log prob like:
# chain = sampler.get_chain()
# log_prob = sampler.get_log_prob()
