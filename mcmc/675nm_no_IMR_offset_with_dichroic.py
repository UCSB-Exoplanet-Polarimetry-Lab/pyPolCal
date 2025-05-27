import numpy as np
import os
import json
import instruments as inst
import mcmc_helper_funcs as mcmc
from functools import partial

# Example file path and configuration
csv_path = "20230914_processed_table.csv"
filter_wavelength = "675-50"

# Load dataset (replace with your file path)
interleaved_values, interleaved_stds, configuration_list = \
    inst.read_csv(csv_path, obs_mode="IPOL", obs_filter=filter_wavelength)

# Define ideal system configuration (this should reflect the setup of your optical train)
system_dict = {
    "components": {
        "wollaston": {
            "type": "wollaston_prism_function",
            "properties": {"beam": "o", "transmission_ratio": 1},
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
with open("675-50_restrictive_HWP_FLC_no_IMR_offset_with_dichroic_best_fit.txt", "r") as f:
    p0 = json.load(f)

# Parse p0 to get keywords and initial values
p0_values, p0_keys = inst.parse_configuration(p0)
ndim = len(p0_values)

# Auto-detect computing resources
pool_processes = max(1, os.cpu_count() - 1) # Leaving one free
nwalkers = max(2 * ndim, pool_processes * 2)
if nwalkers % pool_processes != 0:
    nwalkers += pool_processes - (nwalkers % pool_processes)

print(f"Auto-detected: {pool_processes} processes, {nwalkers} walkers for {ndim} parameters")

# Define bounds and priors for each parameter (customize as needed)
bounds = {
    "wollaston": {
        "transmission_ratio": (0, 2)
    },
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

priors = {}

# Initially setting Gaussian priors for all values
# TODO: Make sure the prior is NOT centered around the starting guess - uniform is better
for component, params in p0.items():
    priors[component] = {}
    for param, val in params.items():
        priors[component][param] = partial(mcmc.gaussian_prior, mu = val, sigma = 0.1 * abs(val))  # Gaussian centered at val

# Custom changing the offset angle priors to be near zero, with 1 degree std
offset_prior = partial(mcmc.gaussian_prior, mu=0, sigma=1)
priors["flc"]["delta_theta"] = offset_prior
priors["hwp"]["delta_theta"] = offset_prior
priors["flc"]["theta"] = offset_prior

# Saving parameters
output_h5_file = "675nm_no_IMR_offset_with_dichroic.h5"
nsteps = 10000
s_in = np.array([1, 0, 0, 0])

# Run MCMC with emcee
sampler, fitted_keys = inst.run_mcmc(
    p0_dict=p0,
    system_mm=system_mm,
    dataset=interleaved_values,
    errors=interleaved_stds,
    configuration_list=configuration_list,
    priors=priors,
    bounds=bounds,
    logl_function=inst.logl,
    output_h5_file=output_h5_file,
    nwalkers=nwalkers,
    nsteps=nsteps,
    pool_processes=pool_processes,
    s_in=s_in,
    process_dataset=inst.process_dataset,
    process_errors=inst.process_errors,
    process_model=inst.process_model
)

# Access chain or log prob like:
# chain = sampler.get_chain()
# log_prob = sampler.get_log_prob()
