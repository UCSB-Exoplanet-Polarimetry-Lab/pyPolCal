import sys
import os
import shutil

# Add the directory containing instruments.py to the Python path
custom_module_path = "/home/rebeccaz/Github/vampires_calibration"
sys.path.append(custom_module_path)
mcmc_helper_funcs_path = ""

import numpy as np
import json
import emcee
import instruments_jax as inst
# from instruments_jax import logl_with_logf, process_dataset, process_errors, process_model
import mcmc_helper_funcs_jax as mcmc
from shared_priors_with_imr_offset_angles_pm45 import (
    build_shared_bounds,
    build_shared_prior_dict,
    prepare_p0_for_shared_priors,
)
from functools import partial


def seed_output_backend_from_previous(previous_h5_file, output_h5_file, nwalkers, ndim):
    """Use a previous emcee backend to seed this run's new output file."""
    if os.path.exists(output_h5_file):
        output_backend = emcee.backends.HDFBackend(output_h5_file)
        if output_backend.iteration > 0:
            if output_backend.shape != (nwalkers, ndim):
                raise ValueError(
                    f"Cannot resume {output_h5_file}: backend shape "
                    f"{output_backend.shape} does not match requested shape "
                    f"{(nwalkers, ndim)}."
                )
            print(
                f"Continuing existing output backend {output_h5_file} "
                f"from iteration {output_backend.iteration}."
            )
            return
        print(f"{output_h5_file} exists but has no samples; checking previous backend.")

    if not os.path.exists(previous_h5_file):
        print(f"No previous backend found at {previous_h5_file}; starting from p0.")
        return

    previous_backend = emcee.backends.HDFBackend(previous_h5_file)
    if previous_backend.iteration == 0:
        print(f"{previous_h5_file} has no saved samples; starting from p0.")
        return
    if previous_backend.shape != (nwalkers, ndim):
        raise ValueError(
            f"Cannot seed from {previous_h5_file}: backend shape "
            f"{previous_backend.shape} does not match requested shape "
            f"{(nwalkers, ndim)}."
        )

    shutil.copy2(previous_h5_file, output_h5_file)
    print(
        f"Seeded {output_h5_file} with {previous_backend.iteration} saved "
        f"iterations from {previous_h5_file}. The sampler will resume in "
        f"the new file."
    )

# Example file path and configuration
filter_wavelength = 720
wavelength_index = 2
nsteps = 20000000
obs_mode = "MBI"
start_txt_folder = "/home/rebeccaz/Github/vampires_calibration/scipy_minimize/intermediate_data_files/"
csv_path = "/home/rebeccaz/Github/vampires_calibration/data/20230914_processed_table.csv"
output_h5_file = "/home/rebeccaz/Github/vampires_calibration/mcmc/results/" + str(filter_wavelength) + "nm_Adjusted_EM_Gain_with_IMR_offset_Angles_PM45_Shared_Priors_" + str(nsteps) + "_steps.h5"
previous_h5_file = "/home/rebeccaz/Github/vampires_calibration/mcmc/results/" + str(filter_wavelength) + "nm_Adjusted_EM_Gain_with_IMR_offset_Angles_PM45_Shared_Priors_1200000_steps.h5"
include_log_f = True

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

# Load p0 dictionary from the matching scipy-minimize best fit.
start_txt_file = start_txt_folder + str(filter_wavelength) + "_with_IMR_offset_pm_1_degree_fixed_EM_gain_with_dichroic_best_fit_old_fit_values.txt"
with open(start_txt_file, "r") as f:
    p0 = json.load(f)
p0.pop("wollaston", None)
p0.setdefault("image_rotator", {}).setdefault("delta_theta", 0.0)

bounds = build_shared_bounds()
prior_dict = build_shared_prior_dict()
p0 = prepare_p0_for_shared_priors(p0, bounds)

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

backend_ndim = ndim + int(include_log_f)
seed_output_backend_from_previous(previous_h5_file, output_h5_file, nwalkers, backend_ndim)

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
    resume=True,
    include_log_f=include_log_f,
    log_f=-3.0
)

# Access chain or log prob like:
# chain = sampler.get_chain()
# log_prob = sampler.get_log_prob()
