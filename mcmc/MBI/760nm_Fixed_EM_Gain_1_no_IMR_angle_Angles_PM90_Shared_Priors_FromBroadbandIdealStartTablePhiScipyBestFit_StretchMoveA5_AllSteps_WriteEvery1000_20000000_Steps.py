#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json

import h5py
import numpy as np

custom_module_path = "/home/rebeccaz/Github/vampires_calibration"
if custom_module_path not in sys.path:
    sys.path.append(custom_module_path)

import instruments_jax as inst
from shared_priors_no_imr_offset_angles_pm90 import (
    build_shared_bounds,
    build_shared_prior_dict,
    prepare_p0_for_shared_priors,
)


filter_wavelength = 760
wavelength_index = 3
target_mcmc_steps = 20000000
backend_flush_interval = 1000
stretch_move_a = 5.0
stretch_move_label = "A5"
nsteps = target_mcmc_steps
obs_mode = "MBI"
start_txt_file = "/home/rebeccaz/Github/vampires_calibration/scipy_minimize/intermediate_data_files/760_no_IMR_offset_fixed_EM_gain_1_no_retardance_constraints_iterate_on_any_improvement_from_broadband_ideal_start_table_phi_with_dichroic_best_fit_old_fit_values.txt"
csv_path = "/home/rebeccaz/Github/vampires_calibration/data/20230914_processed_table.csv"
output_h5_file = "/home/rebeccaz/Github/vampires_calibration/mcmc/results/760nm_Fixed_EM_Gain_1_no_IMR_angle_Angles_PM90_Shared_Priors_FromBroadbandIdealStartTablePhiScipyBestFit_StretchMoveA5_AllSteps_WriteEvery1000_20000000_steps.h5"
include_log_f = True


def get_existing_progress(h5_path):
    if not os.path.exists(h5_path):
        return 0, 0

    with h5py.File(h5_path, "r") as h5:
        if "mcmc" not in h5:
            return 0, 0
        grp = h5["mcmc"]
        saved_steps = int(grp.attrs.get("iteration", 0))
        allocated_steps = saved_steps
        if "chain" in grp:
            allocated_steps = int(grp["chain"].shape[0])

    return saved_steps, allocated_steps


def record_run_metadata(h5_path):
    with h5py.File(h5_path, "a") as h5:
        h5.attrs["filter_wavelength_nm"] = filter_wavelength
        h5.attrs["target_mcmc_steps"] = target_mcmc_steps
        h5.attrs["backend_write_interval"] = 1
        h5.attrs["backend_flush_interval"] = backend_flush_interval
        h5.attrs["stretch_move_a"] = stretch_move_a
        h5.attrs["start_txt_file"] = start_txt_file
        h5.attrs["fixed_em_gain"] = 1.0
        h5.attrs["csv_path"] = csv_path


print(f"Running {filter_wavelength} nm MBI MCMC")
print(f"Starting from scipy best fit: {start_txt_file}")
print(f"Writing backend to: {output_h5_file}")
print(f"Storing every MCMC step and flushing every {backend_flush_interval} stored steps")

with open(start_txt_file, "r") as f:
    p0 = json.load(f)

# The no-IMR-angle shared-prior model does not include Wollaston terms or image-rotator angle offset.
p0.pop("wollaston", None)
p0.setdefault("image_rotator", {}).pop("delta_theta", None)

bounds = build_shared_bounds()
prior_dict = build_shared_prior_dict()
p0 = prepare_p0_for_shared_priors(p0, bounds)
p0_values, p0_keys = inst.parse_configuration(p0)
ndim = len(p0_values)

pool_processes = 7
nwalkers = max(2 * ndim, pool_processes * 2)
if nwalkers % pool_processes != 0:
    nwalkers += pool_processes - (nwalkers % pool_processes)

print(f"Using {pool_processes} processes, {nwalkers} walkers for {ndim} fitted parameters")

saved_steps, allocated_steps = get_existing_progress(output_h5_file)
remaining_mcmc_steps = max(target_mcmc_steps - saved_steps, 0)
print(f"Existing saved steps: {saved_steps}/{target_mcmc_steps}")
print(f"Remaining target steps: {remaining_mcmc_steps}")

fixed_em_gain = 1.0
interleaved_values, interleaved_stds, configuration_list = inst.read_csv(
    csv_path, obs_mode=obs_mode, obs_filter=filter_wavelength
)

system_dict = {
    "components": {
        "wollaston": {
            "type": "wollaston_prism_function",
            "properties": {"beam": "o", "transmission_ratio": fixed_em_gain},
        },
        "dichroic": {
            "type": "diattenuator_retarder_function",
            "properties": {"phi": 0, "epsilon": 0, "theta": 0},
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
system_mm = inst.generate_system_mueller_matrix(system_dict)
s_in = np.array([1, 0, 0, 0])

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
    copy_existing_h5_before_start=False,
    include_log_f=include_log_f,
    log_f=-3.0,
    backend_write_interval=1,
    backend_flush_interval=backend_flush_interval,
    stretch_move_a=stretch_move_a,
)
record_run_metadata(output_h5_file)
print("Finished MCMC run")
