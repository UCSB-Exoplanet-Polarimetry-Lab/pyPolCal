import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import corner
import os
import shutil
import instruments_jax as inst
from instruments_jax import (
    process_dataset, process_errors, parse_configuration,
    generate_system_mueller_matrix, update_system_mm,
    model, plot_data_and_model
)

def load_chain_and_labels(h5_filename, txt_filename, include_logf=False):
    base, ext = os.path.splitext(h5_filename)
    h5_copy = base + "_copy" + ext
    shutil.copy(h5_filename, h5_copy)

    with h5py.File(h5_copy, 'r') as f:
        chain = f['mcmc']['chain'][:]

    with open(txt_filename, 'r') as f:
        p0_dict = json.load(f)

    param_names = [f"{comp}.{param}" for comp, params in p0_dict.items() for param in params]
    if include_logf:
        param_names.append("log_f")

    return chain, param_names

def plot_trace(chain, param_names, step_range=(0, None), max_walkers=None):
    nsteps, nwalkers, ndim = chain.shape
    if max_walkers is None or max_walkers > nwalkers:
        max_walkers = nwalkers

    start, end = step_range
    x = np.arange(nsteps)[start:end]
    fig, axs = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)

    for i in range(ndim):
        for j in range(max_walkers):
            axs[i].plot(x, chain[start:end, j, i], alpha=0.3)
        axs[i].axvline(x[0], color='red', linestyle='--', alpha=0.5)
        axs[i].set_ylabel(param_names[i])
    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()

def plot_corner_flat(chain, param_names, step_range=(0, None), median_or_max="median", num_bins=100):
    flat_chain = chain[step_range[0]:step_range[1], :, :].reshape(-1, chain.shape[-1])
    converted_chain = flat_chain.copy()

    for i, name in enumerate(param_names):
        if ".phi" in name:
            converted_chain[:, i] = converted_chain[:, i] / (2 * np.pi)

    if median_or_max == "median":
        truths = np.median(converted_chain, axis=0)
    elif median_or_max == "max":
        truths = []
        for i in range(converted_chain.shape[1]):
            hist, bin_edges = np.histogram(converted_chain[:, i], bins=num_bins)
            max_index = np.argmax(hist)
            max_val = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2
            truths.append(max_val)
        truths = np.array(truths)
    else:
        raise ValueError("median_or_max must be 'median' or 'max'")

    fig = corner.corner(
        converted_chain,
        labels=[label.replace(".phi", ".phi") for label in param_names],
        truths=truths,
        plot_datapoints=False,     # disables individual scatter points
    )

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.xaxis.labelpad = 40
        ax.yaxis.labelpad = 40

    plt.tick_params(axis='x', which='both', pad=5)
    plt.tick_params(axis='y', which='both', pad=5)
    plt.show()

def summarize_posteriors(chain, param_names, system_dict, txt_file_path=None, txt_save_file_path=None, step_range=(0, None)):
    import instruments_jax as inst

    # Flatten MCMC chain
    flat_chain = chain[step_range[0]:step_range[1], :, :].reshape(-1, chain.shape[-1])

    # Generate the base system Mueller matrix from a known full system_dict
    system_mm = inst.generate_system_mueller_matrix(system_dict)

    # If provided, use the .txt param dict to get parameter keys
    if txt_file_path is not None:
        with open(txt_file_path, "r") as f:
            param_dict = json.load(f)
        _, p_keys = inst.parse_configuration(param_dict)
    else:
        raise ValueError("You must supply txt_file_path to extract parameter keys.")

    # Compute medians
    new_param_values = []
    for i, name in enumerate(param_names):
        val = flat_chain[:, i] / (2 * np.pi) if ".phi" in name else flat_chain[:, i]
        median = np.median(val)
        std = np.std(val)
        new_param_values.append(median)
        print(f"{name} ({'waves' if '.phi' in name else ''}): {median:.5f} ± {std:.5f}")

    if txt_save_file_path is not None:
        updated_mm = inst.update_system_mm(new_param_values, p_keys, system_mm)
        updated_dict = updated_mm.master_property_dict
        with open(txt_save_file_path, "w") as f:
            json.dump(updated_dict, f, indent=4)

def plot_mcmc_fits_double_diff_sum(
    h5_filename,
    txt_filename,
    csv_path,
    filter_wavelength,
    system_dict,
    wavelength_str=None,
    n_samples=50,
    step_range=(0, None),
    imr_theta_filter=None,
):
    # Load CSV data
    interleaved_values, interleaved_stds, configuration_list = inst.read_csv(
        csv_path, obs_mode="IPOL", obs_filter=filter_wavelength
    )
    interleaved_stds_proc = process_errors(interleaved_stds, interleaved_values)
    interleaved_values_proc = process_dataset(interleaved_values)

    # Load initial parameters
    with open(txt_filename, "r") as f:
        p0_dict = json.load(f)
    p0_values, p_keys = parse_configuration(p0_dict)

    # Load and preprocess MCMC chain
    with h5py.File(h5_filename, "r") as f:
        chain = f["mcmc"]["chain"][:]
    # Determine shape
    nsteps, nwalkers, ndim = chain.shape
    start, end = step_range
    if end is None:
        end = nsteps

    # Safety check
    if start >= end or start >= nsteps:
        raise ValueError(f"Invalid step_range {step_range} for chain with {nsteps} steps.")

    # Slice over steps (axis 0), keep all walkers
    flat_chain = chain[start:end, :, :].reshape(-1, ndim)

    # Random sample indices
    if len(flat_chain) < n_samples:
        raise ValueError(f"Only {len(flat_chain)} samples available, but {n_samples} requested.")
    np.random.seed(42)
    sample_indices = np.random.choice(flat_chain.shape[0], size=n_samples, replace=False)

    # Generate system Mueller matrix
    system_mm = generate_system_mueller_matrix(system_dict)

    # Plot only the first sample with legend
    for i, idx in enumerate(sample_indices):
        p_sample = flat_chain[idx]
        model_output = model(
            p_sample[:-1],
            p_keys[:-1],
            system_mm,
            configuration_list,
            s_in=np.array([1, 0, 0, 0]),
            process_model=inst.process_model,
        )
        plot_data_and_model(
            interleaved_values,
            interleaved_stds,
            model_output,
            configuration_list,
            wavelength=wavelength_str,
            imr_theta_filter=imr_theta_filter,
            legend=(i == 0)  # Only show legend on the first one
        )