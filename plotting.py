import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import corner

import json
import h5py
import numpy as np
from instruments import (
    read_csv, parse_configuration, generate_system_mueller_matrix,
    update_system_mm, generate_measurement, process_model, process_dataset,
    process_errors
)
import matplotlib.pyplot as plt

def load_chain_and_labels(h5_filename, txt_filename):
    """Load the MCMC chain and parameter names from HDF5 and .txt files."""
    with h5py.File(h5_filename, 'r') as f:
        chain = f['mcmc']['chain'][:]  # shape should be (nsteps, nwalkers, ndim)

    with open(txt_filename, 'r') as f:
        p0_dict = json.load(f)

    param_names = [f"{comp}.{param}" for comp, params in p0_dict.items() for param in params]
    return chain, param_names

def plot_trace(chain, param_names, burn_in=0, max_walkers=None):
    """Simplified trace plot for MCMC chains."""
    nsteps, nwalkers, ndim = chain.shape
    if max_walkers is None or max_walkers > nwalkers:
        max_walkers = nwalkers

    fig, axs = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
    x = np.arange(nsteps)

    for i in range(ndim):
        for j in range(max_walkers):
            axs[i].plot(x[burn_in:], chain[burn_in:, j, i], alpha=0.3)
        axs[i].axvline(burn_in, color='red', linestyle='--', alpha=0.5)
        axs[i].set_ylabel(param_names[i])
    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import corner

def plot_corner_flat(chain, param_names, burn_in=0, median_or_max="median", num_bins=100):
    """
    Generate a corner plot from a 3D MCMC chain with enhanced styling.
    
    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (nsteps, nwalkers, nparams).
    param_names : list of str
        Names of parameters corresponding to each column of the chain.
    burn_in : int
        Number of initial steps to discard as burn-in.
    median_or_max : str
        Whether to use "median" or "max" of posterior as truth line.
    num_bins : int
        Number of bins to use for histograms.
    """
    flat_chain = chain[burn_in:, :, :].reshape(-1, chain.shape[-1])

    # Determine the 'truths' for vertical lines
    if median_or_max == "median":
        truths = np.median(flat_chain, axis=0)
    elif median_or_max == "max":
        truths = []
        for i in range(flat_chain.shape[1]):
            hist, bin_edges = np.histogram(flat_chain[:, i], bins=num_bins)
            max_index = np.argmax(hist)
            max_val = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2
            truths.append(max_val)
        truths = np.array(truths)
    else:
        raise ValueError("median_or_max must be 'median' or 'max'")

    # Generate corner plot
    fig = corner.corner(
        flat_chain, labels=param_names, truths=truths, plot_datapoints=False
    )

    # Style customization
    large_font_size = 50
    medium_font_size = 40
    label_font_size = 20
    tick_font_size = 15
    default_tick_font_size = 12
    default_font_size = 10
    label_padding = 40
    tick_padding = 5

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=default_tick_font_size)
        ax.xaxis.label.set_size(label_font_size)
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.labelpad = label_padding
        ax.yaxis.labelpad = label_padding

    plt.tick_params(axis='x', which='both', pad=tick_padding)
    plt.tick_params(axis='y', which='both', pad=tick_padding)

    plt.show()


def summarize_posteriors(chain, param_names, burn_in=0):
    """Print median ± std for each parameter."""
    flat_chain = chain[burn_in:, :, :].reshape(-1, chain.shape[-1])
    for i, name in enumerate(param_names):
        median = np.median(flat_chain[:, i])
        std = np.std(flat_chain[:, i])
        print(f"{name}: {median:.5f} ± {std:.5f}")

def plot_mcmc_fits_double_diff_sum(
    h5_filename,
    txt_filename,
    csv_path,
    filter_wavelength,
    system_dict,
    configuration_filter=None,
    wavelength_str=None,
    n_samples=50,
    burn_in=0
):
    # Load dataset
    interleaved_values, interleaved_stds, configuration_list = read_csv(
        csv_path, obs_mode="IPOL", obs_filter=filter_wavelength)

    interleaved_stds_proc = process_errors(interleaved_stds, interleaved_values)
    interleaved_values_proc = process_dataset(interleaved_values)

    # Load and parse parameter dictionary
    with open(txt_filename, "r") as f:
        p0_dict = json.load(f)
    p0_values, p_keys = parse_configuration(p0_dict)

    # Generate system Mueller matrix
    system_mm = generate_system_mueller_matrix(system_dict)

    # Load MCMC chain
    with h5py.File(h5_filename, "r") as f:
        chain = f["mcmc"]["chain"][:]
    if chain.shape[0] > chain.shape[1]:  # (nsteps, nwalkers, ndim)
        chain = np.transpose(chain, (1, 0, 2))  # -> (nwalkers, nsteps, ndim)

    flat_chain = chain[:, burn_in:, :].reshape(-1, chain.shape[-1])
    if len(flat_chain) < n_samples:
        raise ValueError(f"Only {len(flat_chain)} samples available post burn-in, but {n_samples} requested.")

    np.random.seed(42)
    sample_indices = np.random.choice(flat_chain.shape[0], size=n_samples, replace=False)

    # Create model predictions from samples
    model_outputs = []
    for idx in sample_indices:
        p_sample = flat_chain[idx]
        updated_mm = update_system_mm(p_sample, p_keys, system_mm)
        intensities = []

        for config in configuration_list:
            values, keywords = parse_configuration(config)
            updated_mm = update_system_mm(values, keywords, updated_mm)

            updated_mm = update_system_mm(["o"], [["wollaston", "beam"]], updated_mm)
            o_intensity = generate_measurement(updated_mm)[0]

            updated_mm = update_system_mm(["e"], [["wollaston", "beam"]], updated_mm)
            e_intensity = generate_measurement(updated_mm)[0]

            intensities.extend([o_intensity, e_intensity])

        model_values = process_model(intensities)
        model_outputs.append(model_values)

    # Extract double differences and sums from data
    dd_values = interleaved_values_proc[::2]
    ds_values = interleaved_values_proc[1::2]
    dd_stds = interleaved_stds_proc[::2]
    ds_stds = interleaved_stds_proc[1::2]

    # Group by IMR theta
    dd_by_theta = {}
    ds_by_theta = {}

    for i, config in enumerate(configuration_list[::2]):
        hwp_theta = config["hwp"]["theta"]
        imr_theta = round(config["image_rotator"]["theta"], 1)

        if configuration_filter is not None and imr_theta != round(configuration_filter, 1):
            continue

        if imr_theta not in dd_by_theta:
            dd_by_theta[imr_theta] = {"hwp_theta": [], "values": [], "stds": []}
            ds_by_theta[imr_theta] = {"hwp_theta": [], "values": [], "stds": []}

        dd_by_theta[imr_theta]["hwp_theta"].append(hwp_theta)
        dd_by_theta[imr_theta]["values"].append(dd_values[i])
        dd_by_theta[imr_theta]["stds"].append(dd_stds[i])

        ds_by_theta[imr_theta]["hwp_theta"].append(hwp_theta)
        ds_by_theta[imr_theta]["values"].append(ds_values[i])
        ds_by_theta[imr_theta]["stds"].append(ds_stds[i])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot data
    for theta in dd_by_theta:
        dd = dd_by_theta[theta]
        ds = ds_by_theta[theta]
        err_dd = axes[0].errorbar(dd["hwp_theta"], dd["values"], yerr=dd["stds"], fmt='o', label=f"{theta}°")
        color = err_dd[0].get_color()
        axes[1].errorbar(ds["hwp_theta"], ds["values"], yerr=ds["stds"], fmt='o', color=color)

    # Plot models
    for model in model_outputs:
        dd_model = model[::2]
        ds_model = model[1::2]

        for theta in dd_by_theta:
            dd = dd_by_theta[theta]
            ds = ds_by_theta[theta]

            axes[0].plot(dd["hwp_theta"], dd_model[:len(dd["hwp_theta"])], alpha=0.1)
            axes[1].plot(ds["hwp_theta"], ds_model[:len(ds["hwp_theta"])], alpha=0.1)

    axes[0].set_xlabel("HWP θ (deg)")
    axes[0].set_ylabel("Double Difference")
    axes[0].legend(title="IMR θ")

    axes[1].set_xlabel("HWP θ (deg)")
    axes[1].set_ylabel("Double Sum")
    axes[1].legend(title="IMR θ")

    if wavelength_str is not None:
        fig.suptitle(f"{wavelength_str}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




