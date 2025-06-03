import os
import sys
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import corner

# Add the directory containing instruments.py to the Python path
custom_module_path = "/home/rebeccaz/Github/vampires_calibration"
sys.path.append(custom_module_path)

import instruments as inst

def load_sampler_data(h5_filename, txt_filename):
    """Copy the .h5 file and load the MCMC chain and parameter names from a .txt file."""
    copied_filename = h5_filename.replace(".h5", "_copy.h5")
    shutil.copy(h5_filename, copied_filename)

    with h5py.File(copied_filename, 'r') as f:
        chain = f['mcmc']['chain'][:]  # shape might be (nsteps, nwalkers, ndim)
        if chain.shape[0] > chain.shape[1]:  # assume first dim is nsteps
            chain = np.transpose(chain, (1, 0, 2))  # now (nwalkers, nsteps, ndim)

    # Load parameter keys from the .txt file
    with open(txt_filename, 'r') as f:
        p0_dict = json.load(f)

    param_names = []
    for component, params in p0_dict.items():
        for param in params.keys():
            param_names.append(f"{component}.{param}")

    return chain, param_names

def inspect_h5_chain_shape(h5_filename):
    with h5py.File(h5_filename, 'r') as f:
        chain = f['mcmc']['chain']
        print(f"Chain shape: {chain.shape} (nwalkers, nsteps, ndim)")

def plot_mcmc_chains(h5_filename, txt_filename, burn_in=0, n_walkers_to_plot=None):
    """
    Load MCMC chain from file and plot trace plots for each parameter.

    Parameters
    ----------
    h5_filename : str
        Path to the .h5 file containing the MCMC chains.

    txt_filename : str
        Path to the .txt file containing the p0 dictionary.

    burn_in : int
        Number of initial steps to discard visually (shown with a red line).

    n_walkers_to_plot : int or None
        Number of random walkers to plot. If None, plots all walkers.
    """
    copied_filename = h5_filename.replace(".h5", "_copy.h5")
    shutil.copy(h5_filename, copied_filename)

    with h5py.File(copied_filename, 'r') as f:
        chain = f['mcmc']['chain'][:]

    with open(txt_filename, 'r') as f:
        p0_dict = json.load(f)

    param_names = [f"{component}.{param}" for component, params in p0_dict.items() for param in params]
    nwalkers, nsteps, ndim = chain.shape

    # Fix for plotting all steps
    x_range = np.arange(nsteps)

    if n_walkers_to_plot is None or n_walkers_to_plot > nwalkers:
        selected_walkers = np.arange(nwalkers)
    else:
        np.random.seed(42)
        selected_walkers = np.random.choice(nwalkers, n_walkers_to_plot, replace=False)

    fig, axs = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)

    for i in range(ndim):
        for j in selected_walkers:
            axs[i].plot(x_range, chain[j, :, i], alpha=0.3)
        axs[i].axvline(burn_in, color='red', linestyle='--', alpha=0.5)
        axs[i].set_ylabel(param_names[i])

    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()


def plot_corner(h5_filename, txt_filename, burn_in=0):
    chain, param_names = load_sampler_data(h5_filename, txt_filename)
    flat_chain = chain[:, burn_in:, :].reshape(-1, chain.shape[-1])
    fig = corner.corner(flat_chain, labels=param_names, show_titles=True)
    plt.show()

def summarize_posteriors(h5_filename, txt_filename, burn_in=0):
    chain, param_names = load_sampler_data(h5_filename, txt_filename)
    flat_chain = chain[:, burn_in:, :].reshape(-1, chain.shape[-1])
    for i, name in enumerate(param_names):
        median = np.median(flat_chain[:, i])
        std = np.std(flat_chain[:, i])
        print(f"{name}: {median:.5f} ± {std:.5f}")

def plot_mcmc_fits(h5_filename, txt_filename, csv_path, filter_wavelength,
                   system_dict, configuration_filter=None, wavelength_str=None,
                   n_samples=50, burn_in=0):
    """
    Plot model predictions from randomly chosen MCMC samples against observed data.

    Parameters
    ----------
    h5_filename : str
        Path to the .h5 file containing the MCMC chains.

    txt_filename : str
        Path to the .txt file containing the p0 dictionary.

    csv_path : str
        Path to the CSV file containing the observed dataset.

    filter_wavelength : str
        Filter wavelength string (e.g., "675-50") to select the dataset.

    system_dict : dict
        Dictionary defining the optical system components.

    configuration_filter : float, optional
        Image rotator angle to filter the configurations by.

    wavelength_str : str, optional
        Wavelength to display in the plot title.

    n_samples : int
        Number of random chains to plot for model fits.

    burn_in : int
        Number of initial steps to discard from the chain.
    """
    from instruments import read_csv, parse_configuration, update_system_mm, generate_system_mueller_matrix
    from instruments import process_model, plot_data_and_model, process_dataset, process_errors
    # from plotting_utils import plot_data_and_model

    # Load dataset
    interleaved_values, interleaved_stds, configuration_list = read_csv(csv_path, obs_mode="IPOL", obs_filter=filter_wavelength)

    # Load and parse parameter dictionary
    with open(txt_filename, 'r') as f:
        p0_dict = json.load(f)

    p0_values, p_keys = parse_configuration(p0_dict)

    # Generate system Mueller matrix from passed-in dictionary
    system_mm = generate_system_mueller_matrix(system_dict)

    # Load MCMC samples
    copied_filename = h5_filename.replace(".h5", "_copy.h5")
    shutil.copy(h5_filename, copied_filename)
    with h5py.File(copied_filename, 'r') as f:
        chain = f['mcmc']['chain'][:]
    flat_chain = chain[:, burn_in:, :].reshape(-1, chain.shape[-1])

    # Sample random chains
    np.random.seed(42)
    sample_indices = np.random.choice(flat_chain.shape[0], size=n_samples, replace=False)
    models = []

    for idx in sample_indices:
        sample = flat_chain[idx]
        updated_mm = update_system_mm(sample, p_keys, system_mm)
        model_output = process_model(updated_mm, configuration_list)  # Ensure process_model does NOT use s_in
        models.append(model_output)

    # Plot each model on top of the data with alpha=0.1
    for model in models:
        plot_data_and_model(interleaved_values, interleaved_stds, model, configuration_list,
                            imr_theta_filter=configuration_filter, wavelength=wavelength_str)
