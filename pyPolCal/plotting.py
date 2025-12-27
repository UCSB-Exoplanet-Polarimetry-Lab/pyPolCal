import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import corner
import json
import h5py
import numpy as np
from pyPolCal.utils import (
    parse_configuration, generate_system_mueller_matrix,
    update_system_mm, generate_measurement, process_model, process_dataset,
    process_errors
)
from pyPolCal.csv_tools import read_csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
from pathlib import Path
import emcee
from pyPolCal.constants import wavelength_bins
import pandas as pd

#######################################
####### Main Plotting Functions #######
#######################################


def plot_data_and_model(interleaved_values, model, 
    configuration_list, interleaved_stds=None,imr_theta_filter=None, wavelength=None, save_path = None, include_sums=True,title=None):
    """
    Plots double difference and double sum measurements alongside model predictions,
    grouped by image rotator angle (D_IMRANG). Optionally filters by a specific 
    image rotator angle and displays a wavelength in the plot title.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed single difference and single sum values.
        Expected format: [sd1, ss1, sd2, ss2, ...]. 


    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.
    

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

    imr_theta_filter : float, optional
        If provided, only measurements with this image rotator angle (rounded to 0.1°) 
        will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").

    include_sums : bool, optional
        Default is True, plotting the double sums as well as the differences. If false, only the
        double differences are plotted, a residual bar is included, and some other plotting things are updated.

    title : str, optional
        Default is the wavelength.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        A tuple containing the Figure and Axes objects of the plot.
    """
    # Create an array of zeroes if no stds are provided
    if interleaved_stds is None:
        interleaved_stds = np.zeros_like(interleaved_values)
    interleaved_stds = process_errors(interleaved_stds, interleaved_values)
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]
    interleaved_values = process_dataset(interleaved_values)

    # Extract double differences and double sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_model = model[::2]
    ds_model = model[1::2]

    # Group by image_rotator theta
    dd_by_theta = {}
    ds_by_theta = {}

    for i, config in enumerate(configuration_list[::2]):
        hwp_theta = config["hwp"]["theta"]
        imr_theta = round(config["image_rotator"]["theta"], 1)

        if imr_theta_filter is not None and imr_theta != round(imr_theta_filter, 1):
            continue

        if imr_theta not in dd_by_theta:
            dd_by_theta[imr_theta] = {"hwp_theta": [], "values": [], "stds": [], "model": []}
        dd_by_theta[imr_theta]["hwp_theta"].append(hwp_theta)
        dd_by_theta[imr_theta]["values"].append(dd_values[i])
        dd_by_theta[imr_theta]["stds"].append(dd_stds[i])
        dd_by_theta[imr_theta]["model"].append(dd_model[i])

        if imr_theta not in ds_by_theta:
            ds_by_theta[imr_theta] = {"hwp_theta": [], "values": [], "stds": [], "model": []}
        ds_by_theta[imr_theta]["hwp_theta"].append(hwp_theta)
        ds_by_theta[imr_theta]["values"].append(ds_values[i])
        ds_by_theta[imr_theta]["stds"].append(ds_stds[i])
        ds_by_theta[imr_theta]["model"].append(ds_model[i])
    # Create the plots
    if include_sums is True:
        num_plots = 2
        fig, axes = plt.subplots(1, num_plots, figsize=(14, 6), sharex=True)

    elif include_sums is False:
        fig, axarr = plt.subplots(
        2, 1, 
        figsize=(10, 6), 
        gridspec_kw={"height_ratios": [3, 1]}, 
        sharex=True
        )
        ax = axarr[0]
        small_ax = axarr[1]

    # Double Difference plot
    if include_sums is True:
        ax = axes[0]
        for theta, d in dd_by_theta.items():
           err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
           color = err[0].get_color()
           ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel(r"HWP $\theta$ (deg)")
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"IMR $\theta$")
    # Double Sum plot
        ax = axes[1]
        for theta, d in ds_by_theta.items():
            err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
            color = err[0].get_color()
            ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel(r"HWP $\theta$  (deg)")
        ax.set_ylabel("Double Sum")
        ax.legend(title=r"IMR $\theta$")
    elif include_sums is False:
        for theta, d in dd_by_theta.items():
           err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
           color = err[0].get_color()
           ax.plot(d["hwp_theta"], d["model"], color=color)
           residuals =  ((np.array(d["values"]) - np.array(d["model"])))*100
           small_ax.scatter(d['hwp_theta'],residuals,color=color)
        small_ax.axhline(0, color='black', linewidth=1)
        small_ax.set_xlabel(r"HWP $\theta$ (deg)")
        small_ax.set_ylabel(r"Residual ($\%$)", fontsize = 15)
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"IMR $\theta$", fontsize=10)
        ax.grid()

    # Set a suptitle if wavelength is provided
    if wavelength is not None and title is None:
        fig.suptitle(f"{wavelength}nm")
    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    if save_path != None:
        plt.savefig(save_path,dpi=600, bbox_inches='tight')

    plt.show()
    return fig, ax


def plot_config_dict_vs_wavelength(component, parameter, json_dir, custom_ax=None,save_path=None, title=None, axtitle=None):
    """
    Plots a parameter in the JSON configuration dictionaries vs wavelength.
    Only works if all JSON dictionaries are in a directory labeled
    by bin. Sets parameters phi, delta_theta, and theta to degrees.
    Returns an array of the parameters to allow for custom plotting.
    This plot is similar to the one in van Holstein 2020.
    
    Parameters
    ----------
    component : str
        The name of the component (e.g., 'image_rotator', 'hwp', etc.).

    parameter : str
        The key of the parameter in the system dictionary.

    json_dir : str or Path
        The directory containing the JSON configuration dictionaries for all 22 bins.
        Make sure the directory only contains these 22 JSON files. 
        Component names are 'lp' for calibration polarizer, 'image_rotator' for image rotator,
        and 'hwp' for half-wave plate.

    custom_ax : matplotlib Axes, optional
        If provided, the plot will be drawn on this Axes object instead of creating a new one.

    save_path : str or Path, optional
        If specified, saves the plot to this path. Otherwise, displays the plot.

    title : str, optional
        Title for the plot. If not provided, a default title is used.

    axtitle : str, optional
        Title for the y-axis. If not provided, a default title is used.
    
    Returns
    -------
    parameters : np.ndarray
        An array of the parameter values extracted from the JSON files.
        To plot, plot against the default CHARIS wavelength bins (can
        be found in instruments.py).

    fig, ax : matplotlib Figure and Axes
        The Figure and Axes objects of the plot.
    """

    # Check filepaths

    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a valid directory.")
    if save_path is not None:
        save_path = Path(save_path)

    # Load JSON files

    json_files = sorted(json_dir.glob("*.json"))

    # Check for correct file amount

    if len(json_files) != 22:
        raise ValueError(f"Expected 22 JSON files, found {len(json_files)}.")
    
    # Check for bins
 
    for f in json_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not match expected naming convention.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
     # Sort Jsons

    sorted_files = sorted(json_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))

    # Extract parameters

    parameters = []

    for f in sorted_files:
        with open(f, 'r') as file:
            data = json.load(file)
            if component not in data:
                raise ValueError(f"Component '{component}' not found in {f.name}.")
            if parameter not in data[component]:
                raise ValueError(f"Parameter '{parameter}' not found in component '{component}' in {f.name}.")
            # Set relevant components to degrees
            if parameter == 'phi':
                data[component][parameter] = np.degrees(data[component][parameter])
            
            parameters.append(data[component][parameter])

    # Convert to numpy array for plotting

    parameters = np.array(parameters)

    # Plot vs wavelength bins

    fig, ax = plt.subplots(figsize=(10, 6))
    if custom_ax: # set custom ax
        ax = custom_ax
    ax.scatter(wavelength_bins, parameters, marker='x',color='black', linewidths = 1)
    ax.set_xlabel('Wavelength (nm)')
    if axtitle is not None:
        ax.set_ylabel(axtitle)
    else:
        ax.set_ylabel(parameter)
    if title is None:
        ax.set_title(f'{component}: {parameter} vs wavelength')
    else:
        ax.set_title(title)
    ax.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return parameters,fig,ax


def plot_data_and_model_x_imr(interleaved_values, model, 
    configuration_list, interleaved_stds=None, hwp_theta_filter=None, wavelength=None, save_path = None,title=None):
    """
    Plots single differences vs imr angle for some amount of HWP angles. Similar to figure 6 in
    Joost t Hart 2021.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed 
        single differences and sums.


    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.
        If charis use single differences and sums. 

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    interleaved_stds : np.ndarray, optional
        Interleaved array of standard deviations corresponding to the observed values.

    hwp_theta_filter : float, optional
        If provided, only measurements with this hwp angle will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").
    title: str, optional
        Optional title

    Returns
    -------
    fig, ax, small_ax
    """
    # Create an array of zeroes if no stds are provided
    if interleaved_stds is None:
        interleaved_stds = np.zeros_like(interleaved_values)
    # Process into double diffs
    interleaved_stds = process_errors(interleaved_stds,interleaved_values)
    interleaved_values = process_dataset(interleaved_values)

    
    # Extract double differences and double sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]
    dd_model = model[::2]
    ds_model = model[1::2]

    # Group by hwp theta
    dd_by_theta = {}
    ds_by_theta = {}
   
    
    for i, config in enumerate(configuration_list[::2]):
        imr_theta = config["image_rotator"]["theta"]
        hwp_theta = config["hwp"]["theta"]

        if hwp_theta_filter is not None and not np.any(np.isclose(hwp_theta, hwp_theta_filter, atol=1e-2)):
         continue

        if hwp_theta not in dd_by_theta:
            dd_by_theta[hwp_theta] = {"imr_theta": [], "values": [], "stds": [], "model": []}
        dd_by_theta[hwp_theta]["imr_theta"].append(imr_theta)
        dd_by_theta[hwp_theta]["values"].append(dd_values[i])
        dd_by_theta[hwp_theta]["stds"].append(dd_stds[i])
        dd_by_theta[hwp_theta]["model"].append(dd_model[i])

        if hwp_theta not in ds_by_theta:
            ds_by_theta[hwp_theta] = {"imr_theta": [], "values": [], "stds": [], "model": []}
        ds_by_theta[hwp_theta]["imr_theta"].append(imr_theta)
        ds_by_theta[hwp_theta]["values"].append(ds_values[i])
        ds_by_theta[hwp_theta]["stds"].append(ds_stds[i])
        ds_by_theta[hwp_theta]["model"].append(ds_model[i])

   
    num_plots = 1
    sizex=10
    fig, axarr = plt.subplots(
    2, 1, 
    figsize=(sizex, 6), 
    gridspec_kw={"height_ratios": [3, 1]}, 
    sharex=True
    )

    ax = axarr[0]
    small_ax = axarr[1]

    # Double Difference plot
    
    for idx, (theta, d) in enumerate(dd_by_theta.items()):
        hart_cmap = ['cornflowerblue','paleturquoise','orange','red']
        color=hart_cmap[idx]
        err = ax.errorbar(d["imr_theta"], d["values"], color=color,yerr=d["stds"], fmt='o', label=f"{theta}°")
        #color = err[0].get_color
        ax.plot(d["imr_theta"], d["model"], '-', color=color)
        residuals =  (np.array(d["values"]) - np.array(d["model"]))*100
        small_ax.scatter(d['imr_theta'],residuals,color=color)
    small_ax.axhline(0, color='black', linewidth=1)
    small_ax.grid(which='major', axis='y', linestyle='-', linewidth=0.5, color='black')
    small_ax.grid(which='minor',axis='y', linestyle='-', linewidth=0.3, color='gray')
    small_ax.set_xlabel(r"IMR $\theta$ (deg)")
    #small_ax.set_xlim(0,180)
    #ax.invert_yaxis()
    small_ax.set_ylabel(r"Residual ($\%$)", fontsize = 15)
    #small_ax.yaxis.set_minor_locator(MultipleLocator(1))
    small_ax.xaxis.set_major_locator(MultipleLocator(10))
    small_ax.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='gray')
    small_ax.tick_params(axis='y', which='minor', labelleft=False)
    ax.set_ylabel("Double Difference")
    ax.legend(title=r"HWP $\theta$", fontsize=10, loc='upper right')
    ax.grid()

    # Set a suptitle if wavelength is provided
    if wavelength is not None and title is None:
        fig.suptitle(f"{wavelength}nm")
        ax.xaxis.set_major_locator(MultipleLocator(10))    
        ax.xaxis.set_minor_locator(MultipleLocator(1)) 
    plt.tight_layout(rect=[0, 0, 1, 0.95])  #

    if title:
        fig.suptitle(title)

    if save_path != None:
        plt.savefig(save_path,dpi=600, bbox_inches='tight')


    return fig, ax,small_ax

def plot_data_and_model_alt(interleaved_values, model, 
    configuration_list, interleaved_stds=None,
    hwp_theta_filter=None, wavelength=None, save_path=None,
    include_sums=True, title=None):

    """
    Plots double difference and double sum measurements alongside model predictions,
    grouped by altitude angle. Optionally filters by a specific 
    image rotator angle and displays a wavelength in the plot title.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed single difference and single sum values.
        Expected format: [sd1, ss1, sd2, ss2, ...]. 


    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.
        If charis use single differences and sums. 

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

    imr_theta_filter : float, optional
        If provided, only measurements with this image rotator angle (rounded to 0.1°) 
        will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").

    include_sums : bool, optional
        Default is True, plotting the double sums as well as the differences. If false, only the
        double differences are plotted, a residual bar is included, and some other plotting things are updated.

    title : str, optional
        Default is the wavelength.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        A tuple containing the Figure and Axes objects of the plot.
    """
    
    if interleaved_stds is None:
        interleaved_stds = np.zeros_like(interleaved_values)

    interleaved_stds = process_errors(interleaved_stds, interleaved_values)
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]
    interleaved_values = process_dataset(interleaved_values)

    # Extract double differences and sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_model = model[::2]
    ds_model = model[1::2]

    # Group by HWP theta
    dd_by_hwp = {}
    ds_by_hwp = {}

    for i, config in enumerate(configuration_list[::2]):
        hwp_theta = round(config["hwp"]["theta"], 1)
        pa = config["altitude_rot"]["pa"]

        if hwp_theta_filter is not None and hwp_theta != round(hwp_theta_filter, 1):
            continue

        if hwp_theta not in dd_by_hwp:
            dd_by_hwp[hwp_theta] = {"pa": [], "values": [], "stds": [], "model": []}
        dd_by_hwp[hwp_theta]["pa"].append(pa)
        dd_by_hwp[hwp_theta]["values"].append(dd_values[i])
        dd_by_hwp[hwp_theta]["stds"].append(dd_stds[i])
        dd_by_hwp[hwp_theta]["model"].append(dd_model[i])

        if hwp_theta not in ds_by_hwp:
            ds_by_hwp[hwp_theta] = {"pa": [], "values": [], "stds": [], "model": []}
        ds_by_hwp[hwp_theta]["pa"].append(pa)
        ds_by_hwp[hwp_theta]["values"].append(ds_values[i])
        ds_by_hwp[hwp_theta]["stds"].append(ds_stds[i])
        ds_by_hwp[hwp_theta]["model"].append(ds_model[i])

    # Create plots
    if include_sums:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    else:
        fig, axarr = plt.subplots(
            2, 1,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
        ax = axarr[0]
        small_ax = axarr[1]

    # Double Difference
    if include_sums:
        ax = axes[0]
        for hwp, d in dd_by_hwp.items():
            err = ax.errorbar(
                d["pa"], d["values"],
                yerr=d["stds"], fmt='o',
                label=f"{hwp}°"
            )
            color = err[0].get_color()
            ax.plot(d["pa"], d["model"], '-', color=color)

        ax.set_xlabel("Altitude angle PA (deg)")
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"HWP $\theta$")

        # Double Sum
        ax = axes[1]
        for hwp, d in ds_by_hwp.items():
            err = ax.errorbar(
                d["pa"], d["values"],
                yerr=d["stds"], fmt='o',
                label=f"{hwp}°"
            )
            color = err[0].get_color()
            ax.plot(d["pa"], d["model"], '-', color=color)

        ax.set_xlabel("Altitude angle PA (deg)")
        ax.set_ylabel("Double Sum")
        ax.legend(title=r"HWP $\theta$")

    else:
        for hwp, d in dd_by_hwp.items():
            err = ax.errorbar(
                d["pa"], d["values"],
                yerr=d["stds"], fmt='o',
                label=f"{hwp}°"
            )
            color = err[0].get_color()
            ax.plot(d["pa"], d["model"], '-', color=color)

            residuals = (np.array(d["values"]) - np.array(d["model"])) * 100
            small_ax.scatter(d["pa"], residuals, color=color)

        small_ax.axhline(0, color='black', linewidth=1)
        small_ax.set_xlabel("Altitude angle PA (deg)")
        small_ax.set_ylabel(r"Residual ($\%$)")
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"HWP $\theta$")
        ax.grid()

    if wavelength is not None and title is None:
        fig.suptitle(f"{wavelength}nm")
    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
    return fig, ax



####################################
########## MCMC Plotting ###########
####################################

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


def summarize_median_posterior(h5_path, p0_dict, step_range=(0, None)):
    """
    Summarizes the median and 1 sigma credible interval (16th-84th percentile) 
    for each parameter in an MCMC run stored in an emcee HDF5 backend file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the emcee HDF5 backend file.
    p0_dict : dict
        Initial parameter dictionary structured as nested component: {param: val}.
    step_range : tuple
        Optional (start, stop) tuple to slice the chain steps.

    Returns
    -------
    dict
        Dictionary of median and 1 sigma intervals for each parameter.
    """
    h5_path = Path(h5_path)
    reader = emcee.backends.HDFBackend(h5_path)
    chain = reader.get_chain()

    # Flatten the chain over walkers
    flat_chain = chain[step_range[0]:step_range[1], :, :].reshape(-1, chain.shape[-1])

    # Extract parameter keys
    _, param_keys = parse_configuration(p0_dict)

    summary = {}

    print("Posterior Medians and 1 sigma Credible Intervals:")
    for i, comp in enumerate(param_keys):
        samples = flat_chain[:, i]
        median = np.median(samples)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)
        err_low = median - lower
        err_high = upper - median
        component = comp[0]
        key = comp[1]
        if component not in summary:
            summary[component] = {}
        summary[component][key] = {
            "median": median,
            "-1sigma": err_low,
            "+1sigma": err_high
}
        print(f"{component},{key}: {median:.5f} (+{err_high:.5f}/-{err_low:.5f})")

    return summary


#########################################
##### Special Plotting Functions ########
#########################################


def plot_fluxes(csv_path, plot_save_path=None):
    """Plot left and right beam fluxes as a function of the HWP angle for one 
    wavelength bin and derotator angle from a CSV containing headers "LCOUNTS", 
    "RCOUNTS", "RET-ANG1", "D_IMRANG" and "wavelength_bin".
    This can be obtained from the write_fits_info_to_csv function.

    Parameters
    -----------
    csv_path : str or Path
        Path to the specified CSV file.

    plot_save_path : str or Path, optional
        If provided, the plot will be saved to this path. Must end with '.png'.

    Returns
    --------
    fig, ax : matplotlib Figure and Axes
        A tuple containing the Figure and Axes objects of the plot.

    """
    
    # check if csv_path is a valid file path

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    # read csv file into pandas dataframe

    df = pd.read_csv(csv_path)

    # check if necessary columns are present

    required_columns = ['LCOUNTS', 'RCOUNTS', 'RET-ANG1', 'wavelength_bin', 'D_IMRANG']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the CSV file.")
        
    # check for multiple wavelength bins
    unique_bins = df['wavelength_bin'].unique()
    if len(unique_bins) > 1:
        raise ValueError("CSV file contains multiple wavelength bins. Please filter to a single bin before plotting.")
    
    # extract data for plotting

    hwp_angles = df['RET-ANG1'].values
    left_counts = df['LCOUNTS'].values
    right_counts = df['RCOUNTS'].values
    derotator_angles = df['D_IMRANG'].values
    wavelength_bin = df['wavelength_bin'].values[0]

# plot counts as a function of HWP angle for each derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    unique_angles = np.unique(derotator_angles)
    colors = [
    cmap(i / max(len(unique_angles) - 1, 1))
    for i in range(len(unique_angles))
]
    for idx, derotator_angle in enumerate(unique_angles):
        mask = derotator_angles == derotator_angle
        color = colors[idx]  # use a different color for each derotator angle
        sort_order = np.argsort(hwp_angles[mask])
        ax.plot(hwp_angles[mask][sort_order], left_counts[mask][sort_order], marker='o', label=f'{derotator_angle}° (L)', color=color)
        ax.plot(hwp_angles[mask][sort_order], right_counts[mask][sort_order], marker='x', linestyle='--', label=f'{derotator_angle}° (R)', color=color)
    ax.set_xlabel('HWP Angle (degrees)')
    ax.set_ylabel('Counts')
    ax.set_title(f'L/R Counts vs HWP Angle at {wavelength_bin} nm')
    ax.legend(loc='lower right', fontsize='small', title= r'IMR $\theta$')
  


    ax.grid()


    # save plot if a path is provided

    if plot_save_path:
        plot_save_path = Path(plot_save_path)
        if not plot_save_path.parent.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {plot_save_path.parent}")
        if not plot_save_path.suffix == '.png':
            raise ValueError("Plot save path must end with '.png'.")
        fig.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")
    return fig, ax


def plot_polarimetric_efficiency(json_dir, bins, save_path=None, title=None):
    """
    Plots the polarimetric efficiency from JSON configuration dictionaries vs derotator angle.
    Works with the result of the fit_CHARIS_mueller_matrix_by_bin() function
    Only works if all 22 JSON dictionaries are in a directory labeled
    by bin: formatted bin{bin} in the filepath. Returns an array of the polarimetric efficiency values.
    This plot is similar to the one in van Holstein 2020. Also plots
    where measured derotator angles would be with an 'x' marker.
    
    Parameters
    ----------
    json_dir : str or Path
        The directory containing the JSON system dictionaries for all 22 bins.
        Make sure the directory only contains these 22 JSON files. Component names
        are 'lp' for calibration polarizer, 'image_rotator' for image rotator,
        and 'hwp' for half-wave plate.
    bins : np.ndarray
        An array of wavelength bins to simultaneously plot.
    save_path : str or Path, optional
        If specified, saves the plot to this path. 
    title : str, optional
        Title for the plot. If not provided, a default title is used.

    Returns
    -------
    polarimetric_efficiency : np.ndarray
        A 2 dimensional array of the polarimetric efficiency values extracted from the JSON files
        where the first dimension corresponds to the wavelength bins and the second dimension represents the derotator angles.
    fig, ax : matplotlib Figure and Axes
        The Figure and Axes objects of the plot.
    """
    # Check filepaths

    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a valid directory.")
    if save_path is not None:
        save_path = Path(save_path)

    # Load JSON files

    json_files = sorted(json_dir.glob("*.json"))

    # Check for correct file amount

    if len(json_files) != 22:
        raise ValueError(f"Expected 22 JSON files, found {len(json_files)}.")
    
    # Check for bins
 
    for f in json_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not match expected naming convention.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
     # Sort Jsons

    sorted_files = sorted(json_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))

    # Get derotator angles

    derotator_angles_measured = np.array([45, 57.5,70,82.5,95,107.5,120,132.5])
    derotator_angles = np.linspace(0,180,361)

    # Get polarimetric efficiencies


    output = []
    for wavelength_bin in bins:
        # Create efficiency array
        efficiencies = []
        # Extract the Mueller matrix for each derotator angle
        for derotator_angle in derotator_angles:
            # Define system dictionary components for the current wavelength bin and derotator angle
            file = sorted_files[wavelength_bin]
            data = json.load(open(file, 'r'))
           # Parse the dictionary into usable values
            values, keywords = parse_configuration(data)

            # Generate system Mueller matrix without Wollaston prism 

            sys_dict = {
            "components" : {
              #   "wollaston" : {
              #   "type" : "wollaston_prism_function",
              #  "properties" : {"beam": "o"},
              #  "tag": "internal",
              #  },
                "image_rotator" : {
                    "type" : "elliptical_retarder_function",
                    "properties" : {"phi_h": 0,"phi_45": 0, "phi_r": 0, "theta": derotator_angle, "delta_theta": 0},
                    "tag": "internal",
                },
                "hwp" : {
                    "type" : "general_retarder_function",
                    "properties" : {"phi":0, "theta": 0, "delta_theta": 0},
                    "tag": "internal",
                },
                "lp" : {
                    "type": "general_linear_polarizer_function_with_theta",
                    "properties": {"epsilon":1, "delta_theta": 0},
                    "tag": "internal",
                }}
            }

            # generate Mueller matrix object

            system_mm = generate_system_mueller_matrix(sys_dict)

            # Update the Mueller matrix with the model

            updated_system_mm = update_system_mm(values, keywords, system_mm)

            # Calculate the polarimetric efficiency

            M_10 = updated_system_mm.evaluate()[1,0]
            M_20 = updated_system_mm.evaluate()[2,0]
            M_00 = updated_system_mm.evaluate()[0,0]
            efficiency = np.sqrt(M_10**2 + M_20**2) / M_00
            efficiencies.append(efficiency)
        output.append(efficiencies)

    pol_efficiencies = np.array(output)

    # Grab measured angles from pol efficiencies

    measured_indices = []
    for idx,angle in enumerate(derotator_angles):
        if angle in derotator_angles_measured:
            measured_indices.append(idx)
    pol_efficiencies_measured = pol_efficiencies[:, measured_indices]

    # Plot vs derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, wavelength_bin in enumerate(bins):
        ax.plot(derotator_angles, pol_efficiencies[i], label=f"{int(wavelength_bins[wavelength_bin])} nm")
        ax.scatter(derotator_angles_measured, pol_efficiencies_measured[i], marker='x', color='black')
    ax.scatter([], [], marker='x', color='black', label='Angles Used in Data')
    ax.set_xlabel('Derotator Angle (degrees)')
    ax.set_ylabel('Polarimetric Efficiency')
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if title is None:
        ax.set_title('Polarimetric Efficiency vs Derotator Angle')
    else:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return pol_efficiencies, fig, ax


def plot_pol_efficiency_from_data(csv_dir, bins, save_path=None, title=None):
    """Plots polarimetric efficiency as a function of 
    the derotator angle from a csv with relevant columns (can be
    obtained from write_fits_info_to_csv) assuming a total linearly horizontally
    polarized source. This assumes 8 derotator angles and 9 hwp angles. Can plot
    multiple wavelength bins simultaneously.
    
    Parameters
    -----------
    csv_path : str or Path
        Path to the CSV directory containing csvs with relevant bins.

    save_path : str or Path, optional
        If provided, the plot will be saved to this path. Must end with '.png'.

    title : str, optional
        Title of the plot. If None, a default title will be used.
    Returns
    --------
    pol_efficiency : np.array
        Polarimetric efficiency calculated from the interleaved values.
    fig, ax : matplotlib Figure and Axes
        The Figure and Axes objects of the plot.
    """
    csv_dir= Path(csv_dir)

    # Load csvs

    csv_files = sorted(csv_dir.glob("*.csv"))

    # Check for bins
 
    for f in csv_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not contain the bin number.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
    # Sort csvs and extract interleaved values
    derotator_angles = np.linspace(45,132.5,8)
    sorted_files = sorted(csv_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))
    output = []
    for wavelength_bin in bins:
        # Create efficiency array
        efficiencies = []
        file = sorted_files[wavelength_bin]
        interleaved_values = read_csv(file)[0]
        diffs = interleaved_values[::2]
        # Extract the values for each derotator angle
        for i in range(8):  # 8 derotator angles
            start = i * 9
            hwp_diffs = diffs[start:start + 9]  # 9 HWP angles for this derotator
            # Double-difference using HWP 0°, 22.5°, 45°, 67.5°
            try:
                Q = 0.5 * (hwp_diffs[0] - hwp_diffs[4])  # 0° - 45°
                U = 0.5 * (hwp_diffs[2] - hwp_diffs[6])  # 22.5° - 67.5°
            except IndexError:
                print(f"Error: Missing HWP angle at derotator index {i}, file {file}")
                Q = U = 0.0

            pol_eff = np.sqrt(Q**2 + U**2)
            efficiencies.append(pol_eff)
        output.append(efficiencies)
    pol_efficiencies = np.array(output)

    # Plot vs derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, wavelength_bin in enumerate(bins):
        ax.plot(derotator_angles, pol_efficiencies[i], label=f"{int(wavelength_bins[wavelength_bin])} nm")
        ax.scatter(derotator_angles, pol_efficiencies[i], marker='x', alpha=0.7, color='black')
    ax.set_xlabel('Derotator Angle (degrees)')
    ax.set_ylabel('Polarimetric Efficiency')
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if title is None:
        ax.set_title('Polarimetric Efficiency vs Derotator Angle')
    else:
        ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='lower center')
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return pol_efficiencies, fig, ax



    
