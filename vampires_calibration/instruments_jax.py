import numpy as np
import pandas as pd
import pyMuellerMat
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
from scipy.optimize import minimize
import copy
import matplotlib.pyplot as plt
import emcee
from vampires_calibration import mcmc_helper_funcs_jax as mcmc
from multiprocessing import Pool
import os
import jax.numpy as jnp
from jax import jit
import jax
jax.config.update("jax_enable_x64", True)


#######################################################
###### Functions related to reading in .csv values ####
#######################################################

# Function to safely parse the stored array-like strings
def parse_array_string(x):
    if isinstance(x, str):
        x = x.strip("[]")  # Remove brackets
        try:
            return np.array([float(i) for i in x.split()])  # Convert space-separated numbers to float
        except ValueError:
            return np.nan  # Return NaN if conversion fails
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x)  # Already in the correct format
    return np.nan  # If neither, return NaN

# TODO: Test the MBI function
def read_csv_jax(file_path, obs_mode="IPOL", obs_filter=None):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    MBI_filters = [760, 720, 670, 610]

    # Process only one filter if applicable
    if obs_mode == "MBI":
        MBI_index = MBI_filters.index(obs_filter)
        df = df[df["OBS-MOD"] == "IPOL_MBI"]
    elif obs_filter is not None:
        df = df[df["FILTER01"] == obs_filter]

    # print(type(df["diff"].iloc[0]))

    # Convert relevant columns to float (handling possible conversion errors)
    for col in ["RET-POS1", "D_IMRANG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set errors to NaN if not possible

    # Handle MBI mode: Convert stored strings to arrays and extract MBI_index-th value
    if obs_mode == "MBI":
        for col in ["diff", "sum", "diff_std", "sum_std"]:
            df[col] = df[col].apply(parse_array_string)  # Convert string to array
            df[col] = df[col].apply(lambda x: x[MBI_index] if isinstance(x, np.ndarray) and len(x) > MBI_index else np.nan)  # Extract MBI_index-th element safely
    else:
       for col in ["diff", "sum", "diff_std", "sum_std"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set errors to NaN if not possible 

    # Interleave values from "diff" and "sum"
    interleaved_values = np.ravel(np.column_stack((df["diff"].values, df["sum"].values)))

    # Interleave values from "diff_std" and "sum_std"
    interleaved_stds = np.ravel(np.column_stack((df["diff_std"].values, df["sum_std"].values)))

    # Convert each row's values into a list of two-element lists
    configuration_list = []
    for index, row in df.iterrows():
        # Extracting values from relevant columns
        hwp_theta = row["RET-POS1"]
        imr_theta = row["D_IMRANG"]

        flc_theta = row["U_FLC"]

        if flc_theta == "A":
             flc_theta = 0
        elif flc_theta == "B":
             flc_theta = 45

        # Building dictionary
        row_data = {
            "hwp": {"theta": hwp_theta},
            "image_rotator": {"theta": imr_theta},
            "flc": {"theta": flc_theta}
        }

        # Append two configurations for diff and sum
        configuration_list.append(row_data)

    return interleaved_values, interleaved_stds, configuration_list

########################################################################################
###### Functions related to defining, updating, and parsing instrument dictionaries ####
########################################################################################

def generate_system_mueller_matrix(system_dict):
    """
    Parses a system dictionary and generates a System Mueller Matrix object.
    NOTE: The order given must be from downstream to upstream

    Args:
        system_dict (dict): Dictionary containing components, their types, properties, and order.

    Returns:
        SystemMuellerMatrix: An object representing the system Mueller matrix.
    """
    mueller_matrices = []
    
    # Parse the components and construct individual MuellerMatrix objects
    for component_name, component in system_dict["components"].items():
        component_type_str = component["type"]  # Extract the string name of the function

        # Convert string to actual function in pyMuellerMat.common_mm_functions
        try:
            component_function = getattr(pyMuellerMat.common_mm_functions, component_type_str)
        except AttributeError:
            print(f"Error: '{component_type_str}' is not a valid function in pyMuellerMat.common_mm_functions and will be skipped.")
            continue  # Skip this component if function not found
        
        # Create MuellerMatrix object with the retrieved function
        try:
            mm = MuellerMat.MuellerMatrix(component_function, name=component_name)
        except TypeError as e:
            print(f"TypeError for component '{component_name}' with function '{component_function}': {e}")
            continue  # Skip if an error occurs

        # Check and filter valid properties
        if "properties" in component:
            valid_properties = {key: value for key, value in component["properties"].items() if key in mm.properties}
            invalid_properties = {key: value for key, value in component["properties"].items() if key not in mm.properties}

            if invalid_properties:
                print(f"Warning: Component '{component_name}' has invalid properties {list(invalid_properties.keys())}. "
                      f"These properties will be ignored.")

            if valid_properties:
                mm.properties.update(valid_properties)
            else:
                print(f"Warning: Component '{component_name}' has no valid properties and will be skipped.")
                continue  # Skip adding the component if no valid properties exist
        
        mueller_matrices.append(mm)
    
    # Create the SystemMuellerMatrix object from the list of MuellerMatrix objects
    system_mm = MuellerMat.SystemMuellerMatrix(mueller_matrices)
    return system_mm

def update_system_mm(parameter_values, system_parameters, system_mm):
    '''
    Generates a model dataset for the given set of parameters

    Args: 
    parameter_values: list of parameters - NOTE: cannot be numpy array
    system_parameters: list of two element lists [component_name, parameter_name]
                        example: [['Polarizer', 'Theta'], ['Polarizer', 'Phi']]
    systemMM: pyMuellerMat System Mueller Matrix object
    '''

    for i, system_parameter in enumerate(system_parameters):
        # Unpacking each tuple within system_parameters
        component_name = system_parameter[0]
        parameter_name = system_parameter[1]
        # print("Component Name: " + str(component_name))
        # print("Parameter Name: " + str(parameter_name))
        # Check if the component and parameter exist in the system_mm
        if component_name in system_mm.master_property_dict:
            if parameter_name in system_mm.master_property_dict[component_name]:
                # Update the parameter
                # print("system_mm system_mm.master_property_dict[component_name][parameter_name])
                # print("Parameter Value: " + str(parameter_values[i]))
                system_mm.master_property_dict[component_name][parameter_name] = parameter_values[i]
                # print(system_mm.evaluate())
            else:
                print(f"Parameter '{parameter_name}' not found in component '{component_name}'. Skipping...")
        else:
            print(f"Component '{component_name}' not found in System Mueller Matrix. Skipping...")
    return system_mm

def generate_measurement(system_mm, s_in = np.array([1, 0, 0, 0])):
    '''
    Generate a measurement from a given System Mueller Matrix

    Args: 
    system_mm: pyMuellerMat System Mueller Matrix object (numpy array)
    S_in: Stokes vector of the incoming light (numpy array)

    Returns: 
    S_out: Stokes vector of the outgoing light
    '''
    # print("Mueller Matrix type: ", type(system_mm.evaluate()))
    # print("S_in type: ", type(s_in))
    # print("Mueller Matrix dtype: ", system_mm.evaluate().dtype)
    # print("Stokes Vector dtype: ", s_in.dtype)
    output_stokes = system_mm.evaluate() @ s_in
    return output_stokes

#######################################################
###### Functions for MCMC #############################
#######################################################

# Main MCMC function
def run_mcmc(
    p0_dict, system_mm, dataset, configuration_list,
    priors, bounds, output_h5_file,
    nwalkers=64, nsteps=10000, errors=None, pool_processes=None,
    s_in=np.array([1, 0, 0, 0]), process_dataset=None,
    process_errors=None, process_model=None,
    log_f=-3.0, plot=False, include_sums=True
):
    """
    Run MCMC using emcee with support for dictionary-based parameter inputs.

    This function supports standard system Mueller matrix fitting as well as
    extended likelihoods that include a noise-scaling term (`log_f`) in the model.

    Parameters
    ----------
    p0_dict : dict
        Nested dictionary of initial parameter guesses structured by component.
    system_mm : SystemMuellerMatrix
        The optical system's Mueller matrix object.
    dataset : np.ndarray
        Observed data values (interleaved single differences and sums).
    configuration_list : list of dict
        List of per-measurement configurations (e.g., HWP/FLC angles).
    priors : dict
        Dictionary mapping parameter names to prior functions.
    bounds : dict
        Dictionary of (low, high) tuples for each parameter.
    output_h5_file : str
        Path to the output HDF5 file used to store MCMC results.
    nwalkers : int, optional
        Number of walkers (default is max of 2x parameters or process-scaled).
    nsteps : int, optional
        Number of steps for each walker.
    errors : np.ndarray, optional
        Standard deviations associated with each element of `dataset`.
    pool_processes : int, optional
        Number of parallel processes to use.
    s_in : np.ndarray, optional
        Input Stokes vector for the system (default: [1, 0, 0, 0]).
    process_dataset : callable, optional
        Function to process the dataset before likelihood comparison.
    process_errors : callable, optional
        Function to process errors in the same way as the dataset.
    process_model : callable, optional
        Function to process model outputs before likelihood comparison.
    log_f0 : float, optional
        Initial value for `log_f` if `include_log_f` is True.
    plot : bool, optional
        If True, plots every 100 steps. Only works in .py scripts currently.
    include_sums : bool
        Whether or not to take out double sums from modeling. The default is true
        because this works for VAMPIRES. It does not work for CHARIS.
    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler object containing the MCMC chain.
    p_keys : list of tuple
        List of (component, parameter) key pairs used for tracking parameters.
    """


    p0_values, p_keys = parse_configuration(p0_dict)

    
    p0_values = p0_values + [log_f]             

    ndim = len(p0_values)

    #resume = resume and os.path.exists(output_h5_file)
   
    backend = emcee.backends.HDFBackend(output_h5_file)

    # if not resume or backend.iteration == 0:
    #     backend.reset(nwalkers, ndim)
    #if backend.iteration == 0:
    backend.reset(nwalkers, ndim)

    pos = p0_values + 1e-3 * np.random.randn(nwalkers, ndim)

    args = (
        system_mm, dataset, errors, configuration_list, p_keys, s_in,
        process_model, process_dataset, process_errors, 
        priors, bounds, logl_with_logf, include_sums
        )

    with Pool(processes=pool_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc.log_prob, 
            args=args, pool=pool, backend=backend)
        #sampler.run_mcmc(pos, nsteps, progress=True)
        max_steps = nsteps
        index = 0
        autocorr = np.empty(max_steps)
        old_tau = np.inf
        if plot == True:
            plt.ion()
            fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
        for sample in sampler.sample(pos, iterations=max_steps, progress=True):
            if sampler.iteration % 100 != 0:
                continue
        

        # Compute the autocorrelation time so far
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except emcee.autocorr.AutocorrError:
                continue  # Not enough samples yet

            autocorr[index] = np.mean(tau)
            index += 1
            if plot == True:
                chain = sampler.get_chain()[-200:, :, :]
                for i in range(ndim):
                    axes[i].cla()
                    axes[i].plot(chain[:, :, i], alpha=0.5, linewidth=0.5)
                    axes[i].set_ylabel(f"param {i}")
                axes[-1].set_xlabel("step")
                fig.suptitle(f"Iteration {sampler.iteration}")
                plt.pause(0.01)
        # Check convergence
            if sampler.iteration > 100 * np.max(tau):
                if np.all(np.abs(old_tau - tau) / tau < 0.01):
                    print(f"Converged at iteration {sampler.iteration}")
                    break
            old_tau=tau
    return sampler, p_keys


def logl_with_logf(theta, system_mm, dataset, errors, configuration_list, 
                   system_parameters, s_in, process_model, process_dataset, 
                   process_errors, include_sums=True):
    """
    Log-likelihood function that includes a noise inflation parameter log_f.

    Parameters
    ----------
    theta : jnp.ndarray
        Parameter vector, with the last entry being log_f.
    system_mm : SystemMuellerMatrix
        Optical system Mueller matrix.
    dataset : np.ndarray
        Observed data.
    errors : np.ndarray
        Measurement uncertainties.
    configuration_list : list
        List of measurement configurations.
    system_parameters : list
        List of (component, parameter) keys for fitting.
    s_in : np.ndarray
        Input Stokes vector.
    process_model, process_dataset, process_errors : callable
        Optional transformation functions for model, dataset, and errors.
    include_sums : bool
        Whether or not to take out double sums from modeling. The default is true
        because this works for VAMPIRES. It does not work for CHARIS.

    Returns
    -------
    float
        Log-likelihood value.
    """
    log_f = theta[-1]
    theta = theta[:-1]

    # Generate model output
    
    model_output = model(theta, system_parameters, system_mm, configuration_list,
                            s_in=s_in, process_model=process_model)

    dataset = jnp.array(dataset)
    if errors is not None:
        errors = jnp.array(errors)
        if process_errors is not None:
            errors = process_errors(copy.deepcopy(errors), copy.deepcopy(dataset))
    if process_dataset is not None:
        dataset = process_dataset(copy.deepcopy(dataset))

    if include_sums is False:
     dataset = dataset[::2]
     if errors is not None:
         errors = errors[::2]
     model_output = model_output[::2]
    if errors is not None:
        sigma2 = errors**2 + jnp.exp(2 * log_f)
        return -0.5 * jnp.sum((dataset - model_output)**2 / sigma2 + jnp.log(sigma2))
    else:
        return -0.5 * jnp.sum((dataset - model_output)**2 )


# def mcmc_system_mueller_matrix(p0, system_mm, dataset, errors, configuration_list):
#     '''
#     Perform MCMC on a dataset, using a System Mueller Matrix model

#     Example p0: 

#     p0 = {'Polarizer': {'Theta': 0, 'Phi': 0},
#           'Waveplate': {'Theta': 0, 'Phi': 0},
#           'Retarder': {'Theta': 0, 'Phi': 0},
#           'Sample': {'Theta': 0, 'Phi': 0},
#           'Analyzer': {'Theta': 0, 'Phi': 0}}

#     Args: 
#     p0: dictionary of dictionaries of initial parameters
#     systemMM: pyMuellerMat System Mueller Matrix object
#     dataset: list of measurements
#     configuration_list: list of system dictionaries, one for each measurement in dataset
#     '''

#     #####################################################
#     ###### First, define the data model based on p0 #####
#     #####################################################

#     #Parse p0 into a list of values and a list of lists of keywords
#     p0_values = []
#     p0_keywords = []
#     for component, parameters in p0.items():
#         for parameter, value in parameters.items():
#             p0_values.append(value)
#             p0_keywords.append([component, parameter])

#     #TODO: Fill this out more. 

def minimize_system_mueller_matrix(p0, system_mm, dataset, errors, 
    configuration_list, s_in = None, logl_function = None, 
    process_dataset = None, process_errors = None, process_model = None,
    bounds = None):
    '''
    Perform a minimization on a dataset, using a System Mueller Matrix model
    Args:
        p0: Initial parameter dictionary
        system_mm: System Mueller Matrix object
        dataset: List of measured data
        errors: List of measurement errors
        configuration_list: Instrument configuration for each measurement
        s_in: (Optional) Input Stokes vector (default: [1, 0, 0, 0])
    '''
    
    if s_in is None:
        s_in = np.array([1, 0, 0, 0])  # Default s_in

    p0_values, p0_keywords = parse_configuration(p0)

    # print("p0_values: ", p0_values)
    # print("p0_keywords: ", p0_keywords)

    # Running scipy.minimize
    result = minimize(logl, p0_values, 
        args=(p0_keywords, system_mm, dataset, errors, configuration_list, 
            s_in, logl_function, process_dataset, process_errors, process_model), 
            method='Nelder-Mead', bounds = bounds)
    
    # Saving the final result's logl value
    logl_value = logl(result.x, p0_keywords, system_mm, dataset, errors, 
        configuration_list, s_in=s_in, logl_function=logl_function, 
        process_dataset=process_dataset, process_errors = process_errors, 
        process_model = process_model)
    
    return result, logl_value

def parse_configuration(configuration):
    '''
    Parse a configuration dictionary into a list of values and a list of lists of keywords.

    Args: 
    configuration (dict): Dictionary of system components with their properties.

    Returns: 
    values (list of floats): List of all numerical values.
    keywords (list of two-element lists): List of all [component, parameter] pairs.
    '''
    values = []
    keywords = []

    for component, details in configuration.items():
        for parameter, value in details.items():
            values.append(value)
            keywords.append([component, parameter])

    return values, keywords

def update_p0(p0, result):
    """
    Updates the existing p0 dictionary in place using result_values from scipy.optimize,
    based on the parameter order returned by parse_configuration.

    Parameters:
    -----------
    p0 : dict
        The original nested parameter dictionary (will be updated in-place).
    result_values : list or np.ndarray
        Optimized parameter values from scipy.optimize (e.g., result.x).

    Returns:
    --------
    None
        The function modifies `p0` directly.
    """
    # Use the existing parser to get the keyword pairs in order
    _, p0_keywords = parse_configuration(p0)

    # Sanity check
    if len(p0_keywords) != len(result):
        raise ValueError("Mismatch: result_values length does not match number of parameters in p0.")

    # Perform in-place update
    for (component, parameter), value in zip(p0_keywords, result):
        p0[component][parameter] = value


########################################################################################
###### Functions related to fitting ####################################################
########################################################################################

def model(p, system_parameters, system_mm, configuration_list, s_in=None, 
        process_model = None):
    # Default s_in if not provided
    if s_in is None:
        s_in = np.array([1, 0, 0, 0])

    # Update the system Mueller matrix with parameters that we're fitting
    system_mm = update_system_mm(p, system_parameters, system_mm)

    # Generate a model dataset
    output_intensities = []

    # Save the parameters necessary to switch between o and e beam
    o_beam_values, wollaston_beam_keyword = parse_configuration({'wollaston': {'beam': 'o'}})
    e_beam_values, wollaston_beam_keyword = parse_configuration({'wollaston': {'beam': 'e'}})

    for i in range(len(configuration_list)):
        values, keywords = parse_configuration(configuration_list[i])
        system_mm = update_system_mm(values, keywords, system_mm)

        # Compute the intensity for the ordinary beam
        system_mm = update_system_mm(o_beam_values, wollaston_beam_keyword, system_mm)
        s_out_o = generate_measurement(system_mm, s_in)
        o_intensity = s_out_o[0]

        # Compute the intensity for the extraordinary beam
        system_mm = update_system_mm(e_beam_values, wollaston_beam_keyword, system_mm)
        s_out_e = generate_measurement(system_mm, s_in)
        e_intensity = s_out_e[0]
        output_intensities.append(o_intensity)
        output_intensities.append(e_intensity)

    # Optionally parse the intensities into another variable (e.g., normalized difference)
    if process_model is not None:
        output_intensities = process_model(output_intensities)

    # Convert intensities list to numpy arrays
    output_intensities = np.array(output_intensities)

    return output_intensities


@jit
def build_differences_and_sums(intensities):
    intensities = jnp.array(intensities)
    differences = (intensities[::2] - intensities[1::2])
    sums = intensities[::2] + intensities[1::2]

    return differences, sums
@jit
def build_double_differences_and_sums(differences, sums):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''
    # Making sure that differences and sums are numpy arrays
    differences = jnp.array(differences)
    sums = jnp.array(sums)

    double_differences = (differences[::2]-differences[1::2])/(sums[::2]+sums[1::2])
    double_sums = (differences[::2]+differences[1::2])/(sums[::2]+sums[1::2])

    return double_differences, double_sums
@jit
def normalize_diffs(differences,sums):
    diffs=differences/sums
    return diffs,sums
# ONLY USES DIFFERENCES

def process_model(model_intensities):
    """
    Processes the model intensities to compute double differences.
    
    
    Parameters
    ----------
    model_intensities : list or np.ndarray
        List or array of model intensities, expected to be in pairs.
    
    Returns
    --------
    double_differences : np.ndarray
        Array of simulated double differences
    
    """
    
    model_intensities = np.array(model_intensities)


   
    differences, sums = build_differences_and_sums(model_intensities)
    double_differences, double_sums = build_double_differences_and_sums(differences, sums)


    #Format this into one array. 
        # Interleave the double differences and double sums
    interleaved_values = np.ravel(np.column_stack((double_differences, double_sums)))
         
        # NOTE: Subtracting same FLC state orders (A - B) as Miles
    # Take the negative of this as was done before
    interleaved_values = -interleaved_values
    # Extracting differences (done this way for easy reversal to old format of interleaving)
    return interleaved_values

def process_dataset(input_dataset): 
    # Making sure that input_dataset is a numpy array
    input_dataset = np.array(input_dataset)
    sums = input_dataset[1::2]
    differences = input_dataset[::2]
   


    double_differences, double_sums = build_double_differences_and_sums(differences, sums)

    interleaved_values = np.ravel(np.column_stack((double_differences, double_sums)))

    # Format this into one array.
    return interleaved_values

def process_errors(input_errors, input_dataset): 
    """
    Propagates errors through the same transformations as `process_dataset`.
    
    Args:
        input_errors (numpy array): Original errors in intensities.
        input_dataset (numpy array): Original dataset, needed for normalization steps.
        
    Returns:
        numpy array: Propagated errors for double differences and sums.
    """

    # Ensure input is a NumPy array
    input_errors = np.array(input_errors)
    input_dataset = np.array(input_dataset)


    # Compute errors for differences and sums
    # differences_errors = np.sqrt(input_errors[::2]**2 + input_errors[1::2]**2)
    # sums_errors = np.sqrt(input_errors[::2]**2 + input_errors[1::2]**2)
    # Extract difference and sum errors
    differences_errors = input_errors[::2]
    sums_errors = input_errors[1::2]
    
    # Compute single differences and single sums
    differences = input_dataset[::2]
    sums = input_dataset[1::2]
    double_sums = sums[::2] + sums[1::2]
    differences2 = differences[::2]-differences[1::2]
    squared_diff_errors_sqrt = np.sqrt(differences_errors[::2]**2+differences_errors[1::2]**2)
    squared_sum_errors_sqrt = np.sqrt(sums_errors[::2]**2+sums_errors[1::2]**2)
    # using hypot for numerical stability
    num = np.hypot(double_sums*squared_diff_errors_sqrt,differences2*squared_sum_errors_sqrt)
    # Compute propagated errors for double differences
    double_differences_errors = num/double_sums**2

    # Compute propagated errors for double sums
    double_sums_errors = np.hypot(sums_errors[::2],sums_errors[1::2])

    # Interleave errors to maintain order
    interleaved_errors = np.ravel(np.column_stack((double_differences_errors, double_sums_errors)))
    # Double diffs extracted this way for ease of reverting back to the original setup


    return interleaved_errors

#######################################################
###### Functions related to plotting ##################
#######################################################

def plot_data_and_model(interleaved_values, interleaved_stds, model, 
    configuration_list, imr_theta_filter=None, wavelength=None, save_path = None):
    """
    Plots double difference and double sum measurements alongside model predictions,
    grouped by image rotator angle (D_IMRANG). Optionally filters by a specific 
    image rotator angle and displays a wavelength in the plot title.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed double difference and double sum values.
        Expected format: [dd1, ds1, dd2, ds2, ...].

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    imr_theta_filter : float, optional
        If provided, only measurements with this image rotator angle (rounded to 0.1°) 
        will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").

    Returns
    -------
    None
        Displays two subplots: one for double differences and one for double sums,
        including error bars and model curves.
    """
    # Calculate double differences and sums from interleaved single differences
    interleaved_stds = process_errors(interleaved_stds, interleaved_values)
    interleaved_values = process_dataset(interleaved_values)

    # Extract double differences and double sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Double Difference plot
    ax = axes[0]
    for theta, d in dd_by_theta.items():
        err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
        color = err[0].get_color()
        ax.plot(d["hwp_theta"], d["model"], '-', color=color)
    ax.set_xlabel("HWP θ (deg)")
    ax.set_ylabel("Double Difference")
    ax.legend(title="IMR θ")

    # Double Sum plot
    ax = axes[1]
    for theta, d in ds_by_theta.items():
        err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
        color = err[0].get_color()
        ax.plot(d["hwp_theta"], d["model"], '-', color=color)
    ax.set_xlabel("HWP θ (deg)")
    ax.set_ylabel("Double Sum")
    ax.legend(title="IMR θ")

    # Set a suptitle if wavelength is provided
    if wavelength is not None:
        fig.suptitle(f"{wavelength}nm", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    if save_path != None:
        plt.savefig(save_path)

    plt.show()

