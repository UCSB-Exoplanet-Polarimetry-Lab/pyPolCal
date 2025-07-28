import numpy as np
import pandas as pd
import pyMuellerMat
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
from scipy.optimize import minimize
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import emcee
import mcmc_helper_funcs_jax as mcmc
from multiprocessing import Pool
import copy
import os
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit

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
def read_csv(file_path, obs_mode="IPOL", obs_filter=None, flc_theta_a = 0, 
    flc_theta_b = 45):
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
             flc_theta = flc_theta_a
        elif flc_theta == "B":
             flc_theta =flc_theta_b

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
            elif parameter_name != "log_f":
            # Carving out print statement exceptions for log_f
                print(f"Parameter '{parameter_name}' not found in component '{component_name}'. Skipping...")
        elif component_name != "log_f":
        # Carving out print statement exceptions for log_f
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
    p0_dict, system_mm, dataset, errors, configuration_list,
    priors, bounds, logl_function, output_h5_file,
    nwalkers=64, nsteps=10000, pool_processes=None, 
    s_in=np.array([1, 0, 0, 0]), process_dataset=None, 
    process_errors=None, process_model=None, resume=True,
    include_log_f=False, log_f=-3.0
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
        Observed data values (interleaved double differences and sums).
    errors : np.ndarray
        Standard deviations associated with each element of `dataset`.
    configuration_list : list of dict
        List of per-measurement configurations (e.g., HWP/FLC angles).
    priors : dict
        Dictionary mapping parameter names to prior functions.
    bounds : dict
        Dictionary of (low, high) tuples for each parameter.
    logl_function : callable
        Log-likelihood function to evaluate model fit.
    output_h5_file : str
        Path to the output HDF5 file used to store MCMC results.
    nwalkers : int, optional
        Number of walkers (default is max of 2x parameters or process-scaled).
    nsteps : int, optional
        Number of steps for each walker.
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
    resume : bool, optional
        If True and the HDF5 file exists, resume from saved state.
    include_log_f : bool, optional
        If True, appends a `log_f` noise inflation parameter to the parameter list.
    log_f0 : float, optional
        Initial value for `log_f` if `include_log_f` is True.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler object containing the MCMC chain.
    p_keys : list of tuple
        List of (component, parameter) key pairs used for tracking parameters.
    """

    p0_values, p_keys = parse_configuration(p0_dict)

    if include_log_f:
        p0_values = p0_values + [log_f]
        p_keys.append(("log_f", "log_f"))
        
        # Add default bounds for log_f
        bounds["log_f"] = {"log_f": (-10, 0)}

        # Add default uniform prior for log_f
        priors.setdefault("log_f", {})
        priors["log_f"]["log_f"] = {
            "type": "uniform",
            "kwargs": {"low": -10, "high": 0}
        }

    ndim = len(p0_values)

    resume = os.path.exists(output_h5_file)
    backend = emcee.backends.HDFBackend(output_h5_file)

    if not resume or backend.iteration == 0:
        backend.reset(nwalkers, ndim)

    pos = p0_values + 1e-3 * np.random.randn(nwalkers, ndim)

    args = (
        system_mm, dataset, errors, configuration_list, p_keys, s_in,
        process_model, process_dataset, process_errors,
        priors, bounds, logl_with_logf
    )

    with Pool(processes=pool_processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc.log_prob, 
            args=args, pool=pool, backend=backend)
        sampler.run_mcmc(pos, nsteps, progress=True)

    return sampler, p_keys


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

def logl_with_logf(theta, system_mm, dataset, errors, configuration_list, 
                   system_parameters, s_in, process_model, process_dataset, 
                   process_errors):
    """
    Log-likelihood function that includes a noise inflation parameter log_f.
    """
    log_f = theta[-1]
    theta = theta[:-1]

    # Generate model output
    model_output = model(theta, system_parameters, system_mm, configuration_list,
                         s_in=s_in, process_model=process_model)

    # Convert to jax arrays
    dataset = jnp.array(dataset)
    errors = jnp.array(errors)

    # Save raw copies before processing
    raw_dataset = copy.deepcopy(dataset)
    raw_errors = copy.deepcopy(errors)

    # Apply processing
    if process_dataset is not None:
        dataset = process_dataset(raw_dataset)
    if process_errors is not None:
        errors = process_errors(raw_errors, raw_dataset)

    sigma2 = errors**2 + jnp.exp(2 * log_f)
    return -0.5 * jnp.sum((dataset - model_output)**2 / sigma2 + jnp.log(sigma2))

def logl(p, system_parameters, system_mm, dataset, errors, configuration_list, 
         s_in=None, logl_function=None, process_dataset=None, process_errors=None, 
         process_model=None):
    """
    Compute the log-likelihood of a model given a dataset and system configuration.

    This function evaluates how well a set of system Mueller matrix parameters
    (given by `p`) reproduce the observed dataset, using a chi-squared-based 
    likelihood metric or a user-defined log-likelihood function.

    Parameters
    ----------
    p : list of float
        List of current parameter values to optimize (flattened).
    system_parameters : list of [str, str]
        List of [component_name, parameter_name] pairs corresponding to `p`.
    system_mm : pyMuellerMat.MuellerMat.SystemMuellerMatrix
        Mueller matrix model of the optical system.
    dataset : np.ndarray
        Interleaved observed data values (e.g., [dd1, ds1, dd2, ds2, ...]).
    errors : np.ndarray
        Measurement errors associated with `dataset`, in the same order.
    configuration_list : list of dict
        Each dict describes the instrument configuration for a measurement, including 
        settings like HWP angle, FLC state, etc.
    s_in : np.ndarray, optional
        Input Stokes vector, default is unpolarized light [1, 0, 0, 0].
    logl_function : callable, optional
        A custom function with signature `logl_function(p, model, data, errors)` 
        that returns the log-likelihood. If None, default chi-squared is used.
    process_dataset : callable, optional
        Function to transform the dataset (e.g., normalize or reduce dimensionality).
    process_errors : callable, optional
        Function to propagate errors through the same transformation as `process_dataset`.
    process_model : callable, optional
        Function to apply the same transformation to the model predictions as to the data.

    Returns
    -------
    float
        The computed log-likelihood value (higher is better).
    """

    # print("Entered logl")

    # Generating a list of model predicted values for each configuration - already parsed
    output_intensities = model(p, system_parameters, system_mm, configuration_list, 
        s_in=s_in, process_model=process_model)

    # Convert lists to numpy arrays
    dataset = np.array(dataset)
    errors = np.array(errors)

    # print("Output Intensities: ", np.shape(output_intensities))

    # Optionally parse the dataset and output intensities (e.g., normalized difference)
    # print("Pre process_dataset dataset shape: ", np.shape(dataset))
    if process_dataset is not None:
        processed_dataset = process_dataset(copy.deepcopy(dataset))
    # print("Post process_dataset dataset shape: ", np.shape(processed_dataset))

    # Optionally parse the dataset and output intensities (e.g., normalized difference)
    # print("Pre process_errors errors shape: ", np.shape(dataset))
    if process_errors is not None:
        processed_errors = process_errors(copy.deepcopy(errors), 
            copy.deepcopy(dataset))
    # print("Post process_errors errors shape: ", np.shape(processed_errors))

    dataset = copy.deepcopy(processed_dataset)
    errors = copy.deepcopy(processed_errors)

    # Calculate log likelihood
    if logl_function is not None:
        return logl_function(p, output_intensities, dataset, errors)
    else: 
        return 0.5 * np.sum((output_intensities - dataset) ** 2 / errors ** 2)

@jit
def build_differences_and_sums(intensities):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''
    # Making sure that intensities is a numpy array
    intensities = jnp.array(intensities)

    differences = intensities[::2]-intensities[1::2]
    sums = intensities[::2]+intensities[1::2]

    return differences, sums

def build_double_differences_and_sums(differences, sums, normalized = True):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''
    # Making sure that differences and sums are numpy arrays
    differences = jnp.array(differences)
    sums = jnp.array(sums)

    if normalized:
        double_differences = (differences[::2]-differences[1::2])/(sums[::2]+sums[1::2])
        double_sums = (differences[::2]+differences[1::2])/(sums[::2]+sums[1::2])
    else:
        double_differences = differences[::2]-differences[1::2]
        double_sums = differences[::2]+differences[1::2]

    return double_differences, double_sums

def process_model(model_intensities):
    # Making sure that model_intensities is a numpy array
    model_intensities = np.array(model_intensities)
    # print("Entered process_model")

    differences, sums = build_differences_and_sums(model_intensities)

    double_differences, double_sums = build_double_differences_and_sums(differences, sums)

    # print("Differences shape: ", np.shape(differences))
    # print("Sums shape: ", np.shape(sums))
    # print("Double Differences shape: ", np.shape(double_differences))
    # print("Double Sums shape: ", np.shape(double_sums))

    #Format this into one array. 
    interleaved_values = np.ravel(np.column_stack((double_differences, double_sums)))
    
    # Take the negative of this as was done before 
    # NOTE: Subtracting same FLC state orders (A - B) as Miles
    interleaved_values = -interleaved_values

    return interleaved_values

def process_dataset(input_dataset): 
    # Making sure that input_dataset is a numpy array
    # print("Entered process_dataset")
    # print("Pre np.array Input dataset: ", np.shape(input_dataset))
    input_dataset = np.array(input_dataset)
    # print("Post np.array Input dataset: ", np.shape(input_dataset))

    differences = input_dataset[::2]
    sums = input_dataset[1::2]

    # print("Differences: ", differences)
    # print("Sums shape: ", np.shape(sums))

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
    # print("Entered process_errors")

    # Ensure input is a NumPy array
    input_errors = np.array(input_errors)
    input_dataset = np.array(input_dataset)

    # Separating out differences, sums, and corresponding errors
    single_differences = input_dataset[::2]
    single_sums = input_dataset[1::2]
    single_difference_errors = input_errors[::2]
    single_sum_errors = input_errors[1::2]

    # Compute double differences and double sums - changed so that interleaved values are the doubl
    double_differences_numerators, double_sums_numerators = \
        build_double_differences_and_sums(single_differences, single_sums, normalized=False)
    normalized_double_differences, normalized_double_sums = \
        build_double_differences_and_sums(single_differences, single_sums, normalized=True)

    # Compute errors for differences and sums
    # NOTE: All the numerator and denominator errors are all the same
    differences_errors = np.sqrt(single_difference_errors[::2]**2 + single_difference_errors[1::2]**2)
    sums_errors = np.sqrt(single_sum_errors[::2]**2 + single_sum_errors[1::2]**2)
    denominator_errors = sums_errors
    denominator = (single_sums[::2] + single_sums[1::2])  # This is used for normalization

    # Compute propagated errors for double differences
    double_differences_errors = normalized_double_differences ** 2 * np.sqrt(
        (differences_errors / double_differences_numerators) ** 2 + 
        (denominator_errors / denominator) ** 2
    )
    double_sums_errors = normalized_double_sums ** 2 * np.sqrt(
        (sums_errors / double_sums_numerators) ** 2 + 
        (denominator_errors / denominator) ** 2
    )

    # print("Double Differences Errors shape: ", np.shape(double_differences_errors))
    # print("Double Sums Errors shape: ", np.shape(double_sums_errors))

    # Interleave errors to maintain order
    interleaved_errors = np.ravel(np.column_stack((double_differences_errors, double_sums_errors)))

    # print("Final interleaved Errors shape: ", np.shape(interleaved_errors))

    return interleaved_errors

#######################################################
###### Functions related to plotting ##################
#######################################################

def plot_data_and_model(interleaved_values, interleaved_stds, model, 
    configuration_list, imr_theta_filter=None, wavelength=None, 
    save_path=None, legend=True):

    import numpy as np
    import matplotlib.pyplot as plt

    # Accept either a single model or a list of models
    if isinstance(model, np.ndarray) and model.ndim == 1:
        model_outputs = [model]
    else:
        model_outputs = model

    # Calculate double differences and sums from interleaved single differences
    interleaved_stds = process_errors(interleaved_stds, interleaved_values)
    interleaved_values = process_dataset(interleaved_values)

    # Extract double differences and double sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]

    # Group by image_rotator theta
    dd_by_theta = {}
    ds_by_theta = {}

    for i, config in enumerate(configuration_list[::2]):
        hwp_theta = config["hwp"]["theta"]
        imr_theta = round(config["image_rotator"]["theta"], 1)

        if imr_theta_filter is not None and imr_theta != round(imr_theta_filter, 1):
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

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot data
    for theta in dd_by_theta:
        dd = dd_by_theta[theta]
        ds = ds_by_theta[theta]
        err = axes[0].errorbar(dd["hwp_theta"], dd["values"], yerr=dd["stds"], fmt='o', label=f"{theta}°")
        color = err[0].get_color()
        axes[1].errorbar(ds["hwp_theta"], ds["values"], yerr=ds["stds"], fmt='o', color=color)

    # Plot model outputs
    for i, model in enumerate(model_outputs):
        dd_model = model[::2]
        ds_model = model[1::2]
        alpha = 0.1 if i > 0 else 1.0

        for theta in dd_by_theta:
            dd = dd_by_theta[theta]
            ds = ds_by_theta[theta]
            axes[0].plot(dd["hwp_theta"], dd_model[:len(dd["hwp_theta"])], '-', alpha=alpha)
            axes[1].plot(ds["hwp_theta"], ds_model[:len(ds["hwp_theta"])], '-', alpha=alpha)

    axes[0].set_xlabel("HWP θ (deg)")
    axes[0].set_ylabel("Double Difference")
    if legend:
        axes[0].legend(title="IMR θ")

    axes[1].set_xlabel("HWP θ (deg)")
    axes[1].set_ylabel("Double Sum")

    if wavelength is not None:
        fig.suptitle(f"{wavelength}nm", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


