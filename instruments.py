import numpy as np
import pandas as pd
import pyMuellerMat
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
from scipy.optimize import minimize
import copy

#######################################################
###### Functions related to reading in .csv values ####
#######################################################

# TODO: Test the MBI function
def read_csv(file_path, obs_mode="IPOL", obs_filter=None):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    MBI_filters = ["610", "670", "720", "760"]

    # Process only one filter if applicable
    if obs_mode == "MBI":
        df = df[df["OBS-MOD"] == "IPOL_MBI"]
    elif obs_filter is not None:
        df = df[df["FILTER01"] == obs_filter]

    # Convert relevant columns to float (handling possible conversion errors)
    for col in ["RET-POS1", "D_IMRANG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set errors to NaN if not possible

    # Drop rows with NaN values if conversion failed for any critical column
    df.dropna(subset=["RET-POS1", "D_IMRANG"], inplace=True)

    # Handle MBI mode: extract the correct component from numpy arrays
    if obs_mode == "MBI":
        if obs_filter not in MBI_filters:
            raise ValueError(f"Invalid obs_filter '{obs_filter}' for MBI mode. Choose from {MBI_filters}.")

        i = MBI_filters.index(obs_filter)  # Get the index corresponding to the filter
        
        # Convert diff-related columns into numpy arrays if not already
        for col in ["diff", "sum", "diff_std", "sum_std"]:
            df[col] = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
        
        # Extract the i-th element from each numpy array
        df["diff"] = df["diff"].apply(lambda arr: arr[i] if isinstance(arr, np.ndarray) and len(arr) > i else np.nan)
        df["sum"] = df["sum"].apply(lambda arr: arr[i] if isinstance(arr, np.ndarray) and len(arr) > i else np.nan)
        df["diff_std"] = df["diff_std"].apply(lambda arr: arr[i] if isinstance(arr, np.ndarray) and len(arr) > i else np.nan)
        df["sum_std"] = df["sum_std"].apply(lambda arr: arr[i] if isinstance(arr, np.ndarray) and len(arr) > i else np.nan)

    # Convert diff-related columns to numeric after extracting values
    for col in ["diff", "sum", "diff_std", "sum_std"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN values in critical columns
    df.dropna(subset=["diff", "sum", "diff_std", "sum_std"], inplace=True)

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

        # Building dictionary
        row_data = {
            "hwp": {"properties": {"theta": hwp_theta}},
            "image_rotator": {"properties": {"theta": imr_theta}}
        }

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

def mcmc_system_mueller_matrix(p0, system_mm, dataset, errors, configuration_list):
    '''
    Perform MCMC on a dataset, using a System Mueller Matrix model

    Example p0: 

    p0 = {'Polarizer': {'Theta': 0, 'Phi': 0},
          'Waveplate': {'Theta': 0, 'Phi': 0},
          'Retarder': {'Theta': 0, 'Phi': 0},
          'Sample': {'Theta': 0, 'Phi': 0},
          'Analyzer': {'Theta': 0, 'Phi': 0}}

    Args: 
    p0: dictionary of dictionaries of initial parameters
    systemMM: pyMuellerMat System Mueller Matrix object
    dataset: list of measurements
    configuration_list: list of system dictionaries, one for each measurement in dataset
    '''

    #####################################################
    ###### First, define the data model based on p0 #####
    #####################################################

    #Parse p0 into a list of values and a list of lists of keywords
    p0_values = []
    p0_keywords = []
    for component, parameters in p0.items():
        for parameter, value in parameters.items():
            p0_values.append(value)
            p0_keywords.append([component, parameter])

    #TODO: Fill this out more. 


def minimize_system_mueller_matrix(p0, system_mm, dataset, errors, configuration_list):
    '''
    Perform a minimization on a dataset, using a System Mueller Matrix model'''

    p0_values, p0_keywords = parse_configuration(p0)

    print("p0_values: " + str(p0_values))
    print("p0_keywords: " + str(p0_keywords))

    result = minimize(logl, p0_values, args=(p0_keywords, system_mm, dataset, errors, configuration_list), method='Nelder-Mead')

    return result

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
        for parameter, value in details["properties"].items():
            values.append(value)
            keywords.append([component, parameter])

    return values, keywords

########################################################################################
###### Functions related to fitting ####################################################
########################################################################################

def minimize_system_mueller_matrix(p0, system_mm, dataset, errors, configuration_list):
    '''
    Perform a minimization on a dataset, using a System Mueller Matrix model
    '''

    p0_values, p0_keywords = parse_configuration(p0)

    print("p0_values: " + str(p0_values))
    print("p0_keywords: " + str(p0_keywords))

    result = minimize(logl, p0_values, args=(p0_keywords, system_mm, dataset, errors, configuration_list), method='Nelder-Mead')

    return result

def logl(p, system_parameters, system_mm, dataset, errors, configuration_list, 
        logl_function = None, process_dataset = None, process_model = None):
    '''
    Log likelihood function for MCMC

    Args: 
    p: list of parameters
    system_parameters: the list of dictionaries that define what the p parameters are
    system_mm: pyMuellerMat System Mueller Matrix object
    dataset: list of measurements - includes both double and single differences
    for new VAMPIRES
    errors: same length as list of measurements
    confifugration_list: list of dictionaries
    logl_function: for calculating residuls differently
    process dataset:

    Returns: 
    logl: log likelihood (float)
    '''
    # Update the system Mueller matrix with parameters that we're fitting
    system_mm = update_system_mm(p, system_parameters, system_mm)

    # Generate a model dataset
    output_intensities = []

    # Save the parameters necessary to switch between o and e beam
    o_beam_values, wollaston_beam_keyword = parse_configuration({'wollaston': {'properties': {'beam': 'o'}}})
    e_beam_values, wollaston_beam_keyword = parse_configuration({'wollaston': {'properties': {'beam': 'e'}}})

    # print("Dataset Length: " + str(np.shape(dataset)))
    # print("Configuration Length: " + str(np.shape(configuration_list)))

    for i in range(len(configuration_list)):
        values, keywords = parse_configuration(configuration_list[i])

        # Compute the intensity for the ordinary beam
        system_mm = update_system_mm(o_beam_values, wollaston_beam_keyword, system_mm)
        s_out_o = generate_measurement(system_mm, s_in = np.array([1, 1, 0, 0]))
        o_intensity = s_out_o[0]

        # Compute the intensity for the extraordinary beam
        system_mm = update_system_mm(e_beam_values, wollaston_beam_keyword, system_mm)
        s_out_e = generate_measurement(system_mm, s_in = np.array([1, 1, 0, 0]))
        e_intensity = s_out_e[0]

        output_intensities.append(o_intensity)
        output_intensities.append(e_intensity)

    # Optionally parse the dataset and output intensities (e.g. a normalized difference)
    if process_dataset is not None:
        dataset = process_dataset(copy.deepcopy(dataset))
    if process_model is not None:
        output_intensities = process_model(output_intensities)

    # Making sure any lists are converted to numpy arrays
    output_intensities = np.array(output_intensities)
    dataset = np.array(dataset)
    errors = np.array(errors)

    # Calculate the log likelihood - optionally with your own logl function (e.g. to include an extra noise term in p)
    if logl_function is not None:
        return logl_function(p, output_intensities, dataset, errors)
    else: 
        # print("Output Intensities Datatype: " + str(output_intensities.dtype))
        # print("Dataset Datatype: " + str(dataset.dtype))
        # print("Errors Datatype: " + str(errors.dtype))
        return -0.5 * np.sum((np.array(output_intensities) - np.array(dataset)) ** 2 / np.array(errors) ** 2)

def build_differences_and_sums(intensities):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''

    differences = intensities[::2]-intensities[1::2]
    sums = intensities[::2]+intensities[1::2]

    return differences, sums

def build_double_differences_and_sums(differences, sums):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''

    double_differences = (differences[::2]-differences[1::2])/(sums[::2]+sums[1::2])
    double_sums = (sums[::2]-sums[1::2])/(sums[::2]+sums[1::2])

    return double_differences, double_sums

def process_model(model_intensities):

    differences, sums = build_differences_and_sums(model_intensities)

    double_differences, double_sums = build_double_differences_and_sums(differences, sums)

    #Format this into one array. 
    return double_differences, double_sums

def process_dataset(input_dataset): 

    differences = input_dataset[::2]
    sums = input_dataset[1::2]

    double_differences, double_sums = build_double_differences_and_sums(differences, sums)

    #Format this into one array.
    return double_differences, double_sums