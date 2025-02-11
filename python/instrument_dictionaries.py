import numpy as np
import helper_funcs as funcs
from scipy.optimize import minimize
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

def generate_system_mueller_matrix(system_dict):
    """
    Parses a system dictionary and generates a System Mueller Matrix object.

    Args:
        system_dict (dict): Dictionary containing components, their types, properties, and order.

    Returns:
        SystemMuellerMatrix: An object representing the system Mueller matrix.
    """
    mueller_matrices = []
    
    # Parse the components and construct individual MuellerMatrix objects
    for component_name in system_dict["order"]:
        if component_name not in system_dict["components"]:
            raise ValueError(f"Component '{component_name}' is missing in 'components'.")
        
        component = system_dict["components"][component_name]
        component_type = component["type"]
        properties = component["properties"]
        
        # Map the component type to its corresponding function
        if component_type == "Rotator":
            mm_function = rotator_function
        elif component_type == "DiattenuatorRetarder":
            mm_function = diattenuator_retarder_function
        elif component_type == "Retarder":
            mm_function = general_retarder_function
        elif component_type == "WollastonPrism":
            mm_function = wollaston_prism_function
        elif component_type == "LinearPolarizer":
            mm_function = general_linear_polarizer_function_with_theta
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Create a MuellerMatrix object for this component
        mm = MuellerMat.MuellerMatrix(mm_function, name=component_name)
        mm.properties.update(properties)  # Set properties for the component
        mueller_matrices.append(mm)
    
    # Create the SystemMuellerMatrix object from the list of MuellerMatrix objects
    system_mm = MuellerMat.SystemMuellerMatrix(mueller_matrices)
    return system_mm

def update_system_mm(parameter_values, system_mm):
    """
    Updates the system Mueller matrix with the given parameter values.

    Args:
        parameter_values: Dictionary of parameter values to update, where keys are tuples of (component_name, parameter_name).
        system_mm: pyMuellerMat System Mueller Matrix object.
    """
    # Iterate over the parameter updates
    for (component_name, parameter_name), value in parameter_values.items():
        # Check if the component and parameter exist in the systemMM
        if component_name in system_mm.master_property_dict:
            if parameter_name in system_mm.master_property_dict[component_name]:
                # Update the parameter
                system_mm.master_property_dict[component_name][parameter_name] = value
            else:
                print(f"Parameter '{parameter_name}' not found in component '{component_name}'. Skipping...")
        else:
            print(f"Component '{component_name}' not found in System Mueller Matrix. Skipping...")
    return system_mm

def generate_measurement(system_mm, s_in = [1, 0, 0, 0]):
    '''
    Generate a measurement from a given System Mueller Matrix

    Args: 
    systemMM: pyMuellerMat System Mueller Matrix object
    S_in: Stokes vector of the incoming light

    Returns: 
    S_out: Stokes vector of the outgoing light
    '''
    output_stokes = system_mm.evaluate() @ np.array(s_in)
    return output_stokes

# NOTE: This will be used as the "parse_dataset" function to when performing optimizations
def calculate_double_differences_and_sums(configurations, system_mm, s_in = [1, 0, 0, 0]):
    normalized_double_diff_list = []
    normalized_double_sum_list = []
    """
    Args:
        configurations (list): List of system configurations (dictionaries) corresponding to 
        the desired input for update_system_mm
    """

    for i, configuration in enumerate(configurations):
        this_system_mm = update_system_mm(configuration, system_mm)

        system_mm_FL1 = update_system_mm({["flc", "theta"]: 0, ["wollaston", "beam"]: "o"}, system_mm)
        FL1 = generate_measurement(system_mm_FL1)
        system_mm_FR1 = update_system_mm({["flc", "theta"]: 45, ["wollaston", "beam"]: "o"}, system_mm)
        FR1 = generate_measurement(system_mm_FR1)
        system_mm_FL1 = update_system_mm({["flc", "theta"]: 0, ["wollaston", "beam"]: "e"}, system_mm)
        FL2 = generate_measurement(system_mm_FL1)
        system_mm_FR1 = update_system_mm({["flc", "theta"]: 45, ["wollaston", "beam"]: "e"}, system_mm)
        FR2 = generate_measurement(system_mm_FR1)

        normalized_double_diff = ((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))
        normalized_double_sum = ((FL1 - FR1) + (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))

        normalized_double_diff_list.append(normalized_double_diff)
        normalized_double_sum_list.append(normalized_double_sum)

    # TODO: Return this as one long concatenated numpy array, with double_diffs first

def logl(p, systemMM, data, errors, configurations, 
    logl_function=None, parse_dataset=None):
    # TODO: Is p used in the default lg likelihood right now?

    """
    Log-likelihood function for MCMC scipy.minimize

    This function evaluates the log-likelihood by updating the system Mueller matrix
    with the given parameters, generating model predictions, and comparing them 
    to the observed dataset.

    Args:
        p (list): List of parameters being optimized.
        systemMM (SystemMuellerMatrix): The Mueller matrix system model.
        data (list): List of observed measurements.
        errors (list): List of measurement uncertainties.
        configurations (list): List of system configurations corresponding to `data`.
        logl_function (function, optional): Custom log-likelihood function.
        parse_dataset (function, optional): Function to transform dataset before computing likelihood.

    Returns:
        float: Log-likelihood value.
    """
    # Intensities data list
    output_intensities = []

    # Update the system Mueller matrix with parameters that we're fitting
    systemMM = update_systemMM(parameter_values, systemMM)

    # Ensure configurations is a list of system_dicts
    if not isinstance(configurations, list):
        raise ValueError("configurations should be a list of system dictionaries.")

    output_intensities = []
    for i, configuration in enumerate(configurations):
        system_dict = configurations[i]
        configuration = parse_configuration(system_dict)  # This is a single dictionary
        systemMM = update_systemMM(configuration[0], systemMM)
        S_out = generate_measurement(systemMM, data[i][0])
        output_intensities.append(S_out[0])

    # Optionally parse the dataset and output intensities (e.g., normalized difference)
    if parse_dataset is not None:
        data = parse_dataset(copy.deepcopy(data))
        output_intensities = parse_dataset(output_intensities)

    # Compute log-likelihood
    if logl_function is not None:
        return logl_function(p, output_intensities, data, errors)
    else:
        return -0.5 * np.sum(((np.array(output_intensities) - np.array(data)) ** 2) / (np.array(errors) ** 2))

