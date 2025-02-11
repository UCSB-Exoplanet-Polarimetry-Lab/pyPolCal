import numpy as np
import helper_funcs as funcs
from scipy.optimize import minimize
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

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

def update_system_mm(parameter_values, system_parameters, system_mm):
    '''
    Generates a model dataset for the given set of parameters

    Args: 
    parameter_values: list of parameters  - NOTE: cannot be numpy array
    system_parameters: list of two element lists [component_name, parameter_name]
                        example: [['Polarizer', 'Theta'], ['Polarizer', 'Phi']]
    systemMM: pyMuellerMat System Mueller Matrix object
    '''

    #Set up the System Mueller Matrix
    for i, system_parameter in enumerate(system_parameters):
        # Unpacking each tuple within system_parameters
        component_name = system_parameter[0]
        parameter_name = system_parameter[1]
        # Check if the component and parameter exist in the system_mm
        if component_name in system_mm.master_property_dict:
            if parameter_name in system_mm.master_property_dict[component_name]:
                # Update the parameter
                system_mm.master_property_dict[component_name][parameter_name] = parameter_values[i]
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

    # Ensure the dictionary has a 'components' key
    if "components" not in configuration:
        raise ValueError("Invalid configuration format: 'components' key not found.")

    for component, details in configuration["components"].items():
        for parameter, value in details["properties"].items():
            values.append(value)
            keywords.append([component, parameter])

    return values, keywords


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
        system_mm_FL2 = update_system_mm({["flc", "theta"]: 45, ["wollaston", "beam"]: "o"}, system_mm)
        FR1 = generate_measurement(system_mm_FR1)
        system_mm_FR1 = update_system_mm({["flc", "theta"]: 0, ["wollaston", "beam"]: "e"}, system_mm)
        FR1 = generate_measurement(system_mm_FR1)
        system_mm_FR2 = update_system_mm({["flc", "theta"]: 45, ["wollaston", "beam"]: "e"}, system_mm)
        FR2 = generate_measurement(system_mm_FR2)

        normalized_double_diff = ((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))
        normalized_double_sum = ((FL1 - FR1) + (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))

        normalized_double_diff_list.append(normalized_double_diff)
        normalized_double_sum_list.append(normalized_double_sum)

    # Converting lists to numpy arrays
    normalized_double_diff_list = np.array(normalized_double_diff_list)
    normalized_double_sum_list = np.array(normalized_double_sum_list)

    all_data = np.concatenate((normalized_double_diff_list, normalized_double_sum_list), axis = 0)

    return all_data

def logl(p, system_parameters, system_mm, dataset, errors, configuration_list, 
    s_in = [1, 0, 0, 0], logl_function = None, parse_dataset = None):
    '''
    Log likelihood function for MCMC

    Args: 
    p (float np.array): list of all numerical values being optimized 
    system_parameters (list of lists) : list of two-element lists [component_name, parameter_name]
            example: [['Polarizer', 'Theta'], ['Polarizer', 'Phi']]
    system_mm: object with the correct instrument configuration
    dataset (float np.array): list of measurements # NOTE: can be any quantity (intensities/diffs/sums)
    errors (float np.array): list of errors on all measurements

    Returns: 
    logl: log likelihood
    '''
    # Update the system Mueller matrix with parameters that we're fitting
    system_mm = update_system_mm(p, system_mm, system_parameters)

    # Generate a model dataset
    output_intensities = []
    for i in range(len(dataset)):
        # Retrieving values and system_parameters from configurations
        configuration = parse_configuration(configuration_list[i])
        values = configuration[0]
        system_parameters = configuration[1]

        # Updating system Mueller matrix based on i-th configuration
        system_mm = update_system_mm(values, system_parameters, system_mm)

        # Generating and saving intensity measurements
        S_out = generate_measurement(system_mm, s_in = s_in)
        output_intensities.append(S_out[0])

    # Optionally parse the dataset and output intensities (e.g. a normalized difference)
    if parse_dataset is not None:
        dataset = parse_dataset(copy.deepcopy(dataset))
        output_intensities = parse_dataset(output_intensities)

    # Calculate the log likelihood - optionally with your own logl function (e.g. to include an extra noise term in p)
    if logl_function is not None:
        return logl_function(p, output_intensities, dataset, errors)
    else: 
        return -0.5 * np.sum((np.array(output_intensities) - np.array(dataset)) ** 2 / np.array(errors) ** 2)

