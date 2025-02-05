import numpy as np
import helper_funcs as funcs
from scipy.optimize import minimize
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

def perform_optimization(starting_guess=None, bounds=None, configurations=None, data=None, errors=None, system_parameters=None, system_MM=None, tag=None, S_in=[1, 0, 0, 0]):
    """
    Performs optimization using scipy.optimize.minimize to fit specified parameters.
    Only components with the specified tag will be used in the Mueller matrix.

    Args:
        starting_guess: Initial parameter guesses.
        bounds: Parameter bounds for the optimizer.
        configurations: List of system configurations.
        data: List of observed data for comparison.
        errors: List of errors associated with the data.
        system_parameters: List of [component_name, parameter_name] pairs.
        system_MM: System Mueller Matrix object.
        tag: Tag to filter components in the Mueller matrix.
        S_in: Input Stokes vector (default: unpolarized light).
    """

    def filter_system_dict_by_tag(system_dict, tag):
        """
        Filters a system dictionary to include only components with the given tag.
        """
        if tag is not None:
            filtered_components = {
                name: details
                for name, details in system_dict["components"].items()
                if details.get("tag") == tag
            }
            filtered_order = [name for name in system_dict["order"] if name in filtered_components]
            return {"components": filtered_components, "order": filtered_order}
        else:
            return system_dict

    def objective_function(params):
        """
        Wrapper for the log-likelihood function to use with scipy.optimize.minimize.
        """
        # Convert params into a dictionary of parameter values
        parameter_values = {}
        for (component, property_name), value in zip(system_parameters, params):
            parameter_values[(component, property_name)] = value  # Store as tuple keys

        # Call the log-likelihood function
        return logl(parameter_values, system_parameters, system_MM, data, errors, configurations)

    # Perform optimization
    result = minimize(
        objective_function,  # Use the wrapper for log-likelihood
        starting_guess,
        bounds=bounds,
        method="Nelder-Mead",  # You can change the method if needed
    )

    return result

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

def update_systemMM(parameter_values, systemMM):
    """
    Updates the system Mueller matrix with the given parameter values.

    Args:
        parameter_values: Dictionary of parameter values to update, where keys are tuples of (component_name, parameter_name).
        systemMM: pyMuellerMat System Mueller Matrix object.
    """
    # Iterate over the parameter updates
    for (component_name, parameter_name), value in parameter_values.items():
        # Check if the component and parameter exist in the systemMM
        if component_name in systemMM.master_property_dict:
            if parameter_name in systemMM.master_property_dict[component_name]:
                # Update the parameter
                systemMM.master_property_dict[component_name][parameter_name] = value
            else:
                print(f"Parameter '{parameter_name}' not found in component '{component_name}'. Skipping...")
        else:
            print(f"Component '{component_name}' not found in System Mueller Matrix. Skipping...")
    return systemMM

def generate_measurement(systemMM, S_in=[1, 0, 0, 0]):
    '''
    Generate a measurement from a given System Mueller Matrix

    Args: 
    systemMM: pyMuellerMat System Mueller Matrix object
    S_in: Stokes vector of the incoming light

    Returns: 
    S_out: Stokes vector of the outgoing light
    '''
    output_stokes = systemMM.evaluate() @ np.array(S_in)
    return output_stokes

def mcmc_system_mueller_matrix(p0, systemMM, dataset, errors, configuration_list):
    # TODO: Needs to be filled out more
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

def parse_configuration(configuration):
    """
    Parse a single configuration dictionary into values and keywords.

    Args:
        configuration (dict): A dictionary representing a system configuration.

    Returns:
        tuple: (values, keywords)
    """
    if not isinstance(configuration, dict):
        raise ValueError("Expected a dictionary for configuration.")

    values = []
    keywords = []
    for component, parameters in configuration["components"].items():
        for parameter, value in parameters["properties"].items():
            values.append(value)
            keywords.append([component, parameter])
    return values, keywords

# NOTE: This will be used as the "parse_dataset" function to when performing optimizations
def calculate_double_differences_and_sums(configurations, systemMM):
    """
    Args:
        configurations (list): List of system configurations corresponding to '
    """

    for i, configuration in enumerate(configurations):
        this_systemMM = update_system_mm

def logl(parameter_values, system_parameters, systemMM, data, errors, configurations, 
    logl_function=None, parse_dataset=None):
    # TODO: What do I want configurations to be? A list of system Mueller matrices?
    # A list of dictionaries like how update_system_MM is called?

    """
    Log-likelihood function for MCMC scipy.minimize

    This function evaluates the log-likelihood by updating the system Mueller matrix
    with the given parameters, generating model predictions, and comparing them 
    to the observed dataset.

    Args:
        p (list): List of parameters being optimized.
        system_parameters (list): List of two-element lists specifying [component_name, parameter_name].
        systemMM (SystemMuellerMatrix): The Mueller matrix system model.
        data (list): List of observed measurements.
        errors (list): List of measurement uncertainties.
        configurations (list): List of system configurations corresponding to `data`.
        logl_function (function, optional): Custom log-likelihood function.
        parse_dataset (function, optional): Function to transform dataset before computing likelihood.

    Returns:
        float: Log-likelihood value.
    """
    # Update the system Mueller matrix with parameters that we're fitting
    systemMM = update_systemMM(parameter_values, systemMM)

    # Ensure configurations is a list of system_dicts
    if not isinstance(configurations, list):
        raise ValueError("configurations should be a list of system dictionaries.")

    output_intensities = []
    for i, configuration in enumerate(configurations):
        system_dict = configurations[i]
        configuration = parse_configuration(system_dict)  # This is a single dictionary
        print(configuration)
        print(shape(configuration))
        print(configuration[0])
        print(configuration[1])
        systemMM = update_systemMM(configuration[0], systemMM, configuration[1])
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
