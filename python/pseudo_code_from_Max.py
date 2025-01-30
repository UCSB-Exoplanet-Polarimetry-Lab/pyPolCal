import numpy as np
import helper_funcs as funcs
from scipy.optimize import minimize
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

def perform_optimization(starting_guess, bounds, configurations, data, errors, 
    system_parameters, tag, S_in=[1, 0, 0, 0]):
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
        tag: Tag to filter components in the Mueller matrix.
        S_in: Input Stokes vector (default: unpolarized light).
    """

    def filter_system_dict_by_tag(system_dict, tag):
        """
        Filters a system dictionary to include only components with the given tag.
        """
        filtered_components = {
            name: details
            for name, details in system_dict["components"].items()
            if details.get("tag") == tag
        }
        filtered_order = [name for name in system_dict["order"] if name in filtered_components]

        return {
            "components": filtered_components,
            "order": filtered_order,
        }

    def model(parameters, config):
        """
        Generate model output for a given parameter set and configuration.
        """
        # Filter the system dictionary by tag
        filtered_config = filter_system_dict_by_tag(config, tag)

        # Parse parameters into component:property mapping
        parameter_values = {}
        for (component, property_name), value in zip(system_parameters, parameters):
            if component not in parameter_values:
                parameter_values[component] = {}
            parameter_values[component][property_name] = value

        # Update the system configuration with the current parameters
        system_mm = generate_system_mueller_matrix(filtered_config)
        systemMM_updated = update_systemMM(parameter_values, system_mm, system_parameters)

        # Generate measurement
        return generate_measurement(systemMM_updated, S_in)

    def objective_function(params):
        """
        Objective function to minimize: chi-squared between model and data.
        """
        chi_squared = 0
        for i, config in enumerate(configurations):
            model_output = model(params, config)
            chi_squared += np.sum(((model_output - data[i]) / errors[i]) ** 2)
        return chi_squared

    # Perform optimization
    result = minimize(
        objective_function,
        starting_guess,
        bounds=bounds,
        method="Nelder-Mead",
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

def update_systemMM(parameter_values, systemMM, system_parameters):
    """
    Generates a model dataset for the given set of parameters.

    Args: 
        parameter_values: dictionary of parameter values to update.
        systemMM: pyMuellerMat System Mueller Matrix object.
        system_parameters: list of two-element lists [component_name, parameter_name]
                           example: [['Polarizer', 'Theta'], ['Polarizer', 'Phi']]
    """
    # Set up the System Mueller Matrix
    for component_name, parameter_name in system_parameters:
        # Check if the component and parameter exist in the systemMM
        if component_name in systemMM.master_property_dict:
            if parameter_name in systemMM.master_property_dict[component_name]:
                # Update the parameter
                systemMM.master_property_dict[component_name][parameter_name] = parameter_values.get(
                    (component_name, parameter_name),
                    systemMM.master_property_dict[component_name][parameter_name],
                )
            else:
                print(f"Parameter '{parameter_name}' not found in component '{component_name}'. Skipping...")
        else:
            print(f"Component '{component_name}' not found in System Mueller Matrix. Skipping...")

    return systemMM

def generate_measurement(systemMM, S_in=[1,0,0,0]):
    '''
    Generate a measurement from a given System Mueller Matrix

    Args: 
    systemMM: pyMuellerMat System Mueller Matrix object
    S_in: Stokes vector of the incoming light

    Returns: 
    S_out: Stokes vector of the outgoing light
    '''
    output_stokes = systemMM.evaluate() @ np.array(S_in)
    return output_stokes #This is probably systeMM.evaluate@np.array(S_in)

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
    '''
    Parse a configuration dictionary into a list of values and a list of lists of keywords

    Args: 
    configuration: dictionary of dictionaries of initial parameters

    Returns: 
    values: list of values
    keywords: list of lists of keywords
    '''
    values = []
    keywords = []
    for component, parameters in configuration.items():
        for parameter, value in parameters.items():
            values.append(value)
            keywords.append([component, parameter])
    return values, keywords

def logl(p, system_parameters, systemMM, dataset, errors, configuration_list, logl_function = None, parse_dataset = None):
    '''
    Log likelihood function for MCMC

    Args: 
    p: list of parameters
    systemMM: pyMuellerMat System Mueller Matrix object
    dataset: list of measurements

    Returns: 
    logl: log likelihood
    '''
    #Update the system Mueller matrix with parameters that we're fitting
    systemMM = update_systemMM(p, systemMM, system_parameters)

    #Generate a model dataset
    output_intensities = []
    for i in range(len(dataset)):
        configuration = parse_configuration(configuration_list[i])
        systemMM = update_systemMM(configuration[0], systemMM, configuration[1])
        S_out = generate_measurement(systemMM, dataset[i][0])
        output_intensities.append(S_out[0])

    #Optionally parse the dataset and output intensities (e.g. a normalized difference)
    if parse_dataset is not None:
        dataset = parse_dataset(copy.deepcopy(dataset))
        output_intensities = parse_dataset(output_intensities)

    #Calculate the log likelihood - optionally with your own logl function (e.g. to include an extra noise term in p)
    if logl_function is not None:
        return logl_function(p, output_intensities, dataset, errors)
    else: 
        return -0.5 * np.sum((np.array(output_intensities) - np.array(dataset))**2 / np.array(errors)**2)