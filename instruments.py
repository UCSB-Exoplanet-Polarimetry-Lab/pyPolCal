import numpy as np
import pandas as pd
import pyMuellerMat
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
import copy

#######################################################
###### Functions related to reading in .csv values ####
#######################################################

def read_csv(file_path, obs_filter = None):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Process only one filter if applicable
    if obs_filter is not None:
        df = df[df["FILTER01"] == obs_filter]

    # Interleave values from "diff" and "sum"
    interleaved_values = np.ravel(np.column_stack((df["diff"], df["sum"])))

    # Interleave values from "diff_std" and "sum_std"
    interleaved_stds = np.ravel(np.column_stack((df["diff_std"], df["sum_std"])))

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
        row_data = {}
        row_data["hwp"] = {
            "properties": {"theta": hwp_theta},
        }
        row_data["image_rotator"] = {
            "properties": {"theta": imr_theta},
        }
        row_data["flc"] = {
            "properties": {"theta": flc_theta},
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

def update_systemMM(parameter_values, systemMM, system_parameters):
    '''
    Generates a model dataset for the given set of parameters

    Args: 
    p: list of parameters (numpy array of floats)
    systemMM: pyMuellerMat System Mueller Matrix object
    system_parameters: list of two element lists [component_name, parameter_name]
                        example: [['Polarizer', 'Theta'], ['Polarizer', 'Phi']]
    '''

    #Set up the System Mueller Matrix
    for parameter in system_parameters:
        #TODO: Check that this parameter is IN systemMM and if not, don't try to set it. 
        systemMM.set_parameter(parameter, parameter_values[parameter])
    return systemMM.get_M()

def generate_measurement(systemMM, S_in=[1,0,0,0]):
    '''
    Generate a measurement from a given System Mueller Matrix

    Args: 
    systemMM: pyMuellerMat System Mueller Matrix object
    S_in: Stokes vector of the incoming light

    Returns: 
    S_out: Stokes vector of the outgoing light
    '''
    return systemMM.get_S_out(S_in) #This is probably systeMM.evaluate@np.array(S_in)

def mcmc_system_mueller_matrix(p0, systemMM, dataset, errors, configuration_list):
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


def minimize_system_mueller_matrix(p0, systemMM, dataset, errors, configuration_list):
    '''
    Perform a minimization on a dataset, using a System Mueller Matrix model'''

    p0_values, p0_keywords = parse_configuration(p0)

    result = minimize(logl, p0_values, args=(p0_keywords, systemMM, dataset, errors, configuration_list), method='Nelder-Mead')


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

def logl(p, system_parameters, systemMM, dataset, errors, configuration_list, logl_function = None, process_dataset = None, process_model = None):
    '''
    Log likelihood function for MCMC

    Args: 
    p: list of parameters
    system_parameters: the list of dictionaries that define what the p parameters are. 
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
    if process_dataset is not None:
        dataset = process_dataset(copy.deepcopy(dataset))
    if process_model is not None:
        output_intensities = process_model(output_intensities)

    #Calculate the log likelihood - optionally with your own logl function (e.g. to include an extra noise term in p)
    if logl_function is not None:
        return logl_function(p, output_intensities, dataset, errors)
    else: 
        return -0.5 * np.sum((np.array(output_intensities) - np.array(dataset))**2 / np.array(errors)**2)
    

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
        return output
    
    def process_dataset(input_dataset): 

        differences = input_dataset[::2]
        sums = input_dataset[1::2]

        double_differences, double_sums = build_double_differences_and_sums(differences, sums)

        #Format this into one array.
        return output 