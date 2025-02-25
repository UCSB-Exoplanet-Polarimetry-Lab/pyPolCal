import numpy as np
import pandas as pd
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
import copy

#######################################################
###### Functions related to reading in .csv values ####
#######################################################

def read_and_interleave_csv_data(file_path, obs_filter = None):
    """
    Reads a CSV file and extracts the columns "diff", "sum", "diff_std", and "sum_std".
    It then interleaves values from "diff" and "sum" into one NumPy array,
    and interleaves values from "diff_std" and "sum_std" into another NumPy array.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: Two NumPy arrays:
            - The first array contains interleaved values of "diff" and "sum".
            - The second array contains interleaved values of "diff_std" and "sum_std".
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Process only one filter if applicable
    if obs_filter is not None:
        df = df[df["FILTER01"] == obs_filter]

    # Interleave values from "diff" and "sum"
    interleaved_values = np.ravel(np.column_stack((df["diff"], df["sum"])))

    # Interleave values from "diff_std" and "sum_std"
    interleaved_stds = np.ravel(np.column_stack((df["diff_std"], df["sum_std"])))

    return interleaved_values, interleaved_stds   

def read_csv_configuration_data(file_path, obs_filter = None):
    """
    Reads a CSV file, filters for a specified observation filter, and outputs a
    list of dictionaries correspoding to the HWP, IMR, and FLC configurations.

    Args:
        file_path (str): Path to the CSV file.
        obs_filter (str, optional): Value to filter in the "FILTER01" column.

    Returns:
        list: List of dictionaries in the same format as any system_dictionary
    """
    # Read CSV file
    df = pd.read_csv(file_path)

    # Process only one filter if applicable
    if obs_filter is not None:
        df = df[df["FILTER01"] == obs_filter]

    # Convert each row's values into a list of two-element lists
    configuration_list = []
    for index, row in df.iterrows():
        # Extracting values from relevant columns
        HWP_theta = row["RET-POS1"]
        IMR_theta = row["D_IMRANG"]
        FLC_theta = row["U_FLC"]

        if FLC_theta == "A":
            FLC_theta = 0
        elif FLC_theta == "B":
            FLC_theta = 2 * np.pi * np.radians(45)

        # Building dictoinary
        row_data = {}
        row_data["HWP"] = {
            "type": "Retarder",
            "properties": {"theta": HWP_theta},
            "tag": "internal"
        }
        row_data["IMR"] = {
            "type": "Retarder",
            "properties": {"theta": IMR_theta},
            "tag": "internal"
        }
        row_data["FLC"] = {
            "type": "Retarder",
            "properties": {"theta": FLC_theta},
            "tag": "internal"
        }

        configuration_list.append(row_data)

    return configuration_list

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