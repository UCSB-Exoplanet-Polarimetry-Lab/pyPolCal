import numpy as np
import pyMuellerMat
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat

def stokes_to_deg_pol_and_aolp(Q, U):
    pol_percent = np.sqrt(Q ** 2 + U ** 2) * 100  # Convert to percentage
    aolp = 0.5 * np.arctan2(U, Q) * (180/np.pi)  # Convert to degrees
    return pol_percent, aolp

def deg_pol_and_aolp_to_stokes(pol_percent, aolp):
    # Convert percentage polarization to a fraction
    pol_fraction = pol_percent / 100.0
    
    # Convert aolp from degrees to radians
    aolp_rad = np.deg2rad(aolp * 2)  # Factor of 2 due to the 0.5 factor in arctan2

    # Calculate Q and U
    Q = pol_fraction * np.cos(aolp_rad)
    U = pol_fraction * np.sin(aolp_rad)

    return Q, U

def stokes_to_deg_pol_and_aolp_errors(Q, U, Q_err, U_err):
    pol_percent = 100 * np.sqrt(Q**2 + U**2)
    pol_percent_err = 100 * np.sqrt((Q * Q_err)**2 + (U * U_err)**2) / np.sqrt(Q**2 + U**2)
    aolp_err = 0.5 / (1 + (U/Q)**2) * np.sqrt((U_err / Q)**2 + (U * Q_err / Q**2)**2) * (180 / np.pi)  # Convert to degrees
    return pol_percent_err, aolp_err

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


########################################################
######## Functions for Processing Differences ##########
########################################################

def build_differences_and_sums(intensities, normalized=False):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''
    # Making sure that intensities is a numpy array
    intensities = np.array(intensities)

    differences = (intensities[::2]-intensities[1::2])
    sums = intensities[::2]+intensities[1::2]
    if normalized==True:
        # Normalize the differences and sums
       differences = differences / sums
    return differences, sums

def build_double_differences_and_sums(differences, sums):
    '''
    Assume that the input intensities are organized in pairs. Such that
    '''
    # Making sure that differences and sums are numpy arrays
    differences = np.array(differences)
    sums = np.array(sums)

    double_differences = (differences[::2]-differences[1::2])/(sums[::2]+sums[1::2])
    double_sums = (differences[::2]+differences[1::2])/(sums[::2]+sums[1::2])

    return double_differences, double_sums

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

