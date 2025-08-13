import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
from photutils.aperture import RectangularAperture
from photutils.aperture import aperture_photometry
from pathlib import Path
import re
import pandas as pd
import pyMuellerMat
from pyMuellerMat.common_mm_functions import *
from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import emcee
import mcmc_helper_funcs as mcmc
from multiprocessing import Pool
import copy
import os
from functools import partial
from pyMuellerMat.physical_models.charis_physical_models import HWP_retardance, IMR_retardance
import json
from scipy.optimize import least_squares

###############################################################
###### Functions related to reading/writing in .csv values ####
###############################################################

# define CHARIS wavelength bins

wavelength_bins = np.array([1159.5614, 1199.6971, 1241.2219, 1284.184 , 1328.6331, 1374.6208,
1422.2002, 1471.4264, 1522.3565, 1575.0495, 1629.5663, 1685.9701,
1744.3261, 1804.7021, 1867.1678, 1931.7956, 1998.6603, 2067.8395,
2139.4131, 2213.4641, 2290.0781, 2369.3441])

def single_sum_and_diff(fits_cube_path, wavelength_bin):
    """Calculate single difference and sum between left and right beam 
    rectangular aperture photometry from a CHARIS fits cube. Add L/R counts and stds to array.
    
    Parameters:
    -----------
    fits_cube_path : str or Path
        Path to the CHARIS fits cube file.
        
    wavelength_bin : int
        Index of the wavelength bin to analyze (0-based).
    
    Returns:
    --------
    np.ndarray
        Array with six elements:
            [0] single_sum : float
                Single sum of left and right beam apertures:
                (R + L)
            [1] single_diff : float
                Single difference of left and right beam apertures:
                (R - L) / (R + L)
            [2] left_counts : float
                Left beam aperture counts.
            [3] right_counts : float
                Right beam aperture counts.
            [4] sum_std : float
                Standard deviation of the single sum.
            [5] diff_std : float
                Standard deviation of the single difference.
    """
    
    # check if fits_cube_path is a valid file path
    fits_cube_path = Path(fits_cube_path)
    if not fits_cube_path.is_file():
        raise FileNotFoundError(f"File not found: {fits_cube_path}")
    
    # retrieve fits cube data
    hdul = fits.open(fits_cube_path)
    cube_data = hdul[1].data

    # check if data is a 3d cube (wavelength, x, y)

    if cube_data.ndim != 3:
        raise ValueError("Input data must be a 3D cube (wavelength, x, y).")
        
    # check if wavelength_bin is within bounds
    if not (0 <= wavelength_bin < cube_data.shape[0]):
        raise ValueError(f"wavelength_bin must be between 0 and {cube_data.shape[0] - 1}.")
    
    image_data = cube_data[wavelength_bin]

    # define rectangular apertures for left and right beams
    # these values are based on ds9 pixel by pixel analysis
    centroid_lbeam = [71.75, 86.25]
    centroid_rbeam = [131.5, 116.25]
    aperture_width = 44.47634202584561
    aperture_height = 112.3750880855165
    theta = 0.46326596610192305

    # define apertures perform aperture photometry 
    aperture_lbeam = RectangularAperture(centroid_lbeam, aperture_width, aperture_height, theta=theta)
    aperture_rbeam = RectangularAperture(centroid_rbeam, aperture_width, aperture_height, theta=theta)
    phot_lbeam = aperture_photometry(image_data, aperture_lbeam)
    phot_rbeam = aperture_photometry(image_data, aperture_rbeam)

    # calculate normalized single difference and sum
    single_sum = phot_rbeam['aperture_sum'][0] + phot_lbeam['aperture_sum'][0]
    norm_single_diff = (phot_rbeam['aperture_sum'][0] - phot_lbeam['aperture_sum'][0]) #/ single_sum

    # get left and right counts
    left_counts = phot_lbeam['aperture_sum'][0]
    right_counts = phot_rbeam['aperture_sum'][0]

    # Assume Poissanian noise and propagate error
    sum_std = diff_std = np.sqrt(left_counts+right_counts)
    return (single_sum, norm_single_diff, left_counts, right_counts, sum_std, diff_std)

# function to fix corrupted hwp data
def fix_hwp_angles(csv_file_path, nderotator=8):
    '''Take corrupted HWP angles and replace them with assumed values
    in a new csv titled {old_title}_fixed.

    Parameters:
    -----------
    csv_file_path : str or Path
        Path to the specified CSV file containing the corrupted HWP angles.
    
    nderotator : int
        Number of derotator angles (assumed to be 8).
    Returns:
    --------
    None
    '''
    csv_file_path = Path(csv_file_path)
    if not csv_file_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    
    # read csv file into pandas dataframe
    df = pd.read_csv(csv_file_path)

    # check if 'RET-ANG1' column is present
    if 'RET-ANG1' not in df.columns:
        raise ValueError("Column 'RET-ANG1' is missing from the CSV file.")
    
    hwp_angles = np.linspace(0, 90, 9) # define assumed HWP angles
    hwp_angles_assumed = np.tile(hwp_angles, nderotator)  # repeat for n derotator angles
    df["RET-ANG1"] = hwp_angles_assumed # replace 'RET-ANG1' with assumed values
    # save to new csv file with '_fixed' suffix
    
    fixed_csv_path = csv_file_path.with_name(csv_file_path.stem + '_fixed.csv')
    df.to_csv(fixed_csv_path, index=False)
  

    print(f"Fixed HWP angles saved to {fixed_csv_path}")

def arr_csv_HWP(csv_path, hwp_order, todelete=None, new_csv_path=None):
    """Arranges CSVs by a custom HWP order. Deletes selected angles.
    
    Parameters:
    -----------
    csv_path: str or Path
        CSV containing relevant headers, can be obtained from
        write_fits_info_to_csv().
    hwp_order: list or np.ndarray
        List of desired HWP order. 
    todelete: list or np.ndarray, optional
        Optional list of HWP angles to delete.
    new_csv_path: str or Path, optional
        Optional path to create the new csv. If set to None,
        the csv will be edited in place.
    
    Returns:
    ---------
    df: Pandas DataFrame
        Returns DataFrame for visual inspection of csv changes.
        
    """
    hwp_order = np.array(hwp_order)

    # Load to a DF and sort
    df = pd.read_csv(csv_path)
    hwp_angles = df['RET-ANG1']
    if todelete:
        todelete = np.array(todelete)
        indices = np.where(np.isin(df['RET-ANG1'],todelete))[0]
        df = df.drop(indices)

    # Ensure the pattern loops correctly
    npattern = len(hwp_angles) // len(hwp_order)
    remainder = len(hwp_angles) % len(hwp_order)
    hwp_pattern = np.tile(hwp_order,npattern)
    hwp_pattern = np.concatenate((hwp_pattern,hwp_order[:remainder]))
    
    # Modify the DF
    df["RET-ANG1"] = pd.Categorical(df['RET-ANG1'],categories=hwp_order,ordered=True)
    df = df.sort_values(by=['D_IMRANG','RET-ANG1'])

    if new_csv_path:
        df.to_csv(new_csv_path,index=False)
    else:
        df.to_csv(csv_path,index=False)
    return df

    

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


def write_fits_info_to_csv(cube_directory_path, raw_cube_path, output_csv_path, wavelength_bin,hwp_order=[[0,45,11.25,56.25,22.5,67.5,33.75,78.75]],hwp_angles_to_delete=[90]):
    """Write filepath, D_IMRANG (derotator angle), RET-ANG1 (HWP angle), 
    single sum, single difference, LCOUNTS, RCOUNTS, difference std,
    sum std, and wavelength values for a wavelength bin from each fits cube in the directory.
    Default HWP order and deletion works for future double difference calculation. 

    FITS parameters are extracted from raw files, while single sum and difference are calculated using the
    fits cube data and the defined rectangular apertures.
    If the necessary header keywords are not present, the values will be set to NaN.

    Note - This function assumes that the raw and extracted cubes have the same number in the filepath. If
    you processed your cubes in the CHARIS DPP, this is not the case. 
    
    Parameters:
    -----------
    fits_directory_path : str or Path
        Path to the directory containing CHARIS fits cubes.
        
    raw_cube_path : str or Path
        Path to the directory containing the matching raw CHARIS FITS files.

    output_csv_path : str or Path
        Path where the output csv will be created.

    wavelength_bin : int
        Index of the wavelength bin to analyze (0-based).

    hwp_order: list or np.ndarray
        List of desired HWP order. Default works for double difference calculations.

    todelete: list or np.ndarray
        List of HWP angles to delete. Default works
        for double difference calculations. Set to None if you want to keep them all. 

    Returns:
    --------
    None
        Write all info to a csv with these columns: "filepath", "D-IMRANG", "RET-ANG1", "single_sum", "single_diff",
        "LCOUNTS","RCOUNTS", "sum_std", "diff_std", "wavelength_bin"
    """
    # check for valid file paths
    cube_directory_path = Path(cube_directory_path)
    raw_cube_path = Path(raw_cube_path)
    output_csv_path = Path(output_csv_path)

    if not cube_directory_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {cube_directory_path}")
    if output_csv_path.suffix != '.csv':
        raise ValueError(f"Output path must be a CSV file, got {output_csv_path}")
    if not raw_cube_path.is_dir():
        raise NotADirectoryError(f"Raw cube directory does not exist: {raw_cube_path}")
    if wavelength_bin > 21:
        raise ValueError(f"This function is currently only compatible with lowres mode, with 22 wavelength bins.")
    
    # prepare output csv file
    output_csv_path = Path(output_csv_path)
    with open(output_csv_path, 'w') as f:
        f.write("filepath,D_IMRANG,RET-ANG1,single_sum,single_diff,LCOUNTS,RCOUNTS,sum_std,diff_std,wavelength_bin\n")

        # iterate over all fits files in the directory
        for fits_file in sorted(cube_directory_path.glob('*.fits')):
            try:

                # check if corresponding raw fits file exists
                match = re.search(r"(\d{8})", fits_file.name)
                if not match:
                    raise ValueError(f"Could not extract 8-digit ID from filename {fits_file.name}")
                fits_id = match.group(1)
                raw_candidates = list(raw_cube_path.glob(f"*{fits_id}*.fits"))
                if not raw_candidates:
                    raise FileNotFoundError(f"No raw FITS file found for ID {fits_id}")
                raw_fits = raw_candidates[0]
                
                with fits.open(raw_fits) as hdul_raw:
                    raw_header = hdul_raw[0].header
                    d_imrang = raw_header.get("D_IMRANG", np.nan)
                    ret_ang1 = raw_header.get("RET-ANG1", np.nan)

                # round d_imrang to nearest 0.5
                d_imrang = (np.round(d_imrang * 2) / 2)

                # calculate single sum and normalized single difference
                single_sum, single_diff, LCOUNTS, RCOUNTS, sum_std, diff_std = single_sum_and_diff(fits_file, wavelength_bin)

                # wavelength bins for lowres mode
                bins = wavelength_bins
                
                # write to csv file
                f.write(f"{fits_file}, {d_imrang}, {ret_ang1}, {single_sum}, {single_diff}, {LCOUNTS}, {RCOUNTS}, {sum_std}, {diff_std}, {bins[wavelength_bin]}\n")

                # sort HWP angles
                if hwp_order:
                    arr_csv_HWP(output_csv_path,hwp_order,todelete=hwp_angles_to_delete)

            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
                
    print(f"CSV file written to {output_csv_path}")


def read_csv(file_path, mode= 'standard'):
    """Takes a CSV file path containing "D_IMRANG", 
    "RET-ANG1", "single_sum", "single_diff", "diff_std", and "sum_std",
    for one wavelength bin and returns interleaved values, standard deviations, 
    and configuration list.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV.
    mode : str, optional
        If mode = 'physical_model_CHARIS', the wavelengths will be added
        to the configuration list for physical model fitting.

    Returns:
    -----------
    interleaved_values : np.ndarray
        Interleaved values from "single_diff" and "single_sum".
    interleaved_stds : np.ndarray
        Interleaved standard deviations from "diff_std" and "sum_std".
    configuration_list : list
        List of dictionaries containing configuration data for each row.
        
    """
    file_path = Path(file_path)
     
    # Read CSV file
    
    df = pd.read_csv(file_path)
    
    # Convert relevant columns to float (handling possible conversion errors)
    for col in ["RET-ANG1", "D_IMRANG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set errors to NaN if not possible


    # Interleave values from "diff" and "sum"
    interleaved_values = np.ravel(np.column_stack((df["single_diff"].values, df["single_sum"].values)))

    # Interleave values from "diff_std" and "sum_std"
    interleaved_stds = np.ravel(np.column_stack((df["diff_std"].values, df["sum_std"].values)))

    # Convert each row's values into a list of two-element lists
    configuration_list = []
    for index, row in df.iterrows():
        # Extracting values from relevant columns
        hwp_theta = row["RET-ANG1"]
        imr_theta = row["D_IMRANG"]
        if mode == 'physical_model_CHARIS': # add wavelength
            wavelength = row["wavelength_bin"]
            # Building dictionary with wavelength
            row_data = {
                "hwp": {"theta": hwp_theta, "wavelength": wavelength},
                "image_rotator": {"theta": imr_theta, "wavelength": wavelength}
            }
        else:
            # Building dictionary
            row_data = {
                "hwp": {"theta": hwp_theta},
                "image_rotator": {"theta": imr_theta}
            }

        # Append two configurations for diff and sum
        configuration_list.append(row_data)
    if mode == 'physical_model_CHARIS':
        return interleaved_values, interleaved_stds, configuration_list
    else:
        return interleaved_values, interleaved_stds, configuration_list
    

def read_csv_physical_model_all_bins(csv_dir):
    """Does the same thing as read_csv() but reads all 22 csvs written
    in a directory for all 22 CHARIS wavelength bins and puts everything into one array.
    Also adds wavelength bin to the configuration dictionary for use with custom
    pyMuellerMat common mm functions. 
    Parameters:
    -----------
    csv_dir : Path or str
        The directory where the csv files are stored. Will check for bins in the title
        and for 22 files.

    Returns:
    -----------
    interleaved_values_all : list
        A list of interleaved values for all wavelength bins.
    interleaved_stds_all : list
        A list of interleaved standard deviations for all interleaved values.
    configuration_list_all : list
        A list of configuration dictionaries.
    """
    # Check if the directory exists
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"The directory {csv_dir} does not exist.")
        # Load csvs

    csv_files = sorted(csv_dir.glob("*.csv"))

    # Check for bins and sort files
 
    for f in csv_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not contain the bin number.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
    sorted_files = sorted(csv_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))
    if len(sorted_files) != 22:
        raise ValueError("Expected 22 CSV files for all wavelength bins, but found {}".format(len(sorted_files)))
    
    interleaved_values_all = []
    interleaved_stds_all = []
    configuration_list_all = []
    for file in sorted_files:
        interleaved_values, interleaved_stds, configuration_list= read_csv(file, mode='physical_model_CHARIS')
        interleaved_values_all = np.append(interleaved_values_all, interleaved_values)
        interleaved_stds_all = np.append(interleaved_stds_all, interleaved_stds)
        configuration_list_all.extend(configuration_list)

    return interleaved_values_all, interleaved_stds_all, configuration_list_all

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

# MAIN MCMC FUNCTIONS ARE IN INSTRUMENTS_JAX.PY


def summarize_median_posterior(h5_path, p0_dict, step_range=(0, None)):
    """
    Summarizes the median and 1 sigma credible interval (16th-84th percentile) 
    for each parameter in an MCMC run stored in an emcee HDF5 backend file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the emcee HDF5 backend file.
    p0_dict : dict
        Initial parameter dictionary structured as nested component: {param: val}.
    step_range : tuple
        Optional (start, stop) tuple to slice the chain steps.

    Returns
    -------
    dict
        Dictionary of median and 1 sigma intervals for each parameter.
    """
    h5_path = Path(h5_path)
    reader = emcee.backends.HDFBackend(h5_path)
    chain = reader.get_chain()

    # Flatten the chain over walkers
    flat_chain = chain[step_range[0]:step_range[1], :, :].reshape(-1, chain.shape[-1])

    # Extract parameter keys
    _, param_keys = parse_configuration(p0_dict)

    summary = {}

    print("Posterior Medians and 1 sigma Credible Intervals:")
    for i, comp in enumerate(param_keys):
        samples = flat_chain[:, i]
        median = np.median(samples)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)
        err_low = median - lower
        err_high = upper - median
        component = comp[0]
        key = comp[1]
        summary[key] = {
            "median": median,
            "-1sigma": err_low,
            "+1sigma": err_high
        }
        print(f"{component},{key}: {median:.5f} (+{err_high:.5f}/-{err_low:.5f})")

    return summary

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








########################################################################################
###### Functions related to fitting ####################################################
########################################################################################

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
        
def minimize_system_mueller_matrix(p0, system_mm, dataset, errors, 
    configuration_list, s_in = None, custom_function = None, 
    process_dataset = None, process_errors = None, process_model = None, include_sums=True,
    bounds = None, mode = 'minimize'):
    '''
    Perform a minimization on a dataset, using a System Mueller Matrix model. This function
    is highly customizable depending on fitting needs. All customizations are explained in detail
    in the parameter descriptions.

    Parameters
    ----------

    p0: dict of dict
        Dictionary of dictionaries detailing components, parameters to fit, and starting guesses.

        Example p0 dict: 
            {
            "image_rotator" : 
                {"d": 259.7, "delta_theta": 0.014},
            "hwp" :  
                {"w_SiO2": 1.652, "w_MgF2": 1.283, "delta_theta": 0.008},
            "lprot" : 
                {"pa": 0.233},
            }

    system_mm: pyMuellerMat.MuellerMat.SystemMuellerMatrix
        System Mueller matrix object generated from system dictionary and generate_system_mueller_matrix().
        Components in p0 dict and configuration list will be updated. More detailed instructions on how
        to generate these can be found in the minimization notebooks.

    dataset: list or np.ndarray
        Measured data. We currently use interleaved single differences and sums as the initial input.

    errors: list or np.ndarray
        Corresponding errors to measurements.

    configuration_list: list of dict of dict
        Instrument configuration for each measurement. Each component corresponds to a 
        component in the system dictionary used to generate the system Mueller
        Matrix and modifies the Mueller matrix to account for changing parts such as HWP angles, 
        derotator angles, etc. Can be obtained from read_csv().

        Example list for 2 measurements: 
            [{'hwp': {'theta': 0.0}, 'image_rotator': {'theta': 45.0}}, 
            {'hwp': {'theta': 11.25}, 'image_rotator': {'theta': 45.0}}]

    s_in: list or np.ndarray, optional
        Input Stokes vector (default: [1, 0, 0, 0])
    
    custom_function: function, optional
        Custom negative likelihood or cost function. Use a negative likelihood function for
        the default mode = 'minimize' and use a cost function for mode = 'least_squares'.
        The default functions used are logl() and cost(), respectively. 

    process_dataset: function, optional
        Function to process your data. Default is None, leaving your data as is for fitting. 
        Use the function process_dataset() in this module to convert interleaved single sums and differences to double differences.

    process_errors: function, optional
        Function to process errors. Default is none, leaving your errors as is for fitting
        Use the function process_errors() in this module to convert interleaved single sum and difference errors
        to double difference errors. 
    
    process_model: function, optional
        Function to process modeled L/R Wollaston beam intensities. Default is none, leaving L/R beams as is for fitting.
        Use the function process_model() in this module to convert L/R intensities to double differences

    include_sums: bool, optional
        Whether or not to index out the second element of interleaved differences and sums.
        Only use if model, data, and errors are processed. This is set as True because 
        it works with VAMPIRES. This must be set to false for CHARIS.

    mode : str, optional
        "minimize" (default): uses scipy minimize and does not return errors. Minimizes
        a negative log likelihood function.
        "least_squares": returns errors and uses scipy least squares, which takes
        a cost function cost() as an input. Error estimation
        procedure from Van Holstein et al. 2020. 

    '''
    
    if s_in is None:
        s_in = np.array([1, 0, 0, 0])  # Default s_in

    p0_values, p0_keywords = parse_configuration(p0)


    # Running scipy.minimize
    if mode == 'minimize':
        result = minimize(logl, p0_values, 
            args=(p0_keywords, system_mm, dataset, errors, configuration_list, 
                s_in, custom_function, process_dataset, process_errors, process_model,include_sums), 
                method='L-BFGS-B', bounds = bounds)
        

    elif mode == 'least_squares':
        lower_bounds = [bound[0] for bound in bounds]
        upper_bounds = [bound[1] for bound in bounds]
        result = least_squares(cost, p0_values, 
            args=(p0_keywords, system_mm, dataset, errors, configuration_list,
                s_in,custom_function,process_dataset,process_errors,process_model,include_sums), method='trf', bounds = (lower_bounds, upper_bounds),verbose=2)
        J = result.jac
        residual = result.fun
        dof = len(residual) - len(result.x)  # degrees of freedom
        s_res_squared = np.sum(residual**2) / dof
        try:
            cov = s_res_squared * np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            print("Warning: Jacobian is singular — using pseudo-inverse instead.")
            cov = s_res_squared * np.linalg.pinv(J.T @ J)
        errors = np.sqrt(np.diag(cov))
                    
            
    
    # Saving the final result's logl value
    if mode == 'minimize':
        logl_value = logl(result.x, p0_keywords, system_mm, dataset, errors, 
            configuration_list, s_in=s_in, custom_function=custom_function, 
            process_dataset=process_dataset, process_errors = process_errors, 
            process_model = process_model,include_sums=include_sums)
    if mode == 'least_squares':
        logl_value = -result.cost
        return result, logl_value,errors
    elif mode =='minimize':
        return result, logl_value
    
def logl(p, system_parameters, system_mm, dataset, errors, configuration_list, 
         s_in=None, custom_function=None, process_dataset=None, process_errors=None, 
         process_model=None,include_sums=True):
    """
    Compute the negative log-likelihood of a model given a dataset and system configuration
    for later use in scipy minimize.
    This function evaluates how well a set of system Mueller matrix parameters
    (given by `p`) reproduce the observed dataset, using a chi-squared-based 
    likelihood metric or a user-defined function.

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

    configuration_list: list of dict of dict
        Instrument configuration for each measurement. Each component corresponds to a 
        component in the system dictionary used to generate the system Mueller
        Matrix and modifies the Mueller matrix to account for changing parts such as HWP angles, 
        derotator angles, etc. Can be obtained from read_csv().

        Example list for 2 measurements: 
            [{'hwp': {'theta': 0.0}, 'image_rotator': {'theta': 45.0}}, 
            {'hwp': {'theta': 11.25}, 'image_rotator': {'theta': 45.0}}]

    s_in: list or np.ndarray, optional
        Input Stokes vector (default: [1, 0, 0, 0])
    
    custom_function: function, optional
        Custom negative likelihood or cost function. Use a negative likelihood function for
        the default mode = 'minimize' and use a cost function for mode = 'least_squares'.
        The default functions used are logl() and cost(), respectively. 

    process_dataset: function, optional
        Function to process your data. Default is None, leaving your data as is for fitting. 
        Use the function process_dataset() in this module to convert interleaved single sums and differences to double differences.

    process_errors: function, optional
        Function to process errors. Default is none, leaving your errors as is for fitting
        Use the function process_errors() in this module to convert interleaved single sum and difference errors
        to double difference errors. 
    
    process_model: function, optional
        Function to process modeled L/R Wollaston beam intensities. Default is none, leaving L/R beams as is for fitting.
        Use the function process_model() in this module to convert L/R intensities to double differences

    include_sums: bool, optional
        Whether or not to index out the second element of interleaved differences and sums.
        Only use if model, data, and errors are processed. This is set as True because 
        it works with VAMPIRES. This must be set to false for CHARIS.
    Returns
    -------
    float
        The computed negative log-likelihood value (lower is better).
    """


    # Generating a list of model predicted values for each configuration - already parsed
    output_intensities = model(p, system_parameters, system_mm, configuration_list, 
        s_in=s_in, process_model=process_model)

    # Convert lists to numpy arrays
    dataset = np.array(dataset)
    errors = np.array(errors)

    # Optionally parse the dataset and output intensities (e.g., normalized difference)
    if process_dataset is not None:
        processed_dataset = process_dataset(copy.deepcopy(dataset))
    elif process_dataset is None:
        processed_dataset = copy.deepcopy(dataset)
    # Optionally parse the dataset and output intensities (e.g., normalized difference)
    if process_errors is not None:
        processed_errors = process_errors(copy.deepcopy(errors), 
            copy.deepcopy(dataset))
    elif process_errors is None:
        processed_errors = copy.deepcopy(errors)

    dataset = copy.deepcopy(processed_dataset)
    errors = copy.deepcopy(processed_errors)
    # Note - raised floor fromm 1e-3 to 1e-7 to be compatible with small normalized errors
    errors = np.maximum(errors, 1e-7)
    # Calculate log likelihood
    if include_sums is True: # take out differences
        dataset=dataset[::2]
        errors=errors[::2]
        output_intensities=output_intensities[::2]

    if custom_function is not None:
        return custom_function(p, output_intensities, dataset, errors)
    else:
        return 0.5 * np.sum((output_intensities - dataset) ** 2 / errors ** 2)

def cost(p, system_parameters, system_mm, dataset, errors, configuration_list, 
         s_in=None,custom_function=None,process_dataset=None,process_errors=None,process_model=None,include_sums=True):
    """
    Cost function that describes how well Mueller matrix parameters fit data.


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
    s_in: list or np.ndarray, optional
        Input Stokes vector (default: [1, 0, 0, 0])
    
    custom_function: function, optional
        Custom negative likelihood or cost function. Use a negative likelihood function for
        the default mode = 'minimize' and use a cost function for mode = 'least_squares'.
        The default functions used are logl() and cost(), respectively. 

    process_dataset: function, optional
        Function to process your data. Default is None, leaving your data as is for fitting. 
        Use the function process_dataset() in this module to convert interleaved single sums and differences to double differences.

    process_errors: function, optional
        Function to process errors. Default is none, leaving your errors as is for fitting
        Use the function process_errors() in this module to convert interleaved single sum and difference errors
        to double difference errors. 
    
    process_model: function, optional
        Function to process modeled L/R Wollaston beam intensities. Default is none, leaving L/R beams as is for fitting.
        Use the function process_model() in this module to convert L/R intensities to double differences

    include_sums: bool, optional
        Whether or not to index out the second element of interleaved differences and sums.
        Only use if model, data, and errors are processed. This is set as True because 
        it works with VAMPIRES. This must be set to false for CHARIS.
    Returns
    -------
    np.ndarray
        1D vector of standardized residuals suitable for scipy.optimize.least_squares.
    """

    # print("Entered logl")

    # Generating a list of model predicted values for each configuration - already parsed
    output_intensities = model(p, system_parameters, system_mm, configuration_list, process_model=process_model, 
        s_in=s_in)
    
    # Processing model converts raw L/R intensities to double differences
    # Processing errors converts interleaved single sum/difference errors to an array
    # of double difference errors
    if process_errors is not None:
        processed_errors = process_errors(copy.deepcopy(errors), 
            copy.deepcopy(dataset))
    elif process_errors is None:
        processed_errors = copy.deepcopy(errors)
    if process_dataset is not None:
        processed_dataset = process_dataset(copy.deepcopy(dataset))
    elif process_dataset is None:
        processed_dataset = copy.deepcopy(dataset)

    dataset = copy.deepcopy(processed_dataset)
    errors = copy.deepcopy(processed_errors)
    # Numerical floor to avoid division by tiny errors
    errors = np.maximum(errors, 1e-7)
    if include_sums is True:
        dataset=dataset[::2]
        errors=errors[::2]
        output_intensities=output_intensities[::2]
    # Convert lists to numpy arrays, only differences used
    residuals = output_intensities - dataset
    #chi_squared = np.sum((residuals / errors) ** 2)
    cost = residuals / errors

    if custom_function is not None:
        return custom_function(p, output_intensities, dataset, errors)
    else:
        return cost

def model(p, system_parameters, system_mm, configuration_list, s_in=None, 
        process_model = None):
    """Returns simulated L/R wollaston beam intensities for a given set of parameters based on
    parameter values, a dictionary detailing those values based on pyMuellerMat, 
    a pyMuellerMat system Mueller matrix, and a list of configurations for the 
    system Mueller matrix.

    Parameters
    ----------
    p : list of float
        List of parameter values. One list of values per parameter.
    
    system_parameters : list of [str, str]
        List of ['component_name', 'parameter_name'] pairs corresponding to `p`.
        For example, [['Polarizer', 'Theta'], ['Polarizer', 'Phi']] if your p
        is two lists of parameters Theta and Phi for a Polarizer component.

    system_mm : pyMuellerMat system Mueller matrix object
        Mueller matrix model of the optical system. Any parameters
        that are specified in p and system_parameters will be replaced.

    configuration_list : list of dict
        Each dict will update parameters of your current system_mm 
        to generate measurements. For example, if you want 9 HWP angles
        and one derotator angle, each config dict will have this form:
        {"hwp": {"theta": hwp_angle},
        "image_rotator": {"theta": 45.0}
        Append each unique configuration to your configuration_list.

    s_in : np.ndarray, optional
        Input Stokes vector, default is unpolarized light [1, 0, 0, 0].

    process_model : callable, optional
        Converts output intensities to double differences. 

    """
    
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
    differences_errors = np.sqrt(input_errors[::2]**2 + input_errors[1::2]**2)
    sums_errors = np.sqrt(input_errors[::2]**2 + input_errors[1::2]**2)


    # Compute double differences and double sums
    differences = input_dataset[::2] - input_dataset[1::2]
    sums = input_dataset[::2] + input_dataset[1::2]

    denominator = (sums[::2] + sums[1::2])  # This is used for normalization

    # Compute propagated errors for double differences
    double_differences_errors = np.sqrt(
        (sums[::2] + sums[1::2])**2 * (differences_errors[::2]**2 + differences_errors[1::2]**2) + 
        (differences[::2] - differences[1::2])**2 * (sums_errors[::2]**2 + sums_errors[1::2]**2)
    ) / (denominator**2)

    # Compute propagated errors for double sums
    double_sums_errors = np.sqrt(
        (sums[::2] + sums[1::2])**2 * (sums_errors[::2]**2 + sums_errors[1::2]**2) + 
        (sums[::2] - sums[1::2])**2 * (sums_errors[::2]**2 + sums_errors[1::2]**2)
    ) / (denominator**2)


    # Interleave errors to maintain order
    interleaved_errors = np.ravel(np.column_stack((double_differences_errors, double_sums_errors)))
    # Double diffs extracted this way for ease of reverting back to the original setup


    return interleaved_errors

# streamline process for all wavelength bins
# SET UP FOR DDs from NSD
def fit_CHARIS_Mueller_matrix_by_bin(csv_path, wavelength_bin, new_config_dict_path,plot_path=None):
    """
    Mainly a wrapper function for minimize_system_mueller_matrix(). I find it easier to just modify
    this function every time I do a new fit than to use minimize_system_mueller_matrix().
    Fits a Mueller matrix for one wavelength bin from internal calibration data and saves
    the updated configuratio dictionary to a JSON file. Creates a plot
    of each updated model vs the data. Initial guesses for all fits are from Joost t Hart 2021.
    Note that following the most recent model update these guesses should be updated.
    The csv containing the calibration data and relevant headers can be obtained by 
    the write_fits_info_to_csv function in instruments.py. This code is always being modified to fit
    different things. What is being fitted for can be found in the p0 dictionary in the code.
    It can be modified relatively easily to fit for other parameters as well. 

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing the calibration data. Must contain the columns "D_IMRANG", 
    "RET-ANG1", "single_sum", "single_diff", "diff_std", and "sum_std".

    wavelength_bin : int
        The index of the wavelength bin to fit (0-21 for CHARIS).

    new_system_dict_path : str or Path
        Path to save the new system dictionary as a JSON file. The config dict
        component names will be 'lp' for calibration polarizer, 'image_rotator' for image rotator,
        and 'hwp' for half-wave plate.

    plot_path : str or Path, optional
        Path to save the plot of the observed vs modeled data. If not provided, no plot will be saved.
        Must have a .png extension.
    
    Returns
    -------
    error : np.array
      An array of the errors for each parameter. 
    fig : MatPlotLib figure object
    ax : MatPlotLib axis object
    """
    # Check file paths
    filepath = Path(csv_path)
    new_config_dict_path=Path(new_config_dict_path)
    if not filepath.exists() or filepath.suffix != ".csv":
        raise ValueError("Please provide a valid .csv file.")
    if plot_path:
        plot_path = Path(plot_path)
        if plot_path.suffix != ".png":
            raise ValueError("Please provide a valid .png file for plotting.")
    if new_config_dict_path.suffix != ".json":
        raise ValueError("Please provide a valid .json file for saving the new system dictionary.")
    new_config_dict_path = Path(new_config_dict_path)

    # Read in data
    interleaved_values, interleaved_stds, configuration_list = read_csv(filepath)

    # this works, not really sure why
    interleaved_values_forplotfunc = copy.deepcopy(interleaved_values)
    interleaved_stds_forlplotfunc = copy.deepcopy(interleaved_stds)
    #interleaved_stds = process_errors(interleaved_stds,interleaved_values)[::2] # just diffs
    #interleaved_values = process_dataset(interleaved_values)[::2] # just diffs
    #configuration_list = configuration_list 

    # Loading in past fits 

    offset_imr = -0.0118 # derotator offset
    offset_hwp = -0.002# HWP offset
    offset_cal = -0.035 # calibration polarizer offset
    imr_theta = 0 # placeholder 
    hwp_theta = 0 # placeholder
    imr_phi = IMR_retardance(wavelength_bins)[wavelength_bin]
    hwp_phi = HWP_retardance(wavelength_bins)[wavelength_bin]
    epsilon_cal = 1

    # Define instrument configuration as system dictionary
    # Wollaston beam, imr theta/phi, and hwp theta/phi will all be updated within functions, so don't worry about their values here

    system_dict = {
        "components" : {
            "wollaston" : {
            "type" : "wollaston_prism_function",
            "properties" : {"beam": 'o'}, 
            "tag": "internal",
            },
            "image_rotator" : {
                "type" : "general_retarder_function",
                "properties" : {"phi": 0, "theta": imr_theta, "delta_theta": offset_imr},
                "tag": "internal",
            },
            "hwp" : {
                "type" : "general_retarder_function",
                "properties" : {"phi": 0, "theta": hwp_theta, "delta_theta": offset_hwp},
                "tag": "internal",
            },
            "lp_rot": { # changed from delta_theta to match Joost t Hart
                "type": "rotator_function",
                "properties" : {'pa':offset_cal},
                "tag": "internal",
            },
            "lp" : {  # calibration polarizer for internal calibration source
                "type": "diattenuator_retarder_function",
                "properties": {"epsilon":1},
                "tag": "internal",
            }}
    }


    # Converting system dictionary into system Mueller Matrix object

    system_mm = generate_system_mueller_matrix(system_dict)

    # Define initial guesses for our parameters 
    # Modify this if you want to change the parameters

    p0 = {
        "image_rotator" : 
            {"phi": imr_phi, "delta_theta": offset_imr},
        "hwp" :
            {"phi": hwp_phi, "delta_theta": offset_hwp},
        "lp_rot":
            {'pa':offset_cal},
        "lp" : 
            {"epsilon": epsilon_cal},
    }

    # Define some bounds
    # Modify this if you want to change the parameters or minimization bounds
    offset_bounds = (-5,5)
    hwpstd = 0.1*np.abs(hwp_phi)
    hwp_phi_bounds = (hwp_phi-hwpstd, hwp_phi+hwpstd)
    imrstd = 0.1*np.abs(imr_phi)
    imr_phi_bounds = (imr_phi-imrstd, imr_phi+imrstd)
    #imrostd = 0.1*np.abs(offset_imr)
    #offset_imr_bounds = (offset_imr-imrostd, offset_imr+imrostd)
    #hwpostd = 0.1*np.abs(offset_hwp)
    #offset_hwp_bounds = (offset_hwp-hwpostd, offset_hwp+hwpostd)
    epsilon_cal_bounds = (0.9*epsilon_cal, 1)
    #calostd = 0.1 *np.abs(offset_cal)
    #offset_cal_bounds = (-15, 15)
    #dichroic_phi_bounds = (0,np.pi)

    # Minimize the system Mueller matrix using the interleaved values and standard deviations
    # Modify this if you want to change the parameters

    # Counters for iterative fitting

    iteration = 1
    previous_logl = 1000000
    new_logl = 0

    # Perform iterative fitting

    while abs(previous_logl - new_logl) > 0.01*abs(previous_logl):
        if iteration > 1:
            previous_logl = new_logl
        # Configuring minimization function for CHARIS
        result, new_logl, error = minimize_system_mueller_matrix(p0, system_mm, interleaved_values, 
            interleaved_stds, configuration_list, process_dataset=process_dataset,process_model=process_model,process_errors=process_errors,include_sums=True, bounds = [imr_phi_bounds,offset_bounds,hwp_phi_bounds,offset_bounds,offset_bounds,epsilon_cal_bounds],mode='least_squares')
        print(result)

        # Update p0 with new values

        update_p0(p0, result.x)
        iteration += 1


    # Update p dictionary with the fitted values

    update_p0(p0, result.x)

    # Process model

    p0_values, p0_keywords = parse_configuration(p0)

    # Generate modeled left and right beam intensities

    updated_system_mm = update_system_mm(result.x, p0_keywords, system_mm)

    # Generate modeled left and right beam intensities

    LR_intensities2 = model(p0_values, p0_keywords, updated_system_mm, configuration_list)

    # Process these into interleaved single normalized differences and sums

    diffs_sums2 = process_model(LR_intensities2)

    # Plot the modeled and observed values
    if plot_path:
        fig , ax = plot_data_and_model(interleaved_values_forplotfunc, interleaved_stds_forlplotfunc, diffs_sums2,configuration_list, wavelength= wavelength_bins[wavelength_bin], mode='CHARIS',save_path=plot_path)
    else:
        fig , ax = plot_data_and_model(interleaved_values_forplotfunc, interleaved_stds_forlplotfunc, diffs_sums2,configuration_list, wavelength= wavelength_bins[wavelength_bin], mode='CHARIS')
    
    # Print the Mueller matrix

    print("Updated Mueller Matrix:")
    print(updated_system_mm.evaluate())

    # Print residuals
    print(len(interleaved_values), len(diffs_sums2))
    data_dd = process_dataset(interleaved_values)[::2]
    model_dd = diffs_sums2[::2]
    residuals = data_dd - model_dd
    print("Residuals range:", residuals.min(), residuals.max())
    print("Error:", error)

    # Save system dictionary to a json file

    with open (new_config_dict_path, 'w') as f:
        json.dump(p0, f, indent=4)
    error = np.array(error)
    return error, fig, ax





#######################################################
###### Functions related to plotting ##################
#######################################################

# function for calculating single sum and difference

    

def plot_data_and_model(interleaved_values, interleaved_stds, model, 
    configuration_list, imr_theta_filter=None, wavelength=None, save_path = None, include_sums=True,title=None):
    """
    Plots double difference and double sum measurements alongside model predictions,
    grouped by image rotator angle (D_IMRANG). Optionally filters by a specific 
    image rotator angle and displays a wavelength in the plot title.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed single difference and single sum values.
        Expected format: [sd1, ss1, sd2, ss2, ...]. 

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.
        If charis use single differences and sums. 

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    imr_theta_filter : float, optional
        If provided, only measurements with this image rotator angle (rounded to 0.1°) 
        will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").

    include_sums : bool, optional
        Default is True, plotting the double sums as well as the differences. If false, only the
        double differences are plotted, a residual bar is included, and some other plotting things are updated.

    title : str, optional
        Default is the wavelength.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        A tuple containing the Figure and Axes objects of the plot.
    """
    # Calculate double differences and sums from interleaved single differences if in VAMPIRES mode 
    
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
    if include_sums is True:
        num_plots = 2
        fig, axes = plt.subplots(1, num_plots, figsize=(14, 6), sharex=True)

    elif include_sums is False:
        fig, axarr = plt.subplots(
        2, 1, 
        figsize=(10, 6), 
        gridspec_kw={"height_ratios": [3, 1]}, 
        sharex=True
        )
        ax = axarr[0]
        small_ax = axarr[1]

    # Double Difference plot
    if include_sums is True:
        ax = axes[0]
        for theta, d in dd_by_theta.items():
           err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
           color = err[0].get_color()
           ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel(r"HWP $\theta$ (deg)")
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"IMR $\theta$")
    # Double Sum plot
        ax = axes[1]
        for theta, d in ds_by_theta.items():
            err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
            color = err[0].get_color()
            ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel(r"HWP $\theta$  (deg)")
        ax.set_ylabel("Double Sum")
        ax.legend(title=r"IMR $\theta$")
    elif include_sums is False:
        for theta, d in dd_by_theta.items():
           err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
           color = err[0].get_color()
           ax.plot(d["hwp_theta"], d["model"], '-', color=color)
           residuals =  ((np.array(d["values"]) - np.array(d["model"])))*100
           small_ax.scatter(d['hwp_theta'],residuals,color=color)
        small_ax.axhline(0, color='black', linewidth=1)
        small_ax.set_xlabel(r"HWP $\theta$ (deg)")
        small_ax.set_ylabel(r"Residual ($\%$)", fontsize = 15)
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"IMR $\theta$", fontsize=10)
        ax.grid()

    # Set a suptitle if wavelength is provided
    if wavelength is not None and title is None:
        fig.suptitle(f"{wavelength}nm")
    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    if save_path != None:
        plt.savefig(save_path,dpi=600, bbox_inches='tight')

    plt.show()
    return fig, ax

def plot_fluxes(csv_path, plot_save_path=None):
    """Plot left and right beam fluxes as a function of the HWP angle for one 
    wavelength bin and derotator angle from a CSV containing headers "LCOUNTS", 
    "RCOUNTS", "RET-ANG1", "D_IMRANG" and "wavelength_bin".
    This can be obtained from the write_fits_info_to_csv function.

    Parameters:
    -----------
    csv_path : str or Path
        Path to the specified CSV file.

    plot_save_path : str or Path, optional
        If provided, the plot will be saved to this path. Must end with '.png'.

    Returns: 
    --------
    fig, ax : matplotlib Figure and Axes
        A tuple containing the Figure and Axes objects of the plot.

    """
    
    # check if csv_path is a valid file path

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    # read csv file into pandas dataframe

    df = pd.read_csv(csv_path)

    # check if necessary columns are present

    required_columns = ['LCOUNTS', 'RCOUNTS', 'RET-ANG1', 'wavelength_bin', 'D_IMRANG']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the CSV file.")
        
    # check for multiple wavelength bins
    unique_bins = df['wavelength_bin'].unique()
    if len(unique_bins) > 1:
        raise ValueError("CSV file contains multiple wavelength bins. Please filter to a single bin before plotting.")
    
    # extract data for plotting

    hwp_angles = df['RET-ANG1'].values
    left_counts = df['LCOUNTS'].values
    right_counts = df['RCOUNTS'].values
    derotator_angles = df['D_IMRANG'].values
    wavelength_bin = df['wavelength_bin'].values[0]

# plot counts as a function of HWP angle for each derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    unique_angles = np.unique(derotator_angles)
    colors = [
    cmap(i / max(len(unique_angles) - 1, 1))
    for i in range(len(unique_angles))
]
    for idx, derotator_angle in enumerate(unique_angles):
        mask = derotator_angles == derotator_angle
        color = colors[idx]  # use a different color for each derotator angle
        sort_order = np.argsort(hwp_angles[mask])
        ax.plot(hwp_angles[mask][sort_order], left_counts[mask][sort_order], marker='o', label=f'{derotator_angle}° (L)', color=color)
        ax.plot(hwp_angles[mask][sort_order], right_counts[mask][sort_order], marker='x', linestyle='--', label=f'{derotator_angle}° (R)', color=color)
    ax.set_xlabel('HWP Angle (degrees)')
    ax.set_ylabel('Counts')
    ax.set_title(f'L/R Counts vs HWP Angle at {wavelength_bin} nm')
    ax.legend(loc='lower right', fontsize='small', title= r'IMR $\theta$')
  


    ax.grid()


    # save plot if a path is provided

    if plot_save_path:
        plot_save_path = Path(plot_save_path)
        if not plot_save_path.parent.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {plot_save_path.parent}")
        if not plot_save_path.suffix == '.png':
            raise ValueError("Plot save path must end with '.png'.")
        fig.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")
    return fig, ax

    
def plot_config_dict_vs_wavelength(component, parameter, json_dir, save_path=None, title=None, axtitle=None):
    """
    Plots a parameter in the JSON configuration dictionaries vs wavelength.
    Only works if all JSON dictionaries are in a directory labeled
    by bin. Sets parameters phi, delta_theta, and theta to degrees.
    Returns an array of the parameters to allow for custom plotting.
    This plot is similar to the one in van Holstein 2020.
    
    Parameters
    ----------
    component : str
        The name of the component (e.g., 'image_rotator', 'hwp', etc.).
    parameter : str
        The key of the parameter in the system dictionary.
    json_dir : str or Path
        The directory containing the JSON configuration dictionaries for all 22 bins.
        Make sure the directory only contains these 22 JSON files. 
        Component names are 'lp' for calibration polarizer, 'image_rotator' for image rotator,
        and 'hwp' for half-wave plate.
    save_path : str or Path, optional
        If specified, saves the plot to this path. Otherwise, displays the plot.
    title : str, optional
        Title for the plot. If not provided, a default title is used.
    axtitle : str, optional
        Title for the y-axis. If not provided, a default title is used.
    
    Returns
    -------
    parameters : np.ndarray
        An array of the parameter values extracted from the JSON files.
        To plot, plot against the default CHARIS wavelength bins (can
        be found in instruments.py).
    fig, ax : matplotlib Figure and Axes
        The Figure and Axes objects of the plot.
    """

    # Check filepaths

    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a valid directory.")
    if save_path is not None:
        save_path = Path(save_path)

    # Load JSON files

    json_files = sorted(json_dir.glob("*.json"))

    # Check for correct file amount

    if len(json_files) != 22:
        raise ValueError(f"Expected 22 JSON files, found {len(json_files)}.")
    
    # Check for bins
 
    for f in json_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not match expected naming convention.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
     # Sort Jsons

    sorted_files = sorted(json_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))

    # Extract parameters

    parameters = []

    for f in sorted_files:
        with open(f, 'r') as file:
            data = json.load(file)
            if component not in data:
                raise ValueError(f"Component '{component}' not found in {f.name}.")
            if parameter not in data[component]:
                raise ValueError(f"Parameter '{parameter}' not found in component '{component}' in {f.name}.")
            # Set relevant components to degrees
            if parameter == 'theta' or parameter == 'delta_theta' or parameter == 'phi':
                data[component][parameter] = np.degrees(data[component][parameter])
            
            parameters.append(data[component][parameter])

    # Convert to numpy array for plotting

    parameters = np.array(parameters)

    # Plot vs wavelength bins

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(wavelength_bins, parameters, marker='x',color='black', linewidths = 1)
    ax.set_xlabel('Wavelength (nm)')
    if axtitle is not None:
        ax.set_ylabel(axtitle)
    else:
        ax.set_ylabel(parameter)
    if title is None:
        ax.set_title(f'{component}: {parameter} vs wavelength')
    else:
        ax.set_title(title)
    ax.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return parameters,fig,ax
    
   
def plot_polarimetric_efficiency(json_dir, bins, save_path=None, title=None):
    """
    Plots the polarimetric efficiency from JSON configuration dictionaries vs derotator angle.
    Works with the result of the fit_CHARIS_mueller_matrix_by_bin() function
    Only works if all 22 JSON dictionaries are in a directory labeled
    by bin: formatted bin{bin} in the filepath. Returns an array of the polarimetric efficiency values.
    This plot is similar to the one in van Holstein 2020. Also plots
    where measured derotator angles would be with an 'x' marker.
    
    Parameters
    ----------
    json_dir : str or Path
        The directory containing the JSON system dictionaries for all 22 bins.
        Make sure the directory only contains these 22 JSON files. Component names
        are 'lp' for calibration polarizer, 'image_rotator' for image rotator,
        and 'hwp' for half-wave plate.
    bins : np.ndarray
        An array of wavelength bins to simultaneously plot.
    save_path : str or Path, optional
        If specified, saves the plot to this path. 
    title : str, optional
        Title for the plot. If not provided, a default title is used.

    Returns
    -------
    polarimetric_efficiency : np.ndarray
        A 2 dimensional array of the polarimetric efficiency values extracted from the JSON files
        where the first dimension corresponds to the wavelength bins and the second dimension represents the derotator angles.
    fig, ax : matplotlib Figure and Axes
        The Figure and Axes objects of the plot.
    """
    # Check filepaths

    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a valid directory.")
    if save_path is not None:
        save_path = Path(save_path)

    # Load JSON files

    json_files = sorted(json_dir.glob("*.json"))

    # Check for correct file amount

    if len(json_files) != 22:
        raise ValueError(f"Expected 22 JSON files, found {len(json_files)}.")
    
    # Check for bins
 
    for f in json_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not match expected naming convention.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
     # Sort Jsons

    sorted_files = sorted(json_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))

    # Get derotator angles

    derotator_angles_measured = np.array([45, 57.5,70,82.5,95,107.5,120,132.5])
    derotator_angles = np.linspace(0,180,361)

    # Get polarimetric efficiencies


    output = []
    for wavelength_bin in bins:
        # Create efficiency array
        efficiencies = []
        # Extract the Mueller matrix for each derotator angle
        for derotator_angle in derotator_angles:
            # Define system dictionary components for the current wavelength bin and derotator angle
            file = sorted_files[wavelength_bin]
            data = json.load(open(file, 'r'))
           # Parse the dictionary into usable values
            values, keywords = parse_configuration(data)

            # Generate system Mueller matrix without Wollaston prism 

            sys_dict = {
            "components" : {
              #   "wollaston" : {
              #   "type" : "wollaston_prism_function",
              #  "properties" : {"beam": "o"},
              #  "tag": "internal",
              #  },
                "image_rotator" : {
                    "type" : "general_retarder_function",
                    "properties" : {"phi": 0, "theta": derotator_angle, "delta_theta": 0},
                    "tag": "internal",
                },
                "hwp" : {
                    "type" : "general_retarder_function",
                    "properties" : {"phi":0, "theta": 0, "delta_theta": 0},
                    "tag": "internal",
                },
                "lp" : {
                    "type": "general_linear_polarizer_function_with_theta",
                    "properties": {"delta_theta": 0},
                    "tag": "internal",
                }}
            }

            # generate Mueller matrix object

            system_mm = generate_system_mueller_matrix(sys_dict)

            # Update the Mueller matrix with the model

            updated_system_mm = update_system_mm(values, keywords, system_mm)

            # Calculate the polarimetric efficiency

            M_10 = updated_system_mm.evaluate()[1,0]
            M_20 = updated_system_mm.evaluate()[2,0]
            M_00 = updated_system_mm.evaluate()[0,0]
            efficiency = np.sqrt(M_10**2 + M_20**2) / M_00
            efficiencies.append(efficiency)
        output.append(efficiencies)

    pol_efficiencies = np.array(output)

    # Grab measured angles from pol efficiencies

    measured_indices = []
    for idx,angle in enumerate(derotator_angles):
        if angle in derotator_angles_measured:
            measured_indices.append(idx)
    pol_efficiencies_measured = pol_efficiencies[:, measured_indices]

    # Plot vs derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, wavelength_bin in enumerate(bins):
        ax.plot(derotator_angles, pol_efficiencies[i], label=f"{int(wavelength_bins[wavelength_bin])} nm")
        ax.scatter(derotator_angles_measured, pol_efficiencies_measured[i], marker='x', color='black')
    ax.scatter([], [], marker='x', color='black', label='Angles Used in Data')
    ax.set_xlabel('Derotator Angle (degrees)')
    ax.set_ylabel('Polarimetric Efficiency')
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if title is None:
        ax.set_title('Polarimetric Efficiency vs Derotator Angle')
    else:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return pol_efficiencies, fig, ax


def plot_pol_efficiency_from_data(csv_dir, bins, save_path=None, title=None):
    """Plots polarimetric efficiency as a function of 
    the derotator angle from a csv with relevant columns (can be
    obtained from write_fits_info_to_csv) assuming a total linearly horizontally
    polarized source. This assumes 8 derotator angles and 9 hwp angles. Can plot
    multiple wavelength bins simultaneously.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV directory containing csvs with relevant bins.

    save_path : str or Path, optional
        If provided, the plot will be saved to this path. Must end with '.png'.

    title : str, optional
        Title of the plot. If None, a default title will be used.
    Returns:
    --------
    pol_efficiency : np.array
        Polarimetric efficiency calculated from the interleaved values.
    fig, ax : matplotlib Figure and Axes
        The Figure and Axes objects of the plot.
    """
    csv_dir= Path(csv_dir)

    # Load csvs

    csv_files = sorted(csv_dir.glob("*.csv"))

    # Check for bins
 
    for f in csv_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not contain the bin number.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
    # Sort csvs and extract interleaved values
    derotator_angles = np.linspace(45,132.5,8)
    sorted_files = sorted(csv_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))
    output = []
    for wavelength_bin in bins:
        # Create efficiency array
        efficiencies = []
        file = sorted_files[wavelength_bin]
        interleaved_values = read_csv(file)[0]
        diffs = interleaved_values[::2]
        # Extract the values for each derotator angle
        for i in range(8):  # 8 derotator angles
            start = i * 9
            hwp_diffs = diffs[start:start + 9]  # 9 HWP angles for this derotator
            # Double-difference using HWP 0°, 22.5°, 45°, 67.5°
            try:
                Q = 0.5 * (hwp_diffs[0] - hwp_diffs[4])  # 0° - 45°
                U = 0.5 * (hwp_diffs[2] - hwp_diffs[6])  # 22.5° - 67.5°
            except IndexError:
                print(f"Error: Missing HWP angle at derotator index {i}, file {file}")
                Q = U = 0.0

            pol_eff = np.sqrt(Q**2 + U**2)
            efficiencies.append(pol_eff)
        output.append(efficiencies)
    pol_efficiencies = np.array(output)

    # Plot vs derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, wavelength_bin in enumerate(bins):
        ax.plot(derotator_angles, pol_efficiencies[i], label=f"{int(wavelength_bins[wavelength_bin])} nm")
        ax.scatter(derotator_angles, pol_efficiencies[i], marker='x', alpha=0.7, color='black')
    ax.set_xlabel('Derotator Angle (degrees)')
    ax.set_ylabel('Polarimetric Efficiency')
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    if title is None:
        ax.set_title('Polarimetric Efficiency vs Derotator Angle')
    else:
        ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='lower center')
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    return pol_efficiencies, fig, ax


def model_data(json_dir, csv_path=None):
    """
    Creates a Pandas DataFrame of the fitted IMR/HWP retardances and 
    calibration polarizer diattenuation per wavelength bin from a directory of 22 JSON 
    dictionaries. Optionally saves the DataFrame to a CSV file. CURRENT PARAMETERS:
    hwp_retardance, imr_retardance, calibration_polarizer_diattenuation.
    
    Parameters
    ----------
    json_dir : str or Path
        The directory containing the JSON system dictionaries for all 22 bins.
        Make sure the directory only contains these 22 JSON files. Component names
        are 'lp' for calibration polarizer, 'image_rotator' for image rotator,
        and 'hwp' for half-wave plate.

    csv_path : str or Path, optional
        If specified, saves the DataFrame to this path as a CSV file.
        
    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing all fitted retardances by wavelength and offset angles with errors.
    """
    # Check filepaths

    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a valid directory.")
    if csv_path is not None:
        csv_path = Path(csv_path)
    
    # Create dataframe

    df = pd.DataFrame(columns=['wavelength_bin', 'hwp_retardance',  'imr_retardance', 'calibration_polarizer_diattenuation',
                      'hwp_offset', 'hwp_offset_std','imr_offset','imr_offset_std','cal_offset','cal_offset_std'])

   # Load JSON files

    json_files = sorted(json_dir.glob("*.json"))

    # Check for correct file amount

    if len(json_files) != 22:
        raise ValueError(f"Expected 22 JSON files, found {len(json_files)}.")
    
    # Check for bins
 
    for f in json_files:
     try:
        match = re.search(r'bin(\d+)', f.name)
        if not match:
            raise ValueError(f"File {f.name} does not match expected naming convention.")
     except Exception as e:
        raise ValueError(f"Error processing file {f.name}: {e}")
     
     # Sort Jsons

    sorted_files = sorted(json_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))

    # Extract retardances and offsets

    hwp_retardances = []
    imr_retardances = []
    hwp_offsets = []
    imr_offsets = []
    lp_offsets = []
    lp_epsilons = []
    for f in sorted_files:
        with open(f, 'r') as file:
            data = json.load(file)
            if 'hwp' not in data or 'image_rotator' not in data or 'lp' not in data:
                raise ValueError(f"Required components not found in {f.name}.")
            
            # Extract retardances
            hwp_retardance = data['hwp']['phi']
            imr_retardance = data['image_rotator']['phi']
            # Extract lp diattenuation
            lp_epsilon = data['lp']['epsilon']
            # Extract offset angles 
            hwp_offset = data['hwp']['delta_theta']
            imr_offset = data['image_rotator']['delta_theta']
            lp_offset = data['lp_rot']['pa'] 
            hwp_retardances.append(hwp_retardance)
            imr_retardances.append(imr_retardance)
            hwp_offsets.append(hwp_offset)
            imr_offsets.append(imr_offset)
            lp_offsets.append(lp_offset)
            lp_epsilons.append(lp_epsilon)

    # Find offset averages/errors

    hwp_offset_error = np.std(hwp_offsets)
    imr_offset_error = np.std(imr_offsets)
    lp_offset_error = np.std(lp_offsets)
    hwp_offset = np.mean(hwp_offsets)
    imr_offset = np.mean(imr_offsets)
    lp_offset = np.mean(lp_offsets)

    # Replace offset angles with averages

    hwp_offsets = [hwp_offset] * len(hwp_offsets)
    imr_offsets = [imr_offset] * len(imr_offsets)
    lp_offsets = [lp_offset] * len(lp_offsets)

    # Make errors lists

    hwp_offset_errors = [hwp_offset_error] * len(hwp_offsets)
    imr_offset_errors = [imr_offset_error] * len(imr_offsets)
    lp_offset_errors = [lp_offset_error] * len(lp_offsets)

    # Fill DataFrame

    df['wavelength_bin'], df['hwp_retardance'], df['imr_retardance'], \
    df['calibration_polarizer_diattenuation'], df['hwp_offset'],df['hwp_offset_std'], \
    df['imr_offset'], df['imr_offset_std'], df['cal_offset'], df['cal_offset_std'] = \
        (wavelength_bins, hwp_retardances, imr_retardances, lp_epsilons, hwp_offsets , hwp_offset_errors, imr_offsets,
         imr_offset_errors, lp_offsets, lp_offset_errors)   
    
    # Save to CSV if specified

    if csv_path is not None:
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    
    return df
    
def plot_data_and_model_x_imr(interleaved_values, interleaved_stds, model, 
    configuration_list, hwp_theta_filter=None, wavelength=None, save_path = None,title=None):
    """
    Plots single differences vs imr angle for some amount of HWP angles. Similar to figure 6 in
    Joost t Hart 2021.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed 
        single differences and sums.

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.
        If charis use single differences and sums. 

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    hwp_theta_filter : float, optional
        If provided, only measurements with this hwp angle will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").
    title: str, optional
        Optional title

    Returns
    -------
    fig, ax, small_ax
    """
    # Calculate double differences and sums from interleaved single differences if in VAMPIRES mode 
    

    # Extract double differences and double sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]
    dd_model = model[::2]
    ds_model = model[1::2]

    # Group by hwp theta
    dd_by_theta = {}
    ds_by_theta = {}
   
    
    for i, config in enumerate(configuration_list):
        imr_theta = config["image_rotator"]["theta"]
        hwp_theta = config["hwp"]["theta"]

        if hwp_theta_filter is not None and not np.any(np.isclose(hwp_theta, hwp_theta_filter, atol=1e-2)):
         continue

        if hwp_theta not in dd_by_theta:
            dd_by_theta[hwp_theta] = {"imr_theta": [], "values": [], "stds": [], "model": []}
        dd_by_theta[hwp_theta]["imr_theta"].append(imr_theta)
        dd_by_theta[hwp_theta]["values"].append(dd_values[i])
        dd_by_theta[hwp_theta]["stds"].append(dd_stds[i])
        dd_by_theta[hwp_theta]["model"].append(dd_model[i])

        if hwp_theta not in ds_by_theta:
            ds_by_theta[hwp_theta] = {"imr_theta": [], "values": [], "stds": [], "model": []}
        ds_by_theta[hwp_theta]["imr_theta"].append(imr_theta)
        ds_by_theta[hwp_theta]["values"].append(ds_values[i])
        ds_by_theta[hwp_theta]["stds"].append(ds_stds[i])
        ds_by_theta[hwp_theta]["model"].append(ds_model[i])

   
    num_plots = 1
    sizex=10
    fig, axarr = plt.subplots(
    2, 1, 
    figsize=(sizex, 6), 
    gridspec_kw={"height_ratios": [3, 1]}, 
    sharex=True
    )

    ax = axarr[0]
    small_ax = axarr[1]

    # Double Difference plot
    
    for idx, (theta, d) in enumerate(dd_by_theta.items()):
        hart_cmap = ['cornflowerblue','paleturquoise','orange','red']
        color=hart_cmap[idx]
        err = ax.errorbar(d["imr_theta"], d["values"], color=color,yerr=d["stds"], fmt='o', label=f"{theta}°")
        #color = err[0].get_color
        ax.plot(d["imr_theta"], d["model"], '-', color=color)
        residuals =  (np.array(d["values"]) - np.array(d["model"]))*100
        small_ax.scatter(d['imr_theta'],residuals,color=color)
    small_ax.axhline(0, color='black', linewidth=1)
    small_ax.grid(which='major', axis='y', linestyle='-', linewidth=0.5, color='black')
    small_ax.grid(which='minor',axis='y', linestyle='-', linewidth=0.3, color='gray')
    small_ax.set_xlabel(r"IMR $\theta$ (deg)")
    #small_ax.set_xlim(0,180)
    #ax.invert_yaxis()
    small_ax.set_ylabel(r"Residual ($\%$)", fontsize = 15)
    #small_ax.yaxis.set_minor_locator(MultipleLocator(1))
    small_ax.xaxis.set_major_locator(MultipleLocator(10))
    small_ax.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='gray')
    small_ax.tick_params(axis='y', which='minor', labelleft=False)
    ax.set_ylabel("Single Difference")
    ax.legend(title=r"HWP $\theta$", fontsize=10, loc='upper right')
    ax.grid()

    # Set a suptitle if wavelength is provided
    if wavelength is not None and title is None:
        fig.suptitle(f"{wavelength}nm")
        ax.xaxis.set_major_locator(MultipleLocator(10))    
        ax.xaxis.set_minor_locator(MultipleLocator(1)) 
    plt.tight_layout(rect=[0, 0, 1, 0.95])  #

    if title:
        fig.suptitle(title)

    if save_path != None:
        plt.savefig(save_path,dpi=600, bbox_inches='tight')


    return fig, ax,small_ax

    

    

    
    


          
                

            
                    



        