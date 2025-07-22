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
from CHARIS.physical_models import HWP_retardance, IMR_retardance
import json

###############################################################
###### Functions related to reading/writing in .csv values ####
###############################################################

# define CHARIS wavelength bins
wavelength_bins = np.array([1159.5614, 1199.6971, 1241.2219, 1284.184 , 1328.6331, 1374.6208,
1422.2002, 1471.4264, 1522.3565, 1575.0495, 1629.5663, 1685.9701,
1744.3261, 1804.7021, 1867.1678, 1931.7956, 1998.6603, 2067.8395,
2139.4131, 2213.4641, 2290.0781, 2369.3441])
def single_sum_and_diff(fits_cube_path, wavelength_bin):
    """Calculate normalized single difference and sum between left and right beam 
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
            [1] norm_single_diff : float
                Normalized single difference of left and right beam apertures:
                (R - L) / (R + L)
            [2] left_counts : float
                Left beam aperture counts.
            [3] right_counts : float
                Right beam aperture counts.
            [4] sum_std : float
                Standard deviation of the single sum.
            [5] diff_std : float
                Standard deviation of the normalized single difference.
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
    # note- these values are based on rough analysis and may need adjustment for high precision

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
    norm_single_diff = (phot_rbeam['aperture_sum'][0] - phot_lbeam['aperture_sum'][0]) / single_sum

    # get left and right counts

    left_counts = phot_lbeam['aperture_sum'][0]
    right_counts = phot_rbeam['aperture_sum'][0]

    # calculate standard deviations of single sum and normalized single difference

    sum_std = np.sqrt(single_sum) # Assuming Poisson noise for counts
    diff_std = np.sqrt((4*(left_counts**2)*right_counts + 4*(right_counts**2)*left_counts) / (single_sum**4)) # error propagation for normalized difference
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
    # check if csv_file_path is a valid file path

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


def write_fits_info_to_csv(cube_directory_path, raw_cube_path, output_csv_path, wavelength_bin):
    """Write filepath, D_IMRANG (derotator angle), RET-ANG1 (HWP angle), 
    single sum, single difference, LCOUNTS, RCOUNTS, difference std,
    sum std, and wavelength values for a wavelength bin from each fits cube in the directory.

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

    Returns:
    --------
    None
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
        f.write("filepath,D_IMRANG,RET-ANG1,single_sum,norm_single_diff,LCOUNTS,RCOUNTS,sum_std,diff_std,wavelength_bin\n")

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

                single_sum, norm_single_diff, LCOUNTS, RCOUNTS, sum_std, diff_std = single_sum_and_diff(fits_file, wavelength_bin)

                # wavelength bins for lowres mode

                bins = wavelength_bins
                
                # write to csv file

                f.write(f"{fits_file}, {d_imrang}, {ret_ang1}, {single_sum}, {norm_single_diff}, {LCOUNTS}, {RCOUNTS}, {sum_std}, {diff_std}, {bins[wavelength_bin]}\n")
            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
    print(f"CSV file written to {output_csv_path}")


def read_csv(file_path):
    """Takes a CSV file path containing "D_IMRANG", 
    "RET-ANG1", "single_sum", "norm_single_diff", "diff_std", and "sum_std",
    for one wavelength bin and returns interleaved values, standard deviations, 
    and configuration list.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV.

    Returns:
    -----------
    interleaved_values : np.ndarray
        Interleaved values from "norm_single_diff" and "single_sum".
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
    interleaved_values = np.ravel(np.column_stack((df["norm_single_diff"].values, df["single_sum"].values)))

    # Interleave values from "diff_std" and "sum_std"
    interleaved_stds = np.ravel(np.column_stack((df["diff_std"].values, df["sum_std"].values)))

    # Convert each row's values into a list of two-element lists
    configuration_list = []
    for index, row in df.iterrows():
        # Extracting values from relevant columns
        hwp_theta = row["RET-ANG1"]
        imr_theta = row["D_IMRANG"]

        # Building dictionary
        row_data = {
            "hwp": {"theta": hwp_theta},
            "image_rotator": {"theta": imr_theta}
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

# Main MCMC function
def run_mcmc(
    p0_dict, system_mm, dataset, errors, configuration_list,
    priors, bounds, logl_function, output_h5_file,
    nwalkers=64, nsteps=10000, pool_processes=None, 
    s_in=np.array([1, 0, 0, 0]), process_dataset=None, 
    process_errors=None, process_model=None, resume=True
):
    """
    Run MCMC using emcee with support for dictionary-based parameter inputs.
    """

    p0_values, p_keys = parse_configuration(p0_dict)
    ndim = len(p0_values)

    log_prior = mcmc.log_prior

    resume = os.path.exists(output_h5_file)
    backend = emcee.backends.HDFBackend(output_h5_file)

    if not resume or backend.iteration == 0:
        backend.reset(nwalkers, ndim)

    pos = p0_values + 1e-3 * np.random.randn(nwalkers, ndim)

    args = (
        system_mm, dataset, errors, configuration_list, p_keys, s_in,
        process_model, process_dataset, process_errors,
        priors, bounds, logl_function
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
    bounds = None, mode = 'VAMPIRES'):
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
    if mode == 'VAMPIRES':
        result = minimize(logl, p0_values, 
            args=(p0_keywords, system_mm, dataset, errors, configuration_list, 
                s_in, logl_function, process_dataset, process_errors, process_model), 
                method='Nelder-Mead', bounds = bounds)
    elif mode == 'CHARIS':
        result = minimize(logl_CHARIS, p0_values, 
            args=(p0_keywords, system_mm, dataset, errors, configuration_list, 
                s_in), method='Powell', bounds = bounds)
    
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
    """Returns simulated L/R intensities for a given set of parameters based on
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
        Converts output intensities to double differences. CURRENTLY NOT
        COMPATIBLE WITH SINGLE DIFFERENCES. 

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


def generate_CHARIS_mueller_matrix(wavelength_bin, hwp_angle, imr_angle, beam, dict=False):
    """ 
    Generate a pyMuellerMuellerMat matrix object for CHARIS based on the given
    wavelength bin, HWP angle, and derotator angle. Currently only works for lowres mode. 
    Based on Joost 't Hart 2021

    Parameters:
    -----------
    wavelength_bin : int
        The index of the wavelength bin, zero based. (0 to 21 for 22 bins)
    hwp_angle : float
        The rotation angle of the half-wave plate in degrees.
    imr_angle : float
        The angle of the image rotator in degrees.
    beam : str
        The beam type, either 'o' for ordinary or 'e' for extraordinary.
    dict : bool, optional
        If True, returns a system dictionary instead of a Mueller matrix object. Default is False.
    Returns:
    --------
    sys_mm : pyMuellerMat.MuellerMatrix
        A Mueller matrix object representing the CHARIS system.
    """
    # check that it is in lowres mode

    if wavelength_bin < 0 or wavelength_bin > 21:
        raise ValueError("Wavelength bin must be between 0 and 21 for lowres mode.")
    
    # constants based on Joost 't Hart 2021

    offset_imr = -0.0118 # derotator offset
    offset_hwp = -0.002 # HWP offset
    offset_cal = -0.035 # calibration polarizer offset
    wavelength_bins = np.array([1159.5614, 1199.6971, 1241.2219, 1284.184 , 1328.6331, 1374.6208,
                1422.2002, 1471.4264, 1522.3565, 1575.0495, 1629.5663, 1685.9701,
                1744.3261, 1804.7021, 1867.1678, 1931.7956, 1998.6603, 2067.8395,
                2139.4131, 2213.4641, 2290.0781, 2369.3441])
    theta_imr = imr_angle
    theta_hwp = hwp_angle
    theta_cal = 0  # calibration polarizer angle
    hwp_retardances = HWP_retardance(wavelength_bins) # based on physical model
    imr_retardances = IMR_retardance(wavelength_bins) # based on physical model

    # create the system dictionary

    sys_dict = {
        "components" : {
            "wollaston" : {
            "type" : "wollaston_prism_function",
            "properties" : {"beam": beam},
            "tag": "internal",
            },
            "image_rotator" : {
                "type" : "general_retarder_function",
                "properties" : {"phi": imr_retardances[wavelength_bin], "theta": theta_imr + offset_imr},
                "tag": "internal",
            },
            "hwp" : {
                "type" : "general_retarder_function",
                "properties" : {"phi": hwp_retardances[wavelength_bin], "theta": theta_hwp + offset_hwp},
                "tag": "internal",
            },
            "lp" : {
                "type": "general_linear_polarizer_function_with_theta",
                "properties": {"theta": offset_cal + theta_cal},
                "tag": "internal",
            }}
    }

    # generate Mueller matrix object

    system_mm = generate_system_mueller_matrix(sys_dict)

    if dict:
        return sys_dict
    else:
        return system_mm


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
    elif process_dataset is None:
        processed_dataset = copy.deepcopy(dataset)
    # print("Post process_dataset dataset shape: ", np.shape(processed_dataset))

    # Optionally parse the dataset and output intensities (e.g., normalized difference)
    # print("Pre process_errors errors shape: ", np.shape(dataset))
    if process_errors is not None:
        processed_errors = process_errors(copy.deepcopy(errors), 
            copy.deepcopy(dataset))
    elif process_errors is None:
        processed_errors = copy.deepcopy(errors)
    # print("Post process_errors errors shape: ", np.shape(processed_errors))

    dataset = copy.deepcopy(processed_dataset)
    errors = copy.deepcopy(processed_errors)
    errors = np.maximum(errors, 1e-3)
    # Calculate log likelihood
    if logl_function is not None:
     return logl_function(p, output_intensities, dataset, errors)
    else:
     return 0.5 * np.sum((output_intensities - dataset) ** 2 / errors ** 2)

def logl_CHARIS(p, system_parameters, system_mm, dataset, errors, configuration_list, 
         s_in=None):
    """
    Compute the log-likelihood of a model given a dataset and system configuration.

    This function evaluates how well a set of system Mueller matrix parameters
    (given by `p`) reproduce the observed dataset, using a chi-squared-based 
    likelihood metric.
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

    Returns
    -------
    float
        The computed log-likelihood value (higher is better).
    """

    # print("Entered logl")

    # Generating a list of model predicted values for each configuration - already parsed
    output_intensities = model(p, system_parameters, system_mm, configuration_list, 
        s_in=s_in)
    # Convert to single differences
    diffssums = process_model(output_intensities, mode='CHARIS')
    # Convert lists to numpy arrays, only differences used
    dataset = np.array(dataset)[::2]
    errors = np.array(errors)[::2]
    diffssums= diffssums[::2]

    processed_dataset = copy.deepcopy(dataset)

    processed_errors = copy.deepcopy(errors)

    dataset = copy.deepcopy(processed_dataset)
    errors = copy.deepcopy(processed_errors)
    residuals = diffssums - dataset
    chi_squared = np.sum((residuals / errors) ** 2)
    return 0.5*chi_squared
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

def process_model(model_intensities, mode= 'VAMPIRES'):
    """
    Processes the model intensities to compute differences and sums,
    and formats them into a single interleaved array. 
    
    Parameters
    ----------
    model_intensities : list or np.ndarray
        List or array of model intensities, expected to be in pairs.
    mode : str, optional
        Mode of processing, either 'VAMPIRES' or 'CHARIS'. 
        Default is 'VAMPIRES'. VAMPIRES returns double differences
        and CHARIS returns single differences."""
    
    # Making sure the mode exists
    if mode not in ['VAMPIRES', 'CHARIS']:
        raise ValueError("Mode must be either 'VAMPIRES' or 'CHARIS'.")
    
    # Making sure that model_intensities is a numpy array

    model_intensities = np.array(model_intensities)

    # print("Entered process_model")

    

    if mode == 'VAMPIRES':
        differences, sums = build_differences_and_sums(model_intensities)
        double_differences, double_sums = build_double_differences_and_sums(differences, sums)

    # print("Differences shape: ", np.shape(differences))
    # print("Sums shape: ", np.shape(sums))
    # print("Double Differences shape: ", np.shape(double_differences))
    # print("Double Sums shape: ", np.shape(double_sums))

    #Format this into one array. 
    if mode == 'VAMPIRES':
        # Interleave the double differences and double sums
        interleaved_values = np.ravel(np.column_stack((double_differences, double_sums)))
         
        # NOTE: Subtracting same FLC state orders (A - B) as Miles
    
    
    if mode == 'CHARIS':
        # Interleave the norm single differences and single sums
        differences, sums = build_differences_and_sums(model_intensities, normalized=True)
        interleaved_values = np.ravel(np.column_stack((differences, sums)))

    # Take the negative of this as was done before
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
    # print("Sums shape: ", np.shape(sums))e

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

    # print("Pre-processing Errors shape: ", np.shape(input_errors))
    # print("Pre-processing Dataset shape: ", np.shape(input_dataset))

    # Compute errors for differences and sums
    differences_errors = np.sqrt(input_errors[::2]**2 + input_errors[1::2]**2)
    sums_errors = np.sqrt(input_errors[::2]**2 + input_errors[1::2]**2)

    # print("Differences Errors shape: ", np.shape(differences_errors))
    # print("Sums Errors shape: ", np.shape(sums_errors))

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

    # print("Double Differences Errors shape: ", np.shape(double_differences_errors))
    # print("Double Sums Errors shape: ", np.shape(double_sums_errors))

    # Interleave errors to maintain order
    interleaved_errors = np.ravel(np.column_stack((double_differences_errors, double_sums_errors)))

    # print("Final interleaved Errors shape: ", np.shape(interleaved_errors))

    return interleaved_errors


# streamline process for all wavelength bins

def fit_CHARIS_Mueller_matrix_by_bin(csv_path, wavelength_bin, new_system_dict_path,plot_path=None):
    """
    Fits a Mueller matrix for one wavelength bin from internal calibration data and saves
    the updated system dictionary to a JSON file. Creates a plot
    of each updated model vs the data. Initial guesses for all fits are from Joost t Hart 2021.
    Note that following the most recent model update these guesses should be updated.
    The csv containing the calibration data and relevant headers can be obtained by 
    the write_fits_info_to_csv function in instruments.py. This code currently fits for
    derotator retardance/offset, HWP retardance/offset, and calibration polarizer offset.
    It can be modified relatively easily to fit for other parameters as well. 

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing the calibration data. Must contain the columns "D_IMRANG", 
    "RET-ANG1", "single_sum", "norm_single_diff", "diff_std", and "sum_std".

    wavelength_bin : int
        The index of the wavelength bin to fit (0-21 for CHARIS).

    new_system_dict_path : str or Path
        Path to save the new system dictionary as a JSON file.

    plot_path : str or Path, optional
        Path to save the plot of the observed vs modeled data. If not provided, no plot will be saved.
        Must have a .png extension.
    
    Returns
    -------
    None
        The function saves the updated system dictionary to a JSON file and optionally saves a plot of the observed vs modeled data.
    """
    # Check file paths
    filepath = Path(csv_path)
    if not filepath.exists() or filepath.suffix != ".csv":
        raise ValueError("Please provide a valid .csv file.")
    plot_path = Path(plot_path)
    if plot_path.suffix != ".png":
        raise ValueError("Please provide a valid .png file for plotting.")
    if new_system_dict_path.suffix != ".json":
        raise ValueError("Please provide a valid .json file for saving the new system dictionary.")
    new_system_dict_path = Path(new_system_dict_path)
    # Read in data

    interleaved_values, interleaved_stds, configuration_list = read_csv(filepath)

    # Loading in past fits from Joost t Hart 2021

    offset_imr = -0.0118 # derotator offset
    offset_hwp = -0.002 # HWP offset
    offset_cal = -0.035 # calibration polarizer offset
    imr_theta = 0 # placeholder 
    hwp_theta = 0 # placeholder
    imr_phi = IMR_retardance(wavelength_bins)[wavelength_bin]
    hwp_phi = HWP_retardance(wavelength_bins)[wavelength_bin]

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
                "lp" : {  # calibration polarizer for internal calibration source
                    "type": "general_linear_polarizer_function_with_theta",
                    "properties": {"delta_theta": offset_cal },
                    "tag": "internal",
                }}
        }

    # Converting system dictionary into system Mueller Matrix object

    system_mm = generate_system_mueller_matrix(system_dict)

    # Define initial guesses for our parameters 
    # Modify this if you want to change the parameters

    p0 = {
        "image_rotator" : 
            {"phi": IMR_retardance(wavelength_bins)[wavelength_bin], "delta_theta": offset_imr},
        "hwp" :  
            {"phi": HWP_retardance(wavelength_bins)[wavelength_bin], "delta_theta": offset_hwp},
        "lp" : 
            {"delta_theta": offset_cal}
    }

    # Define some bounds
    # Modify this if you want to change the parameters or minimization bounds

    hwp_phi_bounds = (0.9*hwp_phi, 1.1*hwp_phi)
    imr_phi_bounds = (0.9*imr_phi, 1.1*imr_phi)
    offset_imr_bounds = (1.1*offset_imr, 0.9*offset_imr)
    offset_hwp_bounds = (1.1*offset_hwp, 0.9*offset_hwp)
    offset_cal_bounds = (1.1*offset_cal, 0.9*offset_cal)

    # Minimize the system Mueller matrix using the interleaved values and standard deviations
    # Modify this if you want to change the parameters

    result, logl_result = minimize_system_mueller_matrix(p0, system_mm, interleaved_values, 
        interleaved_stds, configuration_list, bounds = [imr_phi_bounds, offset_imr_bounds,hwp_phi_bounds, offset_hwp_bounds, offset_cal_bounds],mode='CHARIS')
    print(result)

    # Update p dictionary with the fitted values

    update_p0(p0, result.x)

    # Process model

    p0_values, p0_keywords = parse_configuration(p0)

    # Generate modeled left and right beam intensities

    updated_system_mm = update_system_mm(result.x, p0_keywords, system_mm)

    # Generate modeled left and right beam intensities

    LR_intensities2 = model(p0_values, p0_keywords, updated_system_mm, configuration_list)

    # Process these into interleaved single normalized differences and sums

    diffs_sums2 = process_model(LR_intensities2, 'CHARIS')

    # Plot the modeled and observed values

    plot_data_and_model(interleaved_values, interleaved_stds, diffs_sums2,configuration_list, wavelength= wavelength_bins[wavelength_bin], mode='CHARIS',save_path=plot_path)

    # Print the Mueller matrix

    print("Updated Mueller Matrix:")
    print(updated_system_mm.evaluate())

    # Print residuals

    residuals = interleaved_values[::2] - diffs_sums2[::2]
    print("Residuals range:", residuals.min(), residuals.max())

    # Save system dictionary to a json file

    with open (new_system_dict_path, 'w') as f:
        json.dump(p0, f, indent=4)





#######################################################
###### Functions related to plotting ##################
#######################################################

# function for calculating single sum and difference

    
def plot_single_differences(csv_file_path, plot_save_path=None):
    """Plot norm single differences as a function of the HWP angle for one 
    wavelength bin from a CSV containing headers "D_IMRANG" , "RET-ANG1" , 
    "norm_single_diff", and "wavelength_bin".
    This can be obtained from the write_fits_info_to_csv function.

    Parameters:
    -----------
    csv_file_path : str or Path
        Path to the specified CSV file.

    plot_save_path : str or Path, optional
        If provided, the plot will be saved to this path. Must end with '.png'.

    Returns: 
    --------
    None
    """
    # check if csv_file_path is a valid file path

    csv_file_path = Path(csv_file_path)
    if not csv_file_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    
    # read csv file into pandas dataframe

    df = pd.read_csv(csv_file_path)

    # check if necessary columns are present

    required_columns = ['D_IMRANG', 'RET-ANG1', 'norm_single_diff', 'wavelength_bin']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the CSV file.")
        
    # check for multiple wavelength bins
    unique_bins = df['wavelength_bin'].unique()
    if len(unique_bins) > 1:
        raise ValueError("CSV file contains multiple wavelength bins. Please filter to a single bin before plotting.")
    
    # extract data for plotting

    hwp_angles = df['RET-ANG1'].values
    single_diffs = df['norm_single_diff'].values
    wavelength_bin = df['wavelength_bin'].values[0]
    derotator_angles = df['D_IMRANG'].values

    # plot single differences as a function of HWP angle for each derotator angle

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for derotator_angle in np.unique(derotator_angles):
        mask = derotator_angles == derotator_angle
        # get the data for this derotator angle
        hwp_subset = hwp_angles[mask]
        diffs_subset = single_diffs[mask]
        # sort by HWP angle
        sort_order = np.argsort(hwp_subset)
        ax.plot(hwp_subset[sort_order], diffs_subset[sort_order], 
            marker='o', label=f'{derotator_angle}°')
    ax.set_xlabel('HWP Angle (degrees)')
    ax.set_ylabel('Normalized Single Difference')
    ax.set_title(f'Single Differences vs HWP Angle at {wavelength_bin} nm')
    ax.legend(loc='lower right', fontsize='small', title= r'IMR $\theta$')
    ax.grid()
    plt.show()

    # save plot if a path is provided

    if plot_save_path:
        plot_save_path = Path(plot_save_path)
        if not plot_save_path.parent.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {plot_save_path.parent}")
        if not plot_save_path.suffix == '.png':
            raise ValueError("Plot save path must end with '.png'.")
        fig.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")

# function for quick csv and plotting
def quick_data_all_bins(cube_directory_path, raw_directory_path, csv_directory, plot_directory):
    """Plot norm single differences as a function of the HWP angle 
    for each derotator angle and write a CSV containing headers "D_IMRANG" , "RET-ANG1" , 
    "norm_single_diff", and "wavelength_bin". Perform this for all 21 wavelength bins.
    
    Parameters:
    -----------
    cube_directory_path : str or Path
        Path to the directory containing CHARIS fits cubes.
        
    raw_directory_path : str or Path
        Path to the directory containing the matching raw CHARIS FITS files.

    csv_directory : str or Path
        Directory where the output csv files will be created.

    plot_directory : str or Path
        Directory where the output plots will be created.
        
    Returns:
    --------
    None
    """
    # check if cube_directory_path and raw_directory_path are valid paths
    csv_directory = Path(csv_directory)
    plot_directory = Path(plot_directory)
    cube_directory_path = Path(cube_directory_path)
    raw_directory_path = Path(raw_directory_path)
    
    if not cube_directory_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {cube_directory_path}")
    if not raw_directory_path.is_dir():
        raise NotADirectoryError(f"Raw directory does not exist: {raw_directory_path}")
    if not csv_directory.is_dir():
        raise NotADirectoryError(f"CSV directory does not exist: {csv_directory}")
    if not plot_directory.is_dir():
        raise NotADirectoryError(f"Plot directory does not exist: {plot_directory}")
    # iterate over all wavelength bins

    for bin in range(0, 22):

        # write csv file for each bin

        csv_file_path = Path(csv_directory) / f'charis_cube_info_bin{bin}.csv'
        write_fits_info_to_csv(
            cube_directory_path=cube_directory_path,
            raw_cube_path=raw_directory_path,
            output_csv_path=csv_file_path,
            wavelength_bin=bin
        )

        # plot single differences for each bin

        plot_save_path = Path(plot_directory) / f'diffvshwp_bin{bin}.png'
        plot_single_differences(csv_file_path, plot_save_path)

def plot_data_and_model(interleaved_values, interleaved_stds, model, 
    configuration_list, imr_theta_filter=None, wavelength=None, save_path = None, mode = 'VAMPIRES'):
    """
    Plots double difference and double sum measurements alongside model predictions,
    grouped by image rotator angle (D_IMRANG). Optionally filters by a specific 
    image rotator angle and displays a wavelength in the plot title.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed double difference and double sum values.
        Expected format: [dd1, ds1, dd2, ds2, ...].

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    imr_theta_filter : float, optional
        If provided, only measurements with this image rotator angle (rounded to 0.1°) 
        will be plotted.

    wavelength : str or int, optional
        Wavelength (e.g., 670 or "670") to display as a centered title with "nm" units 
        (e.g., "670nm").

    Returns
    -------
    None
        Displays two subplots: one for double differences and one for double sums,
        including error bars and model curves.
    """
    # Calculate double differences and sums from interleaved single differences if in VAMPIRES mode 
    if mode =='VAMPIRES':
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
    if mode ==  'VAMPIRES':
        index = 2
    if mode == 'CHARIS':
        index = None
    for i, config in enumerate(configuration_list[::index]):
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
    if mode == 'VAMPIRES':
        num_plots = 2
        sizex= 14
    elif mode == 'CHARIS':
        num_plots = 1
        sizex=10
    fig, axes = plt.subplots(1, num_plots, figsize=(sizex, 6), sharex=True)

    # Double Difference plot
    if mode == 'VAMPIRES':
        ax = axes[0]
        for theta, d in dd_by_theta.items():
           err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
           color = err[0].get_color()
           ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel("HWP θ (deg)")
        ax.set_ylabel("Double Difference")
        ax.legend(title="IMR θ")
    # Double Sum plot
        ax = axes[1]
        for theta, d in ds_by_theta.items():
            err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
            color = err[0].get_color()
            ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel("HWP θ (deg)")
        ax.set_ylabel("Double Sum")
        ax.legend(title="IMR θ")
    elif mode == 'CHARIS':
        ax = axes
        for theta, d in dd_by_theta.items():
           err = ax.errorbar(d["hwp_theta"], d["values"], yerr=d["stds"], fmt='o', label=f"{theta}°")
           color = err[0].get_color()
           ax.plot(d["hwp_theta"], d["model"], '-', color=color)
        ax.set_xlabel("HWP θ (deg)")
        ax.set_ylabel("Single Difference")
        ax.legend(title="IMR θ")
        ax.grid()

    # Set a suptitle if wavelength is provided
    if wavelength is not None:
        fig.suptitle(f"{wavelength}nm", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    if save_path != None:
        plt.savefig(save_path)

    plt.show()

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
    None
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
    plt.show()

    # save plot if a path is provided

    if plot_save_path:
        plot_save_path = Path(plot_save_path)
        if not plot_save_path.parent.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {plot_save_path.parent}")
        if not plot_save_path.suffix == '.png':
            raise ValueError("Plot save path must end with '.png'.")
        fig.savefig(plot_save_path)
        print(f"Plot saved to {plot_save_path}")

    