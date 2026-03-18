"""
csv_tools.py

Contains functions for reading and writing CSV files that
contain aperture photometry data and relevant FITS headers.

"""




from pathlib import Path
import numpy as np
from astropy.io import fits
from photutils.aperture import RectangularAperture
from photutils.aperture import aperture_photometry
import pandas as pd
import re
from pyPolCal.constants import wavelength_bins, charis_aperture_l, charis_aperture_r
import json
import copy

###############################################################
###### Functions related to reading/writing in .csv values ####
###############################################################


def single_sum_and_diff(fits_cube_path, wavelength_bin, aperture_l=charis_aperture_l, aperture_r=charis_aperture_r):
    """Calculate single difference and sum between left and right beam 
    rectangular aperture photometry from CHARIS internal calibration
    fits cubes. Add L/R counts and stds to array. As of 3/17/2026, we have switched to using 
    R-L for double differences, since we (Miles) believe this is the correct convention.
    
    Parameters
    -----------
    fits_cube_path : str or Path
        Path to the CHARIS fits cube file.
        
    wavelength_bin : int
        Index of the wavelength bin to analyze (0-based).

    aperture_l: photutils.aperture.Aperture
        Photutils aperture object for the left Wollaston beam. Default is hardcoded CHARIS aperture.

    aperture_r: photutils.aperture.Aperture
        Photutils aperture object for the right Wollaston beam. Default is hardcoded CHARIS aperture.

    Returns
    --------
    np.ndarray
        Array with six elements:
            [0] single_sum : float
                Single sum of left and right beam apertures:
                (R + L)
            [1] single_diff : float
                Single difference of left and right beam apertures:
                (R - L) 
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
    with fits.open(fits_cube_path) as hdul:
        cube_data = hdul[1].data

    # check if data is a 3d cube (wavelength, y, x)

    if cube_data.ndim != 3:
        raise ValueError("Input data must be a 3D cube (wavelength, y, x).")
        
    # check if wavelength_bin is within bounds
    if not (0 <= wavelength_bin < cube_data.shape[0]):
        raise ValueError(f"wavelength_bin must be between 0 and {cube_data.shape[0] - 1}.")
    
    image_data = cube_data[wavelength_bin]

    # perform aperture photometry 
    aperture_lbeam = aperture_l
    aperture_rbeam = aperture_r
    phot_lbeam = aperture_photometry(image_data, aperture_lbeam)
    phot_rbeam = aperture_photometry(image_data, aperture_rbeam)

    # calculate single difference and sum
    single_sum = phot_rbeam['aperture_sum'][0] + phot_lbeam['aperture_sum'][0]
    single_diff =  (phot_rbeam['aperture_sum'][0] -phot_lbeam['aperture_sum'][0]) 

    # get left and right counts
    left_counts = phot_lbeam['aperture_sum'][0]
    right_counts = phot_rbeam['aperture_sum'][0]

    # Assume Poissanian noise and propagate error
    sum_std = diff_std = np.sqrt(left_counts+right_counts)
    return (single_sum, single_diff, left_counts, right_counts, sum_std, diff_std)

# function to fix corrupted hwp data
def fix_hwp_angles(csv_file_path, nderotator=8):
    '''Take corrupted HWP angles and replace them with assumed values
    in a new csv titled {old_title}_fixed.

    Parameters
    -----------
    csv_file_path : str or Path
        Path to the specified CSV file containing the corrupted HWP angles.
    
    nderotator : int
        Number of derotator angles (assumed to be 8).
    Returns
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

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def arr_csv_HWP(csv_path, hwp_order, todelete=None, new_csv_path=None):
    """Arranges CSVs by a custom HWP order, removes blank angles, and preserves cycles."""
    
    # Load to a DF
    df = pd.read_csv(csv_path)

    if 'D_IMRANG' in df.columns:
        df['D_IMRANG'] = pd.to_numeric(df['D_IMRANG'], errors='coerce')
    df['RET-ANG1'] = pd.to_numeric(df['RET-ANG1'], errors='coerce')

    # Delete blank/NaN RET-ANG1 values
    missing_angles = df[df['RET-ANG1'].isna()]
    if not missing_angles.empty:
        print(f"WARNING: Found {len(missing_angles)} row(s) with a blank or invalid RET-ANG1. Deleting them!")
        
        # Print exactly which files are getting nuked
        if 'filepath' in df.columns:
            for bad_file in df.loc[df['RET-ANG1'].isna(), 'filepath']:
                print(f"   -> Dropped: {bad_file}")
        else:
            print(f"   -> Dropped row indices: {missing_angles.index.tolist()}")
            
        # Drop the offending rows and reset index
        df = df.dropna(subset=['RET-ANG1'])
        df = df.reset_index(drop=True)

    # Drop unwanted angles 
    if todelete is not None:
        df = df[~df['RET-ANG1'].isin(todelete)]
        df = df.reset_index(drop=True)

    # Assign "cycle blocks"
    if 'D_IMRANG' in df.columns and df['D_IMRANG'].notna().any():
        # Safely round to nearest 0.5 
        df['imr_round'] = (df['D_IMRANG'] * 2).round() / 2

        # Apply the custom Double Difference categorical order on a helper column
        df['RET_ANG_cat'] = pd.Categorical(df['RET-ANG1'], categories=hwp_order, ordered=True)

        # Sort by IMR group then by categorical HWP angle
        df = df.sort_values(by=['imr_round', 'RET_ANG_cat'])

        # Drop helper columns
        df = df.drop(columns=['imr_round', 'RET_ANG_cat'])
    else:
        # Fallback to original index-chunking behavior when no IMR information is present
        cycle_length = len(hwp_order)
        df['cycle_id'] = np.arange(len(df)) // cycle_length
        df['RET-ANG1'] = pd.Categorical(df['RET-ANG1'], categories=hwp_order, ordered=True)
        df = df.sort_values(by=['cycle_id', 'RET-ANG1'])
        df = df.drop(columns=['cycle_id'])

    # Save
    save_path = new_csv_path if new_csv_path else csv_path
    df.to_csv(save_path, index=False)
    
    return df
    




def write_fits_info_to_csv(cube_directory_path, output_csv_path, wavelength_bin, raw_cube_path=None, hwp_order=[0,45,11.25,56.25,22.5,67.5,33.75,78.75],hwp_angles_to_delete=[90],aperture_l=charis_aperture_l, aperture_r=charis_aperture_r):
    """Write filepath, D_IMRANG (derotator angle), RET-ANG1 (HWP angle without correcting for synchro ADI), 
    RET-POS1 (actual HWP angle), single sum, single difference, LCOUNTS, RCOUNTS, difference std,
    sum std, and wavelength values for a wavelength bin from each fits cube in the directory.
    Default HWP order and deletion works for future double difference calculation. 

    Single sum and difference are calculated using photometry from defined rectangular apertures.
    If the necessary header keywords are not present, the values will be set to NaN.

    For the raw files- this function assumes that the raw files have the same 8-digit ID as either the processed cubes
    filename or 'ORIGNAME' keyword, so make sure you didn't rename your raw files.

    Parameters
    -----------
    cube_directory_path : str or Path
        Path to the directory containing CHARIS fits cubes.

    output_csv_path : str or Path
        Path where the output csv will be created.

    wavelength_bin : int
        Index of the wavelength bin to analyze (0-based).
    
    raw_cube_path : str or Path
        Path to the directory containing the matching raw CHARIS FITS files. You'll need this
        if you processed your cubes in the CHARIS DPP.

    hwp_order: list or np.ndarray
        List of desired HWP order. Default works for double difference calculations.

    todelete: list or np.ndarray
        List of HWP angles to delete. Default works
        for double difference calculations. Set to None if you want to keep them all. 

    Returns
    --------
    None
        Write all info to a csv with these columns: "filepath", "D-IMRANG", "RET-ANG1", "single_sum", "single_diff",
        "LCOUNTS","RCOUNTS", "sum_std", "diff_std", "wavelength_bin"
    """
    # check for valid file paths
    cube_directory_path = Path(cube_directory_path)
    output_csv_path = Path(output_csv_path)

    if not cube_directory_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {cube_directory_path}")
    if output_csv_path.suffix != '.csv':
        raise ValueError(f"Output path must be a CSV file, got {output_csv_path}")
    if wavelength_bin > 21:
        raise ValueError(f"This function is currently only compatible with lowres mode, with 22 wavelength bins.")
    
    # prepare output csv file
    output_csv_path = Path(output_csv_path)
    with open(output_csv_path, 'w') as f:
        f.write("filepath,D_IMRANG,RET-ANG1,RET-POS1,single_sum,single_diff,LCOUNTS,RCOUNTS,sum_std,diff_std,wavelength_bin\n")

        # iterate over all fits files in the directory
        for fits_file in sorted(cube_directory_path.glob('*.fits')):
            try:
                if raw_cube_path:
                    
                    raw_cube_path = Path(raw_cube_path)
                    if not raw_cube_path.is_dir():
                        raise NotADirectoryError(f"Raw cube directory is not a directory: {raw_cube_path}")
                    # check if corresponding raw fits file exists
                    match = re.search(r"(\d{8})", fits_file.name)
                    if not match:
                        # first try to grab from origname keyword
                        origname = fits.getheader(fits_file, 0).get('ORIGNAME', None)
                        if origname:
                            match = re.search(r"(\d{8})", origname)
                        if not match:
                            raise ValueError(f"Could not extract 8-digit ID from filename {fits_file.name} to match raw files. Maybe you renamed your raw files?")
                    fits_id = match.group(1)
                    raw_candidates = list(raw_cube_path.glob(f"*{fits_id}*.fits"))
                    if not raw_candidates:
                        raise FileNotFoundError(f"No raw FITS file found for ID {fits_id}")
                    raw_fits = raw_candidates[0]
                    
                    with fits.open(raw_fits) as hdul_raw:
                        raw_header = hdul_raw[0].header
                        d_imrang = raw_header.get("D_IMRANG", np.nan)
                        ret_ang1 = raw_header.get("RET-ANG1", np.nan)
                        ret_pos1 = raw_header.get("RET-POS1", np.nan)
                else: # if no specified raw files, grab from extension 3
                    with fits.open(fits_file) as hdul:
                        extension_3 = hdul[3].header
                        if not extension_3:
                            raise ValueError(f"Could not find extension 3 in {fits_file.name}. You may be using frames processed in the DPP, which requires you to provide a raw directory.")
                        # use .get to avoid KeyError if header keywords are missing
                        ret_ang1 = extension_3.get('RET-ANG1', np.nan)
                        ret_pos1 = extension_3.get('RET-POS1', np.nan)
                        d_imrang = extension_3.get('D_IMRANG', np.nan)

                # round d_imrang to nearest 0.5 --im gonna not do this as of 2/19/2025
                #d_imrang = (np.round(d_imrang * 2) / 2)

                # calculate single sum and single difference
                single_sum, single_diff, LCOUNTS, RCOUNTS, sum_std, diff_std = single_sum_and_diff(fits_file, wavelength_bin, aperture_l=aperture_l, aperture_r=aperture_r)

                # wavelength bins for lowres mode
                bins = wavelength_bins
                
                # write to csv file
                f.write(f"{fits_file}, {d_imrang}, {ret_ang1}, {ret_pos1}, {single_sum}, {single_diff}, {LCOUNTS}, {RCOUNTS}, {sum_std}, {diff_std}, {bins[wavelength_bin]}\n")

            except Exception as e:
                print(f"Error processing {fits_file}: {e}")

    # sort HWP angles
    if hwp_order:
        arr_csv_HWP(output_csv_path,hwp_order,todelete=hwp_angles_to_delete)

    print(f"CSV file written to {output_csv_path}")


def read_csv(file_path, mode= 'standard'):
    """Takes a CSV file path containing "D_IMRANG", 
    "RET-ANG1", "single_sum", "single_diff", "diff_std", and "sum_std",
    for one wavelength bin and returns interleaved values, standard deviations, 
    and configuration list.

    Parameters
    -----------
    file_path : str or Path
        Path to the CSV.
    mode : str, optional
        If mode = 'wavelength', the wavelengths will be added
        to the configuration list for physical model fitting.
        If mode = 'm3', it will add the parallactic and altitude angles to the configuration list.
        If mode = 'm3_mcmc', it adds parallactic angle, altitude angle, and wavelength to the configuration list.


    Returns
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
    
    # Convert relevant columns to float 
    if mode == 'standard':
        for col in ["RET-ANG1", "D_IMRANG"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set errors to NaN if not possible
    else:
        for col in ["RET-POS1", "D_IMRANG"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float, set errors to NaN if not possible

    # Interleave values from "diff" and "sum"
    interleaved_values = np.ravel(np.column_stack((df["single_diff"].values, df["single_sum"].values)))

    # Interleave values from "diff_std" and "sum_std"
    interleaved_stds = np.ravel(np.column_stack((df["diff_std"].values, df["sum_std"].values)))

    # Convert each row's values into a list of two-element lists
    configuration_list = []
    for index, row in df.iterrows():
        # Extracting values from relevant columns
        if mode== 'standard':
            hwp_theta = row["RET-ANG1"] # now im only using RET-ANG1 for internal calibrations
        else:
            hwp_theta = row["RET-POS1"] # for on sky data, ret-pos1 accounts for hwp tracking laws, but breaks for internal
        imr_theta = row["D_IMRANG"]

        if mode == 'wavelength': # add wavelength
            wavelength = row["wavelength_bin"]
            hwp_theta = row["RET-ANG1"]
            # Building dictionary with wavelength
            row_data = {
                "hwp": {"theta": hwp_theta, "wavelength": wavelength},
                "image_rotator": {"theta": imr_theta, "wavelength": wavelength},
                "wollaston": {"wavelength":wavelength}
            }
        elif mode == 'm3':
            a = row['a']
            p = row['p']
            row_data = {
                "hwp": {"theta": hwp_theta},
                "image_rotator": {"theta": imr_theta},
                "altitude_rot": {"pa":-a}, # negative altitude angle, confirmed with SCExAO people this is right
                "parang_rot": {"pa":p}
            }
        elif mode == 'm3_mcmc':
            wavelength = row["wavelength_bin"]
            a = row['a']
            p = row['p']
            row_data = {
                "hwp": {"theta": hwp_theta, "wavelength": wavelength},
                "image_rotator": {"theta": imr_theta, "wavelength": wavelength},
                "altitude_rot": {"pa":-a}, # negative altitude angle, confirmed with SCExAO people this is right
                "M3": {"wavelength":wavelength},
                "parang_rot": {"pa":p},
                "wollaston": {"wavelength":wavelength}
            }
        else:
            # Building dictionary
            row_data = {
                "hwp": {"theta": hwp_theta},
                "image_rotator": {"theta": imr_theta}
            }
        # Append two configurations for diff and sum (one for diff, one for sum)
        # Use a deepcopy to ensure callers can mutate one entry safely
        configuration_list.append(copy.deepcopy(row_data))
    if mode == 'wavelength':
        return interleaved_values, interleaved_stds, configuration_list
    else:
        return interleaved_values, interleaved_stds, configuration_list
    

def read_csv_physical_model_all_bins(csv_dir,m3=False):
    """
    Does the same thing as read_csv() but reads all 22 csvs written
    in a directory for all 22 CHARIS wavelength bins and puts everything into one array.
    Also adds wavelength bin to the configuration dictionary for use with custom
    pyMuellerMat common mm functions. 

    Parameters
    -----------
    csv_dir : Path or str
        The directory where the csv files are stored. Will check for bins in the title
        and for 22 files.

    m3 : bool, optional
        Adds necessary parameters to the config dict to fit m3's physical model
        parameters. Adds wavelength for M3, IMR, and HWP.

    Returns
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
       print("Expected 22 CSV files for all wavelength bins, but found {}".format(len(sorted_files)))
    
    interleaved_values_all = []
    interleaved_stds_all = []
    configuration_list_all = []
    if m3 is False:
        for file in sorted_files:
            interleaved_values, interleaved_stds, configuration_list= read_csv(file, mode='wavelength')
            interleaved_values_all = np.append(interleaved_values_all, interleaved_values)
            interleaved_stds_all = np.append(interleaved_stds_all, interleaved_stds)
            configuration_list_all.extend(configuration_list)
    if m3 is True:
        for file in sorted_files:
            interleaved_values, interleaved_stds, configuration_list= read_csv(file, mode='m3_mcmc')
            interleaved_values_all = np.append(interleaved_values_all, interleaved_values)
            interleaved_stds_all = np.append(interleaved_stds_all, interleaved_stds)
            configuration_list_all.extend(configuration_list)

    return interleaved_values_all, interleaved_stds_all, configuration_list_all

def match_fits_tags(cubedir):
    """
    Renames DPP processed data to original CHARIS ID.
    
    Parameters
    ----------
    cubedir : str
        Directory containing the processed CHARIS data with format n*.fits.
        Anything not following this format will be ignored.

    Returns
    -------
    None
        Renames files in the cubedir to match the original CHARIS ID.
    """

    cubedir = Path(cubedir)

    # iterate through all fits files in the cubedir, following dpp format of n*.fits
    for proc_file in cubedir.glob('n*.fits'):

        # grab the fits header containing the original id
        with fits.open(proc_file) as hdul:
            header = hdul[0].header
            original_id = header.get('ORIGNAME', None)
            if not original_id:
                raise ValueError(f"No ORIGINAME found in header of {proc_file.name}")
            
        # rename the file to match the original CHARIS ID
        proc_file.rename(cubedir / f"CRSA{original_id}_flat_cube.fits")
        print(f"Renamed {proc_file.name} to CRSA{original_id}_flat_cube.fits")
        

def model_data(json_dir, csv_path=None,offsets=True):
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

    offsets : bool, optional
        If True, includes offset angles in the DataFrame.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing all fitted retardances by wavelength and offset angles with errors.
    """
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"{json_dir} is not a valid directory.")
    if csv_path is not None:
        csv_path = Path(csv_path)
    
    # Load JSON files
    json_files = sorted(json_dir.glob("*.json"))

    # Check for correct file amount
    if len(json_files) != 22:
        raise ValueError(f"Expected 22 JSON files, found {len(json_files)}.")

    # Check for bins and sort
    for f in json_files:
        try:
            match = re.search(r'bin(\d+)', f.name)
            if not match:
                raise ValueError(f"File {f.name} does not match expected naming convention.")
        except Exception as e:
            raise ValueError(f"Error processing file {f.name}: {e}")
    sorted_files = sorted(json_files, key=lambda f: int(re.search(r'bin(\d+)', f.name).group(1)))

    # Find all possible flattened keys
    def flatten_keys(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_keys(v, new_key, sep=sep))
            else:
                items.append(new_key)
        return items

    # Get all keys from all files
    all_keys = set()
    for f in sorted_files:
        with open(f, 'r') as file:
            data = json.load(file)
            keys = flatten_keys(data)
            all_keys.update(keys)

    # Always include wavelength_bin
    columns = ['wavelength_bin'] + sorted(all_keys)
    df_rows = []

    # Extract values for each file
    for f in sorted_files:
        with open(f, 'r') as file:
            data = json.load(file)
            # Flatten values
            def flatten_values(d, parent_key='', sep='_'):
                items = {}
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.update(flatten_values(v, new_key, sep=sep))
                    else:
                        items[new_key] = v
                return items
            values = flatten_values(data)
            # Extract bin number from filename
            match = re.search(r'bin(\d+)', f.name)
            bin_num = int(match.group(1)) if match else None
            wavelength = wavelength_bins[bin_num] if bin_num is not None else None
            row = {'wavelength_bin': wavelength}
            for col in columns:
                if col != 'wavelength_bin':
                    row[col] = values.get(col, None)
            df_rows.append(row)

    df = pd.DataFrame(df_rows, columns=columns)

    # Save to CSV if specified
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

    return df
