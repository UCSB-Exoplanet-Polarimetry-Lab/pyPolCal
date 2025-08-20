from pathlib import Path
import numpy as np
from astropy.io import fits
from photutils.aperture import RectangularAperture
from photutils.aperture import aperture_photometry
import pandas as pd
import re
from vampires_calibration.constants import wavelength_bins

###############################################################
###### Functions related to reading/writing in .csv values ####
###############################################################


def single_sum_and_diff(fits_cube_path, wavelength_bin):
    """Calculate single difference and sum between left and right beam 
    rectangular aperture photometry from CHARIS internal calibration
    fits cubes. Add L/R counts and stds to array.
    
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

    # check if data is a 3d cube (wavelength, y, x)

    if cube_data.ndim != 3:
        raise ValueError("Input data must be a 3D cube (wavelength, y, x).")
        
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
    single_diff = (phot_rbeam['aperture_sum'][0] - phot_lbeam['aperture_sum'][0]) #/ single_sum

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

    




def write_fits_info_to_csv(cube_directory_path, raw_cube_path, output_csv_path, wavelength_bin,hwp_order=[0,45,11.25,56.25,22.5,67.5,33.75,78.75],hwp_angles_to_delete=[90]):
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
    cube_directory_path : str or Path
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

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV.
    mode : str, optional
        If mode = 'wavelength', the wavelengths will be added
        to the configuration list for physical model fitting.
        If mode = 'm3', it will add the parallactic and altitude angles to the configuration list.

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
        if mode == 'wavelength': # add wavelength
            wavelength = row["wavelength_bin"]
            # Building dictionary with wavelength
            row_data = {
                "hwp": {"theta": hwp_theta, "wavelength": wavelength},
                "image_rotator": {"theta": imr_theta, "wavelength": wavelength}
            }
        elif mode == 'm3':
            a = row['a']
            p = row['p']
            row_data = {
                "hwp": {"theta": hwp_theta},
                "image_rotator": {"theta": imr_theta},
                "altitude_rot": {"pa":a},
                "parang_rot": {"pa":p}
            }
        else:
            # Building dictionary
            row_data = {
                "hwp": {"theta": hwp_theta},
                "image_rotator": {"theta": imr_theta}
            }

        # Append two configurations for diff and sum
        configuration_list.append(row_data)
    if mode == 'wavelength':
        return interleaved_values, interleaved_stds, configuration_list
    else:
        return interleaved_values, interleaved_stds, configuration_list
    

def read_csv_physical_model_all_bins(csv_dir):
    """
    Does the same thing as read_csv() but reads all 22 csvs written
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