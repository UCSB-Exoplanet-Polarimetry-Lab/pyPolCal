import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAperture,CircularAnnulus
from photutils.aperture import aperture_photometry
from photutils.psf import fit_fwhm
from astropy.visualization import simple_norm
from pathlib import Path
import re
from photutils.aperture import ApertureStats
from photutils.centroids import (centroid_com, centroid_sources)
from pyPolCal.constants import wavelength_bins
from pyPolCal.utils import generate_system_mueller_matrix,process_dataset,process_errors,process_model,parse_configuration
from pyPolCal.fitting import update_p0,update_system_mm,minimize_system_mueller_matrix,model
from pyPolCal.plotting import plot_data_and_model
import traceback
from pyPolCal.csv_tools import arr_csv_HWP,read_csv, model_data
import json
from pyMuellerMat.physical_models.charis_physical_models import HWP_retardance,IMR_retardance,M3_retardance,M3_diattenuation
import copy
import matplotlib.patches as mpatches

def charis_centroids_one_psf(image_data,initial_guess_l,initial_guess_r,box_size,wavelength_bin):
    """
    Fits centroids for CHARIS specpol mode using center of mass fit, assuming only one PSF per Wollaston beam.
    Fits all wavelength bins using an initial guess. Uses photutils centroid_sources()
    with centroid function centroid_com(). 

    Parameters
    -----------
    image_data: np.ndarray
        CHARIS FITS cube image data. Axes should be (wavelength,y,x)
        Should be background subtracted.

    initial_guess_l: list or np.ndarray
        Initial guess for the PSF centroid for the left beam [x,y].

    initial_guess_l: list or np.ndarray
        Initial guess for the PSF centroid for the right beam [x,y].

    box_size: int
        Length of the square box where the algorithm will search for
        the PSF center. 
    
    wavelength_bin: int
        Which wavelength bin to centroid. 

    
    Returns:
    --------

    centroid_list: np.ndarray
        Array of left and right beam centroids. 

    """

    

    # Iterate through all bins
    
    image_data_bin_n = image_data

    # Grab initial guesses and calculate centroids
    x_init = (initial_guess_l[0],initial_guess_r[0])
    y_init = (initial_guess_l[1], initial_guess_r[1])
    x,y = centroid_sources(image_data_bin_n,x_init,y_init,box_size,centroid_func=centroid_com)
    centroid_list = [[x[0],y[0]],[x[1],y[1]]]
   
    return centroid_list


def single_sum_and_diff_psf(fits_cube_path, wavelength_bin, aperture_l,aperture_r,annulus_l=None,annulus_r=None):
    """Calculate single difference and sum between left and right beam 
    rectangular aperture photometry from a single psf. Add L/R counts and stds to array.
    Masks all pixels above 40,000 counts to correct
    for detector nonlinearity.
    Parameters
    -----------
    fits_cube_path : str or Path
        Path to the CHARIS fits cube file.
        
    wavelength_bin : int
        Index of the wavelength bin to analyze (0-based).

    aperture_l: photutils.aperture.Aperture
        Photutils aperture object for the left Wollaston beam.

    aperture_r: photutils.aperture.Aperture
        Photutils aperture object for the right Wollaston beam.

    annulus_l: photutils.aperture.Annulus, optional
        Photutils annulus object for local background subtraction for left Wollaston beam.
        Provide r and l or it will be skipped.

    annulus_r: photutils.aperture.Annulus, optional
        Photutils annulus object for local background subtraction for right Wollaston beam.
        Provide r and l or it will be skipped.

    Returns
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
    mask = image_data > 40000
    # define apertures perform aperture photometry 
    phot_lbeam = aperture_photometry(image_data, aperture_l,mask=mask)
    phot_rbeam = aperture_photometry(image_data, aperture_r,mask=mask)

    # get left and right counts
    left_counts = phot_lbeam['aperture_sum'][0]
    right_counts = phot_rbeam['aperture_sum'][0]

    # optional bkgd subtraction
    if (annulus_l is not None) and (annulus_r is not None) :
        bkgd_l = ApertureStats(image_data,annulus_l).median
        bkgd_r = ApertureStats(image_data, annulus_r).median
        left_counts -= bkgd_l*aperture_l.area
        right_counts -= bkgd_r*aperture_r.area

    # calculate normalized single difference and sum
    single_sum = right_counts + left_counts
    single_diff = right_counts - left_counts 

    # Get error on each of the apertures
    std_l = ApertureStats(image_data,aperture_l).std
    std_r = ApertureStats(image_data,aperture_r).std
    sum_std = diff_std = np.sqrt(std_l**2+std_r**2)
    
    return (single_sum, single_diff, left_counts, right_counts, sum_std, diff_std)


def write_fits_info_to_csv_psf(cube_directory_path, raw_cube_path, output_csv_path,centroid_guesses, box_size,wavelength_bin,aperture_radii=None,bkgd_annuli_radii=None,auto_annuli=False, plot_every_x=None, max_fwhm=None):
    """
    
    Write filepath, D_IMRANG (derotator angle), RET-ANG1 (HWP angle), 
    single sum, single difference, LCOUNTS, RCOUNTS, difference std,
    sum std, and wavelength values for a wavelength bin from each fits cube in the directory.
    Default HWP order and deletion works for future double difference calculation. Masks all pixels above 40,000 counts to correct
    for detector nonlinearity.

    FITS parameters are extracted from raw files, while single sum and difference are calculated using the
    fits cube data and the defined rectangular apertures.
    If the necessary header keywords are not present, the values will be set to NaN.

    Note - This function assumes that the raw and extracted cubes have the same number in the filepath. If
    you processed your cubes in the CHARIS DPP, this is not the case. 
    
    Parameters
    -----------

    cube_directory_path : str or Path
        Path to the directory containing CHARIS fits cubes.
        
    raw_cube_path : str or Path
        Path to the directory containing the matching raw CHARIS FITS files.

    output_csv_path : str or Path
        Path where the output csv will be created.

    centroid_guesses : tuple
        [0] left centroid guess: list or np.1darray
            Initial guess for the centroid location of the left Wollaston beam PSF [x,y].
        [1] right centroid guess: list or np.1darray
            Initial guess for the centroid location of the right Wollaston beam PSF [x,y].

    box_size: int
        Length of the square box where the algorithm will search for
        the PSF center. 

    wavelength_bin : int
        Index of the wavelength bin to analyze (0-based).

    aperture_radii : list or np.ndarray
        Radii to use for the circular apertures. [L,R] If None, will be calculated as 3*FWHM of each PSF.

    hwp_order: list or np.ndarray, optional
        List of desired HWP order. Default works for double difference calculations.

    todelete: list or np.ndarray, optional
        List of HWP angles to delete. Default works
        for double difference calculations. Set to None if you want to keep them all. 

    bkgd_annuli_radii: tuple, optional
        [0] left radii: list or np.1darray
        Inside and outside radii length in pixels for the local background subtraction
        annulus of the left Wollaston aperture [inside,outside].
        [1] right radii: list or np.1darray
        Inside and outside radii length in pixels for the local background subtraction
        annulus of the right Wollaston aperture [inside,outside].

    auto_annuli: bool, optional
        If True, will automatically add annuli 5 pixels larger than the aperture radii.

    plot_every_x: int, optional
        Plots apertures against image data every xth file processed.

    max_fwhm: list, optional
        [0] max fwhm left: float
            Maximum FWHM in pixels to accept for the left PSF. If the fitted FWHM is larger than this,
            an error will be raised.
        [1] max fwhm right: float
            Maximum FWHM in pixels to accept for the right PSF. If the fitted FWHM is larger than this,
            an error will be raised.

    Returns
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
        f.write("filepath,D_IMRANG,RET-ANG1,single_sum,single_diff,LCOUNTS,RCOUNTS,sum_std,diff_std,p,a,wavelength_bin\n")

        # iterate over all fits files in the directory
        for idx,fits_file in enumerate(sorted(cube_directory_path.glob('*.fits'))):
            try:

                # check if corresponding raw fits file exists
                match = re.search(r"(\d{8})", fits_file.name)
                if not match:
                    with fits.open(fits_file) as hdul_secondary:
                        header_secondary = hdul_secondary[0].header
                        raw_filename = header_secondary.get('ORIGNAME',None)
                        if raw_filename:
                            m = re.search(r"(\d{8})", raw_filename)
                            if m:
                                match = m
                if match is None:
                    raise ValueError(f"Could not extract 8-digit ID from filename {fits_file.name}")
                fits_id = match.group(1)
                raw_candidates = list(raw_cube_path.glob(f"*{fits_id}*.fits"))
                if not raw_candidates:
                    raise FileNotFoundError(f"No raw FITS file found for ID {fits_id}")
                raw_fits = raw_candidates[0]
                # extract ret and imr ang
                with fits.open(raw_fits) as hdul_raw:
                    raw_header = hdul_raw[0].header
                    d_imrang = raw_header.get("D_IMRANG", np.nan)
                    ret_ang1 = raw_header.get("RET-ANG1", np.nan)

                # extract image data, parang, and altitude
                with fits.open(fits_file) as hdul:
                    cube_header = hdul[0].header
                    d_parang = cube_header.get("PARANG",np.nan)
                    d_alt = cube_header.get("ALTITUDE",np.nan)
                    cube_data = hdul[1].data
                    image_data = cube_data[wavelength_bin]
                
                # find centroids of psfs
                centroids = charis_centroids_one_psf(image_data,centroid_guesses[0],centroid_guesses[1],box_size,wavelength_bin)

                # create circular apertures
                if aperture_radii is not None:
                    aper_l = CircularAperture(centroids[0],r=aperture_radii[0])
                    aper_r = CircularAperture(centroids[1], r=aperture_radii[1])
                    if auto_annuli:
                        bkgd_annuli_radii = ([aperture_radii[0],aperture_radii[0]+5],[aperture_radii[1],aperture_radii[1]+5])

                if aperture_radii is None:
                    fwhm_l = fit_fwhm(image_data,xypos=centroids[0],fit_shape=box_size)
                    fwhm_r = fit_fwhm(image_data,xypos=centroids[1],fit_shape=box_size)
                    if max_fwhm and fwhm_l > max_fwhm[0]:
                        print(f"Fitted FWHM for left PSF of {fits_id} is {fwhm_l}, which is larger than the maximum allowed {max_fwhm[0]}. Max will be used")
                        fwhm_l = max_fwhm[0]
                    if max_fwhm and fwhm_r > max_fwhm[1]:
                        print(f"Fitted FWHM for right PSF of {fits_id} is {fwhm_r}, which is larger than the maximum allowed {max_fwhm[1]}. Max will be used")
                        fwhm_r = max_fwhm[1]
                    print(f"Fitted FWHM left: {fwhm_l}, Fitted FWHM right: {fwhm_r}")
                    aper_l = CircularAperture(centroids[0],r=int(3*fwhm_l))
                    aper_r = CircularAperture(centroids[1], r=int(3*fwhm_r))
                    if auto_annuli:
                        bkgd_annuli_radii = ([int(3*fwhm_l),int(3*fwhm_l+5)],[int(3*fwhm_r),int(3*fwhm_r+5)])

                # calculate single sum and normalized single difference
                if bkgd_annuli_radii or auto_annuli: 
                    bkgd_annulus_l = CircularAnnulus(centroids[0],int(bkgd_annuli_radii[0][0]),int(bkgd_annuli_radii[0][1]))
                    bkgd_annulus_r = CircularAnnulus(centroids[1],int(bkgd_annuli_radii[1][0]),int(bkgd_annuli_radii[1][1]))
                    single_sum, single_diff, LCOUNTS, RCOUNTS, sum_std, diff_std = single_sum_and_diff_psf(fits_file,wavelength_bin,aper_l,aper_r,bkgd_annulus_l,bkgd_annulus_r)
                else:
                    single_sum, single_diff, LCOUNTS, RCOUNTS, sum_std, diff_std = single_sum_and_diff_psf(fits_file,wavelength_bin,aper_l,aper_r)

                # wavelength bins for lowres mode
                bins = wavelength_bins
                
                # write to csv file
                f.write(f"{fits_file}, {d_imrang}, {ret_ang1}, {single_sum}, {single_diff}, {LCOUNTS}, {RCOUNTS}, {sum_std}, {diff_std},{d_parang},{d_alt}, {bins[wavelength_bin]}\n")

                if plot_every_x:
                    if idx % plot_every_x == 0:  # plot every xth file
                        fig, ax = plt.subplots(figsize=(10,6))
                        snorm = simple_norm(image_data,'log',)
                        im = ax.imshow(image_data, origin='lower', cmap='inferno',norm=snorm)
                        mask = image_data > 40000
                        aper_l.plot(ax,color='white')
                        aper_r.plot(ax,color='white')
                        ax.set_title(f"{fits_id} Wavelength bin: {wavelength_bins[wavelength_bin]} nm")
                        if bkgd_annuli_radii or auto_annuli:
                            CircularAnnulus(centroids[0],bkgd_annuli_radii[0][0],bkgd_annuli_radii[0][1]).plot(ax,color='white',alpha=0.5)
                            CircularAnnulus(centroids[1],bkgd_annuli_radii[1][0],bkgd_annuli_radii[1][1]).plot(ax,color='white',alpha=0.5)
                        fig.colorbar(im,ax=ax)
                        ax.imshow(mask, origin='lower', cmap='gray', alpha=0.2, vmin=0, vmax=1)
                        mask_patch = mpatches.Patch(color='gray', label='Masked Pixels > 40000 counts')
                        ax.legend(handles=[mask_patch])

            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
                traceback.print_exc()

    # sort csv by filename
    df = pd.read_csv(output_csv_path)
    df = df.sort_values(by='filepath')
    df.to_csv(output_csv_path,index=False)
    

    print(f"CSV file written to {output_csv_path}")

def plot_data_and_model_alt(interleaved_values, model, 
    configuration_list, interleaved_stds=None,
    hwp_theta_filter=None, wavelength=None, save_path=None,
    include_sums=True, title=None):

    """
    Plots double difference and double sum measurements alongside model predictions,
    grouped by altitude angle. Optionally filters by a specific 
    image rotator angle and displays a wavelength in the plot title.

    Parameters
    ----------
    interleaved_values : np.ndarray
        Interleaved array of observed single difference and single sum values.
        Expected format: [sd1, ss1, sd2, ss2, ...]. 


    model : np.ndarray
        Interleaved array of model-predicted double difference and double sum values.
        If charis use single differences and sums. 

    configuration_list : list of dict
        List of system configurations (one for each measurement), where each dictionary 
        contains component settings like HWP and image rotator angles.

    interleaved_stds : np.ndarray
        Interleaved array of standard deviations corresponding to the observed values.

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
    
    if interleaved_stds is None:
        interleaved_stds = np.zeros_like(interleaved_values)

    interleaved_stds = process_errors(interleaved_stds, interleaved_values)
    dd_stds = interleaved_stds[::2]
    ds_stds = interleaved_stds[1::2]
    interleaved_values = process_dataset(interleaved_values)

    # Extract double differences and sums
    dd_values = interleaved_values[::2]
    ds_values = interleaved_values[1::2]
    dd_model = model[::2]
    ds_model = model[1::2]

    # Group by HWP theta
    dd_by_hwp = {}
    ds_by_hwp = {}

    for i, config in enumerate(configuration_list[::2]):
        hwp_theta = round(config["hwp"]["theta"], 1)
        pa = config["altitude_rot"]["pa"]

        if hwp_theta_filter is not None and hwp_theta != round(hwp_theta_filter, 1):
            continue

        if hwp_theta not in dd_by_hwp:
            dd_by_hwp[hwp_theta] = {"pa": [], "values": [], "stds": [], "model": []}
        dd_by_hwp[hwp_theta]["pa"].append(pa)
        dd_by_hwp[hwp_theta]["values"].append(dd_values[i])
        dd_by_hwp[hwp_theta]["stds"].append(dd_stds[i])
        dd_by_hwp[hwp_theta]["model"].append(dd_model[i])

        if hwp_theta not in ds_by_hwp:
            ds_by_hwp[hwp_theta] = {"pa": [], "values": [], "stds": [], "model": []}
        ds_by_hwp[hwp_theta]["pa"].append(pa)
        ds_by_hwp[hwp_theta]["values"].append(ds_values[i])
        ds_by_hwp[hwp_theta]["stds"].append(ds_stds[i])
        ds_by_hwp[hwp_theta]["model"].append(ds_model[i])

    # Create plots
    if include_sums:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    else:
        fig, axarr = plt.subplots(
            2, 1,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
        ax = axarr[0]
        small_ax = axarr[1]

    # Double Difference
    if include_sums:
        ax = axes[0]
        for hwp, d in dd_by_hwp.items():
            err = ax.errorbar(
                d["pa"], d["values"],
                yerr=d["stds"], fmt='o',
                label=f"{hwp}°"
            )
            color = err[0].get_color()
            ax.plot(d["pa"], d["model"], '-', color=color)

        ax.set_xlabel("Altitude angle PA (deg)")
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"HWP $\theta$")

        # Double Sum
        ax = axes[1]
        for hwp, d in ds_by_hwp.items():
            err = ax.errorbar(
                d["pa"], d["values"],
                yerr=d["stds"], fmt='o',
                label=f"{hwp}°"
            )
            color = err[0].get_color()
            ax.plot(d["pa"], d["model"], '-', color=color)

        ax.set_xlabel("Altitude angle PA (deg)")
        ax.set_ylabel("Double Sum")
        ax.legend(title=r"HWP $\theta$")

    else:
        for hwp, d in dd_by_hwp.items():
            err = ax.errorbar(
                d["pa"], d["values"],
                yerr=d["stds"], fmt='o',
                label=f"{hwp}°"
            )
            color = err[0].get_color()
            ax.plot(d["pa"], d["model"], '-', color=color)

            residuals = (np.array(d["values"]) - np.array(d["model"])) * 100
            small_ax.scatter(d["pa"], residuals, color=color)

        small_ax.axhline(0, color='black', linewidth=1)
        small_ax.set_xlabel("Altitude angle PA (deg)")
        small_ax.set_ylabel(r"Residual ($\%$)")
        ax.set_ylabel("Double Difference")
        ax.legend(title=r"HWP $\theta$")
        ax.grid()

    if wavelength is not None and title is None:
        fig.suptitle(f"{wavelength}nm")
    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
    return fig, ax



def fit_CHARIS_Mueller_matrix_by_bin_m3(csv_path, wavelength_bin, new_config_dict_path,plot_path=None):
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
    "RET-ANG1", "single_sum", "single_diff", "diff_std", and "sum_std", "p","a".

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
      An array of the errors for each parameter. Estimated using the method from van Holstein et al. 2020.
      van Holstein et al. 2020.
    fig : MatPlotLib figure object
    ax : MatPlotLib axis object
    s_res : float
      The polarimetric accuracy of the fit, calculated as the standard deviation of the residuals.
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
    interleaved_values, interleaved_stds, configuration_list = read_csv(filepath,mode='m3')
    # this works, not really sure why
    interleaved_values_forplotfunc = copy.deepcopy(interleaved_values)
    interleaved_stds_forlplotfunc = copy.deepcopy(interleaved_stds)
    #interleaved_stds = process_errors(interleaved_stds,interleaved_values)[::2] # just diffs
    #interleaved_values = process_dataset(interleaved_values)[::2] # just diffs
    #configuration_list = configuration_list 

    # Loading in past fits 
    offset_imr = -0.4506# derotator offset
    offset_hwp = -1.119# HWP offset
    offset_cal = -0.5905 # calibration polarizer offset
    imr_theta = 0 # placeholder 
    hwp_theta = 0 # placeholder
    imr_phi = IMR_retardance(wavelength_bins,259.12694)[wavelength_bin]
    hwp_phi = HWP_retardance(wavelength_bins,1.636,1.278)[wavelength_bin]
    epsilon_cal = 1
    m1, b1, m2, b2 = (1.94073,13.69728,2.07958,13.88817) # from MCMC
    m3_diat = M3_diattenuation(wavelength_bins[wavelength_bin],m1,b1,m2,b2)
    m3_ret = M3_retardance(wavelength_bins[wavelength_bin],m1,b1,m2,b2)

    # Define instrument configuration as system dictionary
    # Wollaston beam, imr theta/phi, and hwp theta/phi will all be updated within functions, so don't worry about their values here
    system_dict = {
        "components" : {
            "wollaston" : {
            "type" : "CHARIS_wollaston_function",
            "properties" : {"wavelength": wavelength_bins[wavelength_bin], "beam": 'o'}, 
            "tag": "internal",
            },
            "image_rotator" : {
                "type" : "elliptical_IMR_function",
                "properties" : {"wavelength": wavelength_bins[wavelength_bin], "theta": imr_theta, "delta_theta": offset_imr},
                "tag": "internal",
            },
            "hwp" : {
                "type" : "two_layer_HWP_function",
                "properties" : {"wavelength": wavelength_bins[wavelength_bin], "w_SiO2": 1.636, "w_MgF2": 1.278, "theta": hwp_theta, "delta_theta": offset_hwp},
                "tag": "internal",
            },
            "altitude_rot" : {
                "type" : "rotator_function",
                "properties" : {"pa":0},
                "tag":"internal",
            },
            "M3" : {
                "type" : "general_diattenuator_function",
                "properties" : {"d_h": m3_diat,"d_45":0,"theta": 0, "delta_theta":0},
                "tag": "internal",
            },

            "parang_rot" : {
                "type" : "rotator_function",
                "properties" : {"pa":0},
                "tag":"internal",
            },
    }
    }

    # Converting system dictionary into system Mueller Matrix object
    system_mm = generate_system_mueller_matrix(system_dict)

    # Define initial guesses for our parameters 

    # MODIFY THIS IF YOU WANT TO CHANGE PARAMETERS
    p0 = {
        "M3": {
            "d_h": m3_diat, "d_45": 0.01,
        },

    }
    # Define some bounds
    # MODIFY THIS IF YOU WANT TO CHANGE PARAMETERS, ADD NEW BOUNDS OR CHANGE THEM
    polbounds = (-np.sqrt(3),np.sqrt(3)) 

    # Minimize the system Mueller matrix using the interleaved values and standard deviations
 

    # Counters for iterative fitting

    iteration = 1
    previous_logl = 1000000
    new_logl = 0

    # Perform iterative fitting
    # MODIFY THE BOUNDS INPUT HERE IF YOU WANT TO CHANGE PARAMETERS
    while abs(previous_logl - new_logl) > 0.01*abs(previous_logl):
        if iteration > 1:
            previous_logl = new_logl
        # Configuring minimization function for CHARIS
        result, new_logl,error = minimize_system_mueller_matrix(p0, system_mm, interleaved_values, configuration_list,
            interleaved_stds, process_dataset=process_dataset,process_model=process_model,process_errors=process_errors,include_sums=False, bounds = [(-1,1),(-1,1)],mode='least_squares')
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
        fig , ax = plot_data_and_model_alt(interleaved_values_forplotfunc, diffs_sums2,configuration_list, interleaved_stds_forlplotfunc, wavelength= wavelength_bins[wavelength_bin], include_sums=False,save_path=plot_path)
    else:
        fig , ax = plot_data_and_model_alt(interleaved_values_forplotfunc, diffs_sums2,configuration_list, interleaved_stds_forlplotfunc, wavelength= wavelength_bins[wavelength_bin],include_sums=False)

    # Print the Mueller matrix

    print("Updated Mueller Matrix:")
    print(updated_system_mm.evaluate())

    # Print residuals
    print(len(interleaved_values), len(diffs_sums2))
    data_dd = process_dataset(interleaved_values)[::2]
    model_dd = diffs_sums2[::2]
    residuals = data_dd*100 - model_dd*100
    print("Residuals range:", residuals.min(), residuals.max())
    print("Error:", error)
    # calculate s_res
    s_res = np.sqrt(np.sum(residuals**2)/(len(data_dd)-len(p0_values)))
    print("s_res:", s_res)

    # Save system dictionary to a json file

    with open (new_config_dict_path, 'w') as f:
        json.dump(p0, f, indent=4)
    #error = np.array(error)
    return error,fig, ax, s_res