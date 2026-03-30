"""
on_sky.py

Functions for performing aperture photometry on CHARIS specpol data and writing to a CSV file
containing aperture photometry results and relevant FITS headers.

"""

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
import traceback
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


def write_fits_info_to_csv_psf(cube_directory_path, output_csv_path,centroid_guesses, box_size,wavelength_bin, raw_cube_path=None,aperture_radii=None,bkgd_annuli_radii=None,auto_annuli=False, plot_every_x=None, max_fwhm=None):
    """
    
    Write filepath, D_IMRANG (derotator angle), RET-ANG1 (HWP angle without correcting for synchro ADI), 
    RET-POS1 (actual HWP angle), single sum, single difference, LCOUNTS, RCOUNTS, difference std,
    sum std, and wavelength values for a wavelength bin from each fits cube in the directory.
    Default HWP order and deletion works for future double difference calculation. 

    Single sum and difference are calculated using aperture photometry. You can either
    fit for apertures using 3X the PSF or provide fixed ones. There are a few things you 
    need to provide to do aperture photometry, and there are a few options, see parameters.
    If the necessary header keywords are not present, the values will be set to NaN.

    For the raw files- this function assumes that the raw files have the same 8-digit ID as either the processed cubes
    filename or 'ORIGNAME' keyword, so make sure you didn't rename your raw files. If you don't provide raw files, 
    the function will try to grab the D_IMRANG, RET-ANG1, and RET-POS1 keywords from extension 3 of the processed cube, 
    which should work for cubes not processed in the DPP. 
    
    Parameters
    -----------

    cube_directory_path : str or Path
        Path to the directory containing CHARIS fits cubes.

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

    raw_cube_path : str or Path, optional
        Path to the directory containing the matching raw CHARIS FITS files.

    aperture_radii : list or np.ndarray, optional
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
        f.write("filepath,D_IMRANG,RET-ANG1,RET-POS1,single_sum,single_diff,LCOUNTS,RCOUNTS,sum_std,diff_std,p,a,wavelength_bin\n")

        # iterate over all fits files in the directory
        for idx,fits_file in enumerate(sorted(cube_directory_path.glob('*.fits'))):
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
                        ret_ang1 = extension_3['RET-ANG1']
                        ret_pos1 = extension_3['RET-POS1']
                        d_imrang = extension_3['D_IMRANG']
                # extract image data, parang, and altitude
                with fits.open(fits_file) as hdul:
                    cube_header = hdul[0].header
                    d_parang = cube_header.get("PARANG",np.nan)
                    d_alt = cube_header.get("ALTITUDE",np.nan)
                    cube_data = hdul[1].data
                    image_data = cube_data[wavelength_bin]
                    origname = cube_header.get("ORIGNAME", None)
                    match = re.search(r"(\d{8})", origname)
                    fits_id = match.group(1)
                
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
                    # Force extraction of the float to avoid NumPy 2.0 casting errors
                    fwhm_l = np.atleast_1d(fwhm_l)[0] 
                    fwhm_r = np.atleast_1d(fwhm_r)[0]
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
                f.write(f"{fits_file}, {d_imrang}, {ret_ang1}, {ret_pos1}, {single_sum}, {single_diff}, {LCOUNTS}, {RCOUNTS}, {sum_std}, {diff_std},{d_parang},{d_alt}, {bins[wavelength_bin]}\n")

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


# wrapper function similar to the ones in instruments.py

