"""

custom_mms.py

User-defined Mueller matrix functions.
Only necessary for MCMC, otherwise you can register custom mms in notebooks.
The custom mueller matrix functions used in the CHARIS fitting tutorial are 
already available here.

"""

import numpy as np
from pyPolCal.mm_registry import register_mm_function

# Example:
# @register_mm_function
# def my_custom_mm(param=0):
#     return np.eye(4)

from pyPolCal.csv_tools import model_data
from pyPolCal.mm_registry import register_mm_function
import numpy as np
from pyPolCal.constants import wavelength_bins
from pyMuellerMat.common_mm_functions import elliptical_retarder_function , wollaston_prism_function
from importlib.resources import files

# Lazy loader for fitted arrays so we don't execute file I/O during import.
_FITS_CACHE = {}

def _load_wavelength_fits():
    if _FITS_CACHE:
        return _FITS_CACHE

    wavelength_fits_dir = files('pyPolCal.tutorial_notebooks.CHARIS_fitting_tutorial').joinpath('wavelength_fit_results')
    wavelength_fits_df = model_data(wavelength_fits_dir)
    _FITS_CACHE['derotator_phi_h'] = wavelength_fits_df['image_rotator_phi_h'].to_numpy()
    _FITS_CACHE['derotator_phi_45'] = wavelength_fits_df['image_rotator_phi_45'].to_numpy()
    _FITS_CACHE['derotator_phi_r'] = wavelength_fits_df['image_rotator_phi_r'].to_numpy()
    _FITS_CACHE['wollaston_eta'] = wavelength_fits_df['wollaston_eta'].to_numpy()
    return _FITS_CACHE

@register_mm_function()
def fitted_derotator_function_12_28_2025(wavelength=500): # Only keyword arguments!
    fits = _load_wavelength_fits()
    phi_h = fits['derotator_phi_h']
    phi_45 = fits['derotator_phi_45']
    phi_r = fits['derotator_phi_r']

    # Interpolating retardance as a function of wavelength
    phi_h_interp = np.interp(wavelength, wavelength_bins, phi_h)
    phi_45_interp = np.interp(wavelength, wavelength_bins, phi_45)
    phi_r_interp = np.interp(wavelength, wavelength_bins, phi_r)

    # using pyMuellerMat's elliptical retarder function
    return elliptical_retarder_function(phi_h_interp, phi_45_interp, phi_r_interp)

@register_mm_function()
def fitted_wollaston_function_12_28_2025(beam='o',wavelength=500): # Only keyword arguments!
    fits = _load_wavelength_fits()
    eta = fits['wollaston_eta']

    # Interpolating retardance as a function of wavelength
    eta_interp = np.interp(wavelength, wavelength_bins, eta)

    # using pyMuellerMat's wollaston function
    return wollaston_prism_function(beam=beam,eta=eta_interp)
