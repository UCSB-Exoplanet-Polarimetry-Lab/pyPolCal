import numpy as np
from pathlib import Path
import copy
from vampires_calibration.constants import wavelength_bins
from pyMuellerMat.physical_models.charis_physical_models import IMR_retardance, HWP_retardance, M3_retardance, M3_diattenuation
from vampires_calibration.csv_tools import read_csv, model_data
from vampires_calibration.utils import generate_system_mueller_matrix,process_dataset,process_errors,process_model,parse_configuration
from vampires_calibration.fitting import update_p0,update_system_mm,minimize_system_mueller_matrix,model
from vampires_calibration.plotting import plot_data_and_model
import json

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
      An array of the errors for each parameter. Estimated using the method from van Holstein et al. 2020.
      van Holstein et al. 2020.
    fig : MatPlotLib figure object
    ax : MatPlotLib axis object
    s_res : float
      The polarimetric accuracy as defined in appendix E of van Holstein et al. 2020.
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
    offset_imr = -0.13959# derotator offset
    offset_hwp = -1.59338# HWP offset
    offset_cal = -0.11835 # calibration polarizer offset
    imr_theta = 0 # placeholder 
    hwp_theta = 0 # placeholder
    imr_phi = IMR_retardance(wavelength_bins,259.11814)[wavelength_bin]
    hwp_phi = HWP_retardance(wavelength_bins,1.63398,1.27711)[wavelength_bin]
    epsilon_cal = 1
    m3_phi = M3_retardance(wavelength_bins[wavelength_bin])
    m3_epsilon = M3_diattenuation(wavelength_bins[wavelength_bin])
    

    # Define instrument configuration as system dictionary
    # Wollaston beam, imr theta/phi, and hwp theta/phi will all be updated within functions, so don't worry about their values here
    system_dict = {
        "components" : {

            "wollaston" : {
            "type" : "wollaston_prism_function",
            "properties" : {"beam": 'o','eta':1}, 
            "tag": "internal",
            },

            "image_rotator" : {
            "type" : "elliptical_retarder_function",
            "properties" : {"phi_45":0,"phi_h": imr_phi, "phi_r": 0,"phi_45":0, "theta": imr_theta, "delta_theta": offset_imr},
            "tag": "internal",
            },
            
            "hwp" : {
                "type" : "general_retarder_function",
                "properties" : {"phi": hwp_phi, "theta": hwp_theta, "delta_theta": offset_hwp},
                "tag": "internal",
            },

            "lp" :{
                "type" : "diattenuator_retarder_function",
                "properties" : {"epsilon": epsilon_cal, "delta_theta": offset_cal},
                "tag": "internal",
            },
}
    }

    # Converting system dictionary into system Mueller Matrix object
    system_mm = generate_system_mueller_matrix(system_dict)

    # Define initial guesses for our parameters 

    # MODIFY THIS IF YOU WANT TO CHANGE PARAMETERS
    p0 = {
            "hwp": {"phi":hwp_phi},
          "image_rotator": {"phi_h":imr_phi, "phi_r":0, "phi_45":0},
          "wollaston": {"eta": 1}
         }

    # Define some bounds
    # MODIFY THIS IF YOU WANT TO CHANGE PARAMETERS, ADD NEW BOUNDS OR CHANGE THEM
    offset_bounds = (-5,5)
    hwpstd = 0.1*np.abs(hwp_phi)
    hwp_phi_bounds = (hwp_phi-hwpstd, hwp_phi+hwpstd)
    imrstd = 0.1*np.abs(imr_phi)
    imr_phi_bounds = (imr_phi-imrstd, imr_phi+imrstd)
    #imrostd = 0.1*np.abs(offset_imr)
    #offset_imr_bounds = (offset_imr-imrostd, offset_imr+imrostd)
    #hwpostd = 0.1*np.abs(offset_hwp)
    #offset_hwp_bounds = (offset_hwp-hwpostd, offset_hwp+hwpostd)
    epsilon_cal_bounds = (-1,1)
    #calostd = 0.1 *np.abs(offset_cal)
    #offset_cal_bounds = (-15, 15)
    dichroic_phi_bounds = (0,np.pi)
    ret_bounds = (0,2*np.pi)

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
        result, new_logl, error = minimize_system_mueller_matrix(p0, system_mm, interleaved_values, 
             configuration_list, process_dataset=process_dataset,process_model=process_model,include_sums=False, bounds = [ret_bounds, ret_bounds,ret_bounds,ret_bounds,(0,1)],mode='least_squares')
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
        fig , ax = plot_data_and_model(interleaved_values_forplotfunc, diffs_sums2,configuration_list, interleaved_stds_forlplotfunc, wavelength= wavelength_bins[wavelength_bin], include_sums=False,save_path=plot_path)
    else:
        fig , ax = plot_data_and_model(interleaved_values_forplotfunc, diffs_sums2,configuration_list, interleaved_stds_forlplotfunc, wavelength= wavelength_bins[wavelength_bin],include_sums=False)
    
    # Print the Mueller matrix
    
    print("Updated Mueller Matrix:")
    print(updated_system_mm.evaluate())

    # Print residuals
    print(len(interleaved_values), len(diffs_sums2))
    data_dd = process_dataset(interleaved_values)[::2]
    model_dd = diffs_sums2[::2]
    residuals = data_dd*100 - model_dd*100
    # calculate s_res as in appendix E of SPHERE cal paper
    s_res = np.sqrt(np.sum(residuals**2)/(len(data_dd)-8))
    print("Residuals range:", residuals.min(), residuals.max())
    print("s_res:", s_res)
    print("Error:", error)

    # Save system dictionary to a json file

    with open (new_config_dict_path, 'w') as f:
        json.dump(p0, f, indent=4)
    error = np.array(error)
    return error, fig, ax, s_res


def fit_CHARIS_Mueller_matrix_by_bin_pickoff(csv_path, wavelength_bin, new_config_dict_path,plot_path=None):
    """
    Same as above but uses physical model as a starting point and fits an additional retarder.
    

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
      An array of the errors for each parameter. Estimated using the method from van Holstein et al. 2020.
      van Holstein et al. 2020.
    fig : MatPlotLib figure object
    ax : MatPlotLib axis object
    s_res : float
      The polarimetric accuracy as defined in appendix E of van Holstein et al. 2020.
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
    offset_imr = -0.13959 # derotator offset
    offset_hwp = -1.59338 # HWP offset
    offset_cal = -0.11835 # calibration polarizer offset
    imr_theta = 0 # placeholder 
    hwp_theta = 0 # placeholder
    imr_phi = IMR_retardance(wavelength_bins)[wavelength_bin]
    hwp_phi = HWP_retardance(wavelength_bins)[wavelength_bin]
    epsilon_cal = 1
    df = model_data('/Users/thomasmcintosh/Desktop/CHARIS-REU/Fitting/naive_fitting/elliptical_imr')
    imr_phi_h = df['image_rotator_phi_h'][wavelength_bin]
    imr_phi_r = df['image_rotator_phi_r'][wavelength_bin]
    imr_phi_45 = df['image_rotator_phi_45'][wavelength_bin]
    wol_eta = df['wollaston_eta'][wavelength_bin]

    # Define instrument configuration as system dictionary
    # Wollaston beam, imr theta/phi, and hwp theta/phi will all be updated within functions, so don't worry about their values here
    system_dict = {
            "components" : {
                "wollaston" : {
                "type" : "wollaston_prism_function",
                "properties" : {"beam": 'o', 'eta':wol_eta}, 
                "tag": "internal",
                },
                "pickoff_ret" : {
                    "type" : "elliptical_retarder_function",
                    "properties" : {"phi_45":0, "phi_h":0, "phi_r": 0, "delta_theta":0},
                    "tag": "internal",
                },
                
                "pickoff" : {
                    "type" : "general_diattenuator_function",
                    "properties" : {"d_h":0, "d_45":0, "d_r": 0, "T_avg":0,"delta_theta":0},
                    "tag": "internal",
                },      
                "image_rotator" : {
                    "type" : "elliptical_retarder_function",
                    "properties" : {"phi_45":0,"phi_h": imr_phi_h, "phi_r": imr_phi_r,"phi_45":imr_phi_45, "theta": imr_theta, "delta_theta": offset_imr},
                    "tag": "internal",
                },
                "hwp" : {
                    "type" : "two_layer_HWP_function",
                    "properties" : {"wavelength": wavelength_bins[wavelength_bin], "w_SiO2":1.66180, "w_MgF2": 1.29757, "theta":hwp_theta, "delta_theta": offset_hwp},
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

    # MODIFY THIS IF YOU WANT TO CHANGE PARAMETERS
    p0 = {
        "pickoff_ret": {"phi_h":0, "phi_45":0, "phi_r": 0, "delta_theta":0},
        "pickoff": {"d_h":0, "d_45":0, "d_r": 0, "T_avg":0,"delta_theta":0},
    }

    # Define some bounds
    # MODIFY THIS IF YOU WANT TO CHANGE PARAMETERS, ADD NEW BOUNDS OR CHANGE THEM
    offset_bounds = (-50,50)
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
    dichroic_phi_bounds = (0,2*np.pi)
    pol_bounds = (0,1)

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
        result, new_logl, error = minimize_system_mueller_matrix(p0, system_mm, interleaved_values, 
            configuration_list, process_dataset=process_dataset,process_model=process_model,include_sums=False, bounds = [dichroic_phi_bounds, dichroic_phi_bounds,dichroic_phi_bounds,offset_bounds,pol_bounds,pol_bounds,pol_bounds,pol_bounds,offset_bounds],mode='least_squares')
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
        fig , ax = plot_data_and_model(interleaved_values_forplotfunc, diffs_sums2,configuration_list, interleaved_stds_forlplotfunc, wavelength= wavelength_bins[wavelength_bin], include_sums=False,save_path=plot_path)
    else:
        fig , ax = plot_data_and_model(interleaved_values_forplotfunc, diffs_sums2,configuration_list, interleaved_stds_forlplotfunc, wavelength= wavelength_bins[wavelength_bin],include_sums=False)
    
    # Print the Mueller matrix

    print("Updated Mueller Matrix:")
    print(updated_system_mm.evaluate())

    # Print residuals
    print(len(interleaved_values), len(diffs_sums2))
    data_dd = process_dataset(interleaved_values)[::2]
    model_dd = diffs_sums2[::2]
    residuals = data_dd*100 - model_dd*100
    s_res = np.sqrt(np.sum(residuals**2)/(len(data_dd)-3))
    print("s_res:", s_res)
    print("Residuals range:", residuals.min(), residuals.max())
    print("Error:", error)

    
    # Save system dictionary to a json file

    with open (new_config_dict_path, 'w') as f:
        json.dump(p0, f, indent=4)
    error = np.array(error)
    return error, fig, ax, s_res






    





    

    
   




    

    
    


          
                

            
                    



        