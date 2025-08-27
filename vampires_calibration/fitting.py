from vampires_calibration.utils import parse_configuration,update_system_mm,generate_measurement
import numpy as np
from scipy.optimize import minimize, least_squares
import copy

########################################################################################
###### Functions related to fitting ####################################################
########################################################################################

def update_p0(p0, result):
    """
    Updates the existing p0 dictionary in place using result_values from scipy.optimize,
    based on the parameter order returned by parse_configuration.

    Parameters
    -----------
    p0 : dict
        The original nested parameter dictionary (will be updated in-place).
    result_values : list or np.ndarray
        Optimized parameter values from scipy.optimize (e.g., result.x).

    Returns
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

    Returns
    --------

    tuple
        [0] result: scipy.optimize.OptimizeResult
            If mode = 'minimize': scipy minimize() result object
            If mode = 'least_sqares': scipy least_squares() result object
        [1] logl_value/cost: float
            If mode = 'minimize': negative log likelihood value
            If mode = 'least_sqares': cost value 
        [2] error (only if mode='least_squares'): float
            Error estimated as in Van Holstein et al. 2020
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
        cost_value = -result.cost
        return result, cost_value,errors
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
    print(output_intensities)
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

    Returns
    --------
    np.ndarray
        Simulated intensities [L,R,L,R..]

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

