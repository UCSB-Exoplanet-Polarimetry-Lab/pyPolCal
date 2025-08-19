from pyMuellerMat import common_mms as cmm
from pyMuellerMat import MuellerMat
import utils as funcs
import numpy as np

def full_system_mueller_matrix(system_dict):
    """
    Returns the Mueller matrix of the optical system based on the provided
    properties and order of components. Handles the `theta` property directly
    within each component.

    Args:
        system_dict: (dict) A dictionary containing:
            - "components": A dictionary where each key is a component name
              (string), and each value is a dictionary specifying the properties
              of that component, including its type and relevant parameters.
            - "order": A list of component names specifying the order in which
              components appear in the system.

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system.
    """
    components_dict = system_dict.get("components", {})
    order = system_dict.get("order", [])

    # List to hold MuellerMatrix objects
    mueller_components = []

    # Build each component based on the specified order
    for comp_name in order:
        comp_info = components_dict.get(comp_name)
        if not comp_info:
            raise ValueError(f"Component '{comp_name}' is not defined in 'components' dictionary.")

        comp_type = comp_info.get("type")
        comp_properties = comp_info.get("properties", {})
        theta = comp_properties.pop("theta", 0)  # Extract theta, default to 0

        # Initialize the component as a MuellerMatrix
        if comp_type == "Retarder":
            mm_function = cmm.general_retarder_function
        elif comp_type == "DiattenuatorRetarder":
            mm_function = cmm.diattenuator_retarder_function
        elif comp_type == "Rotator":
            mm_function = cmm.rotator_function
        elif comp_type == "WollastonPrism":
            mm_function = cmm.wollaston_prism_function
        else:
            raise ValueError(f"Unknown component type: {comp_type}")

        # Create a MuellerMatrix object
        mm = MuellerMat.MuellerMatrix(mueller_matrix_function=mm_function, name=comp_name)
        mm.properties.update(comp_properties)  # Update properties
        mm.properties["theta"] = theta  # Apply theta rotation retroactively
        mueller_components.append(mm)

    # Build the system Mueller matrix - TODO: return SystemMuellerMatrix directly
    sys_mm = MuellerMat.SystemMuellerMatrix(mueller_components)
    inst_matrix = sys_mm.evaluate()

    return inst_matrix

def internal_calibration_mueller_matrix(theta_pol, system_dict, HWP_angs, IMR_angs):
    """
    Returns the double differences, sums, and concatenated list of the two
    based on the Mueller matrix system for internal calibration.

    Args:
        theta_pol: (float) Angle of linear polarization (degrees).
        system_dict: (dict) A dictionary containing:
            - "components": A dictionary where each key is a component name
              (string), and each value is a dictionary specifying the properties
              of that component, including its type, relevant parameters, and tag.
            - "order": A list of component names specifying the order in which
              components appear in the system.
        HWP_angs: (list) List of half-wave plate angles.
        IMR_angs: (list) List of image rotator angles.

    Returns:
        model: A numpy array containing the flattened double differences,
               sums, and concatenated list of both.
    """
    # Extract only "internal" components from the system dictionary
    internal_components = {
        name: details
        for name, details in system_dict["components"].items()
        if details.get("tag") == "internal"
    }
    internal_order = [name for name in system_dict["order"] if name in internal_components]

    # Create a new system dictionary for internal calibration
    internal_system_dict = {
        "components": internal_components,
        "order": internal_order,
    }

    # Q, U from the input Stokes parameters
    Q, U = funcs.deg_pol_and_aolp_to_stokes(100, theta_pol)  # Assuming 100% polarization
    input_stokes = np.array([1, Q, U, 0]).reshape(-1, 1)  # I = 1, V = 0

    double_diffs = np.zeros([len(HWP_angs), len(IMR_angs)])
    double_sums = np.zeros([len(HWP_angs), len(IMR_angs)])

    # Iterate through HWP and IMR angles to calculate double differences and sums
    for i, HWP_ang in enumerate(HWP_angs):
        for j, IMR_ang in enumerate(IMR_angs):
            # Update angles for the system
            internal_system_dict["components"]["hwp"]["properties"]["theta"] = HWP_ang
            internal_system_dict["components"]["image_rotator"]["properties"]["theta"] = IMR_ang

            # Compute Mueller matrices for each FLC state and camera
            FL1_matrix = full_system_mueller_matrix(internal_system_dict)
            internal_system_dict["components"]["flc"]["properties"]["theta"] = 45  # Switch FLC state
            FR1_matrix = full_system_mueller_matrix(internal_system_dict)

            internal_system_dict["components"]["wollaston"]["properties"]["beam"] = "e"  # Camera 2
            FL2_matrix = full_system_mueller_matrix(internal_system_dict)
            internal_system_dict["components"]["flc"]["properties"]["theta"] = 0  # Reset FLC state
            FR2_matrix = full_system_mueller_matrix(internal_system_dict)

            # Calculate intensities
            FL1 = (FL1_matrix @ input_stokes)[0]
            FR1 = (FR1_matrix @ input_stokes)[0]
            FL2 = (FL2_matrix @ input_stokes)[0]
            FR2 = (FR2_matrix @ input_stokes)[0]

            # Compute double differences and sums
            double_diffs[i, j] = ((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))
            double_sums[i, j] = ((FL1 - FR1) + (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))

    # Flatten arrays and concatenate
    double_diffs = np.ndarray.flatten(double_diffs, order="F")
    double_sums = np.ndarray.flatten(double_sums, order="F")
    model = np.concatenate((double_diffs, double_sums))

    return model

def full_system_mueller_matrix_original( 
    delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP, delta_derot, 
    offset_derot, delta_opts, epsilon_opts, rot_opts, delta_FLC, rot_FLC, 
    em_gain, parang, altitude, HWP_ang, IMR_ang, cam_num, FLC_state):
    """
    Returns the double sum and differences based on the physical properties of
    the components for a variety of different wavelengths.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3 - fit from unpolarized standards
        offset_m3: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
        delta_HWP: (float) retardance of the HWP (waves)
        offset_HWP: (float) offset angle of the HWP (degrees)
        delta_derot: (float) retardance of the IMR (waves)
        offset_derot: (float) offset angle of the IMR (degrees)
        delta_opts: (float) retardance of the in-between optics (waves)
        epsilon_opts: (float) diattenuation of the in-between optics
        rot_opts: (float) rotation of the in-between optics (degrees)
        delta_FLC: (float) retardance of the FLC (waves)
        rot_FLC: (float) rotation of the FLC (degrees)
        em_gain: (float) ratio of the effective gain ratio of cam1 / cam2
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)
        cam_num: (int) camera number (1 or 2)
        FLC_state: (int) FLC state (1 or 2)

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system. 
        This matrix describes the change in polarization state as light passes 
        through the system.
    """

    # Parallactic angle rotation
    parang_rot = cmm.Rotator(name = "parang")
    parang_rot.properties['pa'] = parang

    # print("Parallactic Angle: " + str(parang_rot.properties['pa']))

    # One value for polarized standards purposes
    m3 = cmm.DiattenuatorRetarder(name = "m3")
    # TODO: Figure out how this relates to azimuthal angle
    m3.properties['theta'] = 0 ## Letting the parang and altitude rotators do the rotation
    m3.properties['phi'] = 2 * np.pi * delta_m3 ## FREE PARAMETER
    m3.properties['epsilon'] = epsilon_m3 ## FREE PARAMETER

    # Altitude angle rotation
    alt_rot = cmm.Rotator(name = "altitude")
    # Trying Boris' altitude rotation definition
    alt_rot.properties['pa'] = -(altitude + offset_m3)
 
    hwp = cmm.Retarder(name = 'hwp') 
    hwp.properties['phi'] = 2 * np.pi * delta_HWP 
    hwp.properties['theta'] = HWP_ang + offset_HWP
    # print("HWP Angle: " + str(hwp.properties['theta']))

    image_rotator = cmm.Retarder(name = "image_rotator")
    image_rotator.properties['phi'] = 2 * np.pi * delta_derot 
    image_rotator.properties['theta'] = IMR_ang + offset_derot

    optics = cmm.DiattenuatorRetarder(name = "optics") # QWPs are in here too. 
    optics.properties['theta'] = rot_opts 
    optics.properties['phi'] = 2 * np.pi * delta_opts 
    optics.properties['epsilon'] = epsilon_opts 

    flc = cmm.Retarder(name = "flc")
    flc.properties['phi'] = 2 * np.pi * delta_FLC 
    if FLC_state == 1: 
        # print("Entered FLC 1")
        flc.properties['theta'] = rot_FLC
        # print("FLC Angle: " + str(flc.properties['theta']))
    else:
        # print("Entered FLC 2")
        flc.properties['theta'] = rot_FLC + 45
        # print("FLC Angle: " + str(flc.properties['theta']))

    wollaston = cmm.WollastonPrism()
    if cam_num == 1:
        # print("Entered o beam")
        wollaston.properties['beam'] = 'o'
        # print(wollaston.properties['beam'])
    else:
        # print("Entered e beam")
        wollaston.properties['beam'] = 'e'
        # print(wollaston.properties['beam'])

    sys_mm = MuellerMat.SystemMuellerMatrix([wollaston, flc, optics, \
        image_rotator, hwp, alt_rot, m3, parang_rot])
        
    inst_matrix = sys_mm.evaluate()

    # Changing the intensity detection efficiency of just camera1
    if cam_num == 1:
        inst_matrix[:, :] *= em_gain

    return inst_matrix

def internal_calibration_mueller_matrix_original( 
    theta_pol, model, fixed_params, HWP_angs, IMR_angs):
    """
    Returns the double sum and differences based on the physical properties of
    the components for a variety of different wavelengths.

    Args:
        delta_m3: (float) retardance of M3 (waves)
        epsilon_m3: (float) diattenuation of M3 - fit from unpolarized standards
        offset_m3: (float) offset angle of M3 (degrees) - fit from M3 diattenuation fits
        delta_HWP: (float) retardance of the HWP (waves)
        offset_HWP: (float) offset angle of the HWP (degrees)
        delta_derot: (float) retardance of the IMR (waves)
        offset_derot: (float) offset angle of the IMR (degrees)
        delta_opts: (float) retardance of the in-between optics (waves)
        epsilon_opts: (float) diattenuation of the in-between optics
        rot_opts: (float) rotation of the in-between optics (degrees)
        delta_FLC: (float) retardance of the FLC (waves)
        rot_FLC: (float) rotation of the FLC (degrees)
        em_gain: (float) ratio of the effective gain ratio of cam1 / cam2
        parang: (float) parallactic angle (degrees)
        altitude: (float) altitude angle in header (degrees)
        HWP_ang: (float) angle of the HWP (degrees)
        IMR_ang: (float) angle of the IMR (degrees)

    Returns:
        inst_matrix: A numpy array representing the Mueller matrix of the system. 
        This matrix describes the change in polarization state as light passes 
        through the system.
    """

    # TODO: Make this loop through IMR and HWP angles

    # Q, U from the input Stokes parameters
    Q, U = funcs.deg_pol_and_aolp_to_stokes(100, theta_pol)

    # Assumed that I is 1 and V is 0
    input_stokes = np.array([1, Q, U, 0]).reshape(-1, 1)

    double_diffs = np.zeros([len(HWP_angs), len(IMR_angs)])
    double_sums = np.zeros([len(HWP_angs), len(IMR_angs)])

    # Take the observed intensities for each instrument state
    # NOTE: No parallactic angle or altitude rotation
    for i, HWP_ang in enumerate(HWP_angs):
        for j, IMR_ang in enumerate(IMR_angs):
            FL1_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 1, 1) 
            FR1_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang, 2, 1)
            FL2_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang,  1, 2)
            FR2_matrix = model(*fixed_params, 0, 0, HWP_ang, IMR_ang,  2, 2)

            FL1 = (FL1_matrix @ input_stokes)[0]
            FR1 = (FR1_matrix @ input_stokes)[0]
            FL2 = (FL2_matrix @ input_stokes)[0]
            FR2 = (FR2_matrix @ input_stokes)[0]

            double_diffs[i, j] = ((FL1 - FR1) - (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))
            double_sums[i, j] = ((FL1 - FR1) + (FL2 - FR2)) / ((FL1 + FR1) + (FL2 + FR2))

    double_diffs = np.ndarray.flatten(double_diffs, order = "F")
    double_sums = np.ndarray.flatten(double_sums, order = "F")
    model = np.concatenate((double_diffs, double_sums))

    return model

def compare_mueller_matrices(
    delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP,
    delta_derot, offset_derot, delta_opts, epsilon_opts,
    rot_opts, delta_FLC, rot_FLC, em_gain, parang, altitude,
    HWP_ang, IMR_ang, cam_num, FLC_state
):
    """
    Compares the Mueller matrix outputs of the original and updated functions.

    Args:
        Parameters are the same as the original full_system_mueller_matrix_original function.

    Returns:
        None. Prints the matrices and their difference.
    """

    # Call the original function
    inst_matrix_original = full_system_mueller_matrix_original(
        delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP,
        delta_derot, offset_derot, delta_opts, epsilon_opts,
        rot_opts, delta_FLC, rot_FLC, em_gain, parang, altitude,
        HWP_ang, IMR_ang, cam_num, FLC_state
    )

    # Map the parameters to the updated function's format
    optical_system = {
        "components": {
            "parang_rot": {
                "type": "Rotator",
                "properties": {"pa": parang}  # Parallactic angle in degrees
            },
            "m3": {
                "type": "DiattenuatorRetarder",
                "properties": {
                    "phi": 2 * np.pi * delta_m3,  # Retardance in radians
                    "epsilon": epsilon_m3         # Diattenuation
                }
            },
            "alt_rot": {
                "type": "Rotator",
                "properties": {"pa": -(altitude + offset_m3)}  # Altitude angle + offset
            },
            "hwp": {
                "type": "Retarder",
                "properties": {
                    "phi": 2 * np.pi * delta_HWP,  # Retardance in radians
                    "theta": HWP_ang + offset_HWP  # Orientation angle + offset
                }
            },
            "image_rotator": {
                "type": "Retarder",
                "properties": {
                    "phi": 2 * np.pi * delta_derot,  # Retardance in radians
                    "theta": IMR_ang + offset_derot  # Orientation angle + offset
                }
            },
            "optics": {
                "type": "DiattenuatorRetarder",
                "properties": {
                    "phi": 2 * np.pi * delta_opts,  # Retardance in radians
                    "epsilon": epsilon_opts,        # Diattenuation
                    "theta": rot_opts               # Rotation angle
                }
            },
            "flc": {
                "type": "Retarder",
                "properties": {
                    "phi": 2 * np.pi * delta_FLC,  # Retardance in radians
                    "theta": rot_FLC if FLC_state == 1 else rot_FLC + 45  # FLC state handling
                }
            },
            "wollaston": {
                "type": "WollastonPrism",
                "properties": {"beam": "o" if cam_num == 1 else "e"}  # Beam selection
            }
        },
        "order": ["wollaston", "flc", "optics", \
        "image_rotator", "hwp", "alt_rot", "m3", "parang_rot"]
    }

    # Call the updated function
    inst_matrix_updated = full_system_mueller_matrix(optical_system)

    # Print results and their differences
    print("Instrumental Mueller Matrix (Original):")
    print(inst_matrix_original)

    print("\nInstrumental Mueller Matrix (Updated):")
    print(inst_matrix_updated)

    print("\nDifference between the matrices:")
    print(inst_matrix_original - inst_matrix_updated)

def compare_internal_calibration(
    theta_pol, system_dict, fixed_params, HWP_angs, IMR_angs
):
    """
    Compares the outputs of the new and original internal_calibration_mueller_matrix functions.

    Args:
        theta_pol: (float) Angle of linear polarization (degrees).
        system_dict: (dict) A dictionary containing:
            - "components": A dictionary where each key is a component name
              (string), and each value is a dictionary specifying the properties
              of that component, including its type, relevant parameters, and tag.
            - "order": A list of component names specifying the order in which
              components appear in the system.
        fixed_params: (tuple) Parameters passed to the original function.
        HWP_angs: (list) List of half-wave plate angles.
        IMR_angs: (list) List of image rotator angles.

    Returns:
        None. Prints the results and differences.
    """
    # Unpack fixed_params to match full_system_mueller_matrix_original's signature
    (
        delta_m3, epsilon_m3, offset_m3, delta_HWP, offset_HWP,
        delta_derot, offset_derot, delta_opts, epsilon_opts,
        rot_opts, delta_FLC, rot_FLC, em_gain, parang, altitude
    ) = fixed_params

    # Define a wrapper function for the original model
    def original_model(HWP_ang, IMR_ang, cam_num, FLC_state):
        return full_system_mueller_matrix_original(
            [delta_HWP, offset_HWP, delta_derot, offset_derot, delta_opts, epsilon_opts,
            rot_opts, delta_FLC, rot_FLC, em_gain], IMR_ang, cam_num, FLC_state
        )

    # Call the original function with the wrapped model
    original_model_output = internal_calibration_mueller_matrix_original(
        theta_pol, original_model, fixed_params, HWP_angs, IMR_angs
    )

    # Call the new function
    new_model_output = internal_calibration_mueller_matrix(
        theta_pol, system_dict, HWP_angs, IMR_angs
    )

    # Print results
    print("Original Model Output:")
    print(original_model_output)

    print("\nNew Model Output:")
    print(new_model_output)

    # Compare the two models
    difference = original_model_output - new_model_output
    print("\nDifference (Original - New):")
    print(difference)
