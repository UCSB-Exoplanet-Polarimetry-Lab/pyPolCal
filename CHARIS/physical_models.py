import numpy as np
import pandas as pd
import cmath
import matplotlib.pyplot as plt
from pathlib import Path

# all functions adapted from Joost 't Hart 2021

#####################################
### Retrieving refraction indeces ###
#####################################

here = Path(__file__).resolve().parent


# Reading in all indices of refraction 
n_silver_list = pd.read_csv(here / "physical_model_csvs/silver_n.csv")
n_quartz_list = pd.read_csv(here / "physical_model_csvs/Gao.csv")
n_o_quartz_list = pd.read_csv(here / "physical_model_csvs/Ghosh-o.csv")
n_e_quartz_list = pd.read_csv(here / "physical_model_csvs/Ghosh-e.csv")
n_air_list = pd.read_csv(here / "physical_model_csvs/Ciddor.csv")
n_o_MgF2_list = pd.read_csv(here / "physical_model_csvs/MgF2_Dodge_n_o.csv")
n_e_MgF2_list = pd.read_csv(here / "physical_model_csvs/MgF2_Dodge_n_e.csv")

# For the imaginary part of silver's refractive index
k_silver_list = pd.read_csv(here / "physical_model_csvs/silver_k.csv")

# Turning all pandas dataframes 
n_silver_list = pd.DataFrame(n_silver_list).to_numpy()
n_quartz_list = pd.DataFrame(n_quartz_list).to_numpy()
n_o_quartz_list = pd.DataFrame(n_o_quartz_list).to_numpy()
n_e_quartz_list = pd.DataFrame(n_e_quartz_list).to_numpy()
n_air_list = pd.DataFrame(n_air_list).to_numpy()
n_o_MgF2_list = pd.DataFrame(n_o_MgF2_list).to_numpy()
n_e_MgF2_list = pd.DataFrame(n_e_MgF2_list).to_numpy()

# For the imaginary part of silver's refractive index
k_silver_list = pd.DataFrame(k_silver_list).to_numpy()

# Reformating the last value for float conversion - does not affect VAMPIRES model as not within notable
# wavelength range
n_quartz_list[500, 0] = 0
n_quartz_list[500, 1] = 0

# Converting all pandas dataframes into numpy arrays
n_silver_list = np.asarray(n_silver_list, dtype = float)
n_quartz_list = np.asarray(n_quartz_list, dtype = float)
n_o_quartz_list = np.asarray(n_o_quartz_list, dtype = float)
n_e_quartz_list = np.asarray(n_e_quartz_list, dtype = float)
k_silver_list = np.asarray(k_silver_list, dtype = float)
n_o_MgF2_list = np.asarray(n_o_MgF2_list, dtype = float)
n_e_MgF2_list = np.asarray(n_e_MgF2_list, dtype = float)

#####################################
### Fresnel Functions ##############
#####################################

def snell(n1, n2, theta):
    return np.arcsin(n1 * np.sin(theta) / n2)

def r_s(n1, n2, theta_i, theta_t):
    return (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))

def r_p(n1, n2, theta_i, theta_t):
    return (n2 * np.cos(theta_i) - n1 * np.cos(theta_t)) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))

def t_s(n1, n2, theta_i, theta_t):
    return 2 * n1 * np.cos(theta_i) / (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))

def t_p(n1, n2, theta_i, theta_t):
    return 2 * n1 * np.cos(theta_i) / (n2 * np.cos(theta_i) + n1 * np.cos(theta_t))

def r_tot(r01, r10, r12, t01, t10, beta):
    return r01 + (t01 * t10 * r12 * np.exp(beta * 1j)) / (1 - r10 * r12 * np.exp(beta * 1j))

# n is for quartz
def beta(wavelength, d, n, theta):
    return 2 * 2 * np.pi / wavelength * d * n * np.cos(theta)

def retardance(r_tot_s, r_tot_p):
    return cmath.phase(r_tot_s) - cmath.phase(r_tot_p)

#####################################
### Functions for Modeling #########
#####################################

def return_n(wavelength, material):
    """Return the refractive index of a material     at a given wavelength.
    
    Parameters
    ----------
    wavelength : float
        Wavelength in micrometers.
        
    material : str
        Material for which the refractive index is desired. Options include:
        'silver', 'quartz', 'quartz_n_e', 'quartz_n_o', 'air', 'MgF2_n_o', 'MgF2_n_e'.
        
    Returns
    -------
    n : complex
        The refractive index of the material at the specified wavelength.
    """
    if material == "silver":
        index = np.abs(n_silver_list[:, 0] - wavelength).argmin()
        n = n_silver_list[index, 1] + k_silver_list[index, 1] * 1j
    if material == "quartz":
        index = np.abs(n_quartz_list[:, 0] - wavelength).argmin()
        n = n_quartz_list[index, 1]
    if material == "quartz_n_e":
        n = (1 + 0.28851804 + 1.09509924 / (1 - 1.02101864e-2 / wavelength ** 2) + 1.15662475 / \
        (1 - 100 / wavelength ** 2)) ** .5
    if material == "quartz_n_o":
        n = (1 + 0.28604141 + 1.07044083 / (1 - 1.00585997e-2 / wavelength ** 2) + 1.10202242 / \
        (1 - 100 / wavelength ** 2)) ** .5
    if material == "air":
        n = 1 + 0.05792105 / (238.0185 - wavelength ** -2) + 0.00167917 / (57.362 - wavelength ** -2)
    if material == "MgF2_n_o":
        n = (1 + 0.48755108 / (1 - (0.04338408 / wavelength) ** 2) + 0.39875031 / (1 - (0.09461442 / \
        wavelength) ** 2) + 2.3120353 / (1 - (23.793604 / wavelength) ** 2)) ** .5
    if material == "MgF2_n_e":
        n = (1 + 0.41344023 / (1 - (0.03684262 / wavelength) ** 2) + 0.50497499 / (1 - (0.09076162 / \
        wavelength) ** 2) + 2.4904862 / (1 - (23.771995 / wavelength) ** 2)) ** .5
    return n

def HWP_retardance(wavelengths, w_SiO2=1.623, w_MgF2=1.268):
    '''
    Returns the HWP retardance based on eq. 23 from Joost 't Hart 2021

    Args:
        wavelengths: (float list) list of wavelengths (nm)
        w_SiO2: (float) width of SiO2 layer (mm), default is 1.623 mm from Joost 't Hart 2021
        w_MgF2: (float) width of MgF2 layer (mm), default is 1.268 mm from Joost 't Hart 2021

    Returns:
        retardance_list: (float list) list of all corresponding retardances (rad)
    '''

    # Converting input wavelength from nm to um
    wavelengths = np.atleast_1d(wavelengths) / 10 ** 3

    # Converting input widths from mm to um
    w_SiO2 = w_SiO2 * 10 ** 3
    w_MgF2 = w_MgF2 * 10 ** 3

    retardance_list = []
    
    for wavelength in wavelengths:
        retardance = 2 * np.pi / wavelength * (w_SiO2 * (return_n(wavelength, \
            "quartz_n_e") - return_n(wavelength, "quartz_n_o")) - w_MgF2 * \
            (return_n(wavelength, "MgF2_n_e") - return_n(wavelength, "MgF2_n_o")))
        if retardance < 0:
            retardance = -retardance
        retardance_list.append(retardance)

    retardance_list = np.array(retardance_list)
    

    return retardance_list

def IMR_retardance(wavelengths, d=262.56):
    """
    Returns the IMR retardance based on eq. 18 - 21 from Joost 't Hart 2021

    Args:
        wavelengths: (float list) incident wavelengths (nm)
        d: (float) thickness of the quartz layer (nm). Default is 262.56 nm, 
        from Joost 't Hart 2021.

    Returns:
        retardance_list: (float list) corresponding retardances (waves)
    """

    # Converting input wavelength from nm to um
    wavelengths = np.atleast_1d(wavelengths) / 10.0 ** 3
    d = d / 10.0 ** 3

    retardance_list = []

    for wavelength in wavelengths:
        n_silver = return_n(wavelength, "silver")
        n_quartz = return_n(wavelength, "quartz")
        n_air = return_n(wavelength, "air")

        # For the air to quartz transition: 30 degrees
        
        theta_t_01_30 = snell(n_air, n_quartz, np.radians(30))
        
        r_s_01_30 = r_s(n_air, n_quartz, np.radians(30), theta_t_01_30)
        r_p_01_30 = r_p(n_air, n_quartz, np.radians(30), theta_t_01_30)
        
        t_s_01_30 = t_s(n_air, n_quartz, np.radians(30), theta_t_01_30)
        t_p_01_30 = t_p(n_air, n_quartz, np.radians(30), theta_t_01_30)
        
        # For the air to quartz transition: 60 degrees
        
        theta_t_01_60 = snell(n_air, n_quartz, np.radians(60))
        
        r_s_01_60 = r_s(n_air, n_quartz, np.radians(60), theta_t_01_60)
        r_p_01_60 = r_p(n_air, n_quartz, np.radians(60), theta_t_01_60)
        
        t_s_01_60 = t_s(n_air, n_quartz, np.radians(60), theta_t_01_60)
        t_p_01_60 = t_p(n_air, n_quartz, np.radians(60), theta_t_01_60)
        
        # For the air to quartz transition: 30 degrees
        
        theta_t_10_30 = snell(n_quartz, n_air, np.radians(30))
        
        r_s_10_30 = r_s(n_quartz, n_air, theta_t_01_30, theta_t_10_30)
        r_p_10_30 = r_p(n_quartz, n_air, theta_t_01_30, theta_t_10_30)
        
        t_s_10_30 = t_s(n_quartz, n_air, theta_t_01_30, theta_t_10_30)
        t_p_10_30 = t_p(n_quartz, n_air, theta_t_01_30, theta_t_10_30)
        
        # For the air to quartz transition: 60 degrees
        
        r_s_10_60 = r_s(n_quartz, n_air, theta_t_01_60, np.radians(60))
        r_p_10_60 = r_p(n_quartz, n_air, theta_t_01_60, np.radians(60))
        
        t_s_10_60 = t_s(n_quartz, n_air, theta_t_01_60, np.radians(60))
        t_p_10_60 = t_p(n_quartz, n_air, theta_t_01_60, np.radians(60))
        
        # For the quartz to silver transition: 30 degrees
        
        theta_t_12_30 = snell(n_quartz, n_silver, theta_t_01_30)
        
        r_s_12_30 = r_s(n_quartz, n_silver, theta_t_01_30, theta_t_12_30)
        r_p_12_30 = r_p(n_quartz, n_silver, theta_t_01_30, theta_t_12_30)
        
        # For the quartz to silver transition: 60 degrees
        
        theta_t_12_60 = snell(n_quartz, n_silver, theta_t_01_60)
        
        r_s_12_60 = r_s(n_quartz, n_silver, theta_t_01_60, theta_t_12_60)
        r_p_12_60 = r_p(n_quartz, n_silver, theta_t_01_60, theta_t_12_60)
        
        # Betas for 30 and 60 degrees
        
        beta_30 = beta(wavelength, d, n_quartz, theta_t_01_30)
        beta_60 = beta(wavelength, d, n_quartz, theta_t_01_60)
        
        # r_tot for s and p polarization
        
        r_s_tot_30 = r_tot(r_s_01_30, r_s_10_30, r_s_12_30, t_s_01_30, t_s_10_30, \
            beta_30)
        r_s_tot_60 = r_tot(r_s_01_60, r_s_10_60, r_s_12_60, t_s_01_60, t_s_10_60, \
            beta_60)
        r_s_tot = r_s_tot_60 * r_s_tot_30 * r_s_tot_60
        
        r_p_tot_30 = r_tot(r_p_01_30, r_p_10_30, r_p_12_30, t_p_01_30, t_p_10_30, \
            beta_30)
        r_p_tot_60 = r_tot(r_p_01_60, r_p_10_60, r_p_12_60, t_p_01_60, t_p_10_60, \
            beta_60)
        r_p_tot = r_p_tot_60 * r_p_tot_30 * r_p_tot_60

        # Final retardance calculation
        
        def final_IMR_retardance(r_tot_s, r_tot_p):
            return cmath.phase(r_tot_s) - cmath.phase(r_tot_p)
        final_retardance = final_IMR_retardance(r_s_tot, r_p_tot)
        if final_retardance < 0:
            final_retardance = final_retardance + 2 * np.pi
        
        retardance_list.append(final_retardance)

    # print("r_s_tot: " + str(r_s_tot))
    # print("r_p_tot: " + str(r_p_tot))


    retardance_list = np.array(retardance_list)

    return retardance_list