import numpy as np
import copy

def update_systemMM(parameter_values, systemMM, system_parameters):
    '''
    Generates a model dataset for the given set of parameters

    Args: 
    p: list of parameters 
    systemMM: pyMuellerMat System Mueller Matrix object
    system_parameters: list of two element lists [component_name, parameter_name]
                        example: [['Polarizer', 'Theta'], ['Polarizer', 'Phi']]
    '''

    #Set up the System Mueller Matrix
    for parameter in system_parameters:
        #TODO: Check that this parameter is IN systemMM and if not, don't try to set it. 
        systemMM.set_parameter(parameter, parameter_values[parameter])
    return systemMM.get_M()

def generate_measurement(systemMM, S_in=[1,0,0,0]):
    '''
    Generate a measurement from a given System Mueller Matrix

    Args: 
    systemMM: pyMuellerMat System Mueller Matrix object
    S_in: Stokes vector of the incoming light

    Returns: 
    S_out: Stokes vector of the outgoing light
    '''
    return systemMM.get_S_out(S_in) #This is probably systeMM.evaluate@np.array(S_in)

def mcmc_system_mueller_matrix(p0, systemMM, dataset, errors, configuration_list):
    '''
    Perform MCMC on a dataset, using a System Mueller Matrix model

    Example p0: 

    p0 = {'Polarizer': {'Theta': 0, 'Phi': 0},
          'Waveplate': {'Theta': 0, 'Phi': 0},
          'Retarder': {'Theta': 0, 'Phi': 0},
          'Sample': {'Theta': 0, 'Phi': 0},
          'Analyzer': {'Theta': 0, 'Phi': 0}}

    Args: 
    p0: dictionary of dictionaries of initial parameters
    systemMM: pyMuellerMat System Mueller Matrix object
    dataset: list of measurements
    configuration_list: list of system dictionaries, one for each measurement in dataset
    '''

    #####################################################
    ###### First, define the data model based on p0 #####
    #####################################################

    #Parse p0 into a list of values and a list of lists of keywords
    p0_values = []
    p0_keywords = []
    for component, parameters in p0.items():
        for parameter, value in parameters.items():
            p0_values.append(value)
            p0_keywords.append([component, parameter])

    #TODO: Fill this out more. 


def minimize_system_mueller_matrix(p0, systemMM, dataset, errors, configuration_list):
    '''
    Perform a minimization on a dataset, using a System Mueller Matrix model'''

    p0_values, p0_keywords = parse_configuration(p0)

    result = minimize(logl, p0_values, args=(p0_keywords, systemMM, dataset, errors, configuration_list), method='Nelder-Mead')


def parse_configuration(configuration):
    '''
    Parse a configuration dictionary into a list of values and a list of lists of keywords

    Args: 
    configuration: dictionary of dictionaries of initial parameters

    Returns: 
    values: list of values
    keywords: list of lists of keywords
    '''
    values = []
    keywords = []
    for component, parameters in configuration.items():
        for parameter, value in parameters.items():
            values.append(value)
            keywords.append([component, parameter])
    return values, keywords

def logl(p, system_parameters, systemMM, dataset, errors, configuration_list, logl_function = None, process_dataset = None, process_model = None):
    '''
    Log likelihood function for MCMC

    Args: 
    p: list of parameters
    system_parameters: the list of dictionaries that define what the p parameters are. 
    systemMM: pyMuellerMat System Mueller Matrix object
    dataset: list of measurements

    Returns: 
    logl: log likelihood
    '''
    #Update the system Mueller matrix with parameters that we're fitting
    systemMM = update_systemMM(p, systemMM, system_parameters)

    #Generate a model dataset
    output_intensities = []
    for i in range(len(dataset)):
        configuration = parse_configuration(configuration_list[i])
        systemMM = update_systemMM(configuration[0], systemMM, configuration[1])
        S_out = generate_measurement(systemMM, dataset[i][0])
        output_intensities.append(S_out[0])

    #Optionally parse the dataset and output intensities (e.g. a normalized difference)
    if process_dataset is not None:
        dataset = process_dataset(copy.deepcopy(dataset))
    if process_model is not None:
        output_intensities = process_model(output_intensities)

    #Calculate the log likelihood - optionally with your own logl function (e.g. to include an extra noise term in p)
    if logl_function is not None:
        return logl_function(p, output_intensities, dataset, errors)
    else: 
        return -0.5 * np.sum((np.array(output_intensities) - np.array(dataset))**2 / np.array(errors)**2)
    

    def build_differences_and_sums(intensities):
        '''
        Assume that the input intensities are organized in pairs. Such that
        '''

        differences = intensities[::2]-intensities[1::2]
        sums = intensities[::2]+intensities[1::2]

        return differences, sums
    
    def build_double_differences_and_sums(differences, sums):
        '''
        Assume that the input intensities are organized in pairs. Such that
        '''

        double_differences = (differences[::2]-differences[1::2])/(sums[::2]+sums[1::2])
        double_sums = (sums[::2]-sums[1::2])/(sums[::2]+sums[1::2])

        return double_differences, double_sums
    

    def process_model(model_intensities):

        differences, sums = build_differences_and_sums(model_intensities)

        double_differences, double_sums = build_double_differences_and_sums(differences, sums)

        #Format this into one array. 
        return output
    
    def process_dataset(input_dataset): 

        differences = input_dataset[::2]
        sums = input_dataset[1::2]

        double_differences, double_sums = build_double_differences_and_sums(differences, sums)

        #Format this into one array.
        return output 