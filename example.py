### 
## Data is stored in a csv that includes the single differences and sums, and their errors. 
# As well as the instrument configuration for each one. 
# There will be n rows of measurements (by definition this is an even number, since it incldues
# both the single differences and the single sums).

# Step 1 reads in the csv. 


# Step 2: Define a system Muller Matrix object
# This will include all of the optical components. And set any fixed values .

# Step 3: 
    # For each measurement define a configuration list, where each entry in the list
    # is a dictionary of the optical components and their values that change with each 
    # measurement. 
    # This is called "configuration_list"
    # This list will have a length of n
    
# Step 4: 
    # Decide which parameters we're fitting by creating a dictionary of p0 starting guesses. 
    #     Example p0:
    #  p0 = {'Polarizer': {'Theta': 0},
    #       'Waveplate': {'Phi': 0},
    #       'Waveplate': {'Theta': 0},
    #       'Retarder': {'Theta': 0},
    #       'Sample': {''Phi': 0},
    #       'Analyzer': {'Phi': 0}}

# Step 5: 
    # Pass this all in to minimize_system_mueller_matrix. 

