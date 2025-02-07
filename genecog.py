'''
Module: genecog
This module generates synthetic ECoG (Electrocorticography) data samples for different classes and saves them to CSV files. 
The generated data includes noise and activations with specified parameters.
Functions:
    gen_data_sample(class_label, isamp, irow, icol, out_dir, show=True, to_csv=True):
        Generates a single ECoG data sample with noise and activation, and optionally saves it to a CSV file and displays it.
    main():
        Main function to generate a labeled dataset of ECoG data samples for multiple classes and save the dataset information to CSV files.
Usage:
    Run this module as a script to generate the ECoG dataset and save the results to the specified output directory.
'''

import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd
import random
from scipy.stats import multivariate_normal
import time

#
# Get the ECoG class defs.
#
from ecog import ECoGArrayRec
        
#s
# Local Code
#
def gen_data_sample(class_label, isamp, irow, icol, out_dir, show=True, save=True):
            # Set up the ECoG Electrode Array parameters
    idkey = 'ecog_' + str(class_label) + '_' + str(isamp)
    num_samples = 32 # Lenght of time series
    num_rows = 64 # Medial to lateral
    num_cols = 32 # Rostral to caudal
    
            # Create the ECoG array
    ecog_array = ECoGArrayRec(idkey, num_samples, num_rows, num_cols)
    
            # Add a noise floor
    noise_floor_mean = 0
    noise_floor_var = 1
    ecog_array.add_noise(noise_floor_mean, noise_floor_var)
    #ecog_array.implot(0, ecog_array.num_samples, 1)
    
            # Add an activation
    x0 = irow # Matrix position
    y0 = icol # Matrix position
    position_wobble = 2 # In units of matrix positions
    x_var = 2
    y_var = 4
    xy_var = -1
    var_wobble = 0.1 # used to define an amplitude range 
    t_start = 8 # Number of dt time steps
    t_end = 16 # Number of dt time steps
    window_wobble = 4 # Number of dt time steps
    ampl = 10
    ampl_wobble = 0.10 # used to define an amplitude range
    tc_onset = 0.1 # Time constant in seconds
    tc_offset = 0.1 # Time constant in seconds
    dt = 0.05 # Time step in seconds
    ecog_array.add_activation(x0, y0,
                              x_var, y_var, xy_var,
                              t_start, t_end, ampl, tc_onset, tc_offset,
                              dt, position_wobble, ampl_wobble, var_wobble, window_wobble)

        # Show the results
    if show:
        ecog_array.implot(0, ecog_array.num_samples, interval=50)
        
        # Write the results (reshaped)
        # Subsequent readers (e.g., DataLoader) will need to know about the reshape.
    if save:
        ecog_array.save(out_dir)
        
        # Return the idkey
    return ecog_array.idkey

#
# MAIN
#
def main():

        # Set script name for console log
    script_name = 'genecog'
    
        # Start timer
    start_time = time.perf_counter()
    print('*** ' + script_name + '- START ***')

        # For reproducibility
    np.random.seed(42)

        #
        # Generate a labelled dataset.
        #   A 'class' will map to a contiguous subset of the ECoG array.
        #   Each such subset is specified by a center and +/- range (in row and col directions).
        #   Sample individual activations within that area.
        #
    out_dir = 'data'
    num_classes = 6
    num_samples_per_class = 100
    origin_row = [12, 32, 52]
    row_window = 8
    origin_col = [8, 24]
    col_window = 6
    locs = list(itertools.product(origin_row, origin_col))
    
        # Create sample data for each class.
    idkeys_study = []
    labels_study = []
    class_label = -1
    for irow, jcol in locs:
        class_label += 1
        for isamp in range(num_samples_per_class):
            i = random.randint(irow - row_window, irow + row_window)
            j = random.randint(jcol - col_window, jcol + col_window)
            idkey = gen_data_sample(class_label, isamp, i, j, out_dir, show=False, save=True)
            labels_study.append([idkey, class_label])
            idkeys_study.append([out_dir, idkey])
    
        # Write the idkeys_study an labels_study files.
    idkeys_study = pd.DataFrame(idkeys_study, columns=['path', 'idkey'])
    idkeys_study.to_csv(os.path.join(out_dir + os.sep + 'study.csv'), index=False)
    labels_study = pd.DataFrame(labels_study, columns=['idkey', 'label'])
    labels_study.to_csv(os.path.join(out_dir + os.sep + 'labels.csv'), index=False)
            
        # Wrap-up
    print(f'Total elapsed time:  %.4f seconds' % (time.perf_counter() - start_time))
    print('*** ' + script_name + '- END ***')
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()

