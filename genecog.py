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
        
#
# Local Code
#
def gen_data_sample(class_label, isamp, irow, icol, out_dir, show=True, to_csv=True):
            # Set up the ECoG Electrode Array parameters
    idkey = "ecog_" + str(class_label) + "_" + str(isamp)
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
    if to_csv:
        ecog_array.to_csv(out_dir)

#
# MAIN
#
def main():

        # Set script name for console log
    script_name = "genecog"
    
        # Start timer
    start_time = time.perf_counter()

        # For reproducibility
    random.seed(42)

        #
        # Generate a labelled dataset.
        #   A "class" will map to a contiguous subset of the ECoG array.
        #   Each such subset is specified by a center and +/- range (in row and col directions).
        #   Sample individual activations within that area.
        #
    out_dir = "data"
    num_classes = 6
    num_samples_per_class = 1
    labels = range(num_classes)
    origin_row = [12, 32, 52]
    row_window = 8
    origin_col = [8, 24]
    col_window = 6
    locs = list(itertools.product(origin_row, origin_col))
    
        # Create sample data for each class.
    class_label = -1
    for irow, jcol in locs:
        class_label += 1
        for isamp in range(num_samples_per_class):
            i = random.randint(irow - row_window, irow + row_window)
            j = random.randint(jcol - col_window, jcol + col_window)
            gen_data_sample(class_label, isamp, i, j, out_dir, show=False, to_csv=True)
            
        # Wrap-up
    print(f"Total elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + "- END ***")
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()

