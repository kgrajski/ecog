"""
ecog.py

This module defines the ECoGArrayRec class, which represents an Electrocorticography (ECoG) array recording. The class provides methods to manipulate and visualize ECoG data, including adding activation patterns, adding noise, generating Gaussian kernels, plotting animations, saving and loading data, and plotting time series data for specific electrode positions.

Classes:
--------
ECoGArrayRec
    A class to represent an ECoG (Electrocorticography) array recording.

    Attributes:
    -----------
    idkey : str
        Identifier key for the recording.
    num_samples : int
        Number of time samples in the recording.
    num_rows : int
        Number of rows in the ECoG array.
    num_cols : int
        Number of columns in the ECoG array.
    ecog : np.ndarray
        3D numpy array to store the ECoG data.

    Methods:
    --------
    __init__(self, idkey='', num_samples=0, num_rows=0, num_cols=0)
        Initializes the ECoGArrayRec object with the given parameters.

    add_activation(self, x0, y0, x_var, y_var, xy_var, t_start, t_end, ampl, tc_onset, tc_offset, dt, position_wobble, ampl_wobble, var_wobble, window_wobble)
        Adds an activation pattern to the ECoG array.

    add_noise(self, noise_floor_mean, noise_floor_var)
        Adds Gaussian noise to the ECoG array.

    gen_kernel(self, x0, y0, x_var, y_var, xy_var)
        Generates a Gaussian kernel for a given set of parameters.

    implot(self, time_start, time_end, time_step=1, interval=200)
        Plots an animation of the ECoG data over a specified time range.

    load(self, fname)
        Loads ECoG data from a CSV file.

    save(self, out_dir)
        Saves the ECoG data to a CSV file.

    tsplot(self, ix, iy)
        Plots the time series data for a specific electrode position.
"""

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

class ECoGArrayRec:
    '''
    A class to represent an ECoG (Electrocorticography) array recording.
    Attributes
    ----------
    idkey : str
        Identifier key for the recording.
    num_samples : int
        Number of time samples in the recording.
    num_rows : int
        Number of rows in the ECoG array.
    num_cols : int
        Number of columns in the ECoG array.
    ecog : np.ndarray
        3D numpy array to store the ECoG data.
    Methods
    -------
    add_activation(x0, y0, x_var, y_var, xy_var, t_start, t_end, ampl, tc_onset, tc_offset, dt, position_wobble, ampl_wobble, var_wobble, window_wobble):
        Adds an activation pattern to the ECoG array.
    add_noise(noise_floor_mean, noise_floor_var):
        Adds Gaussian noise to the ECoG array.
    gen_kernel(x0, y0, x_var, y_var, xy_var):
        Generates a Gaussian kernel for a given set of parameters.
    implot(time_start, time_end, time_step=1, interval=200):
        Plots an animation of the ECoG data over a specified time range.
    save(out_dir):
        Saves the ECoG data to a CSV file.
    tsplot(ix, iy):
        Plots the time series data for a specific electrode position.
    '''

    def __init__(self, idkey='', num_samples=0, num_rows=0, num_cols=0):
        self.idkey = idkey
        self.num_samples = num_samples
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.ecog = np.zeros(shape=(num_samples, num_rows, num_cols))
        
    def add_activation(self,
                       x0, y0, x_var, y_var, xy_var,
                       t_start, t_end, ampl, tc_onset, tc_offset,
                       dt, position_wobble, ampl_wobble, var_wobble, window_wobble):

            # Randomize the start and end times (just a bit)
        t_start = random.randint(t_start - window_wobble, t_start)
        t_end = random.randint(t_end, t_end + window_wobble)
        
            # Update the ecog array
        for itx in range(t_start, self.num_samples):
            
                # At each time step, randomize (just a bit) the center and spread.
            x0_tmp = random.uniform(x0 - position_wobble, x0 + position_wobble)
            y0_tmp = random.uniform(y0 - position_wobble, y0 + position_wobble)
            x_var_tmp = random.uniform(x_var * (1-var_wobble), x_var * (1+var_wobble))
            y_var_tmp = random.uniform(y_var * (1-var_wobble), y_var * (1+var_wobble))
            xy_var_tmp = random.uniform(xy_var * (1-var_wobble), xy_var * (1+var_wobble))
            
                # Generate the basic kernel for this time slice.
            kernel = self.gen_kernel(x0_tmp, y0_tmp, x_var_tmp, y_var_tmp, xy_var_tmp)
            
                # At each time step, randominze (just a bit) the amplitude.
            amp_term = random.uniform(ampl - ampl_wobble, ampl + ampl_wobble)
            
                # Adjust the amplitude for rise time and fall time.
            if itx < t_end:
                ampl_term = ampl * (1.0 - math.exp(-(itx - t_start) * dt / tc_onset))
            else:
                ampl_term = ampl * math.exp(-(itx - t_end) * dt / tc_offset)
                
                # For the given time slice create the spatial pattern
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    self.ecog[itx, i, j] += ampl_term * kernel[i,j]

    def add_noise(self, noise_floor_mean, noise_floor_var):
        self.ecog += np.random.normal(loc=noise_floor_mean,
                                      scale=noise_floor_var,
                                      size=(self.num_samples, self.num_rows, self.num_cols))
        
    def gen_kernel(self, x0, y0, x_var, y_var, xy_var):
        distr = multivariate_normal(mean=np.array([x0, y0]),
                                    cov=np.array([[x_var, xy_var], [xy_var, y_var]]))
        kernel = np.zeros(shape=(self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                kernel[i, j] = distr.pdf(np.array([i,j]))
        kernel = kernel / np.max(kernel)
        return kernel
    
    def implot(self, time_start, time_end, time_step=1, interval=200):
        
            # Create a new figure and axes explicitly
        fig = plt.figure()
        ax = fig.add_subplot(111)

            # Initialize the plot
        im = ax.imshow(self.ecog[0], cmap='hot', origin='upper')
        cbar = plt.colorbar(im)
        ax.set_title(self.idkey)

            # Animation update function
        def update(itx):
            im.set_data(self.ecog[itx])
            return [im]

        ani = FuncAnimation(fig,
                            update,
                            frames=np.arange(time_start, time_end, time_step),
                            interval=interval,
                            repeat=False,
                            blit=True)
        plt.show()

    def load(self, fname):
            # Read a file name and load a csv file for ecog.
        with open(fname, 'r') as f:
            self.idkey = f.readline().strip()
            self.num_samples, self.num_rows, self.num_cols = map(int, f.readline().strip().split(','))
            self.ecog = np.loadtxt(f, delimiter=",")
            self.ecog = self.ecog.reshape(self.num_samples, self.num_rows, self.num_cols)

    def save(self, out_dir):
            # Create a file name and write a csv file.
        idkey = self.idkey
        fname = os.path.join(out_dir + os.sep + self.idkey + '.csv')
        with open(fname, 'w') as f:
            f.write(f"{idkey}\n")
            ix = [str(x) for x in (self.num_samples, self.num_rows, self.num_cols)]
            f.write(', '.join(ix) + '\n')
            np.savetxt(f, self.ecog.reshape(self.ecog.shape[0], -1), fmt='%.4f', delimiter=',')
        return idkey
        
    def tsplot(self, ix, iy):
            # Create a new figure
        plt.figure()
            # Create the plot
        x = range(self.num_samples)
        y = self.ecog[:,ix,iy]
        plt.plot(x, y)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Time Series For: [' + str(ix) + ', ' + str(iy) + ']')
        plt.draw()