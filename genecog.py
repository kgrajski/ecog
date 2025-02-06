import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd
import random
from scipy.stats import multivariate_normal

class ElectrodeArrayRecording:

    def __init__(self, idkey, num_samples, num_rows, num_cols):
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
        
            # Create a figure and axes
        fig, ax = plt.subplots()

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
        
    def to_csv(self, out_dir):
            # Create a file name and write a csv file for ecog (only).
        fname = os.path.join(out_dir + os.sep + self.idkey + ".csv")
        np.savetxt(fname,
                   self.ecog.reshape(self.ecog.shape[0], -1),
                   delimiter = ',')
        
    def tsplot(self, ix, iy):
            # Create the plot
        x = range(self.num_samples)
        y = self.ecog[:,ix,iy]
        plt.plot(x, y)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Time Series For: [" + str(ix) + ", " + str(iy) + "]")
        plt.show()
        
def gen_data_sample(class_label, isamp, irow, icol, out_dir, show=True, to_csv=True):
            # Set up the ECoG Electrode Array parameters
    idkey = "ecog_" + str(class_label) + "_" + str(isamp)
    num_samples = 32 # Lenght of time series
    num_rows = 64 # Medial to lateral
    num_cols = 32 # Rostral to caudal
    
            # Create the ECoG array
    ecog_array = ElectrodeArrayRecording(idkey, num_samples, num_rows, num_cols)
    
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

def main():

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
            gen_data_sample(class_label, isamp, i, j, out_dir, show=True, to_csv=True)
    
if __name__ == '__main__':
    main()

