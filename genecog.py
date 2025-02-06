import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import random
from scipy.stats import multivariate_normal

class ElectrodeArrayRecording:
    """
    Simulate ECoG (Electrocorticography) recordings from an electrode array.
    
    Attributes:
    -----------
    idkey : str
        Identifier key for the recording.
    num_samples : int
        Number of time samples in the recording.
    num_rows : int
        Number of rows in the electrode array.
    num_cols : int
        Number of columns in the electrode array.
    ecog : numpy.ndarray
        Simulated ECoG data with shape (num_samples, num_rows, num_cols).
    
    Methods:
    --------
    __init__(idkey, num_samples, num_rows, num_cols, noise_floor_mean, noise_floor_sd):
        Initializes the ElectrodeArrayRecording with specified parameters and generates
        simulated ECoG data with Gaussian noise.
    implot():
        Creates an animated plot of the ECoG data over time using matplotlib.
    """
    
    def __init__(self, idkey, num_samples, num_rows, num_cols):
        self.idkey = idkey
        self.num_samples = num_samples
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.ecog = np.zeros(shape=(num_samples, num_rows, num_cols))
        
    def gen_kernel(self, x0, y0, x_var, y_var, xy_var):
        distr = multivariate_normal(mean=np.array([x0, y0]),
                                    cov=np.array([[x_var, xy_var], [xy_var, y_var]]))
        kernel = np.zeros(shape=(self.num_rows, self.num_cols))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                kernel[i, j] = distr.pdf(np.array([i,j]))
        kernel = kernel / np.max(kernel)
        return kernel
        
    def add_activation(self,
                       x0, y0, x_var, y_var, xy_var,
                       t_start, t_end, ampl, tc_onset, tc_offset,
                       dt, position_wobble, ampl_wobble, var_wobble):

            # Update the ecog array
        for itx in range(t_start, self.num_samples):
            
                # At each time step, randomize (just a bit) the center and spread.
            x0_tmp = random.uniform(x0 - position_wobble, x0 + position_wobble)
            y0_tmp = random.uniform(y0 - position_wobble, y0 + position_wobble)
            x_var_tmp = random.uniform(x_var * (1-var_wobble), x_var * (1+var_wobble))
            y_var_tmp = random.uniform(y_var * (1-var_wobble), y_var * (1+var_wobble))
            xy_var_tmp = random.uniform(xy_var * (1-var_wobble), xy_var * (1+var_wobble))
            kernel = self.gen_kernel(x0_tmp, y0_tmp, x_var_tmp, y_var_tmp, xy_var_tmp)
            
                # At each time step, randominze (just a bit) the amplitude.
            amp_term = random.uniform(ampl - ampl_wobble, ampl + ampl_wobble)
            
            if itx < t_end:
                ampl_term = ampl * (1.0 - math.exp(-(itx - t_start) * dt / tc_onset))
            else:
                ampl_term = ampl * math.exp(-(itx - t_end) * dt / tc_offset)
                
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    self.ecog[itx, i, j] += ampl_term * kernel[i,j]

    def add_noise(self, noise_floor_mean, noise_floor_var):
        self.ecog += np.random.normal(loc=noise_floor_mean,
                                      scale=noise_floor_var,
                                      size=(self.num_samples, self.num_rows, self.num_cols))
        
    def implot(self, time_start, time_end, time_step):
        """
        Plots and animates ECoG data.
        This method creates a heatmap animation of ECoG data stored in the `self.ecog` attribute. 
        Each frame of the animation represents a different time point in the ECoG data.
        Attributes:
            self.ecog (numpy.ndarray): A 3D array where each slice along the first axis is a 2D array 
                                       representing ECoG data at a specific time point.
            self.idkey (str): A string used as the title of the plot.
            self.num_samples (int): The number of time points in the ECoG data.
        The method performs the following steps:
            1. Creates a figure and axes for the plot.
            2. Initializes the plot with the first time point of the ECoG data.
            3. Defines an update function to update the plot for each frame of the animation.
            4. Creates an animation using the update function.
            5. Displays the animation.
        """
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

            # Create the animation
        ani = FuncAnimation(fig,
                            update,
                            frames=np.arange(0, self.num_samples),
                            interval=200,
                            repeat=False,
                            blit=True)

            # Show the animation
        plt.show()
        
    def tsplot(self, ix, iy):
            # Create the plot
        x = range(self.num_samples)
        y = self.ecog[:,ix,iy]
        plt.plot(x, y)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Time Series For: [" + str(ix) + ", " + str(iy) + "]")
        plt.show()

def main():
    """
    Run the ECOG simulation.
    """
        # For reproducibility
    random.seed(42)
    
        # Set up the ECoG Electrode Array parameters
    idkey = "Baseline"
    num_samples = 64 # Lenght of time series
    num_rows = 32 # Medial to lateral
    num_cols = 32 # Rostral to caudal
    ecog_array = ElectrodeArrayRecording(idkey, num_samples, num_rows, num_cols)
    
        # Add a noise floor
    noise_floor_mean = 0
    noise_floor_var = 1
    ecog_array.add_noise(noise_floor_mean, noise_floor_var)
    #ecog_array.implot(0, ecog_array.num_samples, 1)
       
        # Add an activation
    x0 = 16 # Matrix position
    y0 = 16 # Matrix position
    position_wobble = 2 # In units of matrix positions
    x_var = 2
    y_var = 4
    xy_var = -1
    var_wobble = 0.1 # used to define an amplitude range 
    t_start = 8 # Number of time steps
    t_end = 16 # Number of time steps
    ampl = 10
    ampl_wobble = 0.10 # used to define an amplitude range
    tc_onset = 0.1 # Time constant in seconds
    tc_offset = 0.1 # Time constant in seconds
    dt = 0.05 # Time step in seconds
    print("Note: ECoG Array Elapsed Time = ", dt * ecog_array.num_samples)
    ecog_array.add_activation(x0, y0,
                              x_var, y_var, xy_var,
                              t_start, t_end, ampl, tc_onset, tc_offset,
                              dt, position_wobble, ampl_wobble, var_wobble)

        # Show the results
    ecog_array.implot(0, ecog_array.num_samples, 1)
    ecog_array.tsplot(x0, y0)
    
if __name__ == '__main__':
    main()

