import numpy
import os

class ElectrodeArray:
    """
    Experimental ECoG array physical parameters.

    Attributes:
        num_rows (int): Number of rows in the electrode array.
        num_cols (int): Number of columns in the electrode array.
        pitch (float): Distance (mm) between adjacent electrodes.
        sampling_period (float): Time interval (sec) between samples.
    """
    def __init__(self, num_rows, num_cols, pitch, sampling_period):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.pitch = pitch
        self.sampling_period = sampling_period

def main():
    """
    Run the ECOG simulation.
    """
    
        # Set up the ECoG Electrode Array
    num_rows = 32 # Medial to lateral
    num_cols = 16 # Rostral to caudal
    pitch = 3.0 # In mm
    sampling_period = 0.001 # In sec
    ecog_array = ElectrodeArray(num_rows, num_cols, pitch, sampling_period)
    print([ecog_array.num_rows, ecog_array.num_cols, ecog_array.pitch, ecog_array.sampling_period])
    
        # Set up the cortical activation zones to be seen by the ECog Electrode Array
    
    
if __name__ == '__main__':
    main()



