
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Get the ECoGDataSet class defs.
#
from ecogds import ECoGDataSet

#
# Local Code
#

#
# MAIN
#
def main():

        # Set script name for console log
    script_name = "devmodel"
    
        #
        # Determine device availability
        #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device=", device)
    
        # Start timer
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")
    
        # For reproducibility
    numpy_seed = 412938
    torch_seed = 293487
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    
        #
        # Learning Parameters
        #
    val_prop = 0.2
    test_prop = 0.2
    train_prop = 1 - val_prop - test_prop
    batch_size = 8
    
        #
        # Directory containing the actual data.
        #
    preproc_dir = "data/"

        #
        # Directory containing the files descriving the data to work with
        #
    study_dir = "data/"
    study_list_fname = "study.csv"
    label_list_fname = "labels.csv"
    
        # Make a study dataset
    study_dataset = ECoGDataSet(study_dir, study_list_fname, label_list_fname, preproc_dir)
    print("** SUMMARY **", study_dataset.samples.groupby(['label'])['label'].count())

        # Make the train, validation, and test splits
    train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])

        # Run DataLoader
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        # OPTIONAL: Quick check
    if 1:
     
        ecog_tensor, label_tensor = train_dataset.__getitem__(0)
        print('Train ECoG shape', ecog_tensor.shape, label_tensor.shape)
        
        ecog_tensor, label_tensor = val_dataset.__getitem__(0)
        print('Val ECoG shape', ecog_tensor.shape, label_tensor.shape)
        
        ecog_tensor, label_tensor = test_dataset.__getitem__(0)
        print('Test ECoG shape', ecog_tensor.shape, label_tensor.shape)          

        print("TRAIN")
        train_batch = next(iter(train_dl))
        print('Batch shape',train_batch[0].shape)
        
        print("VAL")
        val_batch = next(iter(val_dl))
        print('Batch shape',val_batch[0].shape)
        
        print("TEST")
        test_batch = next(iter(test_dl))
        print('Batch shape',test_batch[0].shape)
            
        # Wrap-up
    print(f"Total elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()

