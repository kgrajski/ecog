
import numpy as np
import os
import pandas as pd
import shutil
import time

#
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
    
        # Start timer
    start_time = time.perf_counter()
    print("*** " + script_name + "- START ***")
    
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
    batch_size = 4
    
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
        print('Train image shape', image_tensor.shape, label_tensor.shape)
        
        ecog_tensor, label_tensor = val_dataset.__getitem__(0)
        print('Train image shape', image_tensor.shape, label_tensor.shape)
        
        ecog_tensor, label_tensor = test_dataset.__getitem__(0)
        print('Train image shape', image_tensor.shape, label_tensor.shape)          

        print("TRAIN")
        train_batch = next(iter(train_dl))
        print('Batch shape',train_batch[0].shape)
        print('Batch shape',train_batch[1].shape)
        print('Batch shape',train_batch[2].shape)
        print('Batch shape',train_batch[3].shape)
        
        print("VAL")
        val_batch = next(iter(val_dl))
        print('Batch shape',val_batch[0].shape)
        print('Batch shape',val_batch[1].shape)
        print('Batch shape',val_batch[2].shape)
        print('Batch shape',val_batch[3].shape)
        
        print
        test_batch = next(iter(val_dl))
        print('Batch shape',test_batch[0].shape)
        print('Batch shape',test_batch[1].shape)
        print('Batch shape',test_batch[2].shape)
        print('Batch shape',test_batch[3].shape)
        
        #ecog_array = ECoGArrayRec().load(train_dataset.samples.path[0])
        #ecog_array.implot(0, ecog_array.num_samples, interval=50)
            
        # Wrap-up
    print(f"Total elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + "- END ***")
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()

