
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

#
# Get the ECoGDataSet class defs.
#
from ecogds import ECoGDataSet
from NN_Flat import NN_Flat

#
# Local Code
# Eventually move to a helper function file.
#
def train(dataloader, model, loss_fn, optimizer, device):
        # Taken from PyTorch Getting Started
    size = len(dataloader.dataset)
    model.train()
    avg_batch_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_batch_loss += loss_fn(pred, y).item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    avg_batch_loss /= size
    return avg_batch_loss

def test(dataloader, model, loss_fn, device):
        # Taken from PyTorch Getting Started
    size = len(dataloader.dataset)
    model.eval()
    avg_batch_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_batch_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_batch_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avg_batch_loss:>8f} \n")
    return avg_batch_loss, correct 
    
def zero_mean_unit_sd(x):
    mean = x.mean()
    std = x.std()
    y = (x - mean) / std
    return y

#
# Reminder: nvidia-smi --id=0 --loop=30 --query --display=UTILIZATION
#

#
# Reminder: To view, start TensorBoard on the command line with:
#   tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/

#
# MAIN
#
def main():

        # Set script name for console log
    script_name = "devmodel"
    
        # Start timer
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***\n")
    
        #
        # Determine device availability
        #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device=", device)

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
    batch_size = 60
    
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
    
        #
        # Make a study dataset
        #   Note: Choice of loss function could require one-hot encoding of the label vector.
        #
    study_dataset = ECoGDataSet(study_dir, study_list_fname, label_list_fname, preproc_dir,
                                transform=zero_mean_unit_sd, target_transform=None
                                #target_transform=lambda y: F.one_hot(y, num_classes=study_dataset.num_classes)
                                )

        #
        # Make the train, validation, and test splits
        #
    train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])

        #
        # Run DataLoader
        #
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        #
        # OPTIONAL: Quick check
        #
    if 0:
        
        print("** SUMMARY **", study_dataset.samples.groupby(['label'])['label'].count())
     
        ecog_tensor, label_tensor = train_dataset.__getitem__(0)
        print('Train ECoG shape', ecog_tensor.shape, label_tensor.shape)        

        print("TRAIN")
        train_batch = next(iter(train_dl))
        print('Batch shape',train_batch[0].shape)
        
        if 0:
            for batch_idx, (batch_data, batch_labels) in enumerate(val_batch):
                print(f"Batch index: {batch_idx}")
                print("Batch data shape:", batch_data.shape)
                print("Batch labels:", batch_labels)
                print("Batch labels shape:", batch_labels.shape)
    
        #
        #   Model Training
        #
        
            #
            # Set up tensorboard
            # Default log_dir argument is "runs" - but it's good to be specific
            # torch.utils.tensorboard.SummaryWriter is imported above
            #
    writer = SummaryWriter('runs/dev_ecog')
        
            #
            #   Set up the model.
            #
    input_dim = 32 * 64 * 32
    hidden_dim = 16
    output_dim = 6
    num_layers = 2
    model = NN_Flat(input_dim, hidden_dim, num_layers, output_dim).to(device)
    print(model)
    
            #
            #   Select Loss Function and Optimizer
            #
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
            #
            #   Measure performance on the untrained network.
            #
    if 0:
        test(test_dl, model, loss_fn, device)
    
            #
            #   Do the training.
            #
    num_epochs = 5
    for iter in range(num_epochs):
        print(f"Epoch {iter+1}\n-------------------------------")
        train_avg_batch_loss = train(train_dl, model, loss_fn, optimizer, device)
        print('Test Set')
        test_avg_batch_loss, correct = test(test_dl, model, loss_fn, device)
                    # Log the running loss averaged per batch
        writer.add_scalars('Training vs Test Loss and Correct',
                           {'Train' : train_avg_batch_loss, 'Test' : test_avg_batch_loss,
                            'Correct' : correct}, iter)
    
    print('Validation Set')
    test(val_dl, model, loss_fn, device)
    print("Done!")
    
        #
        # Wrap-up
        #
    print(f"\nTotal elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
    print("*** " + script_name + " - END ***")
            
#
# EXECUTE
#
if __name__ == '__main__':
    main()
    
    