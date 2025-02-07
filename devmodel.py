#
# dev_model.py
#

#
# KAG 26Dec2024
#

#
# References
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://medium.com/@daminininad/intro-to-ai-with-neuroimaging-data-a-end-to-end-tutorial-using-pytorch-f941c6ef547a
# https://medium.com/@jieyoushen/pytorch-nifti-image-data-loader-c41eb4815f08
#

#
# Purpose: Development version 3D Swin Transforer Model for MRI.
#
import ants
from helpers import *

import numpy as np
import os
import pandas as pd
import shutil
import time

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, random_split, Dataset

#
# Constants
#
val_prop = 0.2
test_prop = 0.2
train_prop = 1 - val_prop - test_prop

numpy_seed = 412938
torch_seed = 293487

batch_size = 4

#
# Main
#
script_name = 'dev_model.py'
print("*** " + script_name + "- START ***")

#
# Set the location of the input and output directories
#  We start with nii (preproc) data.
#
preproc_dir = "/home/ubuntu/adni/data/preproc"

#
# Set the input list of images to convert.
#
study_dir = "/home/ubuntu/adni/labbench/dev/"
study_list_fname = "study.txt"
label_list_fname = "labels.csv"

#
# START CLASSES DEFINIION
#
class adni_mri(Dataset):
    """
    PyTorch custom Dataset tuned for ADNI MRI.

    Args:
        Dataset (_type_): _description_
    """
    
    def __init__(self, study_dir, study_list_fname, label_list_fname, preproc_dir, transform=None, target_transform=None):
        self.samples = self.gen_dataset(study_dir, study_list_fname, label_list_fname, preproc_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples.path[idx], self.samples.label[idx]
        print("GET_ITEM=",idx,image_path,label)
        img_ants = ants.image_read(image_path)
        image_tensor = torch.tensor(img_ants.numpy(), dtype=torch.float32).unsqueeze(0) #unsqueeze to add channel dimension
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)
        return image_tensor, label_tensor
    
    def gen_dataset(self, study_dir, study_list_fname, label_list_fname, preproc_dir):
            # Read the list of images in this study
        df_images = pd.read_csv(os.path.join(study_dir, study_list_fname), sep=',', header=0)
            # Read the DX file and do some ADNI-specific processing and map the DX to numbers
        df_labels = pd.read_csv(os.path.join(study_dir, label_list_fname), sep=',', header=0)
        df_labels['tmp_image_id'] = df_labels['image_id'].apply(lambda x: "I" + str(x))
        df_labels['image_id'] = df_labels['tmp_image_id']
        df_labels['label_id'] = df_labels['label'].apply(lambda x: self.dx_code(x))
        df_labels['label'] = df_labels['label_id']
        df_labels = df_labels.drop(['label_id','tmp_image_id'], axis=1)
        df = []
        for row in df_images.itertuples():
                # Create a path
            path = os.path.join(preproc_dir, row.image_id, row.image_id + ".brain_masked.raw.nii.gz")
                # Edit the row.image_id and use that to look up the label (later as a join)
            df.append([row.image_id, path])
        df = pd.DataFrame(df,columns=['image_id','path'])
        df = pd.merge(df, df_labels, on='image_id', how='left').reset_index()
        return df[['path','label']]
    
    def dx_code(self, x):
        if x == 'CN':
            y = 0
        elif x == 'EMCI':
            y = 1
        elif x == 'LMCI':
            y = 2
        elif x == 'AD':
            y = 3
        else:
            y = -1
        return y
    
#
# END CLASSES DEFINIION
#

# 
# MAIN
#
start_time = time.perf_counter()

# Set the random seeds
np.random.seed(numpy_seed)
torch.manual_seed(torch_seed)

# Make a study dataset
study_dataset = adni_mri(study_dir, study_list_fname, label_list_fname, preproc_dir)

# Some quick summaries
print("** SUMMARY **", study_dataset.samples.groupby(['label'])['label'].count())

# Make the train, validation, and test splits
train_dataset, val_dataset, test_dataset = random_split(study_dataset, [train_prop, val_prop, test_prop])

# Run DataLoader
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# OPTIONAL: Quick check
if 1:
    ants_image = ants.image_read(study_dataset.samples.path[16])
    ants.plot(ants_image, figsize=3, axis=1, title="Raw")
    print('Raw image shape',study_dataset.samples.path[16], ants_image.numpy().shape)              
    
    image_tensor, label_tensor = train_dataset.__getitem__(0)
    print('Train image shape', image_tensor.shape, label_tensor.shape)                

    train_batch = next(iter(train_dl))
    print('Batch shape',train_batch[0].shape)
    print('Batch shape',train_batch[1].shape)
    print('Batch shape',train_batch[2].shape)
    print('Batch shape',train_batch[3].shape)             

# Wrap-up
print(f"Total elapsed time:  %.4f seconds" % (time.perf_counter() - start_time))
print("*** " + script_name + "- END ***")