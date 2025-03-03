"""
NN_Conv3D_Simple.py

This module defines the NN_Conv3D_Simple class, which is a simple 3D convolutional neural network implemented using PyTorch. The network consists of a 3D convolutional layer, followed by fully connected layers. It also includes methods for forward propagation.

Classes:
--------
NN_Conv3D_Simple
    A simple 3D convolutional neural network with specified input dimensions, fully connected layer dimensions, and output dimensions.

    Methods:
    --------
    __init__(self, in_depth, in_rows, in_cols, fc_dim, output_dim, pcnt_dropout=0.15)
        Initializes the NN_Conv3D_Simple object with the given parameters and constructs the network layers.

    _conv_layer_set(self, in_c, out_c, conv3d_kernel_size, max_pool_kernel_size)
        Creates a sequential container of a 3D convolutional layer, LeakyReLU activation, and MaxPool3D layer.

    forward(self, x)
        Defines the forward pass of the network.
"""

import torch
import torch.nn as nn

#
# Adapted from: Chen, et al. (2021). https://arxiv.org/abs/2104.06468
# https://medium.com/towards-data-science/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
#
       
#
# The input data has one input "channel"
# In this simple Conv3D exercise just do 2 output channels
#
class NN_Conv3D_Simple(nn.Module):
    
    def __init__(self, in_depth, in_rows, in_cols, fc_dim, output_dim, pcnt_dropout=0.15):
        super(NN_Conv3D_Simple, self).__init__()
        
        num_out_chan = 2
        conv3d_kernel_size = 3
        max_pool_kernel_size = 2
        
        self.conv_layer = self._conv_layer_set(1, num_out_chan, conv3d_kernel_size, max_pool_kernel_size)
        
        in_depth = (in_depth - (conv3d_kernel_size - 1)) // max_pool_kernel_size
        in_rows = (in_rows - (conv3d_kernel_size - 1)) // max_pool_kernel_size
        in_cols = (in_cols - (conv3d_kernel_size - 1)) // max_pool_kernel_size
        fc_in_dim = num_out_chan * in_depth * in_rows * in_cols
        print(in_depth, in_rows, in_cols, fc_in_dim)
        
        self.fc1 = nn.Linear(fc_in_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(fc_dim)
        self.drop=nn.Dropout(p=pcnt_dropout)        
        
    def _conv_layer_set(self, in_c, out_c, conv3d_kernel_size, max_pool_kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(conv3d_kernel_size, conv3d_kernel_size, conv3d_kernel_size), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size)),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out