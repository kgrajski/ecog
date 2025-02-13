import torch
import torch.nn as nn

class NN_Flat(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(NN_Flat, self).__init__()
        
            # Establish the flattening method
        self.flatten = nn.Flatten()
        
            # Set the first (input) layer
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

            # Add additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
            # Add the output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
            # Create the full model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits
    
    def print_params(self, show_values=False):
        print(f"\nModel structure: {self.model}\n")
        for name, param in self.model.named_parameters():
            if show_values:
                print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")
            else:
                print(f"Layer: {name} | Size: {param.size()}")
