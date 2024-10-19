import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),  # First linear layer: 784 -> 256
        nn.ReLU(),                       # Activation function
        nn.Linear(256, 16, bias=True),   # Second linear layer: 256 -> 16
        nn.ReLU(),                       # Activation function
        nn.Linear(16, 10, bias=True)     # Third linear layer: 16 -> 10
        # The last layer has no activation function
    )
    # return model instance (None is just a placeholder)

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
    
