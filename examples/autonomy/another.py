import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time

class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        
        self.actor_mlp = nn.Sequential(
            nn.Linear(23, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.mu = nn.Linear(128, 4)  # Assuming 4 action dimensions
        self.sigma = nn.Parameter(torch.zeros(4))  # Assuming 4 action dimensions
        
        # Value head
        self.value = nn.Linear(128, 1)
        
        # Normalization layers
        self.value_mean_std = nn.BatchNorm1d(1, affine=False)
        self.running_mean_std = nn.BatchNorm1d(23, affine=False)

    def forward(self, x):
        x = self.running_mean_std(x)  # Normalize input
        actor_features = self.actor_mlp(x)
        mu = self.mu(actor_features)
        sigma = torch.exp(self.sigma)
        value = self.value(actor_features)
        value = self.value_mean_std(value)  # Normalize value output
        return mu, sigma, value


# Instantiate the model
model = ActorCriticNet()


# Load the checkpoint
checkpoint = torch.load('Crazyflie.pth')
checkpoint_model = {k.replace('a2c_network.', ''): v for k, v in checkpoint['model'].items()}
unexpected_keys = ['value_mean_std.count', 'running_mean_std.count']
for key in unexpected_keys:
    if key in checkpoint_model:
        del checkpoint_model[key]
model.load_state_dict(checkpoint_model)
model.eval()


print(model)

first_layer = model.actor_mlp[0]  # Adjust this based on your model's architecture
print("Input features:", first_layer.in_features)
