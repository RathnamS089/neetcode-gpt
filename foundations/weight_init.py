import math
import torch
import numpy as np
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = np.sqrt(2 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        weights = torch.round(weights, decimals=4)
        return weights.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = np.sqrt(2 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        weights = torch.round(weights, decimals=4)
        return weights.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        listoact = []
        weights_list = []
        
        # --- PHASE 1: Generate all weights using your dimension tracking logic ---
        for i in range(num_layers):
            if i == 0:
                current = input_dim
                
            if init_type == "xavier":
                std = math.sqrt(2.0 / (current + hidden_dim))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / current)
            else:
                std = 1.0
                
            w1 = torch.randn(hidden_dim, current) * std
            weights_list.append(w1)
            current = hidden_dim

        # --- PHASE 2: Generate input and compute activations sequentially ---
        x = torch.randn(1, input_dim)
        
        for w1 in weights_list:
            z = x @ w1.T
            zact=torch.relu(z)
            
       
                
            listoact.append(zact.std().item())
            x = zact
            
        return [round(value, 2) for value in listoact]

