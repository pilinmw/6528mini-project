import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies):
        """
        Standard Positional Encoding (as used in NeRF).
        Maps an coordinate x to:
        [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.in_features = in_features
        # Original features + 2 (sin, cos) * num_frequencies * in_features
        self.out_features = in_features + 2 * num_frequencies * in_features

    def forward(self, x):
        """
        x: [..., in_features]
        out: [..., out_features]
        """
        if self.num_frequencies == 0:
            return x
            
        coords = [x]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * np.pi
            coords.append(torch.sin(freq * x))
            coords.append(torch.cos(freq * x))
            
        return torch.cat(coords, dim=-1)

class CoordinateMLP(nn.Module):
    def __init__(self, num_frequencies=0, hidden_features=256, hidden_layers=4):
        """
        A simple Multi-Layer Perceptron (MLP) for Implicit Neural Representation.
        
        Args:
            num_frequencies: L frequency bands for positional encoding. 0 means no encoding.
            hidden_features: Number of features in hidden layers.
            hidden_layers: Number of hidden layers.
        """
        super().__init__()
        self.pe = PositionalEncoding(in_features=2, num_frequencies=num_frequencies)
        
        in_dim = self.pe.out_features
        
        layers = []
        # First layer
        layers.append(nn.Linear(in_dim, hidden_features))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))
            
        # Output layer maps to RGB (3 channels)
        # We use Sigmoid to ensure the output is strictly between 0 and 1
        layers.append(nn.Linear(hidden_features, 3))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.pe(x)
        return self.net(x)

if __name__ == "__main__":
    # Test model
    model = CoordinateMLP(num_frequencies=5, hidden_features=128, hidden_layers=3)
    dummy_x = torch.zeros(10, 2)
    out = model(dummy_x)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Input shape: {dummy_x.shape}, Output shape: {out.shape}")
