import torch
import torch.nn as nn
from typing import Tuple
import functools

class FFEncoder(nn.Module):
    def __init__(self, input_size: Tuple[int], latent_dims: int):
        super(FFEncoder, self).__init__()
        total_size = functools.reduce(lambda x, y: x * y, input_size)
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1), # Flattens from (B, C, H, W) to (B, C*H*W)
            nn.Linear(total_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dims)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class FFDecoder(nn.Module):
    def __init__(self, output_size: Tuple[int], latent_dims: int):
        super(FFDecoder, self).__init__()
        total_size = functools.reduce(lambda x, y: x * y, output_size)
        self.layers = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, total_size),
            nn.Unflatten(dim=1, unflattened_size=output_size) # Unflattens from (B, C*H*W) to (B, C, H, W)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class FFAutoEncoder(nn.Module):
    def __init__(self, input_size: Tuple[int], latent_dims: int):
        super(FFAutoEncoder, self).__init__()
        self.encoder = FFEncoder(input_size, latent_dims)
        self.decoder = FFDecoder(input_size, latent_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)