import torch
import torch.nn as nn

# modified from https://github.com/jiankeyang/LaLiGAN/blob/main/src/autoencoder.py
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation="ReLU", classify=False, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            getattr(nn, activation)(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                getattr(nn, activation)(),
            ) for _ in range(n_layers - 1)],
            nn.Linear(hidden_dim, output_dim),
        )
        self.classify = classify

    def forward(self, x):
        if self.classify:
            return torch.sigmoid(self.layers(x))
        else:
            return self.layers(x)