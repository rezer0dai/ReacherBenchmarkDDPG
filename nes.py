import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

def initialize_weights(layer):
    if type(layer) not in [NoisyLinear, ]:
        return
    nn.init.xavier_uniform_(layer.weight)

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

        self.sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(.017))
        self.register_buffer('noise', torch.zeros(out_features, in_features))

        self.apply(initialize_weights)

    def forward(self, data):
        return F.linear(data, self.weight + self.sigma * Variable(self.noise))

    def sample_noise(self):
        self.noise = torch.randn(self.out_features, self.in_features)

    def remove_noise(self):
        self.noise = torch.zeros(self.out_features, self.in_features)

class NoisyNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # TODO : properly expose interface to remove/select noise in selective way
        self.layers = [ NoisyLinear(layer, layers[i+1]) for i, layer in enumerate(layers[:-1]) ]

    def parameters(self):
        return np.concatenate([
                list(super().parameters()),
                np.concatenate([list(layer.parameters()) for layer in self.layers])])

    def sample_noise(self):
        for layer in self.layers:
            layer.sample_noise()

    def remove_noise(self):
        for layer in self.layers:
            layer.remove_noise()

    def forward(self, data):
        for layer in self.layers[:-1]:
            data = F.relu(layer(data))
        return self.layers[-1](data)