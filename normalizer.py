"""
Based on code from Marcin Andrychowicz

                    **** IMPORTANT ****

-> copy-pasted to my project from : https://github.com/vitchyr/rlkit
  >> encouraged to check that nice project
"""
import torch
import torch.nn as nn
import numpy as np

class Normalizer(torch.nn.Module):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=torch.tensor(np.inf),
            mean=0,
            std=1,
    ):
        super().__init__()
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.sum = torch.nn.Parameter(torch.zeros(self.size))
        self.sumsq = torch.nn.Parameter(torch.zeros(self.size))
        self.count = torch.nn.Parameter(torch.ones(1))
        self.mean = torch.nn.Parameter(mean + torch.zeros(self.size))
        self.std = torch.nn.Parameter(std * torch.ones(self.size))

        self.synchronized = True

    def update(self, v):
        assert v.shape[1] == self.size
        self.sum.data = self.sum.data + v.sum(0)
        self.sumsq.data = self.sumsq.data + (v ** 2).sum(0)
        self.count[0] = self.count[0] + v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self._synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range

        # convert back to numpy ( torch is just for sharing data between workers )
        std = self.std.detach()
        mean = self.mean.detach()

        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def _synchronize(self):
        self.mean.data = self.sum.detach() / self.count[0]
        self.std.data = torch.sqrt(
            torch.max(
                torch.tensor(self.eps ** 2),
                self.sumsq.detach() / self.count.detach()[0] - (self.mean.detach() ** 2)
            )
        )
        self.synchronized = True

class BatchNormalizer(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(state_size)
    def forward(self, states):
        return self.bn(states)

class GlobalNormalizerWGrads(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.bn = Normalizer(state_size)
    def forward(self, states):
        self.bn.update(states)
        return self.bn.normalize(states)

class GlobalNormalizer(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.bn = Normalizer(state_size)
    def forward(self, states):
        self.bn.update(states)
        return self.bn.normalize(states).detach()