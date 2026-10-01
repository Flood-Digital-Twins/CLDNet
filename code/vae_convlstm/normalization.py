"""Channel-wise Gaussian normalization for VAE–ConvLSTM arrays."""

import numpy as np


class Normalize_gaussian:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != self.std.shape or np.any(self.std <= 0):
            raise ValueError("Mean and standard deviation must have the same shape and positive std")

    def _parameters(self, values):
        channels = self.mean.size
        if values.ndim == 5 and values.shape[2] == channels:
            shape = (1, 1, channels, 1, 1)
        elif values.ndim == 3 and values.shape[2] == channels:
            shape = (1, 1, channels)
        else:
            raise ValueError(f"Expected (batch, time, {channels}, ...) array, got {values.shape}")
        return self.mean.reshape(shape), self.std.reshape(shape)

    def normalize_forw(self, values):
        mean, std = self._parameters(values)
        return (values - mean) / std

    def normalize_backward(self, values):
        mean, std = self._parameters(values)
        return values * std + mean
