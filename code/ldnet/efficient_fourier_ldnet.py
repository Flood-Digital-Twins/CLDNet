#!/usr/bin/env python3
"""
Efficient Fourier LDNN: MLP dynamics + pointwise MLP decoder with Fourier features.

Efficiency improvements:
- No state_history_ expand over Nx
- Chunked spatial decoding
- Optional streaming decode (per time step)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


def get_activation(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")


def init_linear(layer: nn.Linear, init_name: str) -> None:
    init_name = init_name.lower()
    if init_name in ("xavier_uniform", "glorot uniform"):
        nn.init.xavier_uniform_(layer.weight)
    elif init_name in ("xavier_normal", "glorot normal"):
        nn.init.xavier_normal_(layer.weight)
    elif init_name == "kaiming_uniform":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    elif init_name == "kaiming_normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    elif init_name == "normal":
        nn.init.normal_(layer.weight, std=0.02)
    else:
        raise ValueError(f"Unsupported initializer: {init_name}")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        kernel_initializer: str = "xavier_uniform",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output")
        act = get_activation(activation)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, kernel_initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierEmbedding(nn.Module):
    def __init__(self, in_feats: int, out_feats: int) -> None:
        super().__init__()
        self.encoding = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.encoding(inp)


class EfficientFourierLDNN(nn.Module):
    def __init__(
        self,
        fourier_mapping_size: int,
        layer_sizes_dyn: list[int],
        layer_sizes_rec: list[int],
        activation: str = "relu",
        kernel_initializer: str = "xavier_uniform",
        dropout: float = 0.0,
        chunk_size: int = 10000,
    ) -> None:
        super().__init__()
        if layer_sizes_dyn[-1] <= 0:
            raise ValueError("latent size must be > 0")
        if layer_sizes_rec[0] <= layer_sizes_dyn[-1]:
            raise ValueError("layer_sizes_rec[0] must be latent + coord dims")

        self.num_latent_states = layer_sizes_dyn[-1]
        self.n_coords = layer_sizes_rec[0] - layer_sizes_dyn[-1]
        self.fourier_mapping_size = fourier_mapping_size
        self.chunk_size = chunk_size

        self.dyn = MLP(
            layer_sizes_dyn,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

        rec_input = self.num_latent_states + 2 * self.fourier_mapping_size
        layer_sizes_rec_f = [rec_input] + layer_sizes_rec[1:]
        self.rec = MLP(
            layer_sizes_rec_f,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

        self.B = FourierEmbedding(self.n_coords, fourier_mapping_size)

    def _update_state(
        self,
        u_t: torch.Tensor,
        state: torch.Tensor,
        dt: torch.Tensor,
        equilibrium: bool,
    ) -> torch.Tensor:
        inp = torch.cat([u_t, state], dim=1)
        delta = self.dyn(inp)
        if equilibrium:
            delta = delta - self.dyn(torch.zeros_like(inp))
        return state + dt * delta

    def _decode_points(
        self,
        x_t: torch.Tensor,
        z_t: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # x_t: (B, Nx, n_coords), z_t: (B, latent)
        bsz, nx, _ = x_t.shape
        out_dim = self.rec.net[-1].out_features
        out = torch.empty((bsz, nx, out_dim), device=x_t.device, dtype=x_t.dtype)
        chunk = chunk_size or self.chunk_size

        for start in range(0, nx, chunk):
            end = min(nx, start + chunk)
            x_chunk = x_t[:, start:end, :]
            x_flat = x_chunk.reshape(-1, x_chunk.shape[-1])
            phi = self.B(x_flat * (2 * math.pi))
            phi = phi.reshape(bsz, end - start, self.fourier_mapping_size)
            features = torch.cat([torch.sin(phi), torch.cos(phi)], dim=-1)
            z_expand = z_t[:, None, :].expand(-1, end - start, -1)
            rec_in = torch.cat([features, z_expand], dim=-1)
            rec_out = self.rec(rec_in.reshape(-1, rec_in.shape[-1]))
            out[:, start:end, :] = rec_out.reshape(bsz, end - start, out_dim)

        return out

    def forward(
        self,
        data: dict,
        device: Optional[torch.device] = None,
        equilibrium: bool = False,
        latent_state: bool = False,
        latent_init: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # data["u"]: (B, T, dim_u) or (B, dim_u)
        # data["x"]: (B, T, Nx, n_coords)
        # data["dt"]: scalar tensor or float
        u = data["u"]
        x = data["x"]
        dt = data["dt"]

        if device is None:
            device = x.device

        bsz, t_len, nx, _ = x.shape
        if u.dim() == 2:
            u_is_static = True
        else:
            u_is_static = False

        dt = torch.as_tensor(dt, device=device, dtype=x.dtype)
        if dt.numel() == 1:
            dt = dt.reshape(1)

        if latent_init:
            state = data["latent"]
        else:
            state = torch.zeros(bsz, self.num_latent_states, device=device, dtype=x.dtype)

        if latent_state:
            state_hist = torch.empty((bsz, t_len, self.num_latent_states), device=device, dtype=x.dtype)
        else:
            out_dim = self.rec.net[-1].out_features
            outputs = torch.empty((bsz, t_len, nx, out_dim), device=device, dtype=x.dtype)

        for t in range(t_len):
            u_t = u if u_is_static else u[:, t, :]
            state = self._update_state(u_t, state, dt, equilibrium)
            if latent_state:
                state_hist[:, t, :] = state
            else:
                outputs[:, t, :, :] = self._decode_points(x[:, t, :, :], state, chunk_size=chunk_size)

        return state_hist if latent_state else outputs
