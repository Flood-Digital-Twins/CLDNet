"""Rain sequence forecasting helpers for hydro_surrogate_light."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class RainPropagatorFlatLSTM(nn.Module):
    """Legacy flat-vector rain model kept for backward-compatible checkpoints."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size if output_size is not None else input_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output, hidden = self.lstm(x, hidden)
        return self.fc(output), hidden


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell for one spatial rainfall step."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so spatial resolution is preserved")
        padding = kernel_size // 2
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            self.input_channels + self.hidden_channels,
            4 * self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


class RainPropagatorLSTM(nn.Module):
    """ConvLSTM-based sequence-to-sequence rainfall propagator."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        *,
        grid_shape: tuple[int, int] = (39, 13),
        input_channels: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.flat_input_size = int(input_size)
        self.flat_output_size = int(output_size if output_size is not None else input_size)
        self.hidden_channels = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.input_channels = int(input_channels)
        self.kernel_size = int(kernel_size)

        if self.grid_shape[0] * self.grid_shape[1] != self.flat_input_size:
            raise ValueError(
                f"Grid shape {self.grid_shape} does not match input size {self.flat_input_size}"
            )
        if self.flat_output_size != self.flat_input_size:
            raise ValueError(
                f"Expected output size {self.flat_input_size}, got {self.flat_output_size}"
            )

        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    input_channels=self.input_channels if layer_idx == 0 else self.hidden_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.head = nn.Conv2d(self.hidden_channels, self.input_channels, kernel_size=1)
        self.dropout_layer = nn.Dropout2d(self.dropout) if self.dropout > 0 and self.num_layers > 1 else None

    def _reshape_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:
        if x.ndim == 3:
            batch, time, dim = x.shape
            if dim != self.flat_input_size:
                raise ValueError(f"Expected flat rain input of size {self.flat_input_size}; got {tuple(x.shape)}")
            grid = x.reshape(batch, time, self.input_channels, *self.grid_shape)
            return grid, "flat"
        if x.ndim == 5:
            batch, time, channels, height, width = x.shape
            if channels != self.input_channels or (height, width) != self.grid_shape:
                raise ValueError(
                    f"Expected rain grid with shape (B, T, {self.input_channels}, {self.grid_shape[0]}, {self.grid_shape[1]}); "
                    f"got {tuple(x.shape)}"
                )
            return x, "grid"
        raise ValueError(f"Expected rain sequence with shape (B, T, D) or (B, T, C, H, W); got {tuple(x.shape)}")

    def _init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_channels,
            self.grid_shape[0],
            self.grid_shape[1],
            device=device,
            dtype=dtype,
        )
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x_seq, layout = self._reshape_sequence(x)
        batch, time, _, _, _ = x_seq.shape

        if hidden is None:
            h_state, c_state = self._init_hidden(batch, x_seq.device, x_seq.dtype)
        else:
            h_state, c_state = hidden
            expected_shape = (self.num_layers, batch, self.hidden_channels, self.grid_shape[0], self.grid_shape[1])
            if h_state.shape != expected_shape or c_state.shape != expected_shape:
                raise ValueError(f"Expected hidden state shape {expected_shape}; got {h_state.shape} and {c_state.shape}")

        outputs: list[torch.Tensor] = []
        for time_idx in range(time):
            layer_input = x_seq[:, time_idx]
            next_h: list[torch.Tensor] = []
            next_c: list[torch.Tensor] = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev = h_state[layer_idx]
                c_prev = c_state[layer_idx]
                h_next, c_next = cell(layer_input, (h_prev, c_prev))
                next_h.append(h_next)
                next_c.append(c_next)
                layer_input = h_next
                if layer_idx < self.num_layers - 1 and self.dropout_layer is not None:
                    layer_input = self.dropout_layer(layer_input)

            outputs.append(self.head(layer_input).unsqueeze(1))
            h_state = torch.stack(next_h, dim=0)
            c_state = torch.stack(next_c, dim=0)

        output = torch.cat(outputs, dim=1)
        if layout == "flat":
            output = output.reshape(batch, time, self.flat_output_size)
        return output, (h_state, c_state)


def normalize_sequence(x: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if mean is None or std is None:
        return x
    view_shape = (1,) * (x.ndim - 1) + (mean.shape[0],)
    return (x - mean.reshape(view_shape)) / std.reshape(view_shape)


def denormalize_sequence(x: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if mean is None or std is None:
        return x
    view_shape = (1,) * (x.ndim - 1) + (mean.shape[0],)
    return x * std.reshape(view_shape) + mean.reshape(view_shape)


def rollout_sequence(
    model: nn.Module,
    seed: torch.Tensor,
    total_length: int,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll out a full rain sequence from a seed prefix."""

    if seed.ndim != 3:
        raise ValueError(f"Expected seed with shape (B, T, D); got {tuple(seed.shape)}")
    if total_length <= 0:
        raise ValueError("total_length must be positive")

    seed_work = normalize_sequence(seed, mean, std)
    seed_len = int(seed_work.shape[1])
    if seed_len == 0:
        raise ValueError("Seed prefix must contain at least one time step")

    if total_length <= seed_len:
        full_work = seed_work[:, :total_length, :]
    else:
        _, hidden = model(seed_work)
        current = seed_work[:, -1:, :]
        future: list[torch.Tensor] = []
        for _ in range(total_length - seed_len):
            out_step, hidden = model(current, hidden)
            current = out_step[:, -1:, :]
            future.append(current)
        full_work = torch.cat([seed_work] + future, dim=1)

    return denormalize_sequence(full_work, mean, std)


def rollout_window(
    model: nn.Module | None,
    seed: torch.Tensor,
    window_size: int,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a window that starts from the current seed step."""

    if seed.ndim != 3:
        raise ValueError(f"Expected seed with shape (B, T, D); got {tuple(seed.shape)}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if model is None:
        current = seed[:, -1:, :]
        return current.expand(-1, window_size, -1)

    total_length = int(seed.shape[1] + window_size - 1)
    rolled = rollout_sequence(model, seed, total_length, mean=mean, std=std)
    return rolled[:, -window_size:, :]


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor | None, torch.Tensor | None, dict]:
    """Load a rain model checkpoint and freeze it for inference."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        mean = None
        std = None
        metadata: dict = {}
    else:
        config = dict(checkpoint.get("config", {}))
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            state_dict = checkpoint

        model_type = str(config.get("model_type", checkpoint.get("model_type", "flat_lstm"))).lower()
        has_grid_shape = "grid_shape" in config or "grid_shape" in checkpoint
        input_size = int(config.get("input_size", checkpoint.get("input_size")))
        hidden_size = int(config.get("hidden_size", checkpoint.get("hidden_size")))
        output_size = int(config.get("output_size", checkpoint.get("output_size", input_size)))
        num_layers = int(config.get("num_layers", checkpoint.get("num_layers", 1)))
        dropout = float(config.get("dropout", checkpoint.get("dropout", 0.0)))

        if model_type in {"convlstm", "conv_lstm", "rain_convlstm"} or has_grid_shape:
            grid_shape = config.get("grid_shape", checkpoint.get("grid_shape", (39, 13)))
            if isinstance(grid_shape, list):
                grid_shape = tuple(int(v) for v in grid_shape)
            model = RainPropagatorLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
                grid_shape=grid_shape,
                input_channels=int(config.get("input_channels", checkpoint.get("input_channels", 1))),
                kernel_size=int(config.get("kernel_size", checkpoint.get("kernel_size", 3))),
            )
        else:
            model = RainPropagatorFlatLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
            )

        model.load_state_dict(state_dict)
        mean = checkpoint.get("rain_mean")
        std = checkpoint.get("rain_std")
        metadata = dict(checkpoint)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if mean is not None:
        mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    if std is not None:
        std = torch.as_tensor(std, dtype=torch.float32, device=device)

    return model, mean, std, metadata
