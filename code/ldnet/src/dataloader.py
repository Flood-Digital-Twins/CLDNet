"""Dataset and DataLoader for sample-based training."""
import numpy as np
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset


class DataLoaderX(DataLoader):
    """DataLoader with prefetch for faster data loading."""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class BranchDataset_Sample(Dataset):
    """Dataset that loads trajectories from disk and samples spatial points."""

    def __init__(self, data, device, sample_indices: int = 20000):
        self.data = data
        self.device = device
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        u = np.load(self.data["u"][idx], mmap_mode="r")
        x = np.load(self.data["x"][idx], mmap_mode="r")
        y = np.load(self.data["y"][idx], mmap_mode="r")
        s_paths = self.data.get("s")
        s = None
        if s_paths is not None:
            s = np.load(s_paths[idx], mmap_mode="r")

        num_points = y.shape[2]
        sample_count = min(self.sample_indices, num_points)
        indices = np.random.choice(np.arange(num_points), sample_count, replace=False)

        if s is not None:
            x = np.concatenate([x, s], axis=-1)

        y_idx = self.data.get("y_idx")
        if y_idx is not None:
            y = y[..., y_idx]

        y_mean = self.data.get("y_mean")
        y_std = self.data.get("y_std")
        if y_mean is not None and y_std is not None:
            y_mean = np.asarray(y_mean, dtype=np.float32)
            y_std = np.asarray(y_std, dtype=np.float32)
            if y_idx is not None:
                y_mean = y_mean[y_idx]
                y_std = y_std[y_idx]
            y = (y.astype(np.float32, copy=False) - y_mean.reshape(1, 1, 1, -1)) / y_std.reshape(1, 1, 1, -1)

        u = u.astype(np.float32, copy=False)
        u_mean = self.data.get("u_mean")
        u_std = self.data.get("u_std")
        if u_mean is not None and u_std is not None:
            u_mean = np.asarray(u_mean, dtype=np.float32)
            u_std = np.asarray(u_std, dtype=np.float32)
            if u_mean.shape != (u.shape[-1],) or u_std.shape != (u.shape[-1],):
                raise ValueError(
                    f"Expected u_mean/u_std shape {(u.shape[-1],)}, got {u_mean.shape} and {u_std.shape}"
                )
            u = (u - u_mean.reshape(1, 1, -1)) / u_std.reshape(1, 1, -1)

        t_len = y.shape[1]
        return {
            "u": torch.from_numpy(u[0].copy()),
            "x": torch.from_numpy(np.repeat(x[:, indices, :], t_len, axis=0).astype(np.float32, copy=False)),
            "y": torch.from_numpy(y[0, :, indices, :].transpose(1, 0, 2).astype(np.float32, copy=False)),
            "dt": torch.from_numpy(self.data["dt"].astype(np.float32, copy=False)),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "u": torch.stack([item["u"] for item in batch], dim=0),
            "x": torch.stack([item["x"] for item in batch], dim=0),
            "y": torch.stack([item["y"] for item in batch], dim=0),
            "dt": batch[0]["dt"],
        }
