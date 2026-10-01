# %%
import argparse
from pathlib import Path

from paths import LATENT_DATA_DIR, REPOSITORY_ROOT, VAE_CHECKPOINT_DIR, VAE_FIGURE_DIR

# %%
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import einops
from utils import set_seed
import matplotlib.pyplot as plt
from train_utils import BatchIndicesIterator, plot_prediction_truth_error, load_latest_checkpoint, detect_outliers, limited_gradient
from utils import set_seed, RelativeL2Loss, format_elapsed_time, print_model_size
from ldm_ae.convlstm import *
# %%
set_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
maeloss = torch.nn.L1Loss(reduction='sum')
mseloss = torch.nn.MSELoss(reduction='sum')


class CustomDataset(Dataset):
    def __init__(self, x_data, device):
        self.x_data = x_data
        self.x_data["y"] = torch.Tensor(self.x_data["y"])
        self.x_data["u"] = torch.Tensor(self.x_data["u"])
        self.device = device
    def __len__(self):
        return len(self.x_data["y"])

    def __getitem__(self, idx):
        return self.x_data["y"][idx].to(self.device), self.x_data["u"][idx].to(self.device)

def loss_function(x, x_hat, mean, log_var, mean_dynamics, beta=0.0, lambda_=1.0):
    reconstruction_loss = mseloss(x_hat[:,:,0], x[:,:, 0]) + lambda_ * (mseloss(x_hat[:,:, 1], x[:,:, 1]) + mseloss(x_hat[:,:, 2], x[:,:,2])) + mseloss(x_hat[:,:, 3], x[:,:,3]) + mseloss(mean[:, 1:], mean_dynamics[:, :-1])
    # KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reconstruction_loss# + beta * KLD

def kld(mean1, log_var1, mean2, log_var2):
    return - 0.5 * torch.sum(1+ log_var1 - log_var2 - (mean1 - mean2).pow(2)/log_var2.exp() - log_var1.exp()/log_var2.exp())

import random
def train_epoch(epoch, model, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
        for i, value in enumerate(train_loader):
            # value = einops.rearrange(value, "B T C H W -> (B T) C H W")
            preds = []
            x = value
            target, rain = x
            optimizer.zero_grad()
            y, states = model(torch.cat([target[:, 0:1, :, :, :], rain[:, 0:1]], dim = 2))
            preds.append(target[:, 0:1, :, :, :] + y[2])
            for j in range(1, target.shape[1]):
                if random.random() < (1 - (epoch / 200)):
                    y, states = model(torch.cat([target[:, j:j+1, :, :, :], rain[:, j:j+1]], dim = 2), states)
                else:
                    y, states = model(torch.cat([preds[-1], rain[:, j:j+1]], dim = 2), states)
                preds.append(preds[-1]+y[2])
            # y_orig, _ = model(target)
            # print(mseloss(y_orig[2],  torch.concatenate(preds, dim = 1)))
            loss = mseloss(target[:, 1:], torch.concatenate(preds, dim = 1)[:, :-1])
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)


def valid_epoch(epoch, model, valid_loader):
    losses = []
    model.eval()
    with tqdm(total=len(valid_loader), desc=f"Valid {epoch}: ") as pbar:
        for i, value in enumerate(valid_loader):
            # value = einops.rearrange(value, "B T C H W -> (B T) C H W")
            preds = []
            x = value
            target, rain = x
            optimizer.zero_grad()
            y, states = model(torch.cat([target[:, 0:1, :, :, :], rain[:, 0:1]], dim = 2))
            preds.append(target[:, 0:1, :, :, :] + y[2])
            for j in range(1, target.shape[1]):
                if random.random() < (1 - (epoch / 200)):
                    y, states = model(torch.cat([target[:, j:j+1, :, :, :], rain[:, j:j+1]], dim = 2), states)
                else:
                    y, states = model(torch.cat([preds[-1], rain[:, j:j+1]], dim = 2), states)
                preds.append(preds[-1]+y[2])
            # y_orig, _ = model(target)
            # print(mseloss(y_orig[2],  torch.concatenate(preds, dim = 1)))
            loss = mseloss(target[:, 1:], torch.concatenate(preds, dim = 1)[:, :-1])
            losses.append(loss.item())
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)

# %%
from einops import rearrange

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Texas residual ConvLSTM")
    parser.add_argument("--data-root", type=Path, default=LATENT_DATA_DIR,
                        help="Directory holding VAE latent and rain arrays")
    args = parser.parse_args()
    # %%
    data_root = args.data_root if args.data_root.is_absolute() else REPOSITORY_ROOT / args.data_root
    dataset_directory = data_root / "vae_and_latents_texas" if (data_root / "vae_and_latents_texas").is_dir() else data_root
    required = ("latent_train.npy", "latent_test.npy", "rain_train.npy", "rain_test.npy")
    missing = [str(dataset_directory / name) for name in required if not (dataset_directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing VAE latent inputs: " + ", ".join(missing))
    latent_train = np.load(dataset_directory / "latent_train.npy", mmap_mode="r")
    latent_test = np.load(dataset_directory / "latent_test.npy", mmap_mode="r")
    rain_train_raw = np.load(dataset_directory / "rain_train.npy", mmap_mode="r")
    rain_test_raw = np.load(dataset_directory / "rain_test.npy", mmap_mode="r")
    for split, latent, rain in (("train", latent_train, rain_train_raw), ("test", latent_test, rain_test_raw)):
        if latent.ndim != 5 or rain.ndim != 3 or latent.shape[:2] != rain.shape[:2]:
            raise ValueError(
                f"{split} VAE latent/rain mismatch: latent shape {latent.shape}, rain shape {rain.shape}; "
                "provide matching trajectories and time steps before training"
            )
        if latent.shape[2] < 8 or latent.shape[3:] != (25, 25) or rain.shape[2] != 1:
            raise ValueError(f"Unexpected {split} latent/rain feature shapes: {latent.shape}, {rain.shape}")
    X_train = np.asarray(latent_train[:, :, :8], dtype=np.float32)
    X_test = np.asarray(latent_test[:, :, :8], dtype=np.float32)
    rain_train = np.broadcast_to(rain_train_raw[..., np.newaxis, np.newaxis],
                                 (*rain_train_raw.shape, 25, 25)).copy()
    rain_test = np.broadcast_to(rain_test_raw[..., np.newaxis, np.newaxis],
                                (*rain_test_raw.shape, 25, 25)).copy()
    print(rain_train.shape)
    
    data_train = {"u": rain_train, "y":X_train}
    data_valid = {"u": rain_test, "y":X_test}
    # X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[2]*8, X_train.shape[3], X_train.shape[4])
    # X_test = X_test.reshape(X_test.shape[0], -1, X_test.shape[2]*8, X_test.shape[3], X_test.shape[4])
    print("X_train shape:", X_train.shape)  
    print("X_test shape:", X_test.shape)

    train_dataset = CustomDataset(data_train, device)
    valid_dataset = CustomDataset(data_valid, device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = ConvLSTM(input_dim = 9, hidden_dim = [8*2, 8*2, 8], kernel_size = (5, 5), num_layers = 3, batch_first = True, bias = True, return_all_layers = True)
    model = model.to(device)

    _, _ = print_model_size(model)


    from soap import SOAP
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = SOAP(model.parameters(), lr = 1e-4, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) 

    # %%
    checkpoint_dir = VAE_CHECKPOINT_DIR
    figure_dir = VAE_FIGURE_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            checkpoint = load_latest_checkpoint(checkpoint_dir, device)
        except FileNotFoundError:
            checkpoint = torch.load(checkpoint_dir / "checkpoint_convlstm.pth", map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_loss_list = checkpoint['epoch_loss_list']
        frames = checkpoint.get('frames', [])
    except FileNotFoundError:
        print("No checkpoint found. Initializing model from dual training checkpoint.")
        start_epoch = 1
        epoch_loss_list = []
        frames = []
        # model, optimizer, lr_scheduler are already initialized

    # %%
    num_epochs = 301
    for epoch in range(start_epoch, num_epochs+1):
        print('EPOCH {}:'.format(epoch))
        train_loss = train_epoch(epoch, model, optimizer, train_loader)
        # lr_scheduler.step()
        valid_loss = valid_epoch(epoch, model, valid_loader)
        epoch_loss_list.append((train_loss, valid_loss))
        # === Optionally: Save checkpoint every few epochs ===
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch_loss_list': epoch_loss_list,
                'frames': frames
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

    # %%
    def compute_relative_l2_error(prediction: np.ndarray, ground_truth: np.ndarray):
        """
        Compute the relative L2 error between predictions and labels using NumPy.

        Inputs:
        predictions: (batch_size, num_timestamps, num_features, height, width)
        labels: (batch_size, num_timestamps, num_features, height, width)

        Outputs:
        relative_error: (batch_size, num_timestamps, num_features)
        """
        prediction = np.asarray(prediction, dtype=np.float64)
        ground_truth = np.asarray(ground_truth, dtype=np.float64)

        diff_norm = np.sqrt(np.sum((prediction - ground_truth) ** 2, axis=(3, 4)))
        gt_norm = np.sqrt(np.sum(ground_truth ** 2, axis=(3, 4)))

        # gt_norm = np.where(gt_norm == 0, 1e-12, gt_norm)

        return diff_norm / gt_norm
