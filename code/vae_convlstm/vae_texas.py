# %%
import os
import sys
from pathlib import Path

from paths import REPOSITORY_ROOT, TEXAS_TRAIN_DIR, TEXAS_TEST_DIR, VAE_CHECKPOINT_DIR, VAE_FIGURE_DIR

repo_path = REPOSITORY_ROOT

# %%
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from normalization import Normalize_gaussian

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
        self.window_length = 20
        self.x_data = x_data
        self.x_data["y"] = torch.Tensor(self.x_data["y"])
        self.x_data["u"] = torch.Tensor(self.x_data["u"])
        self.window_start_array = np.concatenate([np.zeros(self.window_length-1), np.ones(self.window_length-1)*(x_data["y"].shape[1] - self.window_length), np.arange(x_data["y"].shape[1])]).astype(np.int_)
        self.device = device
    def __len__(self):
        return len(self.x_data["y"])

    def __getitem__(self, idx):
        start_index = np.random.choice(self.window_start_array)
        return self.x_data["y"][idx, start_index:start_index+self.window_length].to(self.device), self.x_data["u"][idx, start_index:start_index+self.window_length].to(self.device)

def loss_function(x, x_hat, mean, log_var, mean_dynamics, beta=0.0, lambda_=1.0):
    reconstruction_loss = mseloss(x_hat[:,:,0], x[:,:, 0]) + lambda_ * (mseloss(x_hat[:,:, 1], x[:,:, 1]) + mseloss(x_hat[:,:, 2], x[:,:,2])) + mseloss(mean[:, 1:], mean_dynamics[:, :-1])

    return reconstruction_loss# + beta * KLD

def kld(mean1, log_var1, mean2, log_var2):
    return - 0.5 * torch.sum(1+ log_var1 - log_var2 - (mean1 - mean2).pow(2)/log_var2.exp() - log_var1.exp()/log_var2.exp())


def train_epoch(epoch, model, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
        for i, value in enumerate(train_loader):
            # value = einops.rearrange(value, "B T C H W -> (B T) C H W")
            x = value
            target = x
            optimizer.zero_grad()

            x_hat, mean, log_var, mean_dynamics = model(target)
            loss = loss_function(target[0], x_hat, mean, log_var, mean_dynamics)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            losses.append(loss.item())
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
            x = value
            target = x
            optimizer.zero_grad()
            with torch.no_grad():
                x_hat, mean, log_var, mean_dynamics = model(target)
                loss = loss_function(target[0], x_hat, mean, log_var, mean_dynamics)
                loss_val = loss.item()
                losses.append(loss_val)
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss_val:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)

# %%
# from ldm_ae.model import Encoder, Decoder
# from ldm_ae.distributions import DiagonalGaussianDistribution

# %%
def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """
    swish activation function

    """
    return x * torch.sigmoid(x)

# %%
def Normalize(in_channels, num_groups=2):
    """
    the return GroupNorm splits in_channels into num_groups groups, and applies normalization across each group per sample

    GroupNorm: (B, C, H, W) => (B, C, H, W)
    """
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# %%
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        input: 
            x: (B, C, H, W) 
        output:
            (B, C, 2H, 2W)
        """
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest") # (B, C, 2H, 2W)
        if self.with_conv:
            x = self.conv(x) # (B, C, 2H, 2W)
        return x

# %%
class Upsample_4x(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        input: 
            x: (B, C, H, W) 
        output:
            (B, C, 4H, 4W)
        """
        x = torch.nn.functional.interpolate(x, scale_factor=4.0, mode="nearest") # (B, C, 4H, 4W)
        if self.with_conv:
            x = self.conv(x) # (B, C, 4H, 4W)
        return x

class Upsample_5x(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        input: 
            x: (B, C, H, W) 
        output:
            (B, C, 4H, 4W)
        """
        x = torch.nn.functional.interpolate(x, scale_factor=5.0, mode="nearest") # (B, C, 4H, 4W)
        if self.with_conv:
            x = self.conv(x) # (B, C, 4H, 4W)
        return x

# %%
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        """
        input:
            x: (B, C, H, W)
        output:
            (B, C, H//2, W//2)
        """
        if self.with_conv:
            pad = (0,1,0,1) # (left, right, top, bottom)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0) # (B, C, H+1, W+1)
            x = self.conv(x) # (B, C, H//2, W//2) 
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2) # (B, C, H//2, W//2)
        return x

# %%
class Downsample_4x(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=4,
                                        padding=0)

    def forward(self, x):
        """
        input:
            x: (B, C, H, W)
        output:
            (B, C, H//4, W//4)
        """
        if self.with_conv:
            pad = (0,1,0,1) # (left, right, top, bottom)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0) # (B, C, H+1, W+1)
            x = self.conv(x) # (B, C, H//4, W//4) 
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=4, stride=4) # (B, C, H//4, W//4)
        return x

class Downsample_5(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=7,
                                        stride=5,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (3, 3, 3, 3)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
# %%
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        """ 
        input:
            x: (B, C_in, H, W)
            temb: (B, T)

        output: 
            (B, C_out, H, W)
        """
        h = x
        h = self.norm1(h) # (B, C_in, H, W)
        h = nonlinearity(h) # (B, C_in, H, W)
        h = self.conv1(h) # (B, C_out, H, W)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None] # (B, C_out, 1, 1)

        h = self.norm2(h) # (B, C_out, H, W)
        h = nonlinearity(h) # (B, C_out, H, W)
        h = self.dropout(h) # (B, C_out, H, W)
        h = self.conv2(h) # (B, C_out, H, W)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut: 
                x = self.conv_shortcut(x) # (B, C_out, H, W)
            else:
                x = self.nin_shortcut(x) # (B, C_out, H, W)

        return x+h

# %%
from einops import rearrange

# %%
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(in_channels=dim, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        """
        input: 
            x: (B, C, H, W)
        output:
            (B, dim, H, W)
        """

        b, c, h, w = x.shape
        qkv = self.to_qkv(x) # (B, 3 * heads * dim_head, H, W)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)# q: (B, heads, dim_head, H*W), k: (B, heads, dim_head, H*W), v: (B, heads, dim_head, H*W)
        k = k.softmax(dim=-1)  # (B, heads, dim_head, H*W)
        context = torch.einsum('bhdn,bhen->bhde', k, v) # (B, heads, dim_head, dim_head)
        out = torch.einsum('bhde,bhdn->bhen', context, q) # (B, heads, dim_head, H*W)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w) # (B, heads * dim_head, H, W)
        return self.to_out(out) # (B, dim, H, W)

# %%
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels) # (B, in_channels, H, W) => (B, in_channels, H, W)

# %%
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        """"
        input:
            x: (B, C, H, W)
        output:
            (B, C, H, W)
        """
        h_ = x
        h_ = self.norm(h_) # (B, C, H, W)
        q = self.q(h_) # (B, C, H, W)
        k = self.k(h_) # (B, C, H, W)
        v = self.v(h_) # (B, C, H, W)   

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w) # (B, C, H*W)
        q = q.permute(0,2,1)   # (B, H*W, C)
        k = k.reshape(b,c,h*w) # (B, C, H*W)
        w_ = torch.bmm(q,k)   # (B, H*W, H*W)    
        w_ = w_ * (int(c)**(-0.5)) # (B, H*W, H*W)
        w_ = torch.nn.functional.softmax(w_, dim=2) # (B, H*W, H*W)

        # attend to values
        v = v.reshape(b,c,h*w) # (B, C, H*W)
        w_ = w_.permute(0,2,1)   # (B, H*W, H*W) 
        h_ = torch.bmm(v,w_)     # (B, C, H*W) 
        h_ = h_.reshape(b,c,h,w) # (B, C, H, W)

        h_ = self.proj_out(h_) # (B, C, H, W)

        return x+h_ # (B, C, H, W)

# %%
def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)

# %%
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,4,16), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.attn_type = attn_type

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if self.curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            if i_level == 0:
                down.downsample = Downsample_5(block_in, resamp_with_conv)
                self.curr_res = self.curr_res // 5
            if i_level == 1:
                down.downsample = Downsample_4x(block_in, resamp_with_conv)
                self.curr_res = self.curr_res // 4
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        input:
            x: (B, C_in, H, W)
        output:
            (B, 2*z_channels or z_channels, H/2^n, W/2^n), where n = len(ch_mult) - 1
        """


        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)] # (B, ch, H, W)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1] 
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) #
        return h

# %%
class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,4,16), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.attn_type = attn_type

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        self.curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,self.curr_res,self.curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if self.curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level == 1:
                up.upsample = Upsample_4x(block_in, resamp_with_conv)
                self.curr_res = self.curr_res * 4
            
            if i_level == 2:
                up.upsample = Upsample_5x(block_in, resamp_with_conv)
                self.curr_res = self.curr_res * 5
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        """"
        input:
            z: (B, 2*z_channels or z_channels, H/2^n, W/2^n), where n = len(ch_mult) - 1
        output:
            (B, out_ch, H, W)
        """

        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

# %%
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class ConvDynamics_Window(nn.Module):
    """
    x_{t+1} = x_t + f(x_t)
    f: three 5x5 convs (no norm), preserves spatial size and channels.

    Args:
        channels:  input/output channel count
        hidden1:   channels of conv layer 1
        hidden2:   channels of conv layer 2
        periodic:  use circular padding if True, else zero padding
        act:       activation function (e.g., nn.SiLU(), nn.ReLU())
    """
    def __init__(self, channels: int, out: int, hidden1: int = 64, hidden2: int = 64,
                 act=nn.ReLU()):
        super().__init__()
        self.act = act

        # No in-module padding so we can switch between circular/zero
        self.c1 = nn.Conv2d(channels, hidden1, kernel_size=5, stride=1, padding=2, bias=True)
        self.c2 = nn.Conv2d(hidden1, hidden2, kernel_size=5, stride=1, padding=2, bias=True)
        self.c3 = nn.Conv2d(hidden2, out, kernel_size=5, stride=1, padding=2, bias=True)

        # Inits: near-identity at start (last layer zero), first two Kaiming
        nn.init.kaiming_normal_(self.c1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.c2.weight, nonlinearity='relu')
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

        # 5x5 same-padding = 2 on each side
        self.pad2 = (2, 2, 2, 2)

    #should be dealing with batch, time steps, c, h, w
    def forward(self, input):
        x = input.view(input.shape[0]*input.shape[1], input.shape[2], input.shape[3], input.shape[4])
        x = self.c1(x); x = self.act(x)
        x = self.c2(x); x = self.act(x)
        x = self.c3(x)
        #putting really weird format here as a dummy here to match convlstm
        return [x.reshape(input.shape[0], input.shape[1], -1, input.shape[3], input.shape[4])], None

    
# %%
use_observation_encoder = False
class VAE_LSTM(nn.Module):
    def __init__(self,
                 resolution, 
                 in_channel,
                 ch_mult,
                 num_res_blocks,
                 hidden_dim,
                 embed_dim,
                 latent_dim):
        super().__init__()

        self.resolution = resolution
        self.in_channel = in_channel
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.encoder = Encoder(ch=hidden_dim, 
                               out_ch=in_channel, 
                               ch_mult=ch_mult, 
                               num_res_blocks=num_res_blocks,
                               attn_resolutions=[], 
                               dropout=0.0, 
                               resamp_with_conv=True, 
                               in_channels=in_channel,
                               resolution=resolution, 
                               z_channels=hidden_dim, 
                               double_z=True, 
                               use_linear_attn=False, 
                               attn_type="vanilla", 
                               downsample = True)

        self.decoder = Decoder(ch=hidden_dim, 
                               out_ch=in_channel, 
                               ch_mult=ch_mult, 
                               num_res_blocks=num_res_blocks,
                               attn_resolutions = [], 
                               dropout=0.0, 
                               resamp_with_conv=True, 
                               in_channels=in_channel,
                               resolution=resolution, 
                               z_channels=hidden_dim, 
                               give_pre_end=False, 
                               tanh_out=False, 
                               use_linear_attn=False,
                               attn_type="vanilla")
        if use_observation_encoder:
            self.encoder = Encoder(ch=hidden_dim, 
                out_ch=in_channel, 
                ch_mult=ch_mult, 
                num_res_blocks=num_res_blocks,
                attn_resolutions=[], 
                dropout=0.0, 
                resamp_with_conv=True, 
                in_channels=in_channel,
                resolution=resolution, 
                z_channels=hidden_dim, 
                double_z=True, 
                use_linear_attn=False, 
                attn_type="vanilla", 
                downsample = True)

        self.quant_conv = torch.nn.Conv2d(2*hidden_dim, embed_dim*2, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hidden_dim, 1)
        # self.encoder_linear = torch.nn.Linear(embed_dim*16*16, 2*latent_dim)
        # self.decoder_linear = torch.nn.Linear(latent_dim, embed_dim*16*16)
        # self.dynamics = ConvLSTM(input_dim = embed_dim+1, hidden_dim = [embed_dim*2, embed_dim*2, embed_dim], kernel_size = (5, 5), num_layers = 3, batch_first = True, bias = True, return_all_layers = False)
        self.dynamics = ConvDynamics_Window(channels = embed_dim+1, out = embed_dim, hidden1 = embed_dim*2, hidden2 = embed_dim*2)
        print(f'encoder attention type: {self.encoder.attn_type}')
        print(f'decoder attention type: {self.decoder.attn_type}')
        print(f'latent dim: {embed_dim, self.encoder.curr_res, self.encoder.curr_res}')

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # moments = moments.view(moments.shape[0], -1)
        # moments = self.encoder_linear(moments.view(moments.shape[0], -1))
        return moments
    

    def decode(self, z):
        # z = self.decoder_linear(z).view(z.shape[0], self.embed_dim, 16, 16)
        # z = z.view(z.shape[0], self.embed_dim, 16, 16)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def split(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def forward(self, input, sample_posterior=True):
        input_x, input_u = input
        input_reshaped = input_x.view(input_x.shape[0]*input_x.shape[1], input_x.shape[2], input_x.shape[3], input_x.shape[4])

        moments = self.encode(input_reshaped)
        mean, logvar = self.split(moments)
        mean_reshaped = mean.view(input_x.shape[0], input_x.shape[1], mean.shape[1], mean.shape[2], mean.shape[3])
        # batch, 96, C, H, W
        # Conv-LSTM
        # list[Tensor]
        mean_dynamics = self.propagate_dynamics(torch.cat([mean_reshaped, input_u.view(1, input_u.shape[1], 1, 1, 1).repeat(1, 1, 1, mean.shape[2], mean.shape[3])], dim = 2))
        # batch, 96, C, H, W
        
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)

        dec = dec.view(input_x.shape[0], input_x.shape[1], dec.shape[1], dec.shape[2], dec.shape[3])
        logvar = logvar.view(input_x.shape[0], input_x.shape[1], -1)
        return dec, mean_reshaped, logvar, mean_dynamics
    
    def propagate_dynamics(self, latent_states):
        mean_dynamics, _ = self.dynamics(latent_states)
        return mean_dynamics[-1]
    
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, proj_size=0, bias=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        #x : batch_size, num_time_steps = 51, observation_dim = 3x10x10 = 300
        output, _ = self.lstm(x)
        # batch_size, num_time_steps, hidden_size = 256
        output = self.fc(output) #256 -> 12
        # batch_size, num_time_steps, output_size = latent_dim = 12

        return output
    
    def forward_5d(self, x):
        x_reshaped = x.view(x.shape[0], x.shape[1], -1)
        #x : batch_size, num_time_steps = 51, observation_dim = 3x10x10 = 300
        output, _ = self.lstm(x_reshaped)
        # batch_size, num_time_steps, hidden_size = 256
        output = self.fc(output) #256 -> 12
        # batch_size, num_time_steps, output_size = latent_dim = 12

        return output.view(x.shape)
if __name__ == "__main__":
    # %%
    results_path = os.path.join(repo_path, "results")
    dataset_directory =  os.path.join(results_path, "train_dataset/perlin_dam_break_4x_resolution/acaa_00001-00400")
    dataset_directory = TEXAS_TRAIN_DIR
    sample_start_index = 1
    sample_end_index   = 100
    burn_in_length = 1

    rain_source = [None] * len(range(sample_start_index, sample_end_index + 1))
    flow_variables = [None] * len(range(sample_start_index, sample_end_index + 1))
    for i in tqdm(range(sample_start_index, sample_end_index + 1)):
        sample_directory = os.path.join(dataset_directory, f"sample_{i:05d}")
        rain_source[i - sample_start_index] = (np.load(os.path.join(sample_directory, "rain_source.npy")))
        flow_variables[i - sample_start_index] = np.load(os.path.join(sample_directory, "flow_variables.npy"))
        rain_source[i - sample_start_index] = np.repeat(rain_source[i - sample_start_index][:-1, 1], 4)
    flow_variables = np.array(flow_variables).astype(np.float32)
    
    burn_in_length = 1
    flow_variables = flow_variables[:, burn_in_length:, :, :]
    rain_source = np.array(rain_source).astype(np.float32)[:, :, np.newaxis]
    num_time_instants = flow_variables.shape[1]

    mean = np.mean(flow_variables, axis=(0, 1, 3, 4), keepdims = True)
    std = np.std(flow_variables, axis=(0, 1, 3, 4), keepdims = True)
    
    checkpoint_dir = VAE_CHECKPOINT_DIR
    figure_dir = VAE_FIGURE_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok= True)
    np.savez(checkpoint_dir / "mean_std.npz", mean=mean, std=std)
    normalize_y = Normalize_gaussian(mean, std)
    flow_variables = normalize_y.normalize_forw(flow_variables)

    mean = np.mean(rain_source, axis=(0, 1), keepdims = True)
    std = np.std(rain_source, axis=(0, 1), keepdims = True)
    np.savez(checkpoint_dir / "mean_std_rain_source.npz", mean=mean, std=std)
    normalize_u = Normalize_gaussian(mean, std)
    rain_source = normalize_u.normalize_forw(rain_source)

    coords = np.repeat(np.stack([np.repeat(2*(np.arange(0, flow_variables.shape[3], 1)/flow_variables.shape[3] - 0.5), flow_variables.shape[4]), np.tile(2*(np.arange(0, flow_variables.shape[4], 1)/flow_variables.shape[4] - 0.5), flow_variables.shape[3])], axis = 1)[None, :, :], flow_variables.shape[0], axis = 0).astype(np.float32)
    data_train = {"u": rain_source, "y":flow_variables}


    dataset_directory = TEXAS_TEST_DIR
    sample_start_index = 101
    sample_end_index   = 120
    burn_in_length = 1

    rain_source = [None] * len(range(sample_start_index, sample_end_index + 1))
    flow_variables = [None] * len(range(sample_start_index, sample_end_index + 1))
    for i in tqdm(range(sample_start_index, sample_end_index + 1)):
        sample_directory = os.path.join(dataset_directory, f"sample_{i:05d}")
        rain_source[i - sample_start_index] = (np.load(os.path.join(sample_directory, "rain_source.npy")))
        flow_variables[i - sample_start_index] = np.load(os.path.join(sample_directory, "flow_variables.npy"))
        rain_source[i - sample_start_index] = np.repeat(rain_source[i - sample_start_index][:-1, 1], 4)
    flow_variables = np.array(flow_variables).astype(np.float32)
    burn_in_length = 1
    flow_variables = flow_variables[:, burn_in_length:, :, :]
    rain_source = np.array(rain_source).astype(np.float32)[:, :, np.newaxis]
    flow_variables = normalize_y.normalize_forw(flow_variables)
    rain_source = normalize_u.normalize_forw(rain_source)
    data_valid = {"u": rain_source, "y":flow_variables}

    train_dataset = CustomDataset(data_train, device)
    valid_dataset = CustomDataset(data_valid, device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # %%
    model = VAE_LSTM(
        resolution=256, 
        in_channel=3, 
        ch_mult=(1,4,16),
        num_res_blocks=3, 
        hidden_dim=16, 
        embed_dim=8,
        latent_dim = 256
    )
    model = model.to(device)
    

    _, _ = print_model_size(model)


    from soap import SOAP
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = SOAP(model.parameters(), lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) 

    try:
        try:
            checkpoint = load_latest_checkpoint(checkpoint_dir, device)
        except FileNotFoundError:
            checkpoint = torch.load(checkpoint_dir / "checkpoint_vae.pth", map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_loss_list = checkpoint['epoch_loss_list']
        frames = checkpoint.get('frames', [])

    except FileNotFoundError:
        print("No checkpoint found. Initializing model from scratch.")
        start_epoch = 1
        epoch_loss_list = []
        frames = []

    # %%
    num_epochs = 2000
    for epoch in range(start_epoch, num_epochs+1):
        print('EPOCH {}:'.format(epoch))
        train_loss = train_epoch(epoch, model, optimizer, train_loader)
        # lr_scheduler.step()
        valid_loss = valid_epoch(epoch, model, valid_loader)
        epoch_loss_list.append((train_loss, valid_loss))
        # === Optionally: Save checkpoint every few epochs ===
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch_loss_list': epoch_loss_list,
                'frames': frames
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))

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


    import io
    import imageio

    colorbar_vmin = {
        'water_depth': 0.0,
        'discharge_x': -15.0,
        'discharge_y': -15.0
    }
    colorbar_vmax = {
        'water_depth': 40.0,
        'discharge_x': 15.0,
        'discharge_y': 15.0
    }


    # %%
    checkpoint_epoch_list = [num_epochs]
    for checkpoint_epoch in checkpoint_epoch_list:
        print(f"Processing checkpoint epoch: {checkpoint_epoch}")
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{checkpoint_epoch}.pth"
        if not checkpoint_path.is_file() and checkpoint_epoch == 1900:
            checkpoint_path = checkpoint_dir / "checkpoint_vae.pth"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        is_include_downsample = False
        model.eval()
        x_hat_list = []
        with tqdm(total=len(valid_loader), desc="Inference on validation dataset") as pbar:
            for i, value in enumerate(valid_loader):
                # value = einops.rearrange(value, "B T C H W -> (B T) C H W")
                x = value
                with torch.no_grad():
                    x_hat, mean, log_var, mean_dynamics = model(x)
                    # x_hat = einops.rearrange(x_hat, "(B T) C H W -> B T C H W", T=num_time_instants)
                    x_hat_list.append(x_hat.cpu().numpy())
                pbar.update(1)
                pbar.set_postfix_str(f"Processed {i+1}/{len(valid_loader)} batches")

        x_hat_array = np.concatenate(x_hat_list, axis=0)
        print("x_hat_array shape:", x_hat_array.shape)

        relative_l2_error = compute_relative_l2_error(x_hat_array, X_test)
        avg_relative_l2_error = np.mean(relative_l2_error, axis=0)

        plt.plot(avg_relative_l2_error[:, 0], label='Water depth', marker='o', markersize=3)
        plt.plot(avg_relative_l2_error[:, 1], label='Discharge x', marker='o', markersize=3)
        plt.plot(avg_relative_l2_error[:, 2], label='Discharge y', marker='o', markersize=3)
        plt.xlabel('Time Step')
        plt.ylabel('Relative L2 Error')
        plt.title(f'Reconstruction relative L2 error ({checkpoint_epoch})')
        plt.legend()
        plt.savefig(os.path.join(figure_dir, f"valid_reconstruction_relative_l2_error_{checkpoint_epoch}.png"))
        plt.close()

        frames = []
        for time_instant in range(1, num_time_instants + 1):
            time_index = time_instant - 1
            fig = plot_prediction_truth_error(x_hat_array[0, time_index, 0],
                                            data_valid["y"][0, time_index, 0], 
                                            title=f"Water depth at time instant {time_instant}", 
                                            vmin=colorbar_vmin['water_depth'], 
                                            vmax=colorbar_vmax['water_depth'])
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.v2.imread(buf))
            buf.close()
            plt.close(fig)
        imageio.mimsave(os.path.join(figure_dir, f"reconstruction_water_depth_{checkpoint_epoch}.gif"), frames, duration=1.0, loop=0)


        frames = []
        for time_instant in range(1, num_time_instants + 1):
            time_index = time_instant - 1
            fig = plot_prediction_truth_error(x_hat_array[0, time_index, 1],
                                            data_valid["y"][0, time_index, 1], 
                                            title=f"Discharge x at time instant {time_instant}", 
                                            vmin=colorbar_vmin['discharge_x'], 
                                            vmax=colorbar_vmax['discharge_x'])
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.v2.imread(buf))
            buf.close()
            plt.close(fig)
        imageio.mimsave(os.path.join(figure_dir, f"reconstruction_discharge_x_{checkpoint_epoch}.gif"), frames, duration=1.0, loop=0)


        frames = []
        for time_instant in range(1, num_time_instants + 1):
            time_index = time_instant - 1
            fig = plot_prediction_truth_error(x_hat_array[0, time_index, 2],
                                            data_valid["y"][0, time_index, 2], 
                                            title=f"Discharge y at time instant {time_instant}", 
                                            vmin=colorbar_vmin['discharge_y'], 
                                            vmax=colorbar_vmax['discharge_y'])
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.v2.imread(buf))
            buf.close()
            plt.close(fig)
        imageio.mimsave(os.path.join(figure_dir, f"reconstruction_discharge_y_{checkpoint_epoch}.gif"), frames, duration=1.0, loop=0)

    # %%

