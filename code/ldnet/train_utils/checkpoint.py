import os
import re
import torch

def load_latest_checkpoint(checkpoint_dir, device, prefix='checkpoint_epoch_', suffix='.pth'):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(suffix)]
    
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")

    # Extract epoch numbers from filenames
    def extract_epoch(fname):
        match = re.search(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}", fname)
        return int(match.group(1)) if match else -1

    # Get the checkpoint with the highest epoch
    latest_ckpt = max(checkpoint_files, key=extract_epoch)
    latest_ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    print(f"Loading checkpoint: {latest_ckpt_path}")

    checkpoint = torch.load(latest_ckpt_path, weights_only=False, map_location = device)
    return checkpoint