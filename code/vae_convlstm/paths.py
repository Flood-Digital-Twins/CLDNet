"""Repository-relative paths shared by the Texas VAE–ConvLSTM scripts."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
TEXAS_DATA_ROOT = REPOSITORY_ROOT / "data/texas"
TEXAS_TRAIN_DIR = TEXAS_DATA_ROOT / "train_dataset"
TEXAS_TEST_DIR = TEXAS_DATA_ROOT / "test_dataset"
LATENT_DATA_DIR = TEXAS_DATA_ROOT / "vae_and_latents_texas"
VAE_CHECKPOINT_DIR = REPOSITORY_ROOT / "checkpoints/vae_convlstm"
VAE_FIGURE_DIR = REPOSITORY_ROOT / "outputs/vae_convlstm/figures"
