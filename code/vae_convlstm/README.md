# VAE–ConvLSTM baseline (Texas benchmark)

A convolutional VAE compresses each 500 × 500 flow snapshot (h, hu, hv) to a 25 × 25 × 8 latent, and a residual
ConvLSTM advances that latent in quarter-hour steps, driven by the rainfall. At inference only the first frame is
encoded; the rest of the storm is rolled out in latent space and decoded. The model has 14,557,515 parameters.

Paths resolve from `paths.py` relative to the repository, so the scripts run from any working directory. Open the
notebooks from the repository or a subdirectory so their setup cells can find `paths.py`.

| Purpose | Repository-relative location |
|---|---|
| Raw training samples (1–100) | `data/texas/train_dataset/` |
| Raw test samples (101–120) | `data/texas/test_dataset/` |
| State, latent, and rain arrays | `data/texas/vae_and_latents_texas/` |
| VAE definition and training | `code/vae_convlstm/vae_texas.py` |
| Residual ConvLSTM training | `code/vae_convlstm/vae_texas_residual.py` |
| VAE checkpoint (epoch 1900) | `checkpoints/vae_convlstm/checkpoint_vae.pth` |
| Residual ConvLSTM checkpoint (epoch 300) | `checkpoints/vae_convlstm/checkpoint_convlstm.pth` |
| Flow and rain normalization | `checkpoints/vae_convlstm/mean_std.npz`, `mean_std_rain_source.npz` |
| Settings | `configs/vae_convlstm/texas.json` |
| Figures | `outputs/vae_convlstm/figures/` |

**What this repository holds.** The code, both checkpoints (weights only), the normalization files and the two
rain arrays `rain_train.npy` and `rain_test.npy`. The raw `flow_variables.npy` simulations (65 GB) are not included,
and neither are the state and latent arrays derived from them (64 GiB and 0.9 GB). With the raw simulations in
`data/texas/`, the two generators below rebuild those arrays.

Beyond the top-level `requirements.txt`, the code needs `einops` and `scikit-learn`. The SOAP optimizer is bundled
as `soap.py`.

## Score the checkpoints

```bash
python scripts/score_vae_convlstm.py
```

This needs only the raw simulations of storms 101–120. It rolls each storm out from its first frame, decodes all
192 frames and compares them with the simulation in physical units. It reproduces the paper's Texas numbers:
aggregate rRMSE 23.41 %, RMSE 0.0837, R² 0.921; water depth 18.74 %, 0.0692, 0.953. rRMSE and RMSE are averaged
over the 20 storms; R² is pooled.

## Regenerate the arrays

```bash
python code/vae_convlstm/generate_states.py
python code/vae_convlstm/generate_latents.py
```

`generate_states.py` skips the initial flow frame and applies `mean_std.npz`. The output `float32` shapes are
`(100, 192, 3, 500, 500)` and `(20, 192, 3, 500, 500)`, about 64 GiB combined. Each split resumes after its last
completed trajectory and replaces the old file only after all rows have been written and spot checked.

`generate_latents.py` loads the VAE checkpoint and the flow normalization, drops the first flow frame, and writes
`float32` arrays with shapes `(100, 192, 16, 25, 25)` and `(20, 192, 16, 25, 25)`. The first eight channels are the
posterior mean used by the residual ConvLSTM. It uses per-sample temporary files so interrupted runs can resume.

Rows of both follow the saved `rain_train.npy` and `rain_test.npy` order, which is checked against each raw sample's
rain source. The training rows are permuted; the test rows are storms 101–120 in order.

## Smoke test and training

With the arrays in place, a short end-to-end check is

```bash
python code/vae_convlstm/smoke_test.py --sample-id 101 --start 64 --steps 16
```

It loads both checkpoints, compares the one-step latent error with persistence over the full storm, rolls forward
16 steps, decodes each step, and reports physical-unit RMSE, relative RMSE and R² against the saved states.

`vae_texas.py` trains the VAE and `vae_texas_residual.py` fine-tunes the residual ConvLSTM on the latents. Both
write `checkpoint_epoch_<n>.pth` into `checkpoints/vae_convlstm/` and resume from the newest such file, falling back
to the released checkpoint. The released files hold weights only, without optimizer state: the ConvLSTM trainer can
continue from its file, but the VAE trainer cannot, so move `checkpoint_vae.pth` aside to train the VAE from scratch.
Because the two trainers share the folder and the file pattern, keep their `checkpoint_epoch_<n>.pth` files apart.
The notebooks are exploratory, and some cells need substantial memory.

## Third-party code

The autoencoder blocks, the ConvLSTM cell and the SOAP optimizer are adapted from MIT-licensed projects; see
`THIRD_PARTY_LICENSES.md`.
