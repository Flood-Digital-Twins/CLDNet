# Des Plaines evaluation grid

`grid.npz` holds the 1,408,587 cells on which the Des Plaines models are trained and evaluated (cells whose maximum
water depth over the training storms exceeds 0.1 m), within the 5075 × 1661 raster of `data/static/dem_5070.tif`:

| Key | Shape, type | Contents |
|---|---|---|
| `aggregate_mask` | (5075, 1661) bool | the cells; model arrays list them in row-major order (`np.flatnonzero`) |
| `coords` | (1408587, 2) float16 | normalized coordinates fed to the decoder |
| `static_features` | (1408587, 3) float16 | standardized elevation, slope magnitude, Manning coefficient × 100 (2 on open water, 5 on land) |
| `static_features_as_trained` | (1408587, 3) float16 | the same, with the Manning channel as it was when the released checkpoints were trained |

`static_scaling.npz` records the scaling (`dem_mean`, `dem_std`, `manning_scale`, `slope_scale`). The CLDNet input per
cell is `[coords, static features]`. See `scripts/predict_cldnet.py`.

**Which one to use.** The two feature arrays differ only in the Manning channel, at 93,816 cells (6.7 %).
`static_features` is correct: Manning 0.02 exactly where `landcover_5070_processed.tif` is water, which is what the
simulator used. In `static_features_as_trained` the channel is mis-registered: the per-cell values of the simulator's
`manning.dat`, which are numbered from the southern row of the raster upward, were written into the grid from the
northern row downward (it marks 1.8 % of the cells as water; 5.0 % are). The released epoch-539 CLDNet was trained
on that array, so **use `static_features_as_trained` with the released checkpoints** (it reproduces the paper) **and
`static_features` to train a new model**. Elevation and slope are identical in both.
