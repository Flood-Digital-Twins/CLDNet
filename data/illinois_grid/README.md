# Des Plaines evaluation grid

`grid.npz` holds the 1,408,587 cells on which the Des Plaines models are trained and evaluated (cells whose maximum
water depth over the training storms exceeds 0.1 m), within the 5075 × 1661 raster of `data/static/dem_5070.tif`:

| Key | Shape, type | Contents |
|---|---|---|
| `aggregate_mask` | (5075, 1661) bool | the cells; model arrays list them in row-major order (`np.flatnonzero`) |
| `coords` | (1408587, 2) float16 | normalized coordinates fed to the decoder |
| `static_features` | (1408587, 3) float16 | standardized elevation, slope magnitude, scaled Manning coefficient |

`static_scaling.npz` records the scaling (`dem_mean`, `dem_std`, `manning_scale`, `slope_scale`). The CLDNet input per
cell is `[coords, static_features]`. See `scripts/predict_cldnet.py`.
