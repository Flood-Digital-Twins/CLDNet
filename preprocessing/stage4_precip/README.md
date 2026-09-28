# Stage IV precipitation preprocessing

**Partially complete.** The two notebooks that crop the storm library to
the simulator's grid are here. The script that *built* that library is not — see
"What is missing" below.

```
stage4_precip/
├── process_event.ipynb                 crops the 113 numbered storms to (39, 13)
└── process_event_for_validation.ipynb  extracts the 2013 flood-of-record window
```

## `process_event.ipynb`

Reads `precipitation_st4_midwest_fill_nan/event_<id>.nc` and writes
`precipitation_st4_midwest_fill_nan_adjust/event_<id>.nc`, the version
`run_hipims_event.py` consumes and the version deposited in `data/forcings/`.

It does one thing: crop every storm to a common `(39, 13)` grid with `pr[:, :39, :13]`.
The library is not uniform — the 113 source files come in four native shapes,
`(39, 14)` × 59, `(39, 13)` × 47, `(40, 13)` × 6 and `(40, 14)` × 1 — so the crop takes a
fixed top-left corner of each. The hard-coded index list at the top encodes the seven
indices that never existed: 7, 17, 44, 60, 66, 67, 72.

**The crop is by array index, not by coordinate.** Each storm's extraction carries its own
x/y origin — 36 distinct origins across the library, differing by up to ~3 km — so cropping
to `[:39, :13]` does not yield a common geographic footprint. Combined with the index-based
rain-to-DEM mapping in `run_hipims_event.py` (see `data/forcings/README.md`), rainfall cell
*j* drives the same DEM cells in every storm while referring to slightly different ground.

## `process_event_for_validation.ipynb`

Extracts the 2013 flood-of-record from the 336-hour file
`precipitation_event_2013-04-10_2013-04-23.nc` with `time=slice(175, 175+96+1)`.

This is the provenance of a window that the source filename misstates. 2013-04-10 00:00 plus
175 hours is **2013-04-17 07:00**, and 97 steps later is **2013-04-21 07:00** — exactly the
span of the deposited `data/forcings/event_2013-04-17_2013-04-21/`. The 13-day range in the
original name was never simulated.

## What is missing

**The Stage IV event-library builder is not in this release, and no copy of it could be
found.** It is the script that queried the Stage IV archive, chose the 120 candidate storm
windows, assigned their event indices and wrote
`precipitation_st4_midwest_fill_nan/event_<id>.nc`. Searches across every reachable tree
(the authors' storage) found only consumers of those files.
The author confirms no copy is available.

What that costs a reader: the storm *selection* cannot be reproduced or audited — which
windows were considered, the sampling rule, and why the library contains repeats. Everything
downstream of the library is fully reproducible from this release.

### What the library is, measured from the data

Since the builder is unavailable, its output was characterised directly. Reproduce with the
the authors' verification notebooks.

| Property | Value |
|---|---|
| Files | 113, indices 0–119 minus 7, 17, 44, 60, 66, 67, 72 |
| Per storm | `prcp(time, y, x)`, 97 hourly steps, float64, **mm/hr**, EPSG:5070, 4 km |
| Native shapes | `(39,14)` × 59, `(39,13)` × 47, `(40,13)` × 6, `(40,14)` × 1 |
| Distinct grid origins | 36 |
| Date range | 2003-07-20 to 2024-09-30; 86 distinct start times |
| Data quality | no NaNs, no negative values anywhere |
| Catchment-mean depth | 22.2 – 288.0 mm |
| **Unique storms** | **93 of 113** |

### The duplicates, and why the cause is untraceable

Twenty of the 113 are byte-identical copies of other entries — same values, same grid origin,
same timestamps:

> 85→25, 86→26, 87→27, 88→28, 89→29, 95→35, 96→36, 97→37, 98→38, 99→39,
> 110→50, 111→51, 112→52, 113→53, 114→54, 115→55, 116→56, 117→57, 118→58, 119→59

Every duplicate sits at an index offset of exactly **−60** from its twin, and they fall in
three contiguous runs (85–89, 95–99, 110–119 mirroring 25–29, 35–39, 50–59). Indices 0–84
contain no duplicates. That regularity is far too structured for chance collision in storm
sampling, so catalog rows were almost certainly re-emitted under new indices rather than
re-sampled — **but this is inference from the pattern, not something read in code**, and with
the builder unavailable it cannot be confirmed.

The duplication is inherited, not introduced downstream: three separate copies of the library
on disk are byte-identical, and the crop notebook above alters nothing but the grid extent.
The deposited `data/forcings/` keeps only the lower index of each pair, 93 numbered storms
plus the flood-of-record.

There is no grey zone: the closest *non*-identical pair sits at a relative L2 distance of
0.637, so the twenty are unambiguous and no near-duplicate escapes a checksum.
