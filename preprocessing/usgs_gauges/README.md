# USGS gauge data and simulator validation (2013 flood-of-record)

`usgs_validation_crosssections.ipynb` retrieves USGS stage and discharge for the Des Plaines basin with `pygeohydro`,
builds a perpendicular cross-section at each gauge along the NHDPlus flowline (60 m, 90 m for wider channels), and
compares simulated and observed water-surface elevation (paper Section 2.4.4 and Fig. 5). Outputs are cleared.

The retrieved records it produced are in `data/usgs_2013_validation/USGS_2013-04-15_2013-04-23/` (`h/` stage in feet,
`q/` discharge), one CSV per gauge. The notebook also reads full-grid simulated and predicted fields for the 2013
event, which are simulation outputs and are not included in this repository.
