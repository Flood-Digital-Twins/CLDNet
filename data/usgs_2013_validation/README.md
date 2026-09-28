# USGS records, April 2013 flood-of-record

Instantaneous (15-min) USGS NWIS records for 2013-04-15 to 2013-04-23 at the gauges in the Des Plaines basin, retrieved
with `pygeohydro` (see `preprocessing/usgs_gauges/`):

- `USGS_2013-04-15_2013-04-23/h/<index>_<site>_height.csv`: gauge height (ft) with site metadata and datum;
- `USGS_2013-04-15_2013-04-23/q/<index>_<site>_discharge.csv`: discharge;
- `gages.geojson`: gauge locations.

The simulator is validated at six of these gauges (05527800, 05528000, 05529000, 05531500, 05532500, 05540130;
paper Section 2.4.4). A few sites have no data for the window (header-only files).
