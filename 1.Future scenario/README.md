# File description

The files are listed below in recommended execution order. Building inventory
construction and future weather construction are independent branches; both
feed the simulation stage.

## Building inventory and outage branch

1. `Building_1_Household_merge.py`: Spatially joins residential building
   footprints to Census block groups, samples building height, allocates
   households and population subject to physical constraints, assigns
   synthetic household income, and writes one building-level Parquet file per
   state.

2. `Building_2_resstock_map.py`: Standardizes real-building types and matches
   each real building to a county-specific ResStock template using scaled
   physical features and dynamic matching penalties. It writes the
   `*_ResStock.parquet` mapping used by simulation and optimization.

3. `Outage.py`: Combines county-level Presto outage simulations with
   historical EAGLE-I statistics, downscales outage events to individual
   buildings, and exports building outage schedules and county hourly
   availability tables.

## Future weather branch

1. `Future_1_TMY_Daily.py`: Loads and crops CMIP6 daily climate projections,
   fuses multiple climate models, and selects representative months to create
   daily typical meteorological year files for SSP1-2.6, SSP2-4.5, and
   SSP5-8.5 in the 2020s-2050s.

2. `Future_2_TMY_Hourly_by_Morphing.py`: Uses historical ERA5 hourly profiles
   to morph the daily future TMY variables to hourly resolution. It also
   includes spatial, temporal, and physical-consistency diagnostics for the
   generated NetCDF files.

3. `Future_3_EPW.py`: Samples the hourly future fields at county locations,
   derives EPW variables, decomposes global radiation into direct and diffuse
   components, writes county EPW files, and performs EPW validation.

## Configuration

All paths use public-safe relative defaults. Override them as needed with:

- `PROJECT_DATA_DIR`
- `SCENARIO_INPUT_DIR`
- `BUILDING_FOOTPRINT_DIR`
- `CENSUS_PROFILE_DIR`
- `BUILDING_HEIGHT_RASTER`
- `BUILDING_OUTPUT_DIR`
- `RESSTOCK_METADATA_FILE`
- `CMIP6_DATA_DIR`
- `ERA5_DATA_DIR`
- `FUTURE_WEATHER_DIR`
- `EPW_OUTPUT_DIR`
- `COUNTY_GAZETTEER_FILE`
- `OUTAGE_SCENARIO_DIR`
- `EAGLEI_EVENTS_FILE`

Use `TARGET_STATE` for the daily weather script. The scripts contain state
lists and scenario lists that can be narrowed for testing before a national
run.
