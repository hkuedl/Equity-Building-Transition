# File description

This folder is a ResStock/BuildStockBatch project extension. It must be used
with the external
[NatLabRockies/resstock](https://github.com/NatLabRockies/resstock)
repository. Use a ResStock release whose OpenStudio, EnergyPlus, workflow
generator, and BuildStockBatch versions are mutually compatible.

## Recommended execution order

1. `project_controll.yml`: Defines the precomputed sampler, annual simulation,
   hourly outputs, custom measure invocation, upgrade measures, costs, and
   apply logic. Update `sample_file` for the prepared building sample. The
   batch controller rewrites weather and output settings for each scenario.

2. `ApplyCountyOutage/measure.rb`: Optional HPXML model measure that reads a
   scenario-specific outage JSON file, identifies the current building, and
   injects unavailable periods into HPXML. `measure.xml` contains its OpenStudio
   metadata, and `resources/` contains the scenario JSON files.

3. `HybridIslandingControl/measure.rb`: EnergyPlus measure used by the provided
   project YAML. It resolves county outage periods, configures equivalent
   grid/island operation, patches HVAC and noncritical-load availability,
   coordinates PV and battery dispatch, and validates the resulting EMS and
   ElectricLoadCenter objects. `measure.xml` defines its arguments, and
   `resources/outage_json/` stores scenario outage inputs.

4. `run_buildstockbatch.py`: Creates scenario-specific copies of the project
   YAML, selects the requested upgrade set, points the run to future EPW files,
   launches local BuildStockBatch workers, and filters output Parquet files.

5. `Metrics.py`: Postprocesses hourly ResStock results into the decade-specific
   energy, carbon, installation-cost, comfort, outage-resilience, and
   energy-burden metrics consumed by optimization.

## ResStock integration

Place the project configuration and custom measure folders under the
`project_inequity` directory used by ResStock, for example:

```text
resstock/
└── project_inequity/
    ├── project_controll.yml
    ├── buildstock_ready_final.csv
    ├── ApplyCountyOutage/
    └── HybridIslandingControl/
```

The controller recognizes these environment variables:

- `RESSTOCK_DIR`: root of the external ResStock checkout
- `RESSTOCK_PROJECT_FILE`: full path to the active project YAML
- `FUTURE_WEATHER_DIR`: root containing scenario EPW archives/files
- `SIMULATION_OUTPUT_DIR`: root for BuildStockBatch outputs

`Metrics.py` additionally uses `PROJECT_DATA_DIR`, optional comma-separated
`TARGET_STATES`, and optional comma-separated `SSP_SCENARIOS`.

The large JSON resources are scenario inputs, not executable code. Replace
them only with files that preserve the building/county identifiers and outage
interval schema expected by the measures.
