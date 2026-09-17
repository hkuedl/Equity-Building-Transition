# Equitable Residential Building Transitions

_This project develops a building-level workflow for evaluating residential
retrofit strategies under future climate and power-outage conditions. It
combines future weather construction, ResStock simulations, post-processed
energy/carbon/comfort/cost metrics, and multi-objective pathway optimization
to identify retrofit decisions for individual buildings across the United
States._

The repository contains the code used to construct future scenarios, run
building simulations, calculate strategy metrics, and optimize cost-, carbon-,
resilience-, and equity-oriented retrofit pathways.

##Overview

The overview of the proposed Equity-oriented Building Transition framework is presented in the following figure.

The framework includes three major parts: future scenario, simulation, and optimization. In the future scenario, three shared socioeconomic pathways, SSP126, SSP245, and SSP585, are utilized to represent different carbon emission levels in the research period 2020-2060. In the Simulation, active and passive technologies at different technical levels are combined to formulate the strategy pool for transition. TMY future weather, coupled with extreme weather and power outages, is taken as input for building energy simulation. The simulation output is used for evaluation of carbon emissions, thermal stress, and total cost. In the Optimization, the recommended retrofit strategy is obtained by minimizing the inequity and mapped for each real-world building.

## Requirements

[Python](https://www.python.org/) 3.10 or newer is recommended.

The Python workflows use `pandas`, `numpy`, `xarray`, `scipy`,
`scikit-learn`, `geopandas`, `rasterio`, `pyproj`, `pyarrow`, `cftime`,
`metpy`, `matplotlib`, `openpyxl`, and `pyomo`. A supported mathematical
programming solver is also required for the optimization workflow; the current
configuration uses Gurobi.

The simulation workflow additionally requires:

- [ResStock](https://github.com/NatLabRockies/resstock)
- OpenStudio and EnergyPlus versions compatible with the selected ResStock release
- `buildstockbatch`
- Ruby, for the custom OpenStudio measures

Exact package versions should be aligned with the ResStock release used for a
reproduction run.

## Data

The complete processed dataset is available from [Google Drive](https://drive.google.com/drive/folders/1x8d4INtwHMS3tLm3ZKz8F94jgDxMNXmX?usp=drive_link) or
[Baidu Netdisk](https://pan.baidu.com/s/1Zmslm9jTQkPtMizqMWV_qg?pwd=p5d3)
(Code: `p5d3`).

### Metrics

`Metrics` contains the postprocessed performance of all ResStock retrofit
strategies. Files are organized by state, county FIPS, SSP scenario, and
upgrade. The tables include the building-level energy, carbon, thermal-comfort,
outage-resilience, installation-cost, and energy-burden indicators used by the
optimization workflow.

Example layout:

```text
Data/
└── Metrics/
    └── AL/
        └── FIPS_01001/
            ├── ssp126_up01.parquet
            ├── ssp245_up01.parquet
            └── ssp585_up01.parquet
```

### Decisions

`Decisions` contains the building-level optimal retrofit strategies selected
under the pathways analyzed in the manuscript, including baseline,
one-size-fits-all, carbon, resilience, cost, and equity-oriented pathways. It
also includes group-level template decisions, county quality-control reports,
county-group maps, fixed real-to-template mappings, and mapping diagnostics.

Example layout:

```text
Data/
└── Decisions/
    └── AL/
        ├── AL_ssp126_fips_group_map.csv
        ├── AL_ssp126_quality_report.csv
        ├── Group_Template/
        │   └── GROUP_01001/
        │       └── GROUP_01001_ssp126_optimal_pathway.parquet
        └── FIPS_01001/
            ├── 01001_ssp126_cost_optimal_pathway_real.parquet
            ├── 01001_ssp126_carbon_optimal_pathway_real.parquet
            ├── 01001_ssp126_resilience_optimal_pathway_real.parquet
            └── 01001_ssp126_Equity_DP_optimal_pathway_real.parquet
```

The local example contains Alabama (`AL`) and uses `FIPS_01001` to illustrate
the county-level structure. The downloadable dataset contains all counties in
the full 49-state study domain.

## Codes

### Reproduction

The workflow is organized into three stages:

```text
1.Future scenario
2.Simulation
3.Optimization
```

[`1.Future scenario`](./1.Future%20scenario/) constructs the real-building
inventory, maps real buildings to ResStock templates, produces daily and hourly
future typical meteorological years, converts them to county EPW files, and
generates building/county outage scenarios.

[`2.Simulation`](./2.Simulation/) contains the ResStock project configuration,
custom outage/islanding OpenStudio measures, the batch controller, and the
metric postprocessor. This stage must be used with the external
[NatLabRockies/resstock](https://github.com/NatLabRockies/resstock) codebase.

[`3.Optimization`](./3.Optimization/) annualizes retrofit costs, solves the
cost-, carbon-, resilience-, and equity-oriented pathway models, evaluates
future cost-trend cases, and maps optimized ResStock templates back to real
buildings.

Detailed file descriptions and execution order are provided in the `README.md`
inside each folder.

## Path configuration

No user-specific absolute paths are stored in the code. By default, the
optimization scripts read metric files from `Data/Metrics/` and write results
to `Data/Decisions/`. Environment variables can point to external datasets or
separate output locations. The most widely used variables are:

- `PROJECT_DATA_DIR`: root directory for project data
- `OPTIMIZATION_INPUT_ROOT`: optional override for the `Metrics` directory
- `OPTIMIZATION_OUTPUT_ROOT`: optional override for the `Decisions` directory
- `TARGET_STATES`: optional comma-separated state abbreviations
- `TARGET_FIPS`: optional comma-separated county FIPS codes; bypasses `upgrade0.csv`
- `SSP_SCENARIOS`: optional comma-separated SSP scenarios
- `RESSTOCK_METADATA_FILE`: path to `upgrade0.csv`
- `PRECOMPUTED_FIXED_MAPPING`: optional reviewed mapping to reuse directly
- `RESSTOCK_DIR`: external ResStock checkout
- `RESSTOCK_PROJECT_FILE`: active BuildStockBatch project YAML
- `FUTURE_WEATHER_DIR`: future weather/EPW root

See the folder-level READMEs for stage-specific variables and expected inputs.

## Quick-start demo

[`Demo`](./Demo/) provides a compact end-to-end reproduction for Autauga
County, Alabama (`FIPS 01001`) under SSP1-2.6. It includes all 17 retrofit
metric files for 66 ResStock template buildings and a reviewed fixed mapping
for 25,573 real buildings. The demo therefore runs the two optimization entry
points without downloading the full national dataset.

From the repository root:

```bash
python -m pip install -r Demo/requirements.txt
python Demo/run_demo.py --max-cost-cases 1
```

The command above is a short end-to-end smoke test. Run all 40 reproducible
future cost-trend cases with:

```bash
python Demo/run_demo.py
```

The launcher validates the bundled inputs before optimization, uses only
repository-relative paths, and writes generated files under `Demo/outputs/`.
A working Gurobi installation and license are required. See the
[`Demo` documentation](./Demo/README.md) for validation-only, staged, and
resume commands, the expected outputs, and the mapping between short output
folder names and complete scenario names.

## Citation

Please cite the associated manuscript and this repository when using the code
or processed data. Full bibliographic information will be added here after
publication.
