# Equitable Residential Building Retrofit Pathways

_This project develops a building-level workflow for evaluating residential
retrofit strategies under future climate and power-outage conditions. It
combines future weather projections, ResStock simulations, postprocessed
energy/carbon/comfort/cost metrics, and pathway optimization
to identify retrofit decisions for individual buildings across the contiguous United
States._

The repository contains the code used to construct future scenarios, run
building simulations, calculate strategy metrics, and optimize cost-, carbon-,
resilience-, and equity-oriented retrofit pathways.

## Requirements

[Python](https://www.python.org/) 3.10 or newer is recommended.

The Python workflows use `pandas`, `numpy`, `xarray`, `scipy`,
`scikit-learn`, `geopandas`, `rasterio`, `pyproj`, `pyarrow`, `cftime`,
`metpy`, `matplotlib`, `openpyxl`, and `pyomo`. A supported mathematical
programming solver is also required for the optimization workflow; the current
configuration uses Gurobi.

The simulation workflow additionally requires:

- [ResStock](https://github.com/NatLabRockies/resstock)
- [buildstockbatch](https://github.com/NatLabRockies/buildstockbatch)
- OpenStudio and EnergyPlus versions compatible with the selected ResStock release

## Data

The complete processed dataset is available from
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
under the pathways, including baseline,
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
the county-level structure. The dataset contains all counties in
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

No user-specific absolute paths are stored in the code. The scripts use a
repository-relative `data/` directory by default and accept environment
variables for external datasets and outputs. The most widely used variables
are:

- `PROJECT_DATA_DIR`: root directory for project data
- `TARGET_STATES`: optional comma-separated state abbreviations
- `RESSTOCK_METADATA_FILE`: path to `upgrade0.csv`
- `RESSTOCK_DIR`: external ResStock checkout
- `RESSTOCK_PROJECT_FILE`: active BuildStockBatch project YAML
- `FUTURE_WEATHER_DIR`: future weather/EPW root

See the folder-level READMEs for stage-specific variables and expected inputs.

## Citation

Please cite the associated manuscript and this repository when using the code
or processed data. Full bibliographic information will be added here after
publication.

```
```
