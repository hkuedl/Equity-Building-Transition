# File description

The files are listed below in recommended execution order.

1. `annualized_retrofit_cost.py`: Core retrofit-cost library. It maps each
   strategy to physical components, allocates total CAPEX by component,
   accounts for component service lives and replacements, and produces
   decade-specific annualized and cash-event cost columns.

2. `Robust_real_mapping.py`: Shared mapping module used by both optimization
   entry points. It maps optimized ResStock template decisions back to real
   buildings while preserving income-bin distributions, limiting template
   reuse, supporting county groups, and writing fixed mappings and diagnostic
   reports.

3. `Main.py`: Base pathway optimization workflow. It loads county strategy
   metrics, performs quality control, merges small or incomplete counties into
   valid groups, solves baseline, one-size-fits-all, carbon, resilience, cost,
   and equity-oriented modes, and maps the selected template pathways back to
   real buildings. Run this file first because it creates the fixed
   real-to-template mappings reused by the prediction workflow.

4. `annualized_retrofit_cost_Prediction.py`: Extends the cost model with
   installation-decade effects and reproducible active/passive component cost
   trends for future cost scenarios.

5. `Main_Prediction.py`: Runs dynamic retrofit-timing and future cost-trend
   experiments. It reuses the fixed mappings produced by `Main.py`, solves the
   configured equity modes across sampled cost cases, summarizes retrofit
   timing by income group, and exports cost-trend tables and figures.

## Inputs and outputs

The optimization inputs are county-level strategy metric Parquet files
produced by `2.Simulation/Metrics.py` and the real-building/ResStock mapping
produced by `1.Future scenario/Building_2_resstock_map.py`.

The base workflow writes quality reports, county-group maps, group template
decisions, fixed real-to-template mappings, and county-level building
decisions. The prediction workflow writes parallel cost-trend decision folders
while preserving the base fixed mapping.

## Configuration

No machine-specific paths are embedded in the scripts. Configure runs with:

- `PROJECT_DATA_DIR`: project input/output root
- `RESSTOCK_METADATA_FILE`: path to `upgrade0.csv`
- `TARGET_STATES`: optional comma-separated states; unset means all configured states
- `ALL_STATE_DATASETS`: optional comma-separated states whose metric directory uses the `{state}_all` suffix
- `REAL_MAPPING_TIMEOUT_SECONDS`: timeout before deterministic mapping fallback

The current optimizer is configured for Gurobi through Pyomo. Solver
availability and licensing must be handled in the execution environment.
