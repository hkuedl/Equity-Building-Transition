# Quick-start optimization demo

This folder is a self-contained, county-scale example for Alabama FIPS `01001`
(Autauga County). It runs the two entry points in `3.Optimization` using only
SSP1-2.6 (`ssp126`). No `upgrade0.csv` or full state mapping dataset is needed.

## Included inputs

```text
Demo/
├── run_demo.py
├── requirements.txt
└── AL/
    ├── 01001_fixed_real_to_template_mapping.parquet
    └── FIPS_01001/
        ├── ssp126_up01.parquet
        ├── ...
        └── ssp126_up17.parquet
```

The 17 metric files contain the optimization inputs for 66 ResStock template
buildings. The fixed mapping links 25,573 real buildings to those templates and
is reused directly; the demo does not rebuild it.

## Requirements

Use Python 3.10 or newer. From the repository root, install the demo packages:

```bash
python -m pip install -r Demo/requirements.txt
```

The optimization is solved with Gurobi through Pyomo. A working Gurobi
installation and license are therefore required.

## Run

From the repository root:

```bash
python Demo/run_demo.py
```

This first runs `3.Optimization/Main.py`, then
`3.Optimization/Main_Prediction.py`. The prediction stage runs all 40
reproducible cost-trend cases by default. For a short end-to-end smoke test:

```bash
python Demo/run_demo.py --max-cost-cases 1
```

Other useful commands are:

```bash
# Check filenames, schemas, scenario labels, FIPS values, and mapping keys only
python Demo/run_demo.py --validate-only

# Run just one optimization entry point
python Demo/run_demo.py --stage main
python Demo/run_demo.py --stage prediction --max-cost-cases 1
```

If `Main.py` has already completed, resume directly from the prediction stage:

```bash
python Demo/run_demo.py --stage prediction
```

## Expected behavior

Before optimization, a valid bundle prints:

```text
Input validation passed: 17 upgrades, 66 template buildings per upgrade,
25,573 mapped real buildings.
```

The supplied metrics contain one incomplete `up16` record for template
building `249540` in the 2030s. The base workflow therefore reports that one
building in the one-size-fits-all mode falls back to the `up17` baseline. This
is an intentional, conservative missing-data rule rather than a failed run;
the remaining optimization modes continue normally.

A complete prediction run creates all 40 folders from `Case_01` through
`Case_40`. Each case contains one group-level template result, one fixed
mapping, one real-building result, and one mapping report. The run also writes
`AL_ssp126_real_retrofit_timing_by_income.csv` at the `Decision_Pre` level.

The launcher resolves every path relative to `run_demo.py`, so it also works
when called from another directory. Generated files will be kept separate from the
bundled inputs under:

```text
Demo/outputs/AL/
├── Decision_Robust/
└── Decision_Pre/
```

`Decision_Robust` contains the base pathway results and the staged canonical
mapping. `Decision_Pre` contains the dynamic timing and cost-trend results.
Its 40 cost cases use the stable short folder names `Case_01` through `Case_40`
to avoid the legacy Windows 260-character path limit. The complete scientific
scenario name and its corresponding output folder are recorded in
`sampled_40_cost_trend_scenarios.csv`.