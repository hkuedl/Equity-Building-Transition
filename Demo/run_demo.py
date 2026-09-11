"""Run the Alabama 01001 SSP1-2.6 optimization demo.

All paths are resolved from this file, so the demo can be launched from any
working directory after the repository is downloaded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


STATE = "AL"
FIPS = "01001"
SSP = "ssp126"
UPGRADES = tuple(f"up{number:02d}" for number in range(1, 18))
DECADES = ("2020s", "2030s", "2040s", "2050s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-contained SSP1-2.6 optimization demo for AL/01001."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "main", "prediction"),
        default="all",
        help="Run both optimization entry points, or only one of them (default: all).",
    )
    parser.add_argument(
        "--max-cost-cases",
        type=int,
        default=None,
        help=(
            "Limit Main_Prediction.py to the first N reproducible cost cases. "
            "By default, all 40 cases are run. Use 1 for a short smoke test."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the bundled Parquet inputs without starting optimization.",
    )
    args = parser.parse_args()
    if args.max_cost_cases is not None and args.max_cost_cases <= 0:
        parser.error("--max-cost-cases must be a positive integer")
    return args


def expected_metric_columns() -> set[str]:
    columns = {"Building_ID", "Income", "FIPS", "sqft"}
    for decade in DECADES:
        columns.update(
            {
                f"Carbon(kg)_{decade}",
                f"Comfort_Extreme_{decade}",
                f"Comfort_Outage_{decade}",
                f"Energy_cost_{decade}",
                f"Install_cost_{decade}",
            }
        )
    return columns


def validate_inputs(demo_dir: Path) -> tuple[Path, list[Path]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'pyarrow'. Install Demo/requirements.txt first."
        ) from exc

    state_dir = demo_dir / STATE
    metrics_dir = state_dir / f"FIPS_{FIPS}"
    mapping_path = state_dir / f"{FIPS}_fixed_real_to_template_mapping.parquet"
    metric_paths = [metrics_dir / f"{SSP}_{upgrade}.parquet" for upgrade in UPGRADES]

    missing_files = [path for path in [mapping_path, *metric_paths] if not path.is_file()]
    if missing_files:
        formatted = "\n".join(f"  - {path.relative_to(demo_dir)}" for path in missing_files)
        raise SystemExit(f"Demo input files are missing:\n{formatted}")

    required_metrics = expected_metric_columns()
    template_ids: set[str] = set()
    expected_rows = None
    for path in metric_paths:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema.names)
        missing_columns = sorted(required_metrics - available)
        if missing_columns:
            raise SystemExit(
                f"{path.name} is missing required columns: {', '.join(missing_columns)}"
            )
        rows = parquet_file.metadata.num_rows
        if expected_rows is None:
            expected_rows = rows
        elif rows != expected_rows:
            raise SystemExit(
                f"Metric row-count mismatch: {path.name} has {rows}, expected {expected_rows}."
            )

        ids = pq.read_table(path, columns=["Building_ID", "FIPS"]).to_pydict()
        fips_values = {str(value).zfill(5) for value in ids["FIPS"] if value is not None}
        if fips_values != {FIPS}:
            raise SystemExit(f"{path.name} contains unexpected FIPS values: {fips_values}")
        template_ids.update(str(value) for value in ids["Building_ID"] if value is not None)

    required_mapping = {
        "BUILD_ID",
        "county_fips",
        "matched_bldg_id",
        "matched_template_fips",
        "Template_Key",
        "sqft",
        "template_sqft",
        "bldg_assigned_income",
        "mapping_reference_ssp",
        "mapping_reference_mode",
    }
    mapping_file = pq.ParquetFile(mapping_path)
    missing_mapping_columns = sorted(required_mapping - set(mapping_file.schema.names))
    if missing_mapping_columns:
        raise SystemExit(
            "The fixed mapping is missing required columns: "
            + ", ".join(missing_mapping_columns)
        )

    mapping = pq.read_table(
        mapping_path,
        columns=[
            "county_fips",
            "Template_Key",
            "mapping_reference_ssp",
            "mapping_reference_mode",
        ],
    ).to_pydict()
    mapping_fips = {
        str(value).zfill(5) for value in mapping["county_fips"] if value is not None
    }
    if mapping_fips != {FIPS}:
        raise SystemExit(f"The fixed mapping contains unexpected FIPS values: {mapping_fips}")
    if set(mapping["mapping_reference_ssp"]) != {SSP}:
        raise SystemExit("The fixed mapping was not generated from ssp126.")
    if set(mapping["mapping_reference_mode"]) != {"Equity_DP"}:
        raise SystemExit("The fixed mapping was not generated from Equity_DP.")

    expected_template_keys = {f"{FIPS}|{building_id}" for building_id in template_ids}
    mapped_template_keys = {
        str(value) for value in mapping["Template_Key"] if value is not None
    }
    unknown_template_keys = mapped_template_keys - expected_template_keys
    if unknown_template_keys:
        examples = ", ".join(sorted(unknown_template_keys)[:5])
        raise SystemExit(
            "The fixed mapping references templates absent from the metrics. "
            f"Examples: {examples}"
        )

    print(
        "Input validation passed: "
        f"17 upgrades, {expected_rows} template buildings per upgrade, "
        f"{mapping_file.metadata.num_rows:,} mapped real buildings."
    )
    return mapping_path, metric_paths


def build_environment(
    demo_dir: Path,
    mapping_path: Path,
    max_cost_cases: int | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_DATA_DIR": str(demo_dir),
            "OPTIMIZATION_INPUT_ROOT": str(demo_dir),
            "OPTIMIZATION_OUTPUT_ROOT": str(demo_dir / "outputs"),
            "PRECOMPUTED_FIXED_MAPPING": str(mapping_path),
            "TARGET_STATES": STATE,
            "TARGET_FIPS": FIPS,
            "SSP_SCENARIOS": SSP,
            # Full scenario names can exceed the legacy Windows 260-character
            # path limit when the repository is stored in a deeply nested folder.
            "SHORT_COST_CASE_PATHS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if max_cost_cases is None:
        env.pop("MAX_COST_TREND_CASES", None)
    else:
        env["MAX_COST_TREND_CASES"] = str(max_cost_cases)
    return env


def run_script(script_path: Path, repo_root: Path, env: dict[str, str]) -> None:
    relative_script = script_path.relative_to(repo_root)
    print(f"\nRunning {relative_script}", flush=True)
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(
            f"{relative_script} failed with exit code {return_code}. "
            "The underlying error is shown above."
        )


def main() -> None:
    args = parse_args()
    demo_dir = Path(__file__).resolve().parent
    repo_root = demo_dir.parent
    optimization_dir = repo_root / "3.Optimization"
    mapping_path, _ = validate_inputs(demo_dir)
    if args.validate_only:
        return

    env = build_environment(demo_dir, mapping_path, args.max_cost_cases)
    scripts = []
    if args.stage in {"all", "main"}:
        scripts.append(optimization_dir / "Main.py")
    if args.stage in {"all", "prediction"}:
        scripts.append(optimization_dir / "Main_Prediction.py")

    for script_path in scripts:
        if not script_path.is_file():
            raise SystemExit(f"Optimization entry point not found: {script_path}")
        run_script(script_path, repo_root, env)

    print(f"\nDemo completed. Outputs: {demo_dir / 'outputs'}")


if __name__ == "__main__":
    main()
