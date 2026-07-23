import os
import sys
import gc
import re
import time
import multiprocessing
from typing import List, Dict, Any, Optional
from buildstockbatch.local import LocalBatch
from pathlib import Path
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


SCRIPT_DIR = Path(__file__).resolve().parent

ROOT_DIR = SCRIPT_DIR.parent

RESSTOCK_DIR = Path(
    os.environ.get(
        "RESSTOCK_DIR",
        ROOT_DIR / "resstock",
    )
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "SIMULATION_OUTPUT_DIR",
        ROOT_DIR / "simulation_output",
    )
)


def set_parallel_cores(max_cores: int = 15) -> None:
    """Set the LOKY worker limit while leaving capacity for the system."""
    os.environ['LOKY_MAX_CPU_COUNT'] = str(max_cores)
    print(f"--> Parallel worker limit: {os.environ['LOKY_MAX_CPU_COUNT']}")

def change_to_project_dir(project_filename: str) -> str:
    """Change to the project YAML directory and return it."""
    project_dir = os.path.dirname(project_filename)
    os.chdir(project_dir)
    print(f"--> Working directory: {os.getcwd()}")
    return project_dir

def detect_config_attr(bsb: LocalBatch) -> str:
    """Detect the LocalBatch configuration attribute."""
    candidate_attrs = ["cfg", "config", "project_config"]
    for attr in candidate_attrs:
        if hasattr(bsb, attr):
            print(f"--> Configuration attribute: {attr}")
            return attr

    print(" Configuration attribute not found!")
    print(f"[Debug] bsb object attributes: {dir(bsb)}")
    sys.exit(1)

def filter_upgrades(bsb: LocalBatch, target_upgrade_names: Optional[List[str]] = None) -> None:
    if not target_upgrade_names:
        print("\n No target upgrades specified; running the baseline or all configured upgrades.\n")
        return

    print(f"\n Filtering upgrades to: {target_upgrade_names}")

    config_attr = detect_config_attr(bsb)
    current_config: Dict[str, Any] = getattr(bsb, config_attr)
    all_upgrades: List[Dict[str, Any]] = current_config.get("upgrades", [])

    targets = target_upgrade_names
    filtered_upgrades = [
        u for u in all_upgrades
        if u.get("upgrade_name") in targets
    ]

    if not filtered_upgrades:
        print(" Error: Requested upgrades were not found in the YAML file!")
        print("   Requested names:")
        for name in targets:
            print(f"   - {name}")
        print("\n   Available upgrades:")
        for u in all_upgrades:
            print(f"   - {u.get('upgrade_name')}")
        sys.exit(1)

    current_config["upgrades"] = filtered_upgrades
    print(f" Upgrade filter applied (: {config_attr}),starting simulations for {len(filtered_upgrades)} upgrade scenarios...\n")

def run_batch_and_process_results(bsb: LocalBatch, low_disk: bool = False, measures_only: bool = False) -> None:
    """"""
    try:
        print("--> 1. Starting run_batch ...")


        n_jobs = int(os.environ.get("SLURM_CPUS_ON_NODE", "512"))
        print(f"-->  n_jobs = {n_jobs}")

        bsb.run_batch(
            n_jobs=n_jobs,
            measures_only=measures_only,
            sampling_only=False,
            low_disk=low_disk,
        )

        print("-->  Batch completed successfully!")
    except Exception as e:
        print(f"\n[Error] Run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_one_file(file: Path, src_dir: Path, dst_dir: Path, keywords):
    """Process parquet file()"""
    try:
        df = pd.read_parquet(file)

        cols = [c for c in df.columns if any(k in c for k in keywords)]
        if not cols:
            return file.name, 0, "no_match", []

        df_filtered = df[cols]


        rel_path = file.relative_to(src_dir)
        out_file = dst_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_parquet(out_file, index=False)


        return file.name, len(cols), "ok", cols

    except Exception as e:

        return file.name, 0, f"error: {e}", []

def filter_parquet_folder_parallel(src_dir, dst_dir, keywords, max_workers=10):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)


    files = list(src_dir.rglob('*.parquet'))
    print(f"Found {len(files)} parquet files in {src_dir} (including subfolders)")

    if not files:
        return


    all_columns = set()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_file, file, src_dir, dst_dir, keywords): file
            for file in files
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                filename, ncols, status, cols = future.result()


                all_columns.update(cols)

                if status == "ok":
                    print(f"[OK] {filename}: saved {ncols} columns.")
                elif status == "no_match":
                    print(f"[SKIP] {filename}: no columns matched keywords.")
                else:  # error
                    print(f"[ERROR] {filename}: {status}")
            except Exception as e:
                print(f"[EXCEPTION] {file.name}: {e}")


    print("\n===== SUMMARY =====")
    print(f"Total unique column names after filtering: {len(all_columns)}")


    for col in sorted(all_columns):
        print(col)


    summary_file = Path(dst_dir) / "all_filtered_columns.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        for col in sorted(all_columns):
            f.write(col + "\n")
    print(f"\nColumn name list saved to: {summary_file}")


def modify_project_yaml(
    project_filename: str,
    climate_scenario: str,
    weather_scenario: str,
    change_weather: bool = True,
    new_weather_root: Optional[str] = None,
) -> tuple[str, str]:

    weather_outage = weather_scenario.split("_")[0]

    """ YAML file"""
    project_path = Path(project_filename)
    if not project_path.is_file():
        raise FileNotFoundError(f"Project file not found: {project_path}")

    text = project_path.read_text(encoding="utf-8")
    new_text = text


    new_text = new_text.replace("Outage|ssp126 2020s", f"Outage|{climate_scenario} {weather_outage}")
    new_text = new_text.replace("ssp_scenario: ssp245", f"ssp_scenario: {climate_scenario}")
    new_text = new_text.replace("scenario_year: 2020s", f"scenario_year: {weather_outage}")


    if change_weather:
        pattern_weather = re.compile(
            r"^(weather_files_path:\s*)(\S+)(\s*(#.*)?)$",
            flags=re.MULTILINE
        )

        def replace_weather(match: re.Match) -> str:
            prefix = match.group(1)
            old_path = match.group(2)
            comment = match.group(3) or ""

            if new_weather_root:
                base_dir = new_weather_root.rstrip('/')
                new_path = os.path.join(
                    base_dir,
                    climate_scenario,
                    f"{weather_scenario}/weather_files.zip",
                )
            else:
                base_dir = os.path.dirname(old_path)
                new_path = os.path.join(
                    base_dir,
                    "new_weather",
                    climate_scenario,
                    f"{weather_scenario}/weather_files.zip",
                )
            return f"{prefix}{new_path}{comment}"

        new_text, n_weather = pattern_weather.subn(replace_weather, new_text)
        if n_weather == 0:
            raise RuntimeError("No 'weather_files_path:' line found in file.")


    pattern_output = re.compile(
        r"^output_directory:.*$",
        flags=re.MULTILINE
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = OUTPUT_ROOT / "buildbatch_output"
    base_output = Path(f"{base_output}_{timestamp}")


    output_folder = str(base_output / climate_scenario / weather_scenario)
    new_output_line = f"output_directory: {output_folder}"
    new_text, _ = pattern_output.subn(new_output_line, new_text)


    stem = project_path.stem
    new_path = project_path.with_name(f"{stem}_{climate_scenario}_{weather_scenario}.yml")
    new_path.write_text(new_text, encoding="utf-8")

    return str(new_path.resolve()), str(output_folder)


def worker_task(project_filename, climate, weather, target_upgrade_names, new_weather_root):
    """Run one climate/weather combination in an isolated worker process."""
    print(f"\n{'='*60}")
    print(f" [Worker started] Climate={climate}, Weather={weather}")
    print(f"   PID: {os.getpid()}")
    print(f"{'='*60}\n")

    try:

        project_filename_new, output_folder = modify_project_yaml(
            project_filename,
            climate,
            weather,
            change_weather=True,
            new_weather_root=new_weather_root,
        )

        print(project_filename)
        print(project_filename_new)


        set_parallel_cores(max_cores=64)
        change_to_project_dir(project_filename_new)


        bsb = LocalBatch(project_filename_new)


        filter_upgrades(bsb, target_upgrade_names=target_upgrade_names)


        if 'baseline' not in bsb.cfg:
            bsb.cfg['baseline'] = {}
        bsb.cfg['baseline']['skip_sims'] = True


        run_batch_and_process_results(bsb, low_disk=True, measures_only=False)


    except Exception as e:
        print(f" [Worker failed] PID {os.getpid()} : {e}")
        import traceback
        traceback.print_exc()
    finally:

        gc.collect()


def main():

    project_filename = os.environ.get(
        "RESSTOCK_PROJECT_FILE",
        str((RESSTOCK_DIR / "project_inequity" / "project_controll.yml").resolve()),
    )

    target_upgrade_level_C = [
        "Upgrade99_Baseline_Poweroff",
        "Upgrade1_Wall_Maintenance",
        "Upgrade2_Wall_Maintenance_HVAC_Replace",
        "Upgrade3_Full_Maintenance_Infil13",
    ]

    target_upgrade_level_B = [
        "Upgrade99_Baseline_Poweroff",
        "Upgrade4_Light_Roof_HVAC",
        "Upgrade5_Light_Wall_HVAC",
        "Upgrade6_Light_Roof_Wall_HVAC",
        "Upgrade7_Light_FullEnvelope_HVAC",
        "Upgrade8_Light_Roof_Elec",
        "Upgrade9_Light_Wall_Elec",
        "Upgrade10_Light_Roof_Wall_Elec",
        "Upgrade11_Light_FullEnvelope_Elec",
        "Upgrade12_Light_Roof_Elec_PV5",
        "Upgrade13_Light_Wall_Elec_PV5",
        "Upgrade14_Light_Roof_Wall_Elec_PV5",
        "Upgrade15_Light_FullEnvelope_Elec_PV5",
        "Upgrade16_Light_Roof_Elec_PV5_B10",
        "Upgrade17_Light_Wall_Elec_PV5_B10",
        "Upgrade18_Light_Roof_Wall_Elec_PV5_B10",
        "Upgrade19_Light_FullEnvelope_Elec_PV5_B10"
    ]

    target_upgrade_level_A = [
        "Upgrade99_Baseline_Poweroff",
        "Upgrade20_Deep_WinIECC_Fossil",
        "Upgrade21_Deep_RoofIECC_Fossil",
        "Upgrade22_Deep_WallIECC_Fossil",
        "Upgrade23_Deep_EnvelopeIECC_Fossil",
        "Upgrade24_Deep_WinIECC_CCHP",
        "Upgrade25_Deep_RoofIECC_CCHP",
        "Upgrade26_Deep_WallIECC_CCHP",
        "Upgrade27_Deep_EnvelopeIECC_CCHP",
        "Upgrade28_Deep_WinIECC_CCHP_PV11",
        "Upgrade29_Deep_RoofIECC_CCHP_PV11",
        "Upgrade30_Deep_WallIECC_CCHP_PV11",
        "Upgrade31_Deep_EnvelopeIECC_CCHP_PV11",
        "Upgrade32_Deep_WinIECC_CCHP_PV11_B20",
        "Upgrade33_Deep_RoofIECC_CCHP_PV11_B20",
        "Upgrade34_Deep_WallIECC_CCHP_PV11_B20",
        "Upgrade35_Deep_FullIECC_CCHP_PV11_B20"
    ]


    weather_scenario = ['2020s_Scenario_EPW']#, '2050s_Scenario_EPW', '2040s_Scenario_EPW', '2050s_Scenario_EPW']
    climate_scenario = ['ssp126'] #, 'ssp245', 'ssp585']

    NEW_WEATHER_ROOT = os.environ.get(
        "FUTURE_WEATHER_DIR",
        str(ROOT_DIR / "data" / "weather" / "epw"),
    )

    print(f" Main process started PID: {os.getpid()}")


    for climate in climate_scenario:
        for weather in weather_scenario:

            if climate == 'ssp126':
                target_upgrade_names = target_upgrade_level_A
            elif climate == 'ssp245':
                target_upgrade_names = target_upgrade_level_B
            elif climate == 'ssp585':
                target_upgrade_names = target_upgrade_level_C
            else:
                target_upgrade_names = [] # Handle default case


            p = multiprocessing.Process(
                target=worker_task,
                args=(project_filename, climate, weather, target_upgrade_names, NEW_WEATHER_ROOT)
            )

            p.start()
            p.join()


            if p.exitcode != 0:
                print(f" Warning: Worker exited with code {p.exitcode}  (Climate={climate}, Weather={weather})")

            print(" Waiting five seconds for I/O buffers...")
            time.sleep(5)

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', force=True)
    main()
