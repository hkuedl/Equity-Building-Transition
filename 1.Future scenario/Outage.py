import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path


PROJECT_DATA_DIR = Path(
    os.environ.get(
        "PROJECT_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data",
    )
)
BASE_DIR = os.environ.get(
    "BUILDING_OUTPUT_DIR",
    str(PROJECT_DATA_DIR / "building_inventory"),
)
OUTAGE_ROOT_DIR = os.environ.get(
    "OUTAGE_SCENARIO_DIR",
    str(PROJECT_DATA_DIR / "outage" / "presto"),
)
GAZETTEER_PATH = os.environ.get(
    "COUNTY_GAZETTEER_FILE",
    str(PROJECT_DATA_DIR / "geography" / "counties_gazetteer.txt"),
)
EAGLEI_FILE = os.environ.get(
    "EAGLEI_EVENTS_FILE",
    str(PROJECT_DATA_DIR / "outage" / "eaglei_outages_with_events.csv"),
)

US_STATES = [
    ("Alabama", "AL", 1), ("Arizona", "AZ", 4), ("Arkansas", "AR", 5),
    ("California", "CA", 6), ("Colorado", "CO", 8), ("Connecticut", "CT", 9),
    ("Delaware", "DE", 10), ("District of Columbia", "DC", 11), ("Florida", "FL", 12),
    ("Georgia", "GA", 13), ("Idaho", "ID", 16), ("Illinois", "IL", 17),
    ("Indiana", "IN", 18), ("Iowa", "IA", 19), ("Kansas", "KS", 20),
    ("Kentucky", "KY", 21), ("Louisiana", "LA", 22), ("Maine", "ME", 23),
    ("Maryland", "MD", 24), ("Massachusetts", "MA", 25), ("Michigan", "MI", 26),
    ("Minnesota", "MN", 27), ("Mississippi", "MS", 28), ("Missouri", "MO", 29),
    ("Montana", "MT", 30), ("Nebraska", "NE", 31), ("Nevada", "NV", 32),
    ("New Hampshire", "NH", 33), ("New Jersey", "NJ", 34), ("New Mexico", "NM", 35),
    ("New York", "NY", 36), ("North Carolina", "NC", 37), ("North Dakota", "ND", 38),
    ("Ohio", "OH", 39), ("Oklahoma", "OK", 40), ("Oregon", "OR", 41),
    ("Pennsylvania", "PA", 42), ("Rhode Island", "RI", 44), ("South Carolina", "SC", 45),
    ("South Dakota", "SD", 46), ("Tennessee", "TN", 47), ("Texas", "TX", 48),
    ("Utah", "UT", 49), ("Vermont", "VT", 50), ("Virginia", "VA", 51),
    ("Washington", "WA", 53), ("West Virginia", "WV", 54), ("Wisconsin", "WI", 55),
    ("Wyoming", "WY", 56)
]

TARGET_STATES = {
    value.strip().upper()
    for value in os.environ.get("TARGET_STATES", "").split(",")
    if value.strip()
}

TARGET_YEARS = [2025, 2035, 2045, 2055]
GRID_HARDENING_FACTORS = {2025: 1.0, 2035: 0.7, 2045: 0.45, 2055: 0.2}
GLOBAL_SEED = 2025


def load_counties_by_state(txt_path, state_abbr):
    """Load county FIPS codes for a state from the national gazetteer."""
    df = pd.read_csv(txt_path, sep='\t', encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    state_df = df[df['USPS'] == state_abbr].copy()
    state_df['GEOID'] = state_df['GEOID'].astype(str).str.zfill(5)
    return state_df['GEOID'].tolist()

def find_presto_file(folder, county_id):
    """Find the Presto outage workbook associated with a county FIPS."""
    search_pattern = os.path.join(folder, f"*{county_id}*.xlsx")
    files = glob.glob(search_pattern)
    return files[0] if files else None

def get_historical_stats(eaglei_path):
    """Aggregate historical EAGLE-I outage customer statistics by county."""
    df_e = pd.read_csv(eaglei_path)
    stats = df_e.groupby('fips').agg({
        'min_customers': 'mean',
        'max_customers': 'mean',
        'mean_customers': 'mean'
    }).reset_index()
    return stats.set_index('fips').to_dict('index')

def downscale_events_to_buildings(county_events, buildings_subset, year, base_seed=42):
    """Assign county outage events to individual buildings reproducibly."""
    outage_records = {b_id: [] for b_id in buildings_subset['BUILD_ID']}
    current_iter_seed = base_seed

    for _, event in county_events.iterrows():
        target_customers = event['customers_out']
        start_time = event['time']
        duration = event['duration_hr']
        end_time = start_time + timedelta(hours=duration)

        candidate_buildings = buildings_subset.copy()
        customers_affected = 0

        while customers_affected < target_customers and not candidate_buildings.empty:
            weights = candidate_buildings['bvi'] if candidate_buildings['bvi'].sum() > 0 else None
            try:
                chosen = candidate_buildings.sample(n=1, weights=weights, random_state=current_iter_seed)
            except:
                chosen = candidate_buildings.sample(n=1, random_state=current_iter_seed)

            idx = chosen.index[0]
            b_id = chosen['BUILD_ID'].iloc[0]
            outage_records[b_id].append({'start': start_time, 'end': end_time})
            customers_affected += chosen['assigned_people'].iloc[0]
            candidate_buildings.drop(idx, inplace=True)
            current_iter_seed += 1

    results = []
    for b_id, outages in outage_records.items():
        if not outages:
            results.append({'BUILD_ID': b_id, f'Year: {year} start': None, f'Year: {year} end': None})
        else:
            sorted_outages = sorted(outages, key=lambda x: x['start'])
            all_starts = "; ".join([o['start'].strftime("%m/%d/%H") for o in sorted_outages])
            all_ends = "; ".join([o['end'].strftime("%m/%d/%H") for o in sorted_outages])
            results.append({'BUILD_ID': b_id, f'Year: {year} start': all_starts, f'Year: {year} end': all_ends})
    return pd.DataFrame(results)


eaglei_stats = get_historical_stats(EAGLEI_FILE)
global_avg_out = sum(d['mean_customers'] for d in eaglei_stats.values()) / len(eaglei_stats)

for state_name, state_abbr, _ in US_STATES:
    if TARGET_STATES and state_abbr not in TARGET_STATES:
        continue
    print(f"\n" + "#"*60)
    print(f"Processing state: {state_name} ({state_abbr})")


    state_outage_dir = os.path.join(OUTAGE_ROOT_DIR, state_abbr)
    building_file = os.path.join(BASE_DIR, state_abbr, f'{state_abbr}_ResStock.parquet')
    output_parquet = os.path.join(BASE_DIR, state_abbr, f'{state_abbr}_ResStock_Outage.parquet')
    output_excel = os.path.join(BASE_DIR, state_abbr, f'{state_abbr}_County_Hourly_Outage_Status.xlsx')

    if not os.path.exists(building_file):
        print(f"Skipping {state_abbr}: Building inventory file not found {building_file}")
        continue


    df_building_master = pd.read_parquet(building_file)
    df_building_master['county_fips'] = df_building_master['county_fips'].astype(str).str.zfill(5)
    for year in TARGET_YEARS:
        df_building_master[f'Year: {year} start'] = None
        df_building_master[f'Year: {year} end'] = None
    final_df = df_building_master.set_index('BUILD_ID')


    county_ids = load_counties_by_state(GAZETTEER_PATH, state_abbr)
    excel_data_container = {year: {} for year in TARGET_YEARS}

    for fips in county_ids:
        print(f"  Processing county: {fips}...")


        county_buildings = df_building_master[df_building_master['county_fips'] == fips].copy()
        if county_buildings.empty: continue


        c_inc_min, c_inc_max = county_buildings['bldg_assigned_income'].min(), county_buildings['bldg_assigned_income'].max()
        county_buildings['bvi'] = 1 - (county_buildings['bldg_assigned_income'] - c_inc_min) / (c_inc_max - c_inc_min) if c_inc_max > c_inc_min else 1.0


        excel_path = find_presto_file(state_outage_dir, fips)
        if not excel_path:
            print(f"    Warning: FIPS file not found {fips} Presto simulation file")
            continue


        df_raw = pd.read_excel(excel_path, sheet_name='Results')
        df_raw.columns = [c.strip().replace(' ', '_') for c in df_raw.columns]
        sim_stats = df_raw.groupby('simulation_#')['duration_hours'].sum().reset_index()
        p90_val = sim_stats['duration_hours'].quantile(0.90)
        best_sim_id = sim_stats.loc[(sim_stats['duration_hours'] - p90_val).abs().idxmin(), 'simulation_#']

        df_base_events = df_raw[df_raw['simulation_#'] == best_sim_id].copy()


        date_config = pd.DataFrame({
            'year': 2025,
            'month': df_base_events['month'],
            'day': df_base_events['day_of_month'],
            'hour': df_base_events['hour_of_day']
        })


        df_base_events['time'] = pd.to_datetime(date_config, errors='coerce')


        invalid_dates = df_base_events[df_base_events['time'].isna()]
        if not invalid_dates.empty:
            print(f"    [] county {fips} removed {len(invalid_dates)} invalid dates outside 2025( 229)")


        df_base_events = df_base_events.dropna(subset=['time']).copy()


        df_base_events.rename(columns={'duration_hours': 'duration_hr'}, inplace=True)


        np.random.seed(GLOBAL_SEED + int(fips))
        county_pop = county_buildings['assigned_people'].sum()
        fips_int = int(fips)
        low_out, high_out = (eaglei_stats[fips_int]['min_customers'], eaglei_stats[fips_int]['max_customers']) if fips_int in eaglei_stats else (global_avg_out*0.5, global_avg_out*1.5)
        sample_ratios = np.random.uniform(low_out/max(county_pop, 1), high_out/max(county_pop, 1), size=len(df_base_events))
        df_base_events['customers_out'] = np.maximum((sample_ratios * county_pop).astype(int), 1)


        for year in TARGET_YEARS:
            hourly_status = np.ones(8760, dtype=int)
            factor = GRID_HARDENING_FACTORS[year]
            num_ev = int(len(df_base_events) * factor)

            if num_ev > 0:
                yr_events = df_base_events.sample(n=num_ev, random_state=year).copy()
                yr_events['duration_hr'] *= factor
                yr_events['customers_out'] = np.maximum((yr_events['customers_out'] * factor).astype(int), 1)


                for _, ev in yr_events.iterrows():
                    delta = ev['time'] - datetime(2025, 1, 1, 0, 0)
                    start_hour = int(delta.total_seconds() // 3600)
                    duration = int(np.ceil(ev['duration_hr']))
                    end_hour = min(start_hour + duration, 8760)
                    if 0 <= start_hour < 8760:
                        hourly_status[start_hour:end_hour] = 0


                building_outage = downscale_events_to_buildings(yr_events, county_buildings, year, base_seed=GLOBAL_SEED + int(fips) + year)
                building_outage.set_index('BUILD_ID', inplace=True)
                final_df.update(building_outage)

            excel_data_container[year][fips] = hourly_status


    final_df.reset_index().to_parquet(output_parquet, index=False)
    year_mapping = {2025: '2020s', 2035: '2030s', 2045: '2040s', 2055: '2050s'}
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        for year in TARGET_YEARS:
            pd.DataFrame(excel_data_container[year]).to_excel(writer, sheet_name=year_mapping[year], index=False)

    print(f"--- state {state_abbr} completed; files saved ---")

print("\n All state simulations completed!")
