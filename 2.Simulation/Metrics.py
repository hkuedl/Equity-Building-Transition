import gzip
import json
import pandas as pd
import glob
import os
import re
import numpy as np
from pathlib import Path


US_STATES = [
    ("Alabama","AL",1),("Arizona","AZ",4),("Arkansas","AR",5),("California","CA",6),("Colorado","CO",8),("Connecticut","CT",9),
    ("Delaware","DE",10),("District of Columbia","DC",11),("Florida","FL",12),("Georgia","GA",13),("Idaho","ID",16),("Illinois","IL",17),
    ("Indiana","IN",18),("Iowa","IA",19),("Kansas","KS",20),("Kentucky","KY",21),("Louisiana","LA",22),("Maine","ME",23),
    ("Maryland","MD",24),("Massachusetts","MA",25),("Michigan","MI",26),("Minnesota","MN",27),("Mississippi","MS",28),("Missouri","MO",29),
    ("Montana","MT",30),("Nebraska","NE",31),("Nevada","NV",32),("New Hampshire","NH",33),("New Jersey","NJ",34),("New Mexico","NM",35),
    ("New York","NY",36),("North Carolina","NC",37),("North Dakota","ND",38),("Ohio","OH",39),("Oklahoma","OK",40),("Oregon","OR",41),
    ("Pennsylvania","PA",42),("Rhode Island","RI",44),("South Carolina","SC",45),("South Dakota","SD",46),("Tennessee","TN",47),("Texas","TX",48),
    ("Utah","UT",49),("Vermont","VT",50),("Virginia","VA",51),("Washington","WA",53),("West Virginia","WV",54),("Wisconsin","WI",55),("Wyoming","WY",56)
]

STATE_TO_DIVISION = {
    # New England
    "CT": "New England", "ME": "New England", "MA": "New England",
    "NH": "New England", "RI": "New England", "VT": "New England",
    # Middle Atlantic
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",
    # East North Central
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",
    # West North Central
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central", "ND": "West North Central", "SD": "West North Central",
    # South Atlantic
    "DE": "South Atlantic", "DC": "South Atlantic", "FL": "South Atlantic",
    "GA": "South Atlantic", "MD": "South Atlantic", "NC": "South Atlantic",
    "SC": "South Atlantic", "VA": "South Atlantic", "WV": "South Atlantic",
    # East South Central
    "AL": "East South Central", "KY": "East South Central", "MS": "East South Central", "TN": "East South Central",
    # West South Central
    "AR": "West South Central", "LA": "West South Central", "OK": "West South Central", "TX": "West South Central",
    # Mountain
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    # Pacific
    "CA": "Pacific", "OR": "Pacific", "WA": "Pacific"
}
CARBON_FACTORS_STATIC = {
    'natural_gas': 53.06,
    'propane': 63.11,
    'fuel_oil': 73.96,
    'wood_cord': 93.80,
    'coal': 95.52,
    'other': 0
}

UNIT_CONVERTERS = {
    'kwh': {'to_mmbtu': 0.003412, 'to_kbtu': 3.412},
    'kbtu': {'to_mmbtu': 0.001, 'to_kbtu': 1.0}
}


def identify_weather_extreme_days(temp_c, t_heat, t_cold):
    """Flag heat or cold events that persist for at least three days."""

    daily_reshaped = temp_c.reshape(-1, 24)
    daily_max, daily_min = daily_reshaped.max(axis=1), daily_reshaped.min(axis=1)

    is_heat, is_cold = daily_max >= t_heat, daily_min <= t_cold

    def get_mask(series):
        mask = np.zeros(len(series), dtype=bool)
        count = 0
        for i, val in enumerate(series):
            if val:
                count += 1
            else:
                if count >= 3:
                    mask[i-count:i] = True
                count = 0
        if count >= 3:
            mask[len(series)-count:] = True
        return mask

    return get_mask(is_heat) | get_mask(is_cold)

def calculate_metrics_dynamic(df, masks, fuel_data_mmbtu,
                              temp_col, elec_ef, prices_dict, LOWER_F, UPPER_F):
    """Calculate comfort, energy, carbon, and cost metrics for one case."""

    def compute_dh(mask):
        if mask.any() and temp_col in df.columns:
            sub_temp = df.loc[mask, temp_col]

            hot_side = (sub_temp[sub_temp > UPPER_F] - UPPER_F).sum()
            cold_side = (LOWER_F - sub_temp[sub_temp < LOWER_F]).sum()
            return hot_side + cold_side
        return 0


    comfort_ext_total = compute_dh(masks['extreme'])
    comfort_out_total = compute_dh(masks['outage'])
    comfort_uni_total = compute_dh(masks['union'])


    total_carbon = 0
    total_cost = 0
    total_energy_kbtu = 0

    price_key_map = {'electricity': 'Electricity', 'natural_gas': 'Natural Gas',
                     'propane': 'Propane', 'fuel_oil': 'Fuel Oil'}

    for f_type, val_mmbtu in fuel_data_mmbtu.items():
        total_energy_kbtu += val_mmbtu * 1000
        if f_type == 'electricity':
            total_carbon += (val_mmbtu * 293.071) * elec_ef
        else:
            ef = CARBON_FACTORS_STATIC.get(f_type, 0)
            total_carbon += val_mmbtu * ef

        p_key = price_key_map.get(f_type, '')
        price_per_mmbtu = prices_dict.get(p_key, 0)
        total_cost += val_mmbtu * price_per_mmbtu

    return (comfort_ext_total, comfort_out_total, comfort_uni_total), total_carbon, total_cost, total_energy_kbtu


def get_decade_prices(price_excel_path, area):
    """
    readfile, Elec, Gas, Propane, Fuel Oil
    : { '2020s': {'Electricity': 50, 'Natural Gas': 10...} }
     $/MMBtu ()
    """
    try:
        df_raw = pd.read_excel(price_excel_path, sheet_name=area)
    except:
        print("Error: Price file not found or sheet name wrong.")
        return {}

    df_raw.columns = [str(c).strip() for c in df_raw.columns]


    def get_price_row(keyword):

        mask = df_raw.iloc[:, 0].astype(str).str.contains(keyword, case=False, na=False)
        if mask.any():
            return df_raw[mask].iloc[0]
        return None


    target_fuels = {
        'Electricity': 'Electricity',
        'Natural Gas': 'Natural Gas',
        'Propane': 'Propane',
        'Fuel Oil': 'Distillate Fuel Oil'
    }

    rows = {k: get_price_row(v) for k, v in target_fuels.items()}

    periods = {
        '2020s': [str(y) for y in range(2024, 2030)],
        '2030s': [str(y) for y in range(2030, 2040)],
        '2040s': [str(y) for y in range(2040, 2050)],
        '2050s': ['2050']
    }

    price_results = {}
    for label, years in periods.items():
        price_results[label] = {}
        valid_years = [y for y in years if y in df_raw.columns]

        if not valid_years: continue

        for key, row in rows.items():
            if row is not None:
                val = row[valid_years].astype(float).mean()
                price_results[label][key] = val
            else:
                price_results[label][key] = 0.0

    return price_results

def get_elec_ef_decades(ef_csv_path, state_name, ssp):
    """Return decade-average electricity emission factors in kg CO2/kWh."""
    try:
        ef_df = pd.read_csv(ef_csv_path, skiprows=1)
    except:
        print("Error: Emission factor file not found.")
        return [0,0,0,0]

    ma_ef = ef_df[ef_df['state'] == state_name].copy()


    if ssp == 'ssp126': path = 'LowRECost'
    elif ssp == 'ssp245': path = 'Electrification'
    else: path = 'mid-case'

    col_name = path + '(co2_rate_avg_gen)'

    def get_avg(years):
        subset = ma_ef[ma_ef['t'].isin(years)]
        if subset.empty: return 0

        return subset[col_name].mean() / 1000

    return [
        get_avg([2020, 2022, 2024, 2026, 2028]),
        get_avg(range(2030, 2040, 2)),
        get_avg(range(2040, 2050, 2)),
        get_avg([2050])
    ]

def identify_fuel_columns(json_df, bldg_id, stratege):
    """Identify fuel columns and installation costs from simulation metadata."""

    try:
        up_num = int(re.findall(r'\d+', stratege)[0])
    except:
        up_num = 0


    target_row = json_df[(json_df['building_id'] == int(bldg_id)) &
                         (json_df['upgrade'] == up_num)]

    if target_row.empty:
        return {}, 0

    row_data = target_row.iloc[0]


    install_cost = row_data.get('UpgradeCosts.upgrade_cost_usd', 0)
    if pd.isna(install_cost): install_cost = 0


    fuel_map = {
        'coal': 'ReportSimulationOutput.fuel_use_coal_total_m_btu',
        'electricity': 'ReportSimulationOutput.fuel_use_electricity_net_m_btu',
        'fuel_oil': 'ReportSimulationOutput.fuel_use_fuel_oil_total_m_btu',
        'natural_gas': 'ReportSimulationOutput.fuel_use_natural_gas_total_m_btu',
        'propane': 'ReportSimulationOutput.fuel_use_propane_total_m_btu',
        'wood_cord': 'ReportSimulationOutput.fuel_use_wood_cord_total_m_btu',
        'wood_pellets': 'ReportSimulationOutput.fuel_use_wood_pellets_total_m_btu'
    }

    active_fuels = {}
    for std_name, json_col in fuel_map.items():
        val = row_data.get(json_col, 0)
        if pd.notna(val) and val > 0:
            active_fuels[std_name] = val

    return active_fuels, install_cost

def parse_income_range(val):
    """
     ResStock income
    for example: '50000-59999' -> 55000
         '<10000' -> 5000
         '180000-199999' -> 190000
    """
    if pd.isna(val) or val == 'None':
        return np.nan

    s = str(val).replace(',', '').replace('$', '')

    if '-' in s:
        nums = re.findall(r'\d+', s)
        return (float(nums[0]) + float(nums[1])) / 2


    if '<' in s:
        num = float(re.findall(r'\d+', s)[0])
        return num / 2


    if '+' in s:
        num = float(re.findall(r'\d+', s)[0])
        return num * 1.2
    return np.nan


base_dir = os.environ.get(
    "PROJECT_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data"),
)

res_path = os.path.join(base_dir, r'CXY_data/upgrade0.csv')


df_res_full = pd.read_csv(res_path)


decades_list = ['2020s', '2030s', '2040s', '2050s']
ssps = [
    value.strip()
    for value in os.environ.get(
        "SSP_SCENARIOS",
        "ssp126,ssp245,ssp585",
    ).split(",")
    if value.strip()
]
target_states = {
    value.strip().upper()
    for value in os.environ.get("TARGET_STATES", "").split(",")
    if value.strip()
}

LOWER_F, UPPER_F = 68-1.5, 75+1.5

for _, state_name, _ in US_STATES:
    if not target_states or state_name in target_states:
        epw_base_dir = os.path.join(base_dir, 'EPW_8760', state_name+'_EPW')
        epw_outage = os.path.join(base_dir, 'EPW_outage', state_name+'_County_Hourly_Outage_Status.xlsx')
        local_area = STATE_TO_DIVISION.get(state_name, "Unknown")
        df_res_ma = df_res_full[df_res_full['in.state'] == state_name].copy()
        df_res_ma['fips'] = df_res_ma['in.county'].str[1:3] + df_res_ma['in.county'].str[4:7]
        unique_fips_list = df_res_ma['fips'].dropna().unique().tolist()
        unique_fips_list.sort()


        valid_building_ids = None

        for ssp in ssps:


            print(f"--- loadscenario {ssp} alloutage ---")
            scenario_masks = {d: {} for d in decades_list}

            for decade in decades_list:

                df_outage_all = pd.read_excel(epw_outage, sheet_name=decade)
                available_cols = [str(c).zfill(5) for c in df_outage_all.columns]

                for fips in unique_fips_list:

                    epw_path = os.path.join(epw_base_dir, ssp, f"{decade}_TMY_EPW", f"{fips}.epw")
                    if not os.path.exists(epw_path): continue

                    df_epw = pd.read_csv(epw_path, skiprows=8, header=None, usecols=[6], names=['temp_c'])
                    extreme_mask = identify_weather_extreme_days(df_epw['temp_c'].values, 32.22, -17.78)
                    extreme_mask_hourly = np.repeat(extreme_mask, 24)


                    if fips in available_cols:
                        col_name = df_outage_all.columns[available_cols.index(fips)]
                        outage_mask = (df_outage_all[col_name].values == 0)


                        scenario_masks[decade][fips] = {
                            'extreme': extreme_mask_hourly,
                            'outage': outage_mask,
                            'union': extreme_mask_hourly | outage_mask
                        }

            if ssp == 'ssp126':
                strage_comb = ['up17','up01','up02','up03','up04','up05','up06','up07','up08','up09','up10','up11','up12','up13','up14','up15','up16']
            elif ssp == 'ssp245':
                strage_comb = ['up17','up01','up02','up03','up04','up05','up06','up07','up08','up09','up10','up11','up12','up13','up14','up15','up16']
            else:
                strage_comb = ['up04','up01','up02','up03']
            baseline_fips_cache = {}
            for stratege_idx, stratege in enumerate(strage_comb):


                all_fips_done = True
                for fips in unique_fips_list:

                    save_path = os.path.join(base_dir, 'CXY_data', '#Results', state_name, f'FIPS_{fips}', f"{ssp}_{stratege}.parquet")
                    if not os.path.exists(save_path):
                        all_fips_done = False
                        break

                if all_fips_done:
                    print(f"Skipping: {ssp} | {stratege} (Already processed)")


                    if stratege_idx == 0:
                        if valid_building_ids is None:
                            valid_building_ids = set()

                        print(f">>>  FIPS  (Baseline Cache) ...")
                        for fips in unique_fips_list:
                            p = os.path.join(base_dir, 'CXY_data', '#Results', state_name, f'FIPS_{fips}', f"{ssp}_{stratege}.parquet")
                            if os.path.exists(p):
                                temp_df = pd.read_parquet(p)

                                valid_building_ids.update(temp_df['Building_ID'].unique())

                                baseline_fips_cache[fips] = temp_df.set_index('Building_ID').to_dict('index')

                        print(f" ({len(valid_building_ids)} building),.")
                    continue


                county_storage = {fips: {} for fips in unique_fips_list}

                for d_idx, decade in enumerate(decades_list):
                    print(f">>> Processing: {ssp} | {decade} | {stratege}")
                    year_str = decade[:4]

                    search_pattern_json = os.path.join(base_dir, f"Run_resstock_{decade[:-1]}_{ssp[3:]}", 'Data_output', "buildbatch_output_*", ssp, f"{decade}_Scenario_EPW", "simulation_output", "results_job0.json.gz")
                    json_files = glob.glob(search_pattern_json)
                    with gzip.open(json_files[0], 'rt', encoding='utf-8') as f:
                        df_json_full = pd.DataFrame(json.load(f))


                    curr_up_num = int(re.findall(r'\d+', stratege)[0])
                    df_json_scenario = df_json_full[df_json_full['upgrade'] == curr_up_num].copy()

                    df_json_scenario['building_id'] = df_json_scenario['building_id'].astype(int)

                    search_pattern = os.path.join(base_dir, f"Run_resstock_{decade[:-1]}_{ssp[3:]}", 'Data_output', f"buildbatch_output_*", ssp, f"{decade}_Scenario_EPW", "simulation_output", "timeseries", stratege)
                    matching_folders = glob.glob(search_pattern)
                    if not matching_folders: continue
                    input_folder = matching_folders[0]


                    price_dict = get_decade_prices(os.path.join(base_dir, r'CXY_data/#Future_Price_Sum.xlsx'), local_area)
                    elec_ef_list = get_elec_ef_decades(os.path.join(base_dir, r'CXY_data/Future.csv'), state_name, ssp)
                    curr_elec_ef = elec_ef_list[d_idx]
                    curr_prices = price_dict.get(decade, {})


                    files = glob.glob(os.path.join(input_folder, "bldg*.parquet"))


                    for file in files:
                        bldg_id = int(re.findall(r'bldg(\d+)', os.path.basename(file))[0])


                        bldg_meta = df_res_ma[df_res_ma['bldg_id'] == bldg_id]
                        if bldg_meta.empty: continue
                        row = bldg_meta.iloc[0]
                        fips = row['fips']

                        if fips not in scenario_masks[decade]: continue
                        masks = scenario_masks[decade][fips]

                        fuel_data, json_install_cost = identify_fuel_columns(df_json_scenario, bldg_id, stratege)
                        if not fuel_data: continue


                        if bldg_id not in county_storage[fips]:
                            income = row.get('in.representative_income')
                            if pd.isna(income): income = parse_income_range(row.get('in.income'))
                            if pd.isna(income): continue

                            county_storage[fips][bldg_id] = {
                                'Building_ID': bldg_id,
                                'Income': income,
                                'FIPS': fips,
                                'sqft': row.get('in.sqft..ft2', 0)
                            }


                        b_entry = county_storage[fips][bldg_id]


                        df_hourly = pd.read_parquet(file)

                        temp_col = 'temperature__conditioned_space__f'

                        comfort_totals, carbon_total, cost_total, energy_kbtu_total = calculate_metrics_dynamic(
                            df_hourly, masks, fuel_data, temp_col, curr_elec_ef, curr_prices, LOWER_F, UPPER_F
                            )

                        c_ext, c_out, c_uni = comfort_totals

                        b_entry[f'Comfort_Extreme_{decade}'] = c_ext / masks['extreme'].sum() if masks['extreme'].any() else 0

                        b_entry[f'Comfort_Outage_{decade}'] = c_out / masks['outage'].sum() if masks['outage'].any() else 0

                        b_entry[f'Comfort_Union_{decade}'] = c_uni / masks['union'].sum() if masks['union'].any() else 0

                        b_entry[f'Carbon(kg)_{decade}'] = carbon_total
                        b_entry[f'Energy_cost_{decade}'] = cost_total
                        b_entry[f'Energy_burden_{decade}'] = cost_total / b_entry['Income']

                        b_entry[f'Install_cost_{decade}'] = json_install_cost/4
                        b_entry[f'Total_burden_{decade}'] = (b_entry[f'Energy_cost_{decade}'] + b_entry[f'Install_cost_{decade}']) / b_entry['Income']


                for fips, bldg_dict in county_storage.items():
                    if not bldg_dict: continue


                    out_df0 = pd.DataFrame(list(bldg_dict.values()))
                    print(len(out_df0))


                    if ssp == 'ssp126' and stratege == 'up17':

                        mask_income = out_df0['Income'] >= 5000
                        print(mask_income.sum())


                        mask_comfort = out_df0['Comfort_Extreme_2020s'] <= out_df0['Comfort_Extreme_2020s'].quantile(0.95)
                        print(mask_comfort.sum())


                        mask_eb = out_df0['Energy_burden_2020s'] <= 0.25
                        print(mask_eb.sum())


                        valid_mask = mask_income & mask_comfort & mask_eb
                        out_df = out_df0[valid_mask].copy()
                        print(len(out_df))


                        if valid_building_ids is None: valid_building_ids = set()
                        valid_building_ids.update(out_df['Building_ID'].tolist())

                        print(f"FIPS {fips} completed:  {len(out_df0)} ->  {len(out_df)}")

                    elif valid_building_ids is not None:

                        initial_count = len(out_df0)
                        out_df = out_df0[out_df0['Building_ID'].isin(valid_building_ids)].copy()


                    if stratege_idx == 0:
                        if 'baseline_fips_cache' not in locals(): baseline_fips_cache = {}

                        baseline_fips_cache[fips] = out_df.set_index('Building_ID').to_dict('index')
                    else:

                        fips_baseline = baseline_fips_cache.get(fips, {})
                        current_ids = set(out_df['Building_ID'])
                        missing_ids = set(fips_baseline.keys()) - current_ids


                        if missing_ids:

                            missing_rows = []
                            for b_id in missing_ids:
                                row_data = fips_baseline[b_id].copy()
                                row_data['Building_ID'] = b_id
                                missing_rows.append(row_data)

                            out_df = pd.concat([out_df, pd.DataFrame(missing_rows)], ignore_index=True)
                            print(f"FIPS {fips} {stratege}  {len(missing_ids)} missingbuilding")


                    print(f"--- FIPS {fips} | {ssp} | {stratege} statistics ---")
                    eb_cols = [f'Energy_burden_{d}' for d in decades_list]
                    print(out_df[eb_cols].describe(percentiles=[.5, .9, .99]))


                    fips_output_dir = os.path.join(base_dir, 'CXY_data', '#Results', state_name, f'FIPS_{fips}')
                    if not os.path.exists(fips_output_dir):
                        os.makedirs(fips_output_dir)

                    save_path = os.path.join(fips_output_dir, f"{ssp}_{stratege}.parquet")
                    out_df.to_parquet(save_path, engine='pyarrow', compression='brotli', index=False)
                    print(f"Successfully saved to: {save_path}")
