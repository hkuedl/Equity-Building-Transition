import xarray as xr
import pandas as pd
import numpy as np
import os
import unicodedata
import matplotlib.pyplot as plt
from pathlib import Path

def sanitize_name(name):
    """Convert names with diacritics to filesystem-safe ASCII."""
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')


PROJECT_DATA_DIR = Path(
    os.environ.get(
        "PROJECT_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data",
    )
)
GAZETTEER_PATH = os.environ.get(
    "COUNTY_GAZETTEER_FILE",
    str(PROJECT_DATA_DIR / "geography" / "counties_gazetteer.txt"),
)
BASE_FUTURE_DIR = os.environ.get(
    "FUTURE_WEATHER_DIR",
    str(PROJECT_DATA_DIR / "weather" / "future"),
)
OUT_BASE_DIR = os.environ.get(
    "EPW_OUTPUT_DIR",
    str(PROJECT_DATA_DIR / "weather" / "epw"),
)


STATE_TIMEZONE_MAP = {
    'ME': -5, 'NH': -5, 'VT': -5, 'MA': -5, 'RI': -5, 'CT': -5, 'NY': -5, 'NJ': -5, 'PA': -5, 'DE': -5, 'MD': -5, 'DC': -5, 'VA': -5, 'WV': -5, 'NC': -5, 'SC': -5, 'GA': -5, 'FL': -5, 'OH': -5, 'MI': -5, 'IN': -5, 'KY': -5,
    'AL': -6, 'MS': -6, 'TN': -6, 'IL': -6, 'WI': -6, 'MN': -6, 'IA': -6, 'MO': -6, 'AR': -6, 'LA': -6, 'OK': -6, 'TX': -6, 'KS': -6, 'NE': -6, 'SD': -6, 'ND': -6,
    'MT': -7, 'WY': -7, 'CO': -7, 'NM': -7, 'AZ': -7, 'UT': -7, 'ID': -7,
    'WA': -8, 'OR': -8, 'NV': -8, 'CA': -8
}

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

def safe_cast_to_int(data_array, min_val, max_val, default_fill=None):
    """Fill missing values, clip to physical limits, and return integers."""

    s = pd.Series(data_array)
    if s.isnull().any():

        s = s.ffill().bfill()
        if default_fill is not None:
            s = s.fillna(default_fill)

    return np.clip(s.values, min_val, max_val).round(0).astype(int)

def process_single_county(point_data, county_info, tz, state_abbr, out_dir):
    fips, name, c_lat, c_lon = county_info

    df = pd.DataFrame()

    df['year'] = point_data.time.dt.year
    df['month'] = point_data.time.dt.month
    df['day'] = point_data.time.dt.day
    df['hour'] = point_data.time.dt.hour + 1
    df['min'] = 60


    df['src'] = '?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9'


    t_db = point_data['tas'].values - 273.15
    t_dp = point_data['tdps'].values - 273.15
    t_dp = np.minimum(t_dp, t_db)

    df['dry_bulb'] = t_db.round(1)
    df['dew_point'] = t_dp.round(1)


    df['rel_hum'] = safe_cast_to_int(point_data['hurs'].values, 0, 100, 50)


    df['atmos_pressure'] = 101325
    df['ext_hor_rad'] = 0
    df['ext_dir_rad'] = 0
    df['sky_ir_rad'] = point_data['rlds'].values.round(0)


    ghi_values = point_data['rsds'].values
    df['glo_hor_rad'] = ghi_values.round(0)

    dni, dhi = decompose_radiation(point_data['rsds'].values, point_data.time.dt.dayofyear.values, df['hour'].values, c_lat, c_lon, tz)
    df['dir_norm_rad'] = dni.astype(int)
    df['dif_hor_rad'] = dhi.astype(int)


    df['glo_hor_illum'] = 999999
    df['dir_norm_illum'] = 999999
    df['dif_hor_illum'] = 999999
    df['zen_lum'] = 999
    df['wind_dir'] = 0
    df['wind_spd'] = point_data['sfcWind'].values.round(1)


    for i in range(22, 35):
        df[f'col_{i}'] = 0


    save_path = os.path.join(out_dir, f"{fips}.epw")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(create_epw_header(name, c_lat, c_lon, fips, state_abbr, tz))
        df.to_csv(f, header=False, index=False, lineterminator='\n')

def load_counties_by_state(txt_path, state_abbr):
    """
     Gazetteer fileloadstatecounty
    """
    df = pd.read_csv(txt_path, sep='\t', encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]


    state_df = df[df['USPS'] == state_abbr].copy()
    state_df['GEOID'] = state_df['GEOID'].astype(str).str.zfill(5)


    counties_dict = {
        row['GEOID']: (row['NAME'], float(row['INTPTLAT']), float(row['INTPTLONG']))
        for _, row in state_df.iterrows()
    }
    return counties_dict

def decompose_radiation(ghi, dob_year, hour, lat, lon, time_zone):
    """Split GHI into DNI and DHI with a numerically stable Erbs model."""
    ghi = np.array(ghi)
    dob_year = np.array(dob_year)
    hour = np.array(hour)

    deg_rad = np.pi / 180.0
    gon = 1367.0


    b = 2 * np.pi * (dob_year - 1) / 365.0
    delta = (0.006918 - 0.399912*np.cos(b) + 0.070257*np.sin(b) -
             0.006758*np.cos(2*b) + 0.000907*np.sin(2*b) -
             0.002697*np.cos(3*b) + 0.00148*np.sin(3*b))
    eot = 229.18 * (0.000075 + 0.001868*np.cos(b) - 0.032077*np.sin(b) -
                    0.014615*np.cos(2*b) - 0.04089*np.sin(2*b))


    std_meridian = time_zone * 15
    solar_time = (hour - 1.0) + (4 * (lon - std_meridian) + eot) / 60.0

    omega = (solar_time - 12) * 15 * deg_rad
    lat_rad = lat * deg_rad
    cos_zenith = np.sin(lat_rad)*np.sin(delta) + np.cos(lat_rad)*np.cos(delta)*np.cos(omega)


    gon_corrected = gon * (1 + 0.033 * np.cos(360 * dob_year / 365 * deg_rad))
    ext_hor = gon_corrected * np.maximum(cos_zenith, 0)


    is_day_robust = (cos_zenith > 0.174) & (ghi > 50.0)


    kt = np.zeros_like(ghi)
    kt[is_day_robust] = np.clip(ghi[is_day_robust] / np.maximum(ext_hor[is_day_robust], 1.0), 0, 1.0)

    kd = np.ones_like(ghi)
    m2 = (kt > 0.22) & (kt <= 0.80) & is_day_robust
    kd[m2] = 0.9511 - 0.1604*kt[m2] + 4.388*(kt[m2]**2) - 16.638*(kt[m2]**3) + 12.336*(kt[m2]**4)
    kd[(kt > 0.80) & is_day_robust] = 0.165


    dhi = ghi * kd
    dni = np.zeros_like(ghi)


    safe_cos = np.maximum(cos_zenith, 0.15)
    dni[is_day_robust] = (ghi[is_day_robust] - dhi[is_day_robust]) / safe_cos[is_day_robust]


    dni = np.clip(dni, 0, gon_corrected)
    dni = np.where(dni > ghi * 2.0, ghi * 1.5, dni)


    dni[~is_day_robust] = 0
    dhi[~is_day_robust] = ghi[~is_day_robust]

    return dni.round(0), dhi.round(0)

def create_epw_header(county_name, lat, lon, fips, state_abbr, tz):
    """Create an EPW header for a county and time zone."""
    return f"LOCATION,{county_name},{state_abbr},USA,Custom-Morphing,{fips},{lat},{lon},{float(tz)},10.0\n" \
           f"DESIGN CONDITIONS,0\n" \
           f"TYPICAL/EXTREME PERIODS,0\n" \
           f"GROUND TEMPERATURES,0\n" \
           f"HOLIDAYS/DAYLIGHT SAVINGS,0\n" \
           f"COMMENTS 1,Generated from CMIP6 Morphing for ResStock analysis\n" \
           f"COMMENTS 2,County FIPS: {fips}\n" \
           f"DATA PERIODS,1,1,Data,Sunday,1/ 1,12/31\n"


for ssp in ['ssp126', 'ssp245', 'ssp585']:
    for decade in ['2020s', '2030s', '2040s', '2050s']:
        for dtype in ['TMY']:
            print(f"\n Current scenario: {ssp} | {decade}")


            md_dataset = None


            for state_full, state_abbr, _ in US_STATES:
                if TARGET_STATES and state_abbr not in TARGET_STATES and not (
                    state_abbr == "MD" and "DC" in TARGET_STATES
                ):
                    continue
                if state_abbr == "DC":
                    continue

                counties = load_counties_by_state(GAZETTEER_PATH, state_abbr)
                nc_file = f"{state_abbr}_{ssp}_{decade}_{dtype}_hourly.nc"
                nc_path = os.path.join(BASE_FUTURE_DIR, state_abbr, ssp, nc_file)

                if not os.path.exists(nc_path):
                    continue

                print(f"Processing {len(counties)} counties in {state_abbr}...")
                ds = xr.open_dataset(nc_path)


                if state_abbr == "MD":
                    md_dataset = ds.copy(deep=True)

                state_out_dir = os.path.join(OUT_BASE_DIR, f"{state_abbr}_EPW", ssp, f"{decade}_{dtype}_EPW")
                if not os.path.exists(state_out_dir): os.makedirs(state_out_dir)

                tz = STATE_TIMEZONE_MAP.get(state_abbr, -5)
                for fips, (name, c_lat, c_lon) in counties.items():
                    safe_name = sanitize_name(name)
                    point_data = ds.sel(lat=c_lat, lon=c_lon, method='nearest').compute()
                    process_single_county(point_data, (fips, safe_name, c_lat, c_lon), tz, state_abbr, state_out_dir)

                ds.close()


            if md_dataset is not None:
                print(f"   Generating proxy EPW files for DC using Maryland data...")
                dc_counties = load_counties_by_state(GAZETTEER_PATH, "DC")
                dc_out_dir = os.path.join(OUT_BASE_DIR, "DC_EPW", ssp, f"{decade}_{dtype}_EPW")
                if not os.path.exists(dc_out_dir): os.makedirs(dc_out_dir)

                for fips, (name, c_lat, c_lon) in dc_counties.items():

                    point_data = md_dataset.sel(lat=c_lat, lon=c_lon, method='nearest').compute()
                    process_single_county(point_data, (fips, name, c_lat, c_lon), -5, "DC", dc_out_dir)

                md_dataset.close()
            else:
                print("   Warning: Maryland data was not found; DC proxy EPW files cannot be generated.")


VALIDATION_STATE = os.environ.get("EPW_VALIDATION_STATE", "CA")
BASE_FUTURE_DIR = os.environ.get(
    "EPW_VALIDATION_DIR",
    str(Path(OUT_BASE_DIR) / f"{VALIDATION_STATE}_EPW"),
)
SSPS = ['ssp126']#, 'ssp585']
DECADES = ['2020s']
VALIDATION_COUNTIES = {"06013": "Contra Costa", "06015": "Del Norte"}

def load_epw_data(file_path):
    """Read the EPW data section after its eight-line header."""
    columns = [
        'year', 'month', 'day', 'hour', 'minute', 'datasource',
        'dry_bulb', 'dew_point', 'rel_hum', 'pressure',
        'ext_hor_rad', 'ext_dir_rad', 'sky_ir_rad', 'glo_hor_rad',
        'dir_norm_rad', 'dif_hor_rad', 'glo_hor_illum', 'dir_norm_illum',
        'dif_hor_illum', 'zen_lum', 'wind_dir', 'wind_spd'
    ]

    df = pd.read_csv(file_path, skiprows=8, header=None, names=columns, usecols=range(22))
    return df

print(" Starting EPW batch validation...")

for ssp in SSPS:
    for decade in DECADES:
        for fips, c_name in VALIDATION_COUNTIES.items():
            print(f"\nAnalyzing location: {c_name} ({fips}) | {ssp} | {decade}")


            path_tmy = os.path.join(BASE_FUTURE_DIR, ssp, f"{decade}_TMY_EPW", f"{fips}.epw")

            if not os.path.exists(path_tmy):
                print(f"   File missing; skipping this combination")
                continue

            df_tmy = load_epw_data(path_tmy)


            print("  [1/4] Physical limit checks:")
            for name, df in [('TMY', df_tmy)]:

                td_violation = (df['dew_point'] > df['dry_bulb']).sum()

                rh_violation = ((df['rel_hum'] < 0) | (df['rel_hum'] > 100)).sum()

                rad_violation = (df['glo_hor_rad'] < 0).sum()

                print(f"      - {name} dew-point conflicts: {td_violation} | humidity violations: {rh_violation} | negative-radiation values: {rad_violation}")
                if td_violation > 0 or rh_violation > 0:
                    print(f"       Warning: {name} has physical consistency errors!")


            print("  [3/4] Generating the mean July diurnal-cycle plot...")
            plt.figure(figsize=(12, 5))


            for name, df, color in [('TMY', df_tmy, 'blue')]:
                july_diurnal = df[df['month'] == 7].groupby('hour').mean(numeric_only=True)


                plt.subplot(1, 2, 1)
                plt.plot(july_diurnal.index, july_diurnal['dry_bulb'], label=f'{name} Temp', color=color)


                plt.subplot(1, 2, 2)
                plt.plot(july_diurnal.index, july_diurnal['glo_hor_rad'], label=f'{name} Solar', color=color, linestyle='--')

            plt.subplot(1, 2, 1)
            plt.title(f'Diurnal Temp Profile (July) - {c_name}')
            plt.xlabel('Hour'); plt.ylabel(' deg C'); plt.legend(); plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.title(f'Solar Phase Check (July) - {c_name}')
            plt.xlabel('Hour'); plt.ylabel('W/m2'); plt.legend(); plt.grid(True, alpha=0.3)

            plt.tight_layout()


            print("  [4/4] Continuity check (checking hour-to-hour changes)...")
            temp_diffs = np.abs(df_tmy['dry_bulb'].diff())
            large_jumps = (temp_diffs > 5).sum()
            print(f"      - Large hourly temperature changes (>5 deg C): {large_jumps}")

print("\n All validation checks completed!Review the generated plots for phase consistency.")


EPW_FILE_PATH = os.environ.get(
    "EPW_VALIDATION_FILE",
    str(Path(BASE_FUTURE_DIR) / "ssp126" / "2020s_TMY_EPW" / "06013.epw"),
)
MONTH_TO_CHECK = 7
SAVE_PLOT = False

def load_epw_radiation(file_path):
    """Read the radiation-related columns from an EPW file."""
    cols = ['year', 'month', 'day', 'hour', 'minute', 'datasource',
            'dry_bulb', 'dew_point', 'rel_hum', 'pressure',
            'ext_hor_rad', 'ext_dir_rad', 'sky_ir_rad',
            'ghi', 'dni', 'dhi']
    df = pd.read_csv(file_path, skiprows=8, header=None, names=cols, usecols=range(16))
    return df


if not os.path.exists(EPW_FILE_PATH):
    raise FileNotFoundError(f"EPW file not found: {EPW_FILE_PATH}")

df_epw = load_epw_radiation(EPW_FILE_PATH)


fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 1, hspace=0.3)


ax1 = fig.add_subplot(gs[0, 0])
m_df = df_epw[df_epw['month'] == MONTH_TO_CHECK]
diurnal = m_df.groupby('hour').mean(numeric_only=True)

ax1.plot(diurnal.index, diurnal['ghi'], 'k-', lw=3, label='GHI (Global Horizontal)', alpha=0.8)
ax1.plot(diurnal.index, diurnal['dni'], 'r-', lw=2, label='DNI (Direct Normal)')
ax1.plot(diurnal.index, diurnal['dhi'], 'b--', lw=2, label='DHI (Diffuse Horizontal)')

ax1.set_title(f"EPW Radiation Components Diurnal Average (Month {MONTH_TO_CHECK})", fontsize=14)
ax1.set_ylabel("Radiation (W/m2)")
ax1.set_xlabel("Hour of Day (1-24)")
ax1.set_xticks(range(1, 25))
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend()


ax2 = fig.add_subplot(gs[1, 0])

sample_days = m_df[(m_df['day'] >= 10) & (m_df['day'] <= 12)]
x_ticks = np.arange(len(sample_days))

ax2.fill_between(x_ticks, 0, sample_days['ghi'], color='gray', alpha=0.1, label='GHI Area')
ax2.plot(x_ticks, sample_days['ghi'], 'k-', lw=1.5, label='GHI')
ax2.plot(x_ticks, sample_days['dni'], 'r-', lw=1.5, label='DNI')
ax2.plot(x_ticks, sample_days['dhi'], 'b--', lw=1.5, label='DHI')

ax2.set_title("3-Day Radiation Time Series (Component Breakdown Check)", fontsize=14)
ax2.set_ylabel("W/m2")
ax2.set_xlabel("Hours from start of July 10th")
ax2.grid(True, ls=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
if SAVE_PLOT:
    plt.savefig('EPW_Radiation_Check.png', dpi=300)
plt.show()


print("\n" + "="*50)
print(f" EPW radiation consistency report")
print("-" * 50)


dhi_exceed_ghi = (df_epw['dhi'] > df_epw['ghi'] + 1).sum()
print(f"1. Diffuse-radiation check:  {dhi_exceed_ghi} hours have DHI greater than GHI (expected 0)")


peak_hour = diurnal['ghi'].idxmax()
print(f"2. GHI peak hour:  {peak_hour}  (normally expected between 12 and 13)")


night_rad = df_epw[df_epw['hour'].isin([1, 2, 3, 4, 21, 22, 23, 24])]['ghi'].max()
print(f"3. Nighttime residual check: maximum {night_rad:.2f} W/m2 ( 0)")

print("="*50)
