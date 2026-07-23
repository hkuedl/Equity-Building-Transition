

import os
import zipfile
import glob
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metpy.calc import relative_humidity_from_dewpoint, dewpoint_from_relative_humidity
from metpy.units import units


PROJECT_DATA_DIR = Path(
    os.environ.get(
        "PROJECT_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data",
    )
)
ERA5_DATA_DIR = Path(
    os.environ.get(
        "ERA5_DATA_DIR",
        PROJECT_DATA_DIR / "weather" / "era5",
    )
)
FUTURE_WEATHER_DIR = Path(
    os.environ.get(
        "FUTURE_WEATHER_DIR",
        PROJECT_DATA_DIR / "weather" / "future",
    )
)

def extract_era5_zips(base_dir,state):


    nc_files = glob.glob(os.path.join(base_dir, 'era5_'+state+'_*'))

    state_dir = os.path.join(base_dir, state)
    os.makedirs(state_dir, exist_ok=True)

    if not nc_files:
        print(f"No files were found in {base_dir} .")
        return

    for file_path in nc_files:

        file_name = os.path.basename(file_path)
        folder_name = os.path.splitext(file_name)[0]
        target_folder = os.path.join(state_dir, folder_name)

        try:

            with open(file_path, 'rb') as f:
                header = f.read(4)

            if header == b'PK\x03\x04':
                print(f"Archive detected: {file_name}")


                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                    print(f"  - Created directory: {folder_name}")


                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)


                    extracted_files = zip_ref.namelist()
                    print(f"  - Extracted {len(extracted_files)} file: {extracted_files}")


                backup_path = file_path + ".bak_zip"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(file_path, backup_path)
                print(f"  - Original archive renamed to: {os.path.basename(backup_path)}")

            else:
                print(f"Skipping: {file_name} (already uses a standard NetCDF or another supported format)")

        except Exception as e:
            print(f"Processing file {file_name} failed: {e}")

    print("\n--- File extraction completed ---")


US_STATES = [
    ("Alabama", "AL", 1), ("Arizona", "AZ", 4), ("Arkansas", "AR", 5),
    ("California", "CA", 6), ("Colorado", "CO", 8), ("Connecticut", "CT", 9),
    ("Delaware", "DE", 10), ("District of Columbia", "DC", 11),("Florida", "FL", 12),
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

STATE_UTC_MAPPING = {
    'ME': -5, 'VT': -5, 'NH': -5, 'MA': -5, 'RI': -5, 'CT': -5, 'NY': -5, 'PA': -5,
    'NJ': -5, 'DE': -5, 'MD': -5, 'VA': -5, 'WV': -5, 'NC': -5, 'SC': -5, 'GA': -5, 'FL': -5,
    'OH': -5, 'MI': -5, 'IN': -5, 'KY': -5,
    'IL': -6, 'WI': -6, 'MN': -6, 'IA': -6, 'MO': -6, 'AR': -6, 'LA': -6, 'MS': -6, 'AL': -6,
    'TN': -6, 'KY': -6, 'TX': -6, 'OK': -6, 'KS': -6, 'NE': -6, 'SD': -6, 'ND': -6,
    'MT': -7, 'WY': -7, 'CO': -7, 'NM': -7, 'ID': -7, 'UT': -7, 'AZ': -7,
    'WA': -8, 'OR': -8, 'NV': -8, 'CA': -8,
    'AK': -9, 'HI': -10
}

for aa_name_full, state, aa_id in US_STATES:
    if TARGET_STATES and state not in TARGET_STATES:
        continue


    SSPS = ['ssp126', 'ssp245', 'ssp585']
    DECADES = ['2020s', '2030s', '2040s', '2050s']
    DATA_TYPES = ['TMY']

    HIST_DIR = str(ERA5_DATA_DIR / state)
    BASE_FUTURE_DIR = str(FUTURE_WEATHER_DIR / state)


    print(f"--- Stage 1:Process {state} historical ERA5 baseline data ---")


    start_year, end_year = 2005, 2024
    expected_months = {f"{y}_{m:02d}" for y in range(start_year, end_year + 1) for m in range(1, 13)}


    all_files = glob.glob(os.path.join(HIST_DIR, '**', '*.nc'), recursive=True)


    import re
    actual_months = set()
    for f in all_files:
        match = re.search(r'(\d{4})_(\d{2})', f)
        if match:
            actual_months.add(f"{match.group(1)}_{match.group(2)}")

    missing = sorted(list(expected_months - actual_months))
    if missing:
        print(f" :{state} Missing {len(missing)} months of data:")
        print(f"   {missing[:6]} ... ( {len(missing)} )")
        print(f"    Using the available {len(actual_months)} months to compute the mean climate profile.")
    else:
        print(f" {state} Data are complete; total {len(actual_months)} .")


    inst_files = [f for f in all_files if 'instant' in f]
    accum_files = [f for f in all_files if 'accum' in f]

    if not inst_files or not accum_files:
        print(f" Error:{state} No valid instant or accum files were found; skipping this state.")
        continue

    print(f"Loading and merging data streams...")

    ds_inst = xr.open_mfdataset(inst_files, combine='by_coords', parallel=True,
                               coords='minimal', chunks={'time': 500})
    ds_accum = xr.open_mfdataset(accum_files, combine='by_coords', parallel=True,
                                coords='minimal', chunks={'time': 500})


    ds_hist = xr.merge([ds_inst, ds_accum], join='inner', compat='override')


    rename_dict = {'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'}
    ds_hist = ds_hist.rename({k: v for k, v in rename_dict.items() if k in ds_hist.coords})


    offset = STATE_UTC_MAPPING.get(state, -5)
    ds_hist['time'] = ds_hist['time'] + pd.Timedelta(hours=offset)

    if 'expver' in ds_hist.dims:
        ds_hist = ds_hist.sel(expver=1)


    print("Computing historical variables (Wind, RH, Radiation conversion)...")
    ds_hist['sfcWind'] = np.sqrt(ds_hist['u10']**2 + ds_hist['v10']**2)


    t2m_q = ds_hist['t2m'].metpy.quantify()
    d2m_q = ds_hist['d2m'].metpy.quantify()
    ds_hist['hurs'] = relative_humidity_from_dewpoint(t2m_q, d2m_q).metpy.dequantify() * 100


    for rad_var in ['ssrd', 'strd']:
        if rad_var in ds_hist:
            ds_hist[rad_var] = ds_hist[rad_var] / 3600.0


    ds_hist = ds_hist.sel(time=~((ds_hist.time.dt.month == 2) & (ds_hist.time.dt.day == 29)))

    print("Computing the mean hourly climatology (Dayofyear x Hour)...")

    hist_profiles = ds_hist.groupby('time.dayofyear').apply(
        lambda x: x.groupby('time.hour').mean('time')
    ).compute()


    print("\n--- Stage 2:future-scenario morphing ---")

    for ssp in SSPS:
        for decade in DECADES:
            for dtype in DATA_TYPES:

                print(f"\n>>> Processing: {ssp} | {decade} | {dtype}")


                future_file = os.path.join(BASE_FUTURE_DIR, ssp, f"{state}_{ssp}_{decade}_{dtype}_daily.nc")
                output_dir = os.path.join(BASE_FUTURE_DIR, ssp)
                output_file = os.path.join(output_dir, f"{state}_{ssp}_{decade}_{dtype}_hourly.nc")

                if not os.path.exists(future_file):
                    print(f" Skipping:Future daily file not found {future_file}")
                    continue


                ds_fut_daily = xr.open_dataset(future_file)


                current_hist = hist_profiles.interp(
                    lat=ds_fut_daily.lat,
                    lon=ds_fut_daily.lon,
                    method='nearest'
                ).compute()


                morphed_days = []
                for doy in range(1, 366):

                    fut_day = ds_fut_daily.sel(time=ds_fut_daily.time.dt.dayofyear == doy).squeeze(drop=True)
                    hist_day = current_hist.sel(dayofyear=doy).drop_vars('dayofyear')

                    day_ds = xr.Dataset()


                    h_temp = hist_day['t2m']
                    h_mean = h_temp.mean('hour')
                    h_range = h_temp.max('hour') - h_temp.min('hour')

                    f_mean = (fut_day['tasmax'] + fut_day['tasmin']) / 2.0
                    f_range = fut_day['tasmax'] - fut_day['tasmin']


                    ratio = xr.where(h_range > 0.01, f_range / h_range, 1.0).clip(0.1, 5.0)


                    day_ds['tas'] = h_temp + (f_mean - h_mean) + ratio * (h_temp - h_mean)


                    var_map = [('sfcWind', 'sfcWind'), ('hurs', 'hurs'), ('rsds', 'ssrd'), ('rlds', 'strd')]
                    for f_v, h_v in var_map:
                        h_m = hist_day[h_v].mean('hour')

                        scale = xr.where(h_m > 1e-5, fut_day[f_v] / h_m, 0.0).clip(0, 10)
                        m = hist_day[h_v] * scale

                        if f_v == 'hurs': m = m.clip(0, 100)
                        if f_v in ['rsds', 'rlds', 'sfcWind']: m = m.where(m > 0, 0)
                        day_ds[f_v] = m

                    morphed_days.append(day_ds)

                print(f"  Rebuilding the time axis...")
                full_ds = xr.concat(morphed_days, dim='dayofyear')


                full_ds = full_ds.stack(time_idx=('dayofyear', 'hour')).transpose('time_idx', 'lat', 'lon')


                time_coords = pd.date_range(f"2001-01-01", periods=8760, freq="h")
                full_ds = full_ds.assign_coords(time_idx=time_coords)


                full_ds = full_ds.rename({'time_idx': 'time'})

                full_ds = full_ds.drop_vars(['dayofyear', 'hour'], errors='ignore')


                print(f"  Computing final dew-point temperature...")
                tas_q = full_ds['tas'].values * units.kelvin
                rh_q = (full_ds['hurs'].values / 100.0) * units.dimensionless
                tdps_val = dewpoint_from_relative_humidity(tas_q, rh_q).to(units.kelvin).magnitude
                full_ds['tdps'] = (('time', 'lat', 'lon'), tdps_val)


                if not os.path.exists(output_dir): os.makedirs(output_dir)
                encoding = {var: {'zlib': True, 'complevel': 4} for var in full_ds.data_vars}
                full_ds.to_netcdf(output_file, encoding=encoding)
                print(f"   Completed; saved to: {os.path.basename(output_file)}")

    print("\n All morphing tasks completed!")


TARGET_VAR = 'tas'
STATE = os.environ.get("VALIDATION_STATE", "CA")
SSP = os.environ.get("VALIDATION_SSP", "ssp126")
DECADE = os.environ.get("VALIDATION_DECADE", "2020s")
DTYPE = os.environ.get("VALIDATION_DATA_TYPE", "TMY")
LAT_IDX, LON_IDX = 3, 5
BASE_DIR = str(FUTURE_WEATHER_DIR)


VAR_META = {
    'tas': {
        'name': 'Air Temperature',
        'unit': ' deg C',
        'cmap': 'RdYlBu_r',
        'offset': -273.15,  # K to C
        'months': [1, 4, 7, 10],
        'color_list': ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    },
    'rsds': {
        'name': 'Solar Radiation',
        'unit': 'W/m2',
        'cmap': 'YlOrRd',
        'offset': 0,
        'months': [1, 4, 7, 10],
        'color_list': ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    }
}


file_path = os.path.join(BASE_DIR, STATE, SSP, f"{STATE}_{SSP}_{DECADE}_{DTYPE}_hourly.nc")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

ds = xr.open_dataset(file_path).isel(lat=LAT_IDX, lon=LON_IDX)
meta = VAR_META[TARGET_VAR]
data_series = ds[TARGET_VAR] + meta['offset']


ds_full = xr.open_dataset(file_path)
data_all = ds_full[TARGET_VAR] + meta['offset']

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)


ax1 = fig.add_subplot(gs[0, 0])
data_all.mean('time').plot(ax=ax1, cmap=meta['cmap'], cbar_kwargs={'label': meta['unit']})
ax1.set_title(f"Annual Mean {meta['name']} (Spatial)")


ax2 = fig.add_subplot(gs[0, 1])
data_all.max('time').plot(ax=ax2, cmap=meta['cmap'], cbar_kwargs={'label': meta['unit']})
ax2.set_title(f"Annual Max {meta['name']} (Spatial)")


ax3 = fig.add_subplot(gs[0, 2])
data_all.std('time').plot(ax=ax3, cmap='viridis', cbar_kwargs={'label': meta['unit']})
ax3.set_title(f"Temporal Standard Deviation (Spatial)")


ax4 = fig.add_subplot(gs[1, 0])
if TARGET_VAR == 'tas' and 'tdps' in ds_full:

    diff = (ds_full['tas'] - ds_full['tdps']).min('time')
    diff.plot(ax=ax4, cmap='RdBu', center=0, cbar_kwargs={'label': 'delta T (K)'})
    ax4.set_title("Min (T_air - T_dew) - Physics Check")
elif TARGET_VAR == 'rsds':

    night_mask = ds_full.time.dt.hour.isin([23, 0, 1, 2, 3])
    night_residue = ds_full['rsds'].sel(time=night_mask).max('time')
    night_residue.plot(ax=ax4, cmap='Reds', cbar_kwargs={'label': 'W/m2'})
    ax4.set_title("Max Nighttime Radiation Residue")


ax5 = fig.add_subplot(gs[1, 1:])

ax5.hist(data_all.values.flatten(), bins=100, color='#34495e', alpha=0.7, edgecolor='white')
ax5.set_title(f"Value Distribution Across All Grids and Time")
ax5.set_xlabel(f"{meta['name']} ({meta['unit']})")
ax5.set_ylabel("Frequency")
ax5.set_yscale('log')

plt.suptitle(f"Comprehensive Grid Diagnostics: {STATE} | {SSP} | {DECADE}", fontsize=18, y=0.95)
plt.show()


print(f"\n" + "="*60)
print(f" Full-grid audit (Total Grid Points: {len(ds_full.lat) * len(ds_full.lon)})")
print("-" * 60)


nan_count = ds_full[TARGET_VAR].isnull().sum().compute().values
print(f"1. Missing values (NaN) total: {nan_count} ({nan_count/(8760*len(ds_full.lat)*len(ds_full.lon))*100:.4f}%)")


g_max = data_all.max().values
g_min = data_all.min().values
print(f"2. Global range: [{g_min:.2f}, {g_max:.2f}] {meta['unit']}")


if TARGET_VAR == 'tas' and 'tdps' in ds_full:
    violation_mask = (ds_full['tas'] - ds_full['tdps'] < -0.01)
    v_count = violation_mask.sum().values
    if v_count > 0:
        print(f" Warning: found {v_count} pixel-hours have dew-point conflicts (Td > T)!")
    else:
        print(" check: The dew-point constraint is satisfied at every grid cell.")

print("="*60)


fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)


ax1 = fig.add_subplot(gs[0, 0])
for i, m in enumerate(meta['months']):
    m_data = data_series.sel(time=data_series.time.dt.month == m)
    diurnal_mean = m_data.groupby('time.hour').mean()
    diurnal_std = m_data.groupby('time.hour').std()

    ax1.plot(diurnal_mean.hour, diurnal_mean.values,
             label=f'Month {m}', color=meta['color_list'][i], lw=2.5)
    ax1.fill_between(diurnal_mean.hour, diurnal_mean - diurnal_std,
                     diurnal_mean + diurnal_std, color=meta['color_list'][i], alpha=0.1)

ax1.set_title(f"Mean Diurnal {meta['name']} ({STATE} {DECADE})", fontsize=14)
ax1.set_xlabel("Hour of Day (Local Time)")
ax1.set_ylabel(f"{meta['name']} ({meta['unit']})")
ax1.grid(True, ls='--', alpha=0.5)
ax1.set_xlim(0, 23)
ax1.legend()


ax2 = fig.add_subplot(gs[0, 1])

start_date = f"{ds.time.dt.year[0].values}-07-01"
end_date = f"{ds.time.dt.year[0].values}-07-07"
ts_data = data_series.sel(time=slice(start_date, end_date))

ax2.plot(ts_data.time, ts_data.values, color='#2c3e50', lw=1.5)
ax2.fill_between(ts_data.time, ts_data.min(), ts_data.values, color='#34495e', alpha=0.1)
ax2.set_title(f"7-Day Time Series Sample (July)", fontsize=14)
ax2.set_ylabel(meta['unit'])
ax2.grid(True, ls='--', alpha=0.5)


ax3 = fig.add_subplot(gs[1, :])
matrix = data_series.values.reshape(365, 24)
im = ax3.imshow(matrix.T, aspect='auto', cmap=meta['cmap'], interpolation='none')
plt.colorbar(im, ax=ax3, label=meta['unit'])

ax3.set_title(f"Annual {meta['name']} Fingerprint (Hour vs. Day of Year)", fontsize=14)
ax3.set_xlabel("Day of Year")
ax3.set_ylabel("Hour of Day")
ax3.invert_yaxis()


print(f"\n" + "="*50)
print(f" Physical consistency report | variable: {meta['name']} | region: {STATE}")
print("-"*50)

v_max = float(data_series.max())
v_min = float(data_series.min())
v_mean = float(data_series.mean())

print(f"1. Range: [{v_min:.2f}, {v_max:.2f}] {meta['unit']}")
print(f"2. Annual mean: {v_mean:.2f} {meta['unit']}")

if TARGET_VAR == 'rsds':

    night_vals = data_series.sel(time=data_series.time.dt.hour.isin([23, 0, 1, 2, 3]))
    print(f"3. Nighttime residual check: maximum {night_vals.max().values:.4f} (expected near zero)")
    if v_min < -0.01: print(" Warning: Negative radiation values were found!")
elif TARGET_VAR == 'tas':

    if 'tdps' in ds:
        td_diff = (ds['tas'] - ds['tdps']).min().values
        print(f"3. Dew-point check: (T - Td)  {td_diff:.4f} ( >= 0)")
        if td_diff < -0.1: print(" Warning: Dew-point temperature exceeds air temperature!")

print("="*50)

plt.tight_layout()
plt.show()
