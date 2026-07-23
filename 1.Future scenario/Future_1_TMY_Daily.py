import xarray as xr
import numpy as np
import pandas as pd
import os
import cftime
import gc
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from pathlib import Path

# --- Configuration ---
PROJECT_DATA_DIR = Path(
    os.environ.get(
        "PROJECT_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data",
    )
)
state = os.environ.get("TARGET_STATE", "CA")
DATA_DIR = os.environ.get(
    "CMIP6_DATA_DIR",
    str(PROJECT_DATA_DIR / "weather" / "cmip6"),
)
FUTURE_WEATHER_DIR = Path(
    os.environ.get(
        "FUTURE_WEATHER_DIR",
        PROJECT_DATA_DIR / "weather" / "future",
    )
)
MODELS = ['MPI-ESM1-2-HR', 'CanESM5', 'GFDL-ESM4']
VARIABLES = ['tasmax', 'tasmin', 'rsds', 'sfcWind', 'hurs', 'rlds']
DECADES = {'2020s': (2020, 2029), '2030s': (2030, 2039), '2040s': (2040, 2049), '2050s': (2050, 2059)}
NUM_SAMPLING_POINTS = 16

US_STATES_BOUNDING_BOXES = {
    'AL': {'min_lat': 30.22, 'max_lat': 35.00, 'min_lon': -88.47, 'max_lon': -84.89},
    'AK': {'min_lat': 51.21, 'max_lat': 71.35, 'min_lon': -179.15, 'max_lon': -129.98},
    'AZ': {'min_lat': 31.33, 'max_lat': 37.00, 'min_lon': -114.82, 'max_lon': -109.05},
    'AR': {'min_lat': 33.00, 'max_lat': 36.50, 'min_lon': -94.62, 'max_lon': -89.64},
    'CA': {'min_lat': 32.53, 'max_lat': 42.01, 'min_lon': -124.41, 'max_lon': -114.13},
    'CO': {'min_lat': 37.00, 'max_lat': 41.00, 'min_lon': -109.06, 'max_lon': -102.04},
    'CT': {'min_lat': 40.99, 'max_lat': 42.05, 'min_lon': -73.73, 'max_lon': -71.79},
    'DE': {'min_lat': 38.45, 'max_lat': 39.84, 'min_lon': -75.79, 'max_lon': -75.05},
    'DC': {'min_lat': 38.79, 'max_lat': 38.99, 'min_lon': -77.12, 'max_lon': -76.91},
    'FL': {'min_lat': 24.40, 'max_lat': 31.00, 'min_lon': -87.64, 'max_lon': -80.03},
    'GA': {'min_lat': 30.36, 'max_lat': 35.00, 'min_lon': -85.61, 'max_lon': -80.84},
    'HI': {'min_lat': 18.91, 'max_lat': 28.40, 'min_lon': -178.33, 'max_lon': -154.81},
    'ID': {'min_lat': 42.00, 'max_lat': 49.00, 'min_lon': -117.24, 'max_lon': -111.04},
    'IL': {'min_lat': 36.97, 'max_lat': 42.51, 'min_lon': -91.51, 'max_lon': -87.50},
    'IN': {'min_lat': 37.77, 'max_lat': 41.76, 'min_lon': -88.09, 'max_lon': -84.78},
    'IA': {'min_lat': 40.38, 'max_lat': 43.50, 'min_lon': -96.64, 'max_lon': -90.14},
    'KS': {'min_lat': 37.00, 'max_lat': 40.00, 'min_lon': -102.05, 'max_lon': -94.59},
    'KY': {'min_lat': 36.50, 'max_lat': 39.15, 'min_lon': -89.57, 'max_lon': -81.93},
    'LA': {'min_lat': 28.93, 'max_lat': 33.02, 'min_lon': -94.04, 'max_lon': -88.82},
    'ME': {'min_lat': 42.97, 'max_lat': 47.46, 'min_lon': -71.08, 'max_lon': -66.95},
    'MD': {'min_lat': 37.91, 'max_lat': 39.72, 'min_lon': -79.49, 'max_lon': -75.05},
    'MA': {'min_lat': 41.24, 'max_lat': 42.89, 'min_lon': -73.51, 'max_lon': -69.93},
    'MI': {'min_lat': 41.70, 'max_lat': 48.31, 'min_lon': -90.42, 'max_lon': -82.12},
    'MN': {'min_lat': 43.50, 'max_lat': 49.38, 'min_lon': -97.24, 'max_lon': -89.49},
    'MS': {'min_lat': 30.17, 'max_lat': 35.00, 'min_lon': -91.65, 'max_lon': -88.10},
    'MO': {'min_lat': 35.99, 'max_lat': 40.61, 'min_lon': -95.77, 'max_lon': -89.10},
    'MT': {'min_lat': 44.36, 'max_lat': 49.00, 'min_lon': -116.05, 'max_lon': -104.04},
    'NE': {'min_lat': 40.00, 'max_lat': 43.00, 'min_lon': -104.05, 'max_lon': -95.31},
    'NV': {'min_lat': 35.00, 'max_lat': 42.00, 'min_lon': -120.01, 'max_lon': -114.04},
    'NH': {'min_lat': 42.70, 'max_lat': 45.31, 'min_lon': -72.56, 'max_lon': -70.58},
    'NJ': {'min_lat': 38.93, 'max_lat': 41.36, 'min_lon': -75.56, 'max_lon': -73.89},
    'NM': {'min_lat': 31.33, 'max_lat': 37.00, 'min_lon': -109.05, 'max_lon': -103.00},
    'NY': {'min_lat': 40.50, 'max_lat': 45.02, 'min_lon': -79.76, 'max_lon': -71.86},
    'NC': {'min_lat': 33.84, 'max_lat': 36.59, 'min_lon': -84.32, 'max_lon': -75.46},
    'ND': {'min_lat': 45.94, 'max_lat': 49.00, 'min_lon': -104.05, 'max_lon': -96.55},
    'OH': {'min_lat': 38.40, 'max_lat': 41.98, 'min_lon': -84.82, 'max_lon': -80.52},
    'OK': {'min_lat': 33.62, 'max_lat': 37.00, 'min_lon': -103.00, 'max_lon': -94.43},
    'OR': {'min_lat': 41.99, 'max_lat': 46.30, 'min_lon': -124.57, 'max_lon': -116.46},
    'PA': {'min_lat': 39.72, 'max_lat': 42.27, 'min_lon': -80.52, 'max_lon': -74.69},
    'RI': {'min_lat': 41.15, 'max_lat': 42.02, 'min_lon': -71.89, 'max_lon': -71.12},
    'SC': {'min_lat': 32.03, 'max_lat': 35.22, 'min_lon': -83.35, 'max_lon': -78.54},
    'SD': {'min_lat': 42.95, 'max_lat': 45.95, 'min_lon': -104.06, 'max_lon': -96.44},
    'TN': {'min_lat': 34.99, 'max_lat': 36.68, 'min_lon': -90.31, 'max_lon': -81.65},
    'TX': {'min_lat': 25.84, 'max_lat': 36.50, 'min_lon': -106.65, 'max_lon': -93.51},
    'UT': {'min_lat': 37.00, 'max_lat': 42.00, 'min_lon': -114.05, 'max_lon': -109.04},
    'VT': {'min_lat': 42.73, 'max_lat': 45.02, 'min_lon': -73.44, 'max_lon': -71.47},
    'VA': {'min_lat': 36.54, 'max_lat': 39.47, 'min_lon': -83.68, 'max_lon': -75.24},
    'WA': {'min_lat': 45.54, 'max_lat': 49.00, 'min_lon': -124.85, 'max_lon': -116.92},
    'WV': {'min_lat': 37.20, 'max_lat': 40.64, 'min_lon': -82.64, 'max_lon': -77.72},
    'WI': {'min_lat': 42.49, 'max_lat': 47.08, 'min_lon': -92.89, 'max_lon': -86.25},
    'WY': {'min_lat': 41.00, 'max_lat': 45.00, 'min_lon': -111.05, 'max_lon': -104.05}
}

STATE_BOUNDING_BOX = US_STATES_BOUNDING_BOXES.get(state)

def get_file_path(var, model, year, SSP):
    ss = 'gr1' if model == 'GFDL-ESM4' else 'gn'
    return os.path.join(DATA_DIR, model, SSP, var, f"{var}_day_{model}_{SSP}_r1i1p1f1_{ss}_{year}_v2.0.nc")

def _convert_lon_360_to_180(ds):
    lon_name = 'lon'
    if lon_name not in ds.coords: return ds
    lon_max = ds[lon_name].max().item()
    if lon_max > 180:
        ds = ds.assign_coords(**{lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)
    return ds

def load_and_crop_data_safe(var, model, years, SSP, ma_bbox):
    """Load valid model files for the requested years and crop to the state."""

    CORRUPTED_BLACKLIST = [
        "rsds_day_CanESM5_ssp585_r1i1p1f1_gn_2046_v2.0.nc"
    ]

    valid_paths = []
    for year in years:
        path = get_file_path(var, model, year, SSP)
        file_name = os.path.basename(path)


        if file_name in CORRUPTED_BLACKLIST:
            print(f" Skippingfile: {file_name}")
            continue


        if os.path.exists(path):
            try:

                with xr.open_dataset(path, engine='netcdf4') as tmp:
                    _ = tmp[var].isel(time=0, lat=0, lon=0).values
                valid_paths.append(path)
            except Exception as e:
                print(f" Warning: foundfile,Skipping: {path}. Error: {e}")
        else:
            print(f" missingfile: {path}")

    if not valid_paths:
        raise FileNotFoundError(f"Unable to {model} {var} file")


    ds = xr.open_mfdataset(
        valid_paths,
        combine='by_coords',
        chunks={'time': 365, 'lat': 100, 'lon': 100},
        decode_times=True
    )


    ds = _convert_lon_360_to_180(ds)
    ds = ds.sel(lat=slice(ma_bbox['min_lat'], ma_bbox['max_lat']),
                lon=slice(ma_bbox['min_lon'], ma_bbox['max_lon']))

    if f"{var}_day" in ds.data_vars: ds = ds.rename({f"{var}_day": var})


    if not isinstance(ds.time.values[0], cftime.DatetimeNoLeap):
        if pd.api.types.is_datetime64_any_dtype(ds.time.dtype):
            non_leap_mask = ~((ds.time.dt.month == 2) & (ds.time.dt.day == 29))
            ds = ds.isel(time=non_leap_mask)
            noleap_times = [cftime.DatetimeNoLeap(t.year, t.month, t.day) for t in ds.time.to_series()]
            ds['time'] = noleap_times
        else:
            noleap_times = [cftime.DatetimeNoLeap(t.year, t.month, t.day) for t in ds.time.values]
            ds['time'] = noleap_times
        ds.time.attrs['calendar'] = 'noleap'

    return ds

def fuse_models_for_decade(variable_name, decade_start, decade_end, SSP, ma_bbox):
    years = list(range(decade_start, decade_end + 1))
    model_data = []
    ref_time = xr.cftime_range(start=f'{decade_start}-01-01', periods=365*len(years), freq='D', calendar='noleap')

    for model in MODELS:
        try:
            da = load_and_crop_data_safe(variable_name, model, years, SSP, ma_bbox)[variable_name]

            aligned_da = da.sortby('time').reindex({'time': ref_time}, method='nearest')

            aligned_da = aligned_da.ffill('time').bfill('time').ffill('lat').bfill('lat')
            model_data.append(aligned_da)
        except Exception as e:
            print(f"Skipping {model} ({variable_name}): {e}")

    if not model_data: return None, None
    fused = xr.concat(model_data, dim='model').mean(dim='model', skipna=True)

    fused = fused.ffill('time').bfill('time').ffill('lat').bfill('lat')
    return model_data, fused


for SSP in ['ssp126', 'ssp245', 'ssp585']:

    OUTPUT_DIR = str(FUTURE_WEATHER_DIR / state / SSP)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    init_ds = load_and_crop_data_safe(VARIABLES[0], MODELS[0], [2020], SSP, STATE_BOUNDING_BOX)
    ma_lats, ma_lons = init_ds['lat'].values, init_ds['lon'].values
    lon_mesh, lat_mesh = np.meshgrid(ma_lons, ma_lats)
    ma_grid_coords = np.c_[lat_mesh.ravel(), lon_mesh.ravel()]
    kmeans = KMeans(n_clusters=NUM_SAMPLING_POINTS, random_state=42, n_init=10).fit(ma_grid_coords)
    sampling_points = ma_grid_coords[cKDTree(ma_grid_coords).query(kmeans.cluster_centers_)[1]]
    nearest_idx = cKDTree(sampling_points).query(ma_grid_coords)[1].reshape(len(ma_lats), len(ma_lons))

    for decade_name, (d_start, d_end) in DECADES.items():

        tmy_out = os.path.join(OUTPUT_DIR, f"{state}_{SSP}_{decade_name}_TMY_daily.nc")


        if os.path.exists(tmy_out):
            print(f" Skipping {decade_name} {SSP}: file")
            continue

        print(f"\n Processdecade: {decade_name} ({SSP})")


        _, f_tmax = fuse_models_for_decade('tasmax', d_start, d_end, SSP, STATE_BOUNDING_BOX)
        _, f_tmin = fuse_models_for_decade('tasmin', d_start, d_end, SSP, STATE_BOUNDING_BOX)

        f_tmax = f_tmax.assign_coords(y_idx=('time', pd.factorize(f_tmax.time.dt.year.values)[0]))
        f_tmin = f_tmin.assign_coords(y_idx=('time', pd.factorize(f_tmin.time.dt.year.values)[0]))


        sp_lat_idx = xr.DataArray([np.abs(ma_lats - p[0]).argmin() for p in sampling_points], dims="sp")
        sp_lon_idx = xr.DataArray([np.abs(ma_lons - p[1]).argmin() for p in sampling_points], dims="sp")
        sp_tmax = f_tmax.isel(lat=sp_lat_idx, lon=sp_lon_idx).compute()
        sp_tmin = f_tmin.isel(lat=sp_lat_idx, lon=sp_lon_idx).compute()

        indices = np.zeros((len(sampling_points), 12, 2), dtype=int)
        for s in range(len(sampling_points)):
            for m in range(1, 13):
                m_max = sp_tmax.sel(time=sp_tmax.time.dt.month == m, sp=s)
                m_min = sp_tmin.sel(time=sp_tmin.time.dt.month == m, sp=s)
                d_avg_max, d_avg_min = m_max.mean().item(), m_min.mean().item()
                dists = [np.sqrt((m_max.where(m_max.y_idx==y, drop=True).mean() - d_avg_max)**2 + (m_min.where(m_min.y_idx==y, drop=True).mean() - d_avg_min)**2).item() for y in range(10)]
                indices[s, m-1, 1] = np.argmin(dists)


        sel_tmy = indices[nearest_idx, :, 1]


        res_tmy = {}
        for var in VARIABLES:
            print(f"   variable: {var}...")
            _, fused_v = fuse_models_for_decade(var, d_start, d_end, SSP, STATE_BOUNDING_BOX)
            fused_v = fused_v.assign_coords(y_idx=('time', pd.factorize(fused_v.time.dt.year.values)[0]))

            t_chunks = []
            for m_idx, m_data in fused_v.groupby('time.month'):

                def get_d(da): return xr.DataArray(np.arange(len(da.time)), dims='time', coords={'time': da.time})
                day_c = m_data.groupby('y_idx').apply(get_d)
                struc = m_data.assign_coords(day=day_c).set_index(time=['y_idx', 'day']).unstack('time').transpose('y_idx', 'day', 'lat', 'lon')


                ida_t = xr.DataArray(sel_tmy[:,:,m_idx-1], dims=['lat','lon'], coords={'lat':ma_lats, 'lon':ma_lons})


                t_chunks.append(struc.isel(y_idx=ida_t).rename({'day':'time'}).drop_vars(['month','y_idx'], errors='ignore'))


            res_tmy[var] = xr.concat(t_chunks, dim='time')
            del fused_v; gc.collect()


        ref_y = xr.cftime_range(start='2001-01-01', periods=365, freq='D', calendar='noleap')


        print("Writing daily TMY NetCDF...")

        xr.Dataset(res_tmy).assign_coords(time=ref_y).to_netcdf(tmy_out)


        print("Writing TMY selection-index NetCDF...")
        month_coords = np.arange(1, 13)


        ds_sel_tmy = xr.Dataset(
            {"year_index": (["lat", "lon", "month"], sel_tmy)},
            coords={"lat": ma_lats, "lon": ma_lons, "month": month_coords}
        )


        ds_sel_tmy.to_netcdf(tmy_out.replace('.nc', '_selection.nc'))

        print(f" {decade_name} {SSP} Processcompleted!")

        print(f"   - file: {os.path.basename(tmy_out)}")
