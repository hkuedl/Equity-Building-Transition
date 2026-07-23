import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer
import numpy as np
import random
import os
from pathlib import Path


PROJECT_DATA_DIR = Path(
    os.environ.get(
        "PROJECT_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data",
    )
)
SCENARIO_INPUT_DIR = Path(
    os.environ.get(
        "SCENARIO_INPUT_DIR",
        PROJECT_DATA_DIR / "future_scenario",
    )
)
BUILDING_FOOTPRINT_DIR = Path(
    os.environ.get(
        "BUILDING_FOOTPRINT_DIR",
        SCENARIO_INPUT_DIR / "building_footprints",
    )
)
CENSUS_PROFILE_DIR = Path(
    os.environ.get(
        "CENSUS_PROFILE_DIR",
        SCENARIO_INPUT_DIR / "census_profiles",
    )
)
BUILDING_OUTPUT_DIR = Path(
    os.environ.get(
        "BUILDING_OUTPUT_DIR",
        PROJECT_DATA_DIR / "building_inventory",
    )
)
path_shp_admin_boundaries = Path(
    os.environ.get(
        "BLOCK_GROUP_SHAPEFILE",
        SCENARIO_INPUT_DIR / "boundaries" / "US.shp",
    )
)
tif_path = Path(
    os.environ.get(
        "BUILDING_HEIGHT_RASTER",
        SCENARIO_INPUT_DIR / "building_heights" / "GBH2020_150m_GEDI.tif",
    )
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

for aa_name_full, aa_name, aa_id in US_STATES:
    if TARGET_STATES and aa_name not in TARGET_STATES:
        continue


    path_gdb_buildings = BUILDING_FOOTPRINT_DIR / aa_name / f"{aa_name}_Structures.gdb"
    layer_name_buildings = f"{aa_name}_Structures"
    path_csv_profiles = CENSUS_PROFILE_DIR / f"{aa_name_full}_Census.xlsx"
    output_fold = BUILDING_OUTPUT_DIR / aa_name
    output_fold.mkdir(parents=True, exist_ok=True)
    output_path = output_fold / f"{aa_name}.parquet"

    print("---  1: Load and preprocess data ---")


    print(f"Loading and filtering {aa_name_full} Block Groups...")
    gdf_bg_us = gpd.read_file(path_shp_admin_boundaries)
    gdf_bg_ma = gdf_bg_us[gdf_bg_us['STATEFP'] == f"{aa_id:02d}"].copy()
    gdf_bg_ma['GEOID'] = gdf_bg_ma['GEOID'].astype(str)
    print(f"Loaded {len(gdf_bg_ma)} Block Groups.")


    del gdf_bg_us


    print("Loading building footprints...")
    try:
        gdf_buildings_ma = gpd.read_file(path_gdb_buildings, layer=layer_name_buildings, engine='pyogrio')
    except Exception as e:
        print(f"pyogrio failed: {e}. Skipping this state.")
        continue

    print(f"Loaded {len(gdf_buildings_ma)} building footprints.")


    gdf_buildings_residential = gdf_buildings_ma[gdf_buildings_ma['OCC_CLS'] == 'Residential'].copy()
    print(f"filter {len(gdf_buildings_residential)} residential buildings.")
    del gdf_buildings_ma

    print("Loading demographic profiles...")
    df_profiles = pd.read_excel(path_csv_profiles)
    df_profiles.rename(columns={'ID': 'GEOID'}, inplace=True)
    df_profiles['GEOID'] = df_profiles['GEOID'].astype(str)
    df_profiles['GEOID'] = df_profiles['GEOID'].replace(r'^\d+US', '', regex=True)
    print(f"Loaded {len(df_profiles)} Block Group.")


    print("Harmonizing coordinate reference systems...")
    if gdf_bg_ma.crs != gdf_buildings_residential.crs:
        print(f"Warning: CRS mismatch!converting building-layer CRS...")
        gdf_buildings_residential = gdf_buildings_residential.to_crs(gdf_bg_ma.crs)
    else:
        print("CRS matched!")

    print("\n---  1: buildingBlock Group ---")
    gdf_buildings_with_bg = gpd.sjoin(gdf_buildings_residential, gdf_bg_ma[['GEOID', 'geometry']], how="inner", predicate="within")
    gdf_buildings_with_bg.drop(columns=['index_right'], inplace=True)
    print(f"Spatial join completed.{len(gdf_buildings_with_bg)} residential buildingssuccess.")


    print("\n---  1.5: Extracting building heights from the GBH2020 150 m raster ---")


    has_height_mask = gdf_buildings_with_bg['HEIGHT'].notna() & (gdf_buildings_with_bg['HEIGHT'] > 0)
    needs_height_mask = ~has_height_mask

    print(f"Existing heights: {has_height_mask.sum()} | Missing heights: {needs_height_mask.sum()}")


    coords_lon_lat = list(zip(gdf_buildings_with_bg['LONGITUDE'], gdf_buildings_with_bg['LATITUDE']))

    try:
        with rasterio.open(tif_path) as src:
            raster_crs = src.crs
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

            print("Transforming coordinates and sampling...")
            lon_lat_array = np.array(coords_lon_lat)
            px, py = transformer.transform(lon_lat_array[:, 0], lon_lat_array[:, 1])
            projected_coords = list(zip(px, py))

            sampled_gen = src.sample(projected_coords)
            tif_heights = np.array([float(v[0]) for v in sampled_gen])


        invalid_mask = (tif_heights <= 0) | (tif_heights < -1e30)
        tif_heights[invalid_mask] = 3.5


        gdf_buildings_with_bg['HEIGHT'] = tif_heights
        print(f"Height extraction completed.mean height: {np.mean(tif_heights):.1f}m")

    except Exception as e:
        print(f"Height extraction failed (check the input path): {e}")
        print("Using the default 3.5 m height for missing values...")
        gdf_buildings_with_bg.loc[needs_height_mask, 'HEIGHT'] = 3.5


    gdf_final = pd.merge(gdf_buildings_with_bg, df_profiles, on='GEOID', how='left')


    print("\n---  2: Computing robust building weights ---")


    gdf_final['estimated_floors'] = np.maximum(1, (gdf_final['HEIGHT'] / 3.0)).round()
    gdf_final['total_floor_area'] = gdf_final['SQFEET'] * gdf_final['estimated_floors']


    avg_sfh_total_area = gdf_final[
        (gdf_final['PRIM_OCC'] == 'Single Family Dwelling') & (gdf_final['total_floor_area'] > 0)
    ].groupby('GEOID')['total_floor_area'].mean().rename('avg_sfh_total_area')

    gdf_final = gdf_final.merge(avg_sfh_total_area, on='GEOID', how='left')
    gdf_final['avg_sfh_total_area'].fillna(gdf_final['avg_sfh_total_area'].mean(), inplace=True)


    def calculate_weight_robust(row):
        if row['PRIM_OCC'] == 'Single Family Dwelling':
            return 1.0

        if row['total_floor_area'] > 0 and row['avg_sfh_total_area'] > 0:

            weight = row['total_floor_area'] / row['avg_sfh_total_area']


            weight = min(weight, 50.0)

            return max(1.0, weight)
        return 1.0

    gdf_final['building_weight'] = gdf_final.apply(calculate_weight_robust, axis=1)


    total_weight_per_bg = gdf_final.groupby('GEOID')['building_weight'].sum().rename('total_bg_weight')
    gdf_final = gdf_final.merge(total_weight_per_bg, on='GEOID', how='left')


    print("\n---  2.3 : Allocation algorithm (minimum occupancy plus weighted allocation) ---")

    random.seed(42)


    gdf_final['assigned_households'] = 1
    gdf_final['assigned_people'] = 3


    bg_counts = gdf_final.groupby('GEOID')['BUILD_ID'].count().rename('bg_bldg_count')
    gdf_final = gdf_final.merge(bg_counts, on='GEOID', how='left')


    gdf_final['residual_total_hh'] = (gdf_final['Total households'] - (gdf_final['bg_bldg_count'] * 1)).clip(lower=0)
    gdf_final['residual_total_pp'] = (gdf_final['Total'] - (gdf_final['bg_bldg_count'] * 3)).clip(lower=0)

    aa = gdf_final.iloc[:10,:]


    gdf_final['alloc_ratio'] = (gdf_final['building_weight'] / gdf_final['total_bg_weight']).fillna(0)

    gdf_final['alloc_residual_hh'] = gdf_final['alloc_ratio'] * gdf_final['residual_total_hh']
    gdf_final['alloc_residual_pp'] = gdf_final['alloc_ratio'] * gdf_final['residual_total_pp']


    gdf_final['assigned_households'] += np.floor(gdf_final['alloc_residual_hh']).astype(int)
    gdf_final['assigned_people'] += np.floor(gdf_final['alloc_residual_pp']).astype(int)


    print("Redistributing residuals...")

    def distribute_remainder(df, target_col, residual_col, alloc_float_col, baseline):


        for geoid, group in df.groupby('GEOID'):
            target_residual = int(group[residual_col].iloc[0])
            current_allocated = int(group[target_col].sum() - (len(group) * baseline))
            diff = target_residual - current_allocated

            if diff > 0:
                fractional = group[alloc_float_col] - np.floor(group[alloc_float_col])

                top_indices = fractional.nlargest(diff).index
                df.loc[top_indices, target_col] += 1

    distribute_remainder(gdf_final, 'assigned_households', 'residual_total_hh', 'alloc_residual_hh', 1)
    distribute_remainder(gdf_final, 'assigned_people', 'residual_total_pp', 'alloc_residual_pp', 3)


    print("\n---  2.5: Applying physical constraints ---")


    gdf_final['max_hh_by_area'] = (gdf_final['total_floor_area'] / 400).apply(np.ceil).clip(upper=300)
    gdf_final['assigned_households'] = np.minimum(gdf_final['assigned_households'], gdf_final['max_hh_by_area']).astype(int)


    max_pp_allowed = gdf_final['assigned_households'] * 5
    gdf_final['assigned_people'] = np.minimum(gdf_final['assigned_people'], max_pp_allowed).astype(int)


    min_pp_required = gdf_final['assigned_households'] * 1
    gdf_final['assigned_people'] = np.maximum(gdf_final['assigned_people'], min_pp_required).astype(int)


    gdf_final.drop(columns=['max_hh_by_area'], inplace=True, errors='ignore')

    print(f"After constraints:maximum households {gdf_final['assigned_households'].max()}, maximum occupants {gdf_final['assigned_people'].max()}")


    print("\n---  2.4 : Generating and assigning synthetic households ---")

    income_bins_config = {
        '<10k': (1000, 10000), '10k-15k': (10000, 15000), '15k-20k': (15000, 20000),
        '20k-25k': (20000, 25000), '25k-30k': (25000, 30000), '30k-35k': (30000, 35000),
        '35k-40k': (35000, 40000), '40k-45k': (40000, 45000), '45k-50k': (45000, 50000),
        '50k-60k': (50000, 60000), '60k-75k': (60000, 75000), '75k-100k': (75000, 100000),
        '100k-125k': (100000, 125000), '125k-150k': (125000, 150000), '150k-200k': (150000, 200000),
        '>200k': (200000, 300000)
    }


    income_cols = list(income_bins_config.keys())
    gdf_final[income_cols] = gdf_final[income_cols].fillna(0)

    synthetic_households_list = []

    for geoid, group_df in gdf_final.groupby('GEOID'):

        household_pool = []
        for col, (low, high) in income_bins_config.items():
            num_households_in_bin = int(group_df[col].iloc[0])
            for _ in range(num_households_in_bin):
                household_pool.append(random.uniform(low, high))


        total_needed = group_df['assigned_households'].sum()
        if len(household_pool) < total_needed:

            shortage = int(total_needed - len(household_pool))
            if len(household_pool) > 0:
                household_pool.extend(random.choices(household_pool, k=shortage))
            else:
                household_pool.extend([50000] * shortage)

        random.shuffle(household_pool)


        pool_idx = 0
        for idx, row in group_df.iterrows():
            n = int(row['assigned_households'])

            assigned = household_pool[pool_idx : pool_idx + n]


            if len(assigned) < n:
                extras = random.choices(household_pool, k=n-len(assigned)) if household_pool else [50000]*n
                assigned.extend(extras)

            for i, inc in enumerate(assigned):
                synthetic_households_list.append({
                    'BUILD_ID': row['BUILD_ID'],
                    'assigned_income': inc
                })
            pool_idx += n

    df_synthetic = pd.DataFrame(synthetic_households_list)
    print("Synthetic household generation completed.")


    print("\n--- Final step: Merge income data ---")


    bldg_median = df_synthetic.groupby('BUILD_ID')['assigned_income'].median().rename('bldg_assigned_income')


    gdf_final = gdf_final.merge(bldg_median, on='BUILD_ID', how='left')

    print(f"NaN values after merge: {gdf_final['bldg_assigned_income'].isnull().sum()}")


    gdf_final['bldg_assigned_income'] = gdf_final.groupby('GEOID')['bldg_assigned_income'].transform(
        lambda x: x.fillna(x.median())
    )

    gdf_final['bldg_assigned_income'] = gdf_final['bldg_assigned_income'].fillna(gdf_final['bldg_assigned_income'].median())

    print(f"Final NaN values: {gdf_final['bldg_assigned_income'].isnull().sum()}")

    #################

    print("\n" + "="*50)
    print("Consistency check (Consistency Check)")
    print("="*50)


    state_assigned_pp = gdf_final['assigned_people'].sum()
    state_census_pp = gdf_final.groupby('GEOID')['Total'].first().sum()

    state_assigned_hh = gdf_final['assigned_households'].sum()
    state_census_hh = gdf_final.groupby('GEOID')['Total households'].first().sum()

    print(f"[State population] allocated: {state_assigned_pp:,} | census: {state_census_pp:,}")
    print(f"[State households] allocated: {state_assigned_hh:,} | census: {state_census_hh:,}")


    pp_diff_pct = (state_assigned_pp - state_census_pp) / state_census_pp * 100 if state_census_pp > 0 else 0
    hh_diff_pct = (state_assigned_hh - state_census_hh) / state_census_hh * 100 if state_census_hh > 0 else 0

    print(f"Population difference: {pp_diff_pct:.4f}%")
    print(f"Household difference: {hh_diff_pct:.4f}%")


    bg_check = gdf_final.groupby('GEOID').agg({
        'assigned_people': 'sum',
        'Total': 'first',
        'assigned_households': 'sum',
        'Total households': 'first'
    })


    mismatch_pp = bg_check[bg_check['assigned_people'] != bg_check['Total']]
    mismatch_hh = bg_check[bg_check['assigned_households'] != bg_check['Total households']]

    print("-" * 50)
    print(f"Block groups with population mismatch: {len(mismatch_pp)} / {len(bg_check)}")
    print(f"Block groups with household mismatch: {len(mismatch_hh)} / {len(bg_check)}")

    if len(mismatch_pp) > 0:
        avg_diff = (mismatch_pp['assigned_people'] - mismatch_pp['Total']).mean()
        print(f"Mean population over-allocation in mismatched areas: {avg_diff:.2f} people/BG (mainly caused by the minimum-occupancy constraint)")

    print("="*50)

    gdf_final.drop(columns=['bg_bldg_count','residual_total_hh','residual_total_pp','alloc_ratio','alloc_residual_hh','alloc_residual_pp'], inplace=True, errors='ignore')

    ax = gdf_final.iloc[:10,:]
    print("\n---  3: Save results ---")
    col_all = gdf_final.columns.tolist()
    col_select = ['BUILD_ID', 'PRIM_OCC', 'PROP_ADDR', 'PROP_CITY', 'PROP_ZIP', 'HEIGHT',
           'SQMETERS', 'LONGITUDE', 'LATITUDE', 'Shape_Length', 'Shape_Area',
           'geometry', 'GEOID', 'Total.1', 'White alone',
           'Black or African American alone',
           'American Indian and Alaska Native alone', 'Asian alone',
           'Native Hawaiian and Other Pacific Islander alone',
           'Some other race alone', 'Two or more races', 'assigned_households',
           'assigned_people', 'bldg_assigned_income']


    aa11 = gdf_final.iloc[:10,:]

    gdf_save = gdf_final[col_select]

    gdf_save.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Saved to: {output_path}")
