import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import os
from pathlib import Path

PROJECT_DATA_DIR = Path(
    os.environ.get(
        "PROJECT_DATA_DIR",
        Path(__file__).resolve().parents[1] / "data",
    )
)
BUILDING_OUTPUT_DIR = Path(
    os.environ.get(
        "BUILDING_OUTPUT_DIR",
        PROJECT_DATA_DIR / "building_inventory",
    )
)
res_path = Path(
    os.environ.get(
        "RESSTOCK_METADATA_FILE",
        PROJECT_DATA_DIR / "resstock" / "upgrade0.csv",
    )
)
df_res_full = pd.read_csv(res_path)

def generate_county_map(df, target_state):


    state_df = df[df['in.state'] == target_state].drop_duplicates(subset=['in.county'])


    def parse_fips(g_code):
        if pd.isna(g_code) or len(str(g_code)) < 8:
            return str(g_code)
        s = str(g_code)
        state_part = s[1:3]
        county_part = s[4:7]
        return state_part + county_part


    county_map = {}
    for _, row in state_df.iterrows():
        fips = parse_fips(row['in.county'])
        name = row['in.county_name']
        county_map[fips] = name


    sorted_map = dict(sorted(county_map.items()))

    return sorted_map


def standardize_bldg_type_forced(row):
    """Map each real building to a ResStock-compatible building type."""
    occ_type = str(row['PRIM_OCC'])
    area_sqm = row['SQMETERS']

    if 'Single Family Dwelling' in occ_type:
        return 'Single-Family Detached'

    elif 'Multi - Family Dwelling' in occ_type:


        if area_sqm < 450:
            return 'Multi-Family with 2 - 4 Units'
        else:
            return 'Multi-Family with 5+ Units'

    elif 'Manufactured home' in occ_type:
        return 'Mobile Home'

    elif 'Nursing Home' in occ_type or 'Institutional Dormitory' in occ_type or 'Temporary Lodging' in occ_type:

        return 'Multi-Family with 5+ Units'

    elif 'Unclassified' in occ_type:

        if area_sqm < 250:

            return 'Single-Family Detached'
        else:

            return 'Multi-Family with 5+ Units'
    else:


        return 'Single-Family Detached'


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
    building_path = BUILDING_OUTPUT_DIR / aa_name / f"{aa_name}.parquet"
    bd_full = pd.read_parquet(building_path)

    out_file = BUILDING_OUTPUT_DIR / aa_name / f"{aa_name}_ResStock.parquet"

    ma_county_map = generate_county_map(df_res_full, aa_name)


    print("Preprocessing real-building data...")


    aa = bd_full.loc[:10,:]

    bd = bd_full[bd_full['GEOID'].str.startswith(f"{aa_id:02d}")].copy()


    bd['county_fips'] = bd['GEOID'].str.slice(0, 5)
    bd['county_name'] = bd['county_fips'].map(ma_county_map)


    bd['stories'] = (bd['HEIGHT'] / 3).round().astype(int)

    bd.loc[bd['stories'] == 0, 'stories'] = 1


    bd['sqft'] = bd['SQMETERS'] * bd['stories'] * 10.7639


    bd['building_type'] = bd.apply(standardize_bldg_type_forced, axis=1)


    bd_match_features = bd[[
        'BUILD_ID', 'county_name', 'building_type',
        'sqft', 'stories', 'assigned_people', 'bldg_assigned_income'
    ]].rename(columns={
        'assigned_people': 'occupants',
        'bldg_assigned_income': 'income'
    }).copy()

    print(f"Real-building preprocessing completed; total {len(bd_match_features)} buildings.")
    print("Real-building feature sample:")
    print(bd_match_features.head())


    print("\nPreprocessing ResStock templates...")


    res_counties = bd_match_features['county_name'].unique()


    df_res = df_res_full[
        (df_res_full['in.state'] == aa_name) &
        (df_res_full['in.county_name'].isin(res_counties))
    ].copy()


    df_res['building_type'] = df_res['in.geometry_building_type_recs'].apply(
        lambda x: 'Multi-Family' if 'Multi-Family' in x else x
    )


    res_match_features = df_res[[
        'bldg_id', 'in.county_name', 'in.geometry_building_type_recs',
        'in.sqft..ft2', 'in.geometry_stories', 'in.occupants', 'in.representative_income'
    ]].rename(columns={
        'in.county_name': 'county_name',
        'in.geometry_building_type_recs': 'building_type',
        'in.sqft..ft2': 'sqft',
        'in.geometry_stories': 'stories',
        'in.occupants': 'occupants',
        'in.representative_income': 'income'
    }).copy()


    numeric_cols = ['sqft', 'stories', 'occupants', 'income']
    for col in numeric_cols:
        res_match_features[col] = pd.to_numeric(res_match_features[col], errors='coerce')

    res_match_features.dropna(inplace=True)

    print(f"ResStock template preprocessing completed; remaining {len(res_match_features)} templates.")
    print("ResStock:")
    print(res_match_features.head())


    print("\nRunning the matching process(using dynamic penalties)...")
    rng = np.random.default_rng(seed=42)

    matches = []
    numeric_features = ['sqft', 'stories', 'occupants', 'income']


    usage_counts = {tid: 0 for tid in res_match_features['bldg_id']}


    for county in res_counties:
        print(f"  -- Processing {county}...")

        bd_county = bd_match_features[bd_match_features['county_name'] == county]
        res_county = res_match_features[res_match_features['county_name'] == county]

        if res_county.empty:
            print(f"    Warning:No ResStock templates found for {county} .")
            for idx, real_building in bd_county.iterrows():
                matches.append({
                    'BUILD_ID': real_building['BUILD_ID'],
                    'matched_bldg_id': None,
                    'match_distance': np.inf,
                    'reason': 'No templates for county'
                })
            continue

        for bldg_type in bd_county['building_type'].unique():
            bd_subset = bd_county[bd_county['building_type'] == bldg_type].copy()
            res_subset = res_county[res_county['building_type'] == bldg_type].copy()

            if res_subset.empty:
                for idx, real_building in bd_subset.iterrows():
                    matches.append({
                        'BUILD_ID': real_building['BUILD_ID'],
                        'matched_bldg_id': None,
                        'match_distance': np.inf,
                        'reason': f'No templates for {bldg_type}'
                    })
                continue


            scaler = MinMaxScaler()
            real_f = bd_subset[numeric_features].values
            cand_f = res_subset[numeric_features].values

            combined = np.vstack([real_f, cand_f])
            scaler.fit(combined)

            real_scaled = scaler.transform(real_f)
            cand_scaled = scaler.transform(cand_f)


            distance_matrix = cdist(real_scaled, cand_scaled, 'euclidean')


            K = min(20, len(res_subset))


            penalty_factor = 0.05

            best_match_indices = []
            min_distances = []


            subset_tids = res_subset['bldg_id'].values

            for i in range(len(real_scaled)):
                dist_to_all = distance_matrix[i]


                counts_array = np.array([usage_counts[tid] for tid in subset_tids])

                adjusted_dists = dist_to_all + (counts_array * penalty_factor)


                top_k_idx = np.argpartition(adjusted_dists, K - 1)[:K]
                top_k_dists_adj = adjusted_dists[top_k_idx]


                weights = 1.0 / (top_k_dists_adj + 1e-6)
                probs = weights / weights.sum()


                chosen_idx = rng.choice(top_k_idx, p=probs)
                chosen_tid = subset_tids[chosen_idx]


                usage_counts[chosen_tid] += 1

                best_match_indices.append(chosen_idx)

                min_distances.append(dist_to_all[chosen_idx])

            matched_bldg_ids = res_subset['bldg_id'].iloc[best_match_indices].values

            for i, real_id in enumerate(bd_subset['BUILD_ID']):
                matches.append({
                    'BUILD_ID': real_id,
                    'matched_bldg_id': matched_bldg_ids[i],
                    'match_distance': min_distances[i],
                    'reason': 'Success'
                })

    print("Dynamic-penalty matching completed!")


    res_features_for_merge = res_match_features.rename(columns={
        'sqft': 'template_sqft',
        'stories': 'template_stories',
        'occupants': 'template_occupants',
        'income': 'template_income',
        'building_type': 'template_building_type'
    })


    res_features_for_merge = res_features_for_merge[[
        'bldg_id', 'template_sqft', 'template_stories',
        'template_occupants', 'template_income', 'template_building_type'
    ]]

    df_matches = pd.DataFrame(matches)


    df_matches_with_template_props = pd.merge(
        df_matches,
        res_features_for_merge,
        left_on='matched_bldg_id',
        right_on='bldg_id',
        how='left'
    )


    df_final_result = pd.merge(
        bd,
        df_matches_with_template_props,
        on='BUILD_ID',
        how='left'
    )


    if 'bldg_id' in df_final_result.columns:
        df_final_result.drop(columns=['bldg_id'], inplace=True)

    print("\nMatch preview:")
    print(df_final_result[[
        'BUILD_ID', 'PROP_ADDR', 'PROP_CITY', 'county_name',
        'building_type', 'sqft', 'stories',
        'matched_bldg_id', 'match_distance', 'reason'
    ]].head(10))


    df_final_result.to_parquet(
        out_file,
        engine='pyarrow',
        compression='snappy'
    )

    aa = df_final_result.iloc[:10,:]


    print("\n" + "="*50)
    print("ResStock template coverage by county")
    print("="*50)


    all_templates = res_match_features[['bldg_id', 'county_name']].drop_duplicates()


    selected_ids = df_matches['matched_bldg_id'].dropna().unique()


    total_counts = all_templates.groupby('county_name').size().reset_index(name='total_templates')


    selected_templates_info = all_templates[all_templates['bldg_id'].isin(selected_ids)]
    selected_counts = selected_templates_info.groupby('county_name').size().reset_index(name='selected_templates')


    stats = pd.merge(total_counts, selected_counts, on='county_name', how='left').fillna(0)
    stats['selected_templates'] = stats['selected_templates'].astype(int)


    stats['unselected_count'] = stats['total_templates'] - stats['selected_templates']
    stats['unselected_ratio'] = (stats['unselected_count'] / stats['total_templates'] * 100).round(2)


    stats_display = stats.copy()
    stats_display['unselected_ratio'] = stats_display['unselected_ratio'].astype(str) + '%'


    total_all = stats['total_templates'].sum()
    selected_all = stats['selected_templates'].sum()
    unselected_all = total_all - selected_all
    print("-" * 50)
    print(f"State total (Grand Total):")
    print(f"Total templates: {total_all}")
    print(f"Selected unique templates: {selected_all}")
    print(f"Unselected templates: {unselected_all}")
    print(f"Statewide unselected share: {(unselected_all/total_all*100):.2f}%")
    print("="*50)
