# -*- coding: utf-8 -*-
"""
robust_real_mapping.py

Distribution-constrained, income-bin-constrained, and capacity-aware mapping
from optimized ResStock template buildings back to real buildings.

This module is intentionally separated from the optimization code because the
real-building mapping logic is mode-specific and more complex than ordinary
nearest-neighbor matching.
"""

import os
import gc
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLS = ["sqft", "stories", "bldg_assigned_income"]
DEFAULT_DECADES = ["2020s", "2030s", "2040s", "2050s"]

# If fixed real-building mapping for one county takes too long, fall back to a
# simple deterministic mapping. You can override it by setting env var:

FIXED_MAPPING_TIMEOUT_SECONDS = int(os.environ.get("REAL_MAPPING_TIMEOUT_SECONDS", "600"))

INCOME_BIN_LABELS = {
    0: "<$25k",
    25000: "$25k-$50k",
    50000: "$50k-$75k",
    75000: "$75k-$100k",
    100000: "$100k-$150k",
    150000: "$150k-$200k",
    200000: ">$200k",
}

INCOME_BIN_ORDER = [0, 25000, 50000, 75000, 100000, 150000, 200000]


def normalize_id(x):
    """Normalize building/template IDs so 128786.0, '128786.0', and 128786 all become '128786'."""
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return pd.NA
        if float(x).is_integer():
            return str(int(x))
        return str(x).strip()

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".")[0]
    return s

def extract_original_building_id(x):
    """
    Safely extract original template Building_ID.
    Handles:
    - None / NaN
    - '01001|12345'
    - '12345'
    - '12345.0'
    """
    xid = normalize_id(x)
    if pd.isna(xid):
        return pd.NA

    s = str(xid).strip()
    if "|" in s:
        s = s.rsplit("|", 1)[-1]

    return normalize_id(s)

def assign_income_bin(income):
    """Assign income to the fixed 7-bin structure used by the optimization."""
    if pd.isna(income):
        return np.nan
    income = float(income)
    if income < 25000:
        return 0
    if income < 50000:
        return 25000
    if income < 75000:
        return 50000
    if income < 100000:
        return 75000
    if income < 150000:
        return 100000
    if income < 200000:
        return 150000
    return 200000


def safe_median(series, fallback=np.nan):
    series = pd.to_numeric(series, errors="coerce")
    med = series.median(skipna=True)
    return fallback if pd.isna(med) else med


def safe_nearest_k_indices(train_f, test_f, k=20, batch_size=1000):
    """
    Robust K-nearest-neighbor search using scipy.cdist.
    This avoids sklearn NearestNeighbors.kneighbors threadpool/BLAS issues.
    Returns an integer array with shape (n_test, k).
    """
    train_f = np.asarray(train_f, dtype=float)
    test_f = np.asarray(test_f, dtype=float)

    if train_f.shape[0] == 0 or test_f.shape[0] == 0:
        return np.empty((0, 0), dtype=int)

    k = min(int(k), train_f.shape[0])
    all_indices = []

    for start in range(0, test_f.shape[0], batch_size):
        end = min(start + batch_size, test_f.shape[0])
        dist = cdist(test_f[start:end], train_f, metric="euclidean")

        if k == 1:
            idx = np.argmin(dist, axis=1)[:, None]
        else:
            idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            row = np.arange(idx.shape[0])[:, None]
            idx = idx[row, np.argsort(dist[row, idx], axis=1)]

        all_indices.append(idx)

    return np.vstack(all_indices)


def _largest_remainder_counts(shares, total):
    """Convert fractional shares to integer counts that sum exactly to total."""
    if total <= 0 or len(shares) == 0:
        return {k: 0 for k in shares.index}

    raw = shares.astype(float) * total
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())

    out = base.to_dict()
    if remainder > 0:
        frac_order = (raw - base).sort_values(ascending=False).index.tolist()
        for k in frac_order[:remainder]:
            out[k] += 1
    return out


def _income_bin_distance(a, b):
    if pd.isna(a) or pd.isna(b):
        return 999
    try:
        return abs(INCOME_BIN_ORDER.index(int(a)) - INCOME_BIN_ORDER.index(int(b)))
    except ValueError:
        return 999


def _prepare_state_map(state_map, group_members, feature_cols=DEFAULT_FEATURE_COLS):
    """Clean real-building map and fill mapping features."""
    out = state_map.copy()
    out["county_fips"] = out["county_fips"].astype(str).str.zfill(5)
    group_members = [str(f).zfill(5) for f in group_members]

    if "matched_bldg_id" not in out.columns:
        raise ValueError("state_map must contain matched_bldg_id.")
    out["matched_bldg_id"] = out["matched_bldg_id"].map(normalize_id).astype("string")

    # Real-building income and income bins are central to the final mapping.
    if "bldg_assigned_income" not in out.columns:
        out["bldg_assigned_income"] = np.nan
    out["bldg_assigned_income"] = pd.to_numeric(out["bldg_assigned_income"], errors="coerce")
    state_income_median = safe_median(out["bldg_assigned_income"], fallback=50000.0)
    out["real_income_imputed_flag"] = out["bldg_assigned_income"].isna()
    out["bldg_assigned_income"] = out["bldg_assigned_income"].fillna(state_income_median)
    out["Real_Income_Bin"] = out["bldg_assigned_income"].apply(assign_income_bin).astype(int)
    out["Real_Income_Bin_Label"] = out["Real_Income_Bin"].map(INCOME_BIN_LABELS)

    out["mapping_feature_imputed_flag"] = False
    group_pool = out[out["county_fips"].isin(group_members)].copy()

    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        missing0 = out[col].isna()
        if not missing0.any():
            continue

        county_median = out.groupby("county_fips")[col].transform("median")
        out.loc[missing0, col] = county_median[missing0]

        missing1 = out[col].isna()
        if missing1.any():
            gm = safe_median(group_pool[col] if col in group_pool.columns else pd.Series(dtype=float), fallback=np.nan)
            out.loc[missing1, col] = gm

        missing2 = out[col].isna()
        if missing2.any():
            sm = safe_median(out[col], fallback=0.0)
            out.loc[missing2, col] = sm

        out.loc[missing0, "mapping_feature_imputed_flag"] = True

    if "template_sqft" not in out.columns:
        out["template_sqft"] = out["sqft"] if "sqft" in out.columns else 1.0
    out["template_sqft"] = (
        pd.to_numeric(out["template_sqft"], errors="coerce")
        .fillna(out["sqft"] if "sqft" in out.columns else 1.0)
        .replace(0, np.nan)
        .fillna(1.0)
    )
    return out


def _prepare_template_options(mode_opt, feature_cols=DEFAULT_FEATURE_COLS):
    """
    Build a unique optimized template pool for one mode.

    The candidate pool comes from df_opt itself rather than from already matched
    real buildings. This avoids concentrating imputed mappings on only the
    templates that happened to appear in direct matches.
    """
    opt = mode_opt.copy()

    if "Building_ID_Original" not in opt.columns:
        opt["Building_ID_Original"] = opt["Building_ID"].map(extract_original_building_id)
    else:
        opt["Building_ID_Original"] = opt["Building_ID_Original"].map(extract_original_building_id)

    opt["Original_FIPS"] = opt["Original_FIPS"].astype(str).str.zfill(5)
    opt["Building_ID_Original"] = opt["Building_ID_Original"].map(normalize_id).astype("string")
    opt = opt.dropna(subset=["Original_FIPS", "Building_ID_Original"]).copy()
    opt["Template_Key"] = opt["Original_FIPS"].astype(str) + "|" + opt["Building_ID_Original"].astype(str)

    if "Income_Bin" not in opt.columns:
        if "Income" in opt.columns:
            opt["Income_Bin"] = pd.to_numeric(opt["Income"], errors="coerce").apply(assign_income_bin)
        else:
            opt["Income_Bin"] = np.nan
    opt = opt.dropna(subset=["Income_Bin"]).copy()
    opt["Income_Bin"] = opt["Income_Bin"].astype(int)

    # Template feature columns: use template physical attributes if available;

    for col in feature_cols:
        if col == "bldg_assigned_income":
            opt[col] = pd.to_numeric(opt.get("Income", np.nan), errors="coerce")
        elif col not in opt.columns:
            opt[col] = np.nan
        else:
            opt[col] = pd.to_numeric(opt[col], errors="coerce")

    # Fill template mapping features within mode.
    for col in feature_cols:
        med = safe_median(opt[col], fallback=0.0)
        opt[col] = opt[col].fillna(med)

    # Use one row per template in mapping; strategy is mode-specific.
    keep_cols = [
        "Template_Key",
        "Original_FIPS",
        "Building_ID_Original",
        "Income_Bin",
        "Optimal_Strategy",
    ] + [c for c in feature_cols if c in opt.columns]
    pool = opt[keep_cols].drop_duplicates("Template_Key").copy()
    return opt, pool


def _target_strategy_counts(template_pool, n_real):
    if n_real <= 0 or template_pool.empty:
        return {}
    shares = template_pool["Optimal_Strategy"].value_counts(normalize=True)
    return _largest_remainder_counts(shares, n_real)


def _template_capacities(template_pool, n_real, reuse_factor=3):
    """
    Capacity is computed within the current real-building group.
    It avoids unlimited reuse of one template but still allows realistic many-to-one mapping.
    """
    keys = template_pool["Template_Key"].dropna().unique().tolist()
    if not keys:
        return {}
    base = int(np.ceil(max(n_real, 1) / len(keys)) * reuse_factor)
    base = max(base, 1)
    return {k: base for k in keys}


def _build_candidate_pool(
    template_pool,
    original_fips,
    real_income_bin,
    group_members,
    allow_group=False,
    allow_adjacent_income=False,
):
    """Candidate hierarchy: county+same bin -> group+same bin -> group+adjacent bin."""
    original_fips = str(original_fips).zfill(5)
    group_members = [str(f).zfill(5) for f in group_members]

    if not allow_group:
        base = template_pool[template_pool["Original_FIPS"] == original_fips].copy()
    else:
        base = template_pool[template_pool["Original_FIPS"].isin(group_members)].copy()

    if base.empty:
        return base, False

    same_bin = base[base["Income_Bin"].astype(int) == int(real_income_bin)].copy()
    if not same_bin.empty:
        return same_bin, False

    if allow_adjacent_income:
        base["_bin_dist"] = base["Income_Bin"].map(lambda b: _income_bin_distance(b, real_income_bin))
        min_dist = base["_bin_dist"].min()
        relaxed = base[base["_bin_dist"] == min_dist].drop(columns=["_bin_dist"]).copy()
        return relaxed, True

    return same_bin, False


def _select_template_for_real(
    real_row,
    candidate_pool,
    feature_cols,
    scaler,
    nn_model,
    nn_indices,
    local_position,
    template_use_count,
    template_capacity,
    strategy_remaining,
    require_strategy_capacity=True,
):
    """
    Select from K nearest candidates with two soft/hard preferences:
    1) strategy target count not yet exhausted;
    2) template capacity not yet exhausted.
    """
    candidate_positions = nn_indices[local_position]
    best_pos = None
    best_score = np.inf

    for rank, pos in enumerate(candidate_positions):
        t = candidate_pool.iloc[pos]
        key = t["Template_Key"]
        strategy = t["Optimal_Strategy"]

        strategy_over = 0
        if require_strategy_capacity and strategy_remaining:
            strategy_over = max(0, 1 - strategy_remaining.get(strategy, 0))

        cap_over = max(0, template_use_count[key] + 1 - template_capacity.get(key, 1))

        # Rank is the distance proxy because kneighbors already sorts by distance.
        score = rank + 1000 * strategy_over + 100 * cap_over
        if score < best_score:
            best_score = score
            best_pos = pos

        if strategy_over == 0 and cap_over == 0:
            break

    if best_pos is None:
        return None

    return candidate_pool.iloc[best_pos]


def _assign_real_group(
    df_real,
    target_idx,
    candidate_pool,
    feature_cols,
    mapping_level,
    template_use_count,
    strategy_remaining,
    template_capacity,
    income_relaxed_flag=False,
    k_neighbors=20,
    mapping_start_time=None,
    timeout_seconds=None,
):
    if len(target_idx) == 0 or candidate_pool.empty:
        return []

    assigned = []
    candidate_pool = candidate_pool.drop_duplicates("Template_Key").copy()
    if candidate_pool.empty:
        return []

    # Build NN over the whole candidate pool.
    n_neighbors = min(k_neighbors, len(candidate_pool))
    scaler = StandardScaler()
    train_f = scaler.fit_transform(candidate_pool[feature_cols])
    test_f = scaler.transform(df_real.loc[target_idx, feature_cols])

    nn_indices = safe_nearest_k_indices(train_f, test_f, k=n_neighbors)


    # Do not assign scalar-by-scalar to pandas StringDtype columns inside the loop.
    # For large CA/TX counties this can be extremely slow because pandas repeatedly
    # calls isna() on extension arrays. Collect values and write them in bulk.
    out_idx = []
    out_matched_id = []
    out_fips = []
    out_template_key = []
    out_income_bin = []
    out_strategy = []
    out_template_sqft = []

    for local_pos, real_idx in enumerate(target_idx):
        if (
            timeout_seconds is not None
            and mapping_start_time is not None
            and local_pos % 1000 == 0
            and time.time() - mapping_start_time > timeout_seconds
        ):
            raise TimeoutError(
                f"Fixed mapping exceeded {timeout_seconds}s during {mapping_level}; "
                "switching to simple fallback mapping."
            )

        selected = _select_template_for_real(
            real_row=df_real.loc[real_idx],
            candidate_pool=candidate_pool,
            feature_cols=feature_cols,
            scaler=scaler,
            nn_model=None,
            nn_indices=nn_indices,
            local_position=local_pos,
            template_use_count=template_use_count,
            template_capacity=template_capacity,
            strategy_remaining=strategy_remaining,
            require_strategy_capacity=True,
        )

        if selected is None:
            continue

        key = selected["Template_Key"]
        strategy = selected["Optimal_Strategy"]

        out_idx.append(real_idx)
        out_matched_id.append(selected["Building_ID_Original"])
        out_fips.append(selected["Original_FIPS"])
        out_template_key.append(selected["Template_Key"])
        out_income_bin.append(int(selected["Income_Bin"]))
        out_strategy.append(strategy)
        out_template_sqft.append(selected["sqft"] if "sqft" in selected.index else np.nan)

        template_use_count[key] += 1
        if strategy in strategy_remaining and strategy_remaining[strategy] > 0:
            strategy_remaining[strategy] -= 1

        assigned.append(real_idx)

    if out_idx:
        idx = pd.Index(out_idx)
        df_real.loc[idx, "matched_bldg_id"] = np.asarray(out_matched_id, dtype=object)
        df_real.loc[idx, "matched_template_fips"] = np.asarray(out_fips, dtype=object)
        df_real.loc[idx, "Template_Key"] = np.asarray(out_template_key, dtype=object)
        df_real.loc[idx, "mapping_level"] = mapping_level
        df_real.loc[idx, "imputed_flag"] = mapping_level != "county_direct"
        df_real.loc[idx, "income_bin_relaxed_flag"] = bool(income_relaxed_flag)
        df_real.loc[idx, "mapped_template_income_bin"] = np.asarray(out_income_bin, dtype=int)
        df_real.loc[idx, "mapped_template_strategy"] = np.asarray(out_strategy, dtype=object)
        if "template_sqft" in df_real.columns:
            df_real.loc[idx, "template_sqft"] = np.asarray(out_template_sqft, dtype=float)

    return assigned


def map_county_for_mode(
    county_map,
    template_pool,
    original_fips,
    group_members,
    feature_cols=DEFAULT_FEATURE_COLS,
    reuse_factor=3,
    k_neighbors=20,
    timeout_seconds=None,
):
    """
    Mode-specific mapping with four constraints:
    - accepted direct matches must be in the same income bin;
    - remapping candidates are same-income-bin first;
    - mapped real-building strategy shares follow template strategy shares within each income bin;
    - template reuse is capacity-limited.
    """
    df = county_map.copy()
    mapping_start_time = time.time()
    original_fips = str(original_fips).zfill(5)
    group_members = [str(f).zfill(5) for f in group_members]

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    # Use object dtype during assignment; pandas StringDtype scalar writes are very slow

    df["matched_bldg_id"] = df["matched_bldg_id"].map(normalize_id).astype(object)
    df["matched_template_fips"] = df["county_fips"].astype(object)
    df["Template_Key"] = df["matched_template_fips"].astype(str) + "|" + df["matched_bldg_id"].astype(str)

    df["mapping_level"] = "unresolved"
    df["imputed_flag"] = False
    df["income_bin_relaxed_flag"] = False
    df["mapped_template_income_bin"] = np.nan
    df["mapped_template_strategy"] = pd.NA

    template_info_df = (
        template_pool[["Template_Key", "Income_Bin", "Optimal_Strategy"]]
        .drop_duplicates("Template_Key")
        .rename(columns={"Income_Bin": "_template_income_bin", "Optimal_Strategy": "_template_strategy"})
    )
    available_keys = set(template_info_df["Template_Key"].unique())
    template_info = template_info_df.set_index("Template_Key")[["_template_income_bin", "_template_strategy"]].to_dict("index")

    # Direct match is accepted only if key exists and income bin matches.
    # Vectorized implementation; the old row-wise df.apply is very slow for TX.
    df = df.merge(template_info_df, on="Template_Key", how="left")
    direct_key_ok = df["_template_income_bin"].notna()
    direct_income_ok = direct_key_ok & (
        df["Real_Income_Bin"].astype("Int64") == df["_template_income_bin"].astype("Int64")
    )
    direct_ok = direct_key_ok & direct_income_ok

    df.loc[direct_ok, "mapping_level"] = "county_direct"
    df.loc[direct_ok, "mapped_template_income_bin"] = df.loc[direct_ok, "_template_income_bin"]
    df.loc[direct_ok, "mapped_template_strategy"] = df.loc[direct_ok, "_template_strategy"]

    # For direct income mismatches, force remapping.
    df.loc[direct_key_ok & ~direct_income_ok, "Template_Key"] = pd.NA
    df = df.drop(columns=[c for c in ["_template_income_bin", "_template_strategy"] if c in df.columns])

    # Process each real income bin separately so strategy distribution is controlled inside bins.
    for real_bin in sorted(df["Real_Income_Bin"].dropna().unique()):
        bin_idx_all = df.index[df["Real_Income_Bin"].astype(int) == int(real_bin)]
        if len(bin_idx_all) == 0:
            continue

        # Candidate reference for target strategy distribution:
        # county+same-bin if possible; otherwise group+same-bin; otherwise nearest income bin.
        ref_pool, ref_relaxed = _build_candidate_pool(
            template_pool=template_pool,
            original_fips=original_fips,
            real_income_bin=real_bin,
            group_members=group_members,
            allow_group=False,
            allow_adjacent_income=False,
        )
        if ref_pool.empty:
            ref_pool, ref_relaxed = _build_candidate_pool(
                template_pool=template_pool,
                original_fips=original_fips,
                real_income_bin=real_bin,
                group_members=group_members,
                allow_group=True,
                allow_adjacent_income=False,
            )
        if ref_pool.empty:
            ref_pool, ref_relaxed = _build_candidate_pool(
                template_pool=template_pool,
                original_fips=original_fips,
                real_income_bin=real_bin,
                group_members=group_members,
                allow_group=True,
                allow_adjacent_income=True,
            )

        if ref_pool.empty:
            continue

        n_real_bin = len(bin_idx_all)
        target_counts = _target_strategy_counts(ref_pool, n_real_bin)

        # Count direct assignments against strategy targets and template capacities.
        template_capacity = _template_capacities(ref_pool, n_real_bin, reuse_factor=reuse_factor)
        template_use_count = defaultdict(int)
        strategy_remaining = defaultdict(int, target_counts)

        direct_idx = df.index[(df.index.isin(bin_idx_all)) & (df["mapping_level"] == "county_direct")]
        for idx in direct_idx:
            key = df.loc[idx, "Template_Key"]
            strategy = df.loc[idx, "mapped_template_strategy"]
            template_use_count[key] += 1
            if strategy in strategy_remaining and strategy_remaining[strategy] > 0:
                strategy_remaining[strategy] -= 1

        unresolved = df.index[(df.index.isin(bin_idx_all)) & (df["mapping_level"] == "unresolved")]
        if len(unresolved) == 0:
            continue

        # 1) county + same income bin
        cpool, relaxed = _build_candidate_pool(
            template_pool, original_fips, real_bin, group_members,
            allow_group=False, allow_adjacent_income=False
        )
        assigned = _assign_real_group(
            df, unresolved, cpool, feature_cols, "county_imputed",
            template_use_count, strategy_remaining, template_capacity,
            income_relaxed_flag=False, k_neighbors=k_neighbors,
            mapping_start_time=mapping_start_time,
            timeout_seconds=timeout_seconds
        )

        unresolved = df.index[(df.index.isin(bin_idx_all)) & (df["mapping_level"] == "unresolved")]
        if len(unresolved) == 0:
            continue

        # 2) group + same income bin
        gpool, relaxed = _build_candidate_pool(
            template_pool, original_fips, real_bin, group_members,
            allow_group=True, allow_adjacent_income=False
        )
        _assign_real_group(
            df, unresolved, gpool, feature_cols, "group_imputed",
            template_use_count, strategy_remaining, template_capacity,
            income_relaxed_flag=False, k_neighbors=k_neighbors,
            mapping_start_time=mapping_start_time,
            timeout_seconds=timeout_seconds
        )

        unresolved = df.index[(df.index.isin(bin_idx_all)) & (df["mapping_level"] == "unresolved")]
        if len(unresolved) == 0:
            continue

        # 3) group + nearest income bin, with explicit flag
        rpool, relaxed = _build_candidate_pool(
            template_pool, original_fips, real_bin, group_members,
            allow_group=True, allow_adjacent_income=True
        )
        _assign_real_group(
            df, unresolved, rpool, feature_cols, "group_income_relaxed",
            template_use_count, strategy_remaining, template_capacity,
            income_relaxed_flag=True, k_neighbors=k_neighbors,
            mapping_start_time=mapping_start_time,
            timeout_seconds=timeout_seconds
        )

    return df


def simple_fast_map_county_for_mode(
    county_map,
    template_pool,
    original_fips,
    group_members,
    feature_cols=DEFAULT_FEATURE_COLS,
):
    """
    Fast fallback mapping used only when the full constrained NN mapping is too slow.

    It keeps the essential consistency needed for downstream merging:
    - accepts valid direct matches with same income bin;
    - otherwise assigns templates by county+same income bin;
    - then group+same income bin;
    - then group+nearest income bin;
    - no NN distance, no strategy-share target, no capacity control.
    """
    df = county_map.copy()
    original_fips = str(original_fips).zfill(5)
    group_members = [str(f).zfill(5) for f in group_members]

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["matched_bldg_id"] = df["matched_bldg_id"].map(normalize_id).astype(object)
    df["matched_template_fips"] = df["county_fips"].astype(object)
    df["Template_Key"] = df["matched_template_fips"].astype(str) + "|" + df["matched_bldg_id"].astype(str)

    df["mapping_level"] = "unresolved"
    df["imputed_flag"] = False
    df["income_bin_relaxed_flag"] = False
    df["mapped_template_income_bin"] = np.nan
    df["mapped_template_strategy"] = pd.NA

    pool = template_pool.drop_duplicates("Template_Key").copy()
    pool["Original_FIPS"] = pool["Original_FIPS"].astype(str).str.zfill(5)
    pool["Building_ID_Original"] = pool["Building_ID_Original"].map(normalize_id).astype(object)
    pool["Template_Key"] = pool["Original_FIPS"].astype(str) + "|" + pool["Building_ID_Original"].astype(str)
    pool["Income_Bin"] = pool["Income_Bin"].astype(int)

    info = (
        pool[["Template_Key", "Income_Bin", "Optimal_Strategy", "Original_FIPS", "Building_ID_Original"]]
        .drop_duplicates("Template_Key")
        .rename(columns={"Income_Bin": "_template_income_bin", "Optimal_Strategy": "_template_strategy"})
    )
    df = df.merge(info, on="Template_Key", how="left")
    direct_ok = df["_template_income_bin"].notna() & (
        df["Real_Income_Bin"].astype("Int64") == df["_template_income_bin"].astype("Int64")
    )

    df.loc[direct_ok, "mapping_level"] = "county_direct"
    df.loc[direct_ok, "mapped_template_income_bin"] = df.loc[direct_ok, "_template_income_bin"]
    df.loc[direct_ok, "mapped_template_strategy"] = df.loc[direct_ok, "_template_strategy"]

    unresolved_idx = df.index[df["mapping_level"] == "unresolved"].tolist()
    if unresolved_idx:
        for real_bin in sorted(df.loc[unresolved_idx, "Real_Income_Bin"].dropna().unique()):
            idx = df.index[(df["mapping_level"] == "unresolved") & (df["Real_Income_Bin"].astype(int) == int(real_bin))]
            if len(idx) == 0:
                continue

            cand = pool[(pool["Original_FIPS"] == original_fips) & (pool["Income_Bin"].astype(int) == int(real_bin))].copy()
            level = "county_imputed"
            relaxed = False

            if cand.empty:
                cand = pool[(pool["Original_FIPS"].isin(group_members)) & (pool["Income_Bin"].astype(int) == int(real_bin))].copy()
                level = "group_imputed"

            if cand.empty:
                tmp = pool[pool["Original_FIPS"].isin(group_members)].copy()
                if not tmp.empty:
                    tmp["_bin_dist"] = tmp["Income_Bin"].map(lambda b: _income_bin_distance(b, real_bin))
                    min_dist = tmp["_bin_dist"].min()
                    cand = tmp[tmp["_bin_dist"] == min_dist].drop(columns=["_bin_dist"]).copy()
                level = "group_income_relaxed"
                relaxed = True

            if cand.empty:
                continue

            cand = cand.sort_values(["Original_FIPS", "Income_Bin", "Template_Key"]).reset_index(drop=True)
            take = cand.iloc[np.arange(len(idx)) % len(cand)].reset_index(drop=True)

            df.loc[idx, "matched_bldg_id"] = take["Building_ID_Original"].to_numpy(dtype=object)
            df.loc[idx, "matched_template_fips"] = take["Original_FIPS"].to_numpy(dtype=object)
            df.loc[idx, "Template_Key"] = take["Template_Key"].to_numpy(dtype=object)
            df.loc[idx, "mapping_level"] = level
            df.loc[idx, "imputed_flag"] = True
            df.loc[idx, "income_bin_relaxed_flag"] = relaxed
            df.loc[idx, "mapped_template_income_bin"] = take["Income_Bin"].to_numpy(dtype=int)
            df.loc[idx, "mapped_template_strategy"] = take["Optimal_Strategy"].to_numpy(dtype=object)
            if "sqft" in take.columns and "template_sqft" in df.columns:
                df.loc[idx, "template_sqft"] = take["sqft"].to_numpy(dtype=float)

    df = df.drop(columns=[c for c in ["_template_income_bin", "_template_strategy", "Original_FIPS", "Building_ID_Original"] if c in df.columns])
    return df


def _fixed_mapping_path(output_dir, fips):
    """One canonical real-building -> template mapping per county."""
    return os.path.join(output_dir, f"{fips}_fixed_real_to_template_mapping.parquet")


def _fixed_mapping_report_path(output_dir, fips):
    return os.path.join(output_dir, f"{fips}_fixed_mapping_reference_report.csv")


def _distribution_report_path(output_dir, fips):
    return os.path.join(output_dir, f"{fips}_fixed_mapping_distribution_check.csv")


def _mapping_merge_cols(mapping_df):
    """Columns from the fixed mapping that should be carried into each final real-building result."""
    cols = [
        "BUILD_ID",
        "county_fips",
        "matched_bldg_id",
        "matched_template_fips",
        "Template_Key",
        "sqft",
        "template_sqft",
        "bldg_assigned_income",
        "Real_Income_Bin",
        "Real_Income_Bin_Label",
        "imputed_flag",
        "mapping_level",
        "mapping_feature_imputed_flag",
        "income_bin_relaxed_flag",
        "mapped_template_income_bin",
        "mapped_template_strategy",
        "mapping_reference_ssp",
        "mapping_reference_mode",
    ]
    return [c for c in cols if c in mapping_df.columns]


def _build_fixed_mapping_from_reference(
    county_map,
    reference_template_pool,
    original_fips,
    group_members,
    feature_cols=DEFAULT_FEATURE_COLS,
    reuse_factor=3,
    k_neighbors=20,
    reference_ssp="ssp126",
    reference_mode="Equity_DP",
):
    """
    Build the canonical mapping once using the reference optimized template pool.

    The current mapping principles are preserved:
    - direct matches are accepted only when the template exists and the income bin matches;
    - unresolved buildings are reassigned by county + same income bin first;
    - group + same income bin is used as fallback;
    - group + nearest income bin is used as the last fallback and explicitly flagged;
    - strategy-count targets and template-capacity limits are computed from the reference pool.
    """
    try:
        fixed = map_county_for_mode(
            county_map=county_map,
            template_pool=reference_template_pool,
            original_fips=original_fips,
            group_members=group_members,
            feature_cols=feature_cols,
            reuse_factor=reuse_factor,
            k_neighbors=k_neighbors,
            timeout_seconds=FIXED_MAPPING_TIMEOUT_SECONDS,
        )
    except TimeoutError as e:
        print(f"  [!] {e}")
        print("  [!] Use SIMPLE_FAST fallback mapping: ignore NN/capacity/strategy-balance constraints.")
        fixed = simple_fast_map_county_for_mode(
            county_map=county_map,
            template_pool=reference_template_pool,
            original_fips=original_fips,
            group_members=group_members,
            feature_cols=feature_cols,
        )

    unresolved = int((fixed["mapping_level"] == "unresolved").sum())
    if unresolved > 0:
        fixed = fixed[fixed["mapping_level"] != "unresolved"].copy()

    fixed["matched_bldg_id"] = fixed["matched_bldg_id"].map(normalize_id).astype("string")
    fixed["matched_template_fips"] = fixed["matched_template_fips"].astype(str).str.zfill(5)
    fixed["Template_Key"] = (
        fixed["matched_template_fips"].astype(str)
        + "|"
        + fixed["matched_bldg_id"].astype(str)
    )
    fixed["mapping_reference_ssp"] = reference_ssp
    fixed["mapping_reference_mode"] = reference_mode
    fixed["unresolved_excluded_in_fixed_mapping"] = unresolved
    return fixed


def _build_strategy_distribution_check(fixed_mapping, reference_template_pool, fips, group_members):
    """
    Compare the reference template optimal-strategy distribution with the mapped real-building
    optimal-strategy distribution under the fixed reference mapping.
    """
    records = []
    fips = str(fips).zfill(5)
    group_members = [str(x).zfill(5) for x in group_members]

    for real_bin in sorted(fixed_mapping["Real_Income_Bin"].dropna().unique()):
        real_bin = int(real_bin)
        real_sub = fixed_mapping[fixed_mapping["Real_Income_Bin"].astype(int) == real_bin].copy()
        n_real = len(real_sub)
        if n_real == 0:
            continue

        ref_pool, _ = _build_candidate_pool(
            reference_template_pool, fips, real_bin, group_members,
            allow_group=False, allow_adjacent_income=False
        )
        if ref_pool.empty:
            ref_pool, _ = _build_candidate_pool(
                reference_template_pool, fips, real_bin, group_members,
                allow_group=True, allow_adjacent_income=False
            )
        if ref_pool.empty:
            ref_pool, _ = _build_candidate_pool(
                reference_template_pool, fips, real_bin, group_members,
                allow_group=True, allow_adjacent_income=True
            )
        if ref_pool.empty:
            continue

        target_counts = _target_strategy_counts(ref_pool, n_real)
        actual_counts = real_sub["mapped_template_strategy"].value_counts(dropna=False).to_dict()
        strategies = sorted(set(target_counts.keys()) | set(actual_counts.keys()), key=lambda x: str(x))

        for strategy in strategies:
            t = int(target_counts.get(strategy, 0))
            a = int(actual_counts.get(strategy, 0))
            records.append({
                "fips": fips,
                "income_bin": real_bin,
                "income_bin_label": INCOME_BIN_LABELS.get(real_bin, str(real_bin)),
                "strategy": strategy,
                "target_count_from_reference_templates": t,
                "actual_count_in_fixed_real_mapping": a,
                "target_share_from_reference_templates": t / n_real if n_real else np.nan,
                "actual_share_in_fixed_real_mapping": a / n_real if n_real else np.nan,
                "abs_count_error": abs(a - t),
                "signed_count_error": a - t,
            })
    return pd.DataFrame(records)


def _write_fixed_mapping_reports(output_dir, fips, fixed_mapping, reference_template_pool, group_members):
    real_buildings = int(fixed_mapping["BUILD_ID"].nunique()) if "BUILD_ID" in fixed_mapping.columns else len(fixed_mapping)
    used_templates = int(fixed_mapping["Template_Key"].nunique()) if "Template_Key" in fixed_mapping.columns else 0
    available_templates_county = int(reference_template_pool[reference_template_pool["Original_FIPS"] == str(fips).zfill(5)]["Template_Key"].nunique())
    available_templates_group = int(reference_template_pool["Template_Key"].nunique())

    report = pd.DataFrame([{
        "fips": str(fips).zfill(5),
        "reference_ssp": "ssp126",
        "reference_mode": "Equity_DP",
        "mapped_real_buildings": real_buildings,
        "used_templates": used_templates,
        "available_templates_county": available_templates_county,
        "available_templates_group": available_templates_group,
        "template_use_ratio_county": used_templates / available_templates_county if available_templates_county else np.nan,
        "template_use_ratio_group": used_templates / available_templates_group if available_templates_group else np.nan,
        "avg_template_reuse": real_buildings / used_templates if used_templates else np.nan,
        "county_direct": int((fixed_mapping["mapping_level"] == "county_direct").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "county_imputed": int((fixed_mapping["mapping_level"] == "county_imputed").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "group_imputed": int((fixed_mapping["mapping_level"] == "group_imputed").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "group_income_relaxed": int((fixed_mapping["mapping_level"] == "group_income_relaxed").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "income_relaxed": int(fixed_mapping["income_bin_relaxed_flag"].sum()) if "income_bin_relaxed_flag" in fixed_mapping.columns else 0,
        "unresolved_excluded": int(fixed_mapping.get("unresolved_excluded_in_fixed_mapping", pd.Series([0])).iloc[0]),
    }])
    report.to_csv(_fixed_mapping_report_path(output_dir, fips), index=False)

    # Print template utilization summary for the fixed real-building -> template mapping.
    # This does not change any mapping/output logic; it only surfaces key diagnostics.
    county_ratio = report.loc[0, "template_use_ratio_county"]
    group_ratio = report.loc[0, "template_use_ratio_group"]
    county_ratio_txt = f"{county_ratio:.2%}" if pd.notna(county_ratio) else "NA"
    group_ratio_txt = f"{group_ratio:.2%}" if pd.notna(group_ratio) else "NA"

    print(
        f"  Fixed mapping template utilization | "
        f"county={county_ratio_txt} "
        f"({used_templates}/{available_templates_county}); "
        f"group={group_ratio_txt} "
        f"({used_templates}/{available_templates_group}); "
        f"avg reuse={report.loc[0, 'avg_template_reuse']:.1f}; "
        f"direct={report.loc[0, 'county_direct']}, "
        f"county_imputed={report.loc[0, 'county_imputed']}, "
        f"group_imputed={report.loc[0, 'group_imputed']}, "
        f"income_relaxed={report.loc[0, 'group_income_relaxed']}"
    )

    reuse_counts = fixed_mapping["Template_Key"].value_counts() if "Template_Key" in fixed_mapping.columns else pd.Series(dtype=int)
    if not reuse_counts.empty:
        print(
            f"  Template reuse summary | "
            f"min={int(reuse_counts.min())}, "
            f"median={reuse_counts.median():.1f}, "
            f"mean={reuse_counts.mean():.1f}, "
            f"max={int(reuse_counts.max())}"
        )

    dist = _build_strategy_distribution_check(fixed_mapping, reference_template_pool, fips, group_members)
    dist.to_csv(_distribution_report_path(output_dir, fips), index=False)


def _prepare_mode_template_for_fixed_merge(mode_opt, feature_cols=DEFAULT_FEATURE_COLS):
    """Prepare one mode's optimized template result for merging with the fixed mapping."""
    mode_opt_clean, _ = _prepare_template_options(mode_opt, feature_cols=feature_cols)
    return mode_opt_clean


def _write_table_chunked(df, output_path, compression="brotli"):
    """Append one pandas dataframe chunk to a parquet file through PyArrow writer."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    return table


def _merge_fixed_mapping_with_mode_result(
    fixed_mapping,
    mode_opt_clean,
    mode,
    ssp,
    fips,
    output_dir,
    optimizer,
    feature_cols=DEFAULT_FEATURE_COLS,
    decades=DEFAULT_DECADES,
    merge_chunk_size=50000,
):
    """
    Merge the fixed real->template mapping with one SSP/mode template-optimization result.

    TX-safe version:
    - validates Template_Key uniqueness on the template side;
    - writes output parquet chunk by chunk, instead of materializing a huge merged dataframe;
    - prints chunk-level progress so "not moving" can be diagnosed;
    - returns None for the dataframe to avoid keeping it in memory; stats are still returned.
    """
    merge_cols = _mapping_merge_cols(fixed_mapping)

    real_side_cols = set(merge_cols) - {"Template_Key"}
    redundant_cols = {
        "sqft", "FIPS", "bldg_assigned_income", "Real_Income_Bin", "Real_Income_Bin_Label",
        "matched_bldg_id", "matched_template_fips", "mapped_template_strategy",
        "mapped_template_income_bin", "mapping_level", "imputed_flag",
        "mapping_feature_imputed_flag", "income_bin_relaxed_flag",
    }
    cols_to_drop = sorted((real_side_cols | redundant_cols) & set(mode_opt_clean.columns))
    mode_opt_for_merge = mode_opt_clean.drop(columns=cols_to_drop)

    if "Template_Key" not in fixed_mapping.columns or "Template_Key" not in mode_opt_for_merge.columns:
        raise ValueError("Both fixed_mapping and mode_opt_clean must contain Template_Key.")

    dup_templates = int(mode_opt_for_merge["Template_Key"].duplicated().sum())
    if dup_templates > 0:
        print(f"  [!] mode={mode}: duplicated Template_Key in mode result = {dup_templates}; keep first to avoid row explosion.")
        mode_opt_for_merge = mode_opt_for_merge.drop_duplicates("Template_Key", keep="first").copy()

    fixed_keys = fixed_mapping["Template_Key"].dropna().astype(str)
    mode_keys = set(mode_opt_for_merge["Template_Key"].dropna().astype(str).unique())
    fixed_templates = set(fixed_keys.unique())
    missing_fixed_templates = len(fixed_templates - mode_keys)

    safe_mode = str(mode).replace("/", "_").replace("\\", "_").replace(" ", "_")
    mode_out_path = os.path.join(output_dir, f"{fips}_{ssp}_{safe_mode}_optimal_pathway_real.parquet")
    if os.path.exists(mode_out_path):
        os.remove(mode_out_path)

    real_buildings = int(fixed_mapping["BUILD_ID"].nunique()) if "BUILD_ID" in fixed_mapping.columns else len(fixed_mapping)
    total_rows = len(fixed_mapping)
    mapped_building_ids = set()
    used_template_keys = set()
    writer = None
    written_rows = 0

    # Small right table is indexed once; each chunk only merges its own real rows.
    mode_opt_for_merge = mode_opt_for_merge.copy()
    mode_opt_for_merge["Template_Key"] = mode_opt_for_merge["Template_Key"].astype(str)

    print(
        f"  Start chunked real merge: FIPS {fips} | {ssp} | {mode} | "
        f"real_rows={total_rows:,}, mode_templates={len(mode_opt_for_merge):,}, "
        f"missing_fixed_templates={missing_fixed_templates:,}"
    )

    for start in range(0, total_rows, int(merge_chunk_size)):
        end = min(start + int(merge_chunk_size), total_rows)
        left = fixed_mapping.iloc[start:end][merge_cols].copy()
        left["Template_Key"] = left["Template_Key"].astype(str)

        merged = pd.merge(left, mode_opt_for_merge, on="Template_Key", how="inner", validate="many_to_one")
        if merged.empty:
            print(f"    chunk {start//int(merge_chunk_size):05d}: rows {start:,}-{end:,}, merged=0")
            continue

        merged["raw_scale_factor"] = merged["sqft"] / merged["template_sqft"].replace(0, 1)
        merged["eff_scale_factor"] = merged["raw_scale_factor"].clip(lower=0.5, upper=1.3)

        extensive_cols = []
        for d in decades:
            extensive_cols.extend([
                f"Energy_cost_{d}",
                f"Install_cost_{d}",
                f"Annualized_CAPEX_{d}",
                f"Carbon(kg)_{d}",
            ])
        if "Total_CAPEX_for_annualization" in merged.columns:
            extensive_cols.append("Total_CAPEX_for_annualization")

        for col in extensive_cols:
            if col in merged.columns:
                merged[col] = merged[col] * merged["eff_scale_factor"]

        if "bldg_assigned_income" not in merged.columns:
            if "bldg_assigned_income_x" in merged.columns:
                merged["bldg_assigned_income"] = merged["bldg_assigned_income_x"]
            elif "bldg_assigned_income_y" in merged.columns:
                merged["bldg_assigned_income"] = merged["bldg_assigned_income_y"]
            else:
                merged["bldg_assigned_income"] = np.nan

        merged["Income"] = pd.to_numeric(merged["bldg_assigned_income"], errors="coerce")
        inc_med = safe_median(merged["Income"], fallback=50000.0)
        merged["real_income_imputed_flag"] = merged["Income"].isna()
        merged["Income"] = merged["Income"].fillna(inc_med)
        merged["Income_Bin"] = merged["Income"].apply(assign_income_bin).astype(int)
        merged["Income_Bin_Label"] = merged["Income_Bin"].map(INCOME_BIN_LABELS)

        for d in decades:
            merged[f"Paid_CAPEX_{d}"] = optimizer.get_paid_capex(merged, d, mode)
            merged[f"Paid_Energy_Cost_{d}"] = optimizer.get_paid_energy_cost(merged, d, mode)
            merged[f"Annual_Total_Cost_{d}"] = optimizer.get_resident_total_cost(merged, d, mode)

        table = pa.Table.from_pandas(merged, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(mode_out_path, table.schema, compression="brotli")
        writer.write_table(table)

        written_rows += len(merged)
        if "BUILD_ID" in merged.columns:
            mapped_building_ids.update(merged["BUILD_ID"].dropna().astype(str).unique().tolist())
        used_template_keys.update(merged["Template_Key"].dropna().astype(str).unique().tolist())

        print(
            f"    chunk {start//int(merge_chunk_size):05d}: "
            f"rows {start:,}-{end:,}, merged={len(merged):,}, total_written={written_rows:,}"
        )

        del left, merged, table
        gc.collect()

    if writer is not None:
        writer.close()

    if written_rows == 0:
        return None, {
            "fips": fips,
            "ssp": ssp,
            "mode": mode,
            "real_buildings": real_buildings,
            "mapped_buildings": 0,
            "real_coverage": 0.0,
            "missing_fixed_templates_in_mode": int(missing_fixed_templates),
            "output_path": mode_out_path,
        }

    mapped_buildings = len(mapped_building_ids) if mapped_building_ids else written_rows
    used_templates = len(used_template_keys)

    stats = {
        "fips": fips,
        "ssp": ssp,
        "mode": mode,
        "real_buildings": real_buildings,
        "mapped_buildings": mapped_buildings,
        "real_coverage": mapped_buildings / real_buildings if real_buildings else 0,
        "used_templates": used_templates,
        "fixed_templates": len(fixed_templates),
        "missing_fixed_templates_in_mode": missing_fixed_templates,
        "avg_template_reuse": mapped_buildings / used_templates if used_templates else np.nan,
        "county_direct": int((fixed_mapping["mapping_level"] == "county_direct").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "county_imputed": int((fixed_mapping["mapping_level"] == "county_imputed").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "group_imputed": int((fixed_mapping["mapping_level"] == "group_imputed").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "group_income_relaxed": int((fixed_mapping["mapping_level"] == "group_income_relaxed").sum()) if "mapping_level" in fixed_mapping.columns else 0,
        "income_relaxed": int(fixed_mapping["income_bin_relaxed_flag"].sum()) if "income_bin_relaxed_flag" in fixed_mapping.columns else 0,
        "output_path": mode_out_path,
    }
    return None, stats


def map_group_results_to_original_counties(
    final_combined_df,
    state_map,
    group_members,
    output_root,
    ssp,
    optimizer,
    feature_cols=DEFAULT_FEATURE_COLS,
    decades=DEFAULT_DECADES,
    reuse_factor=3,
    k_neighbors=10,
    reference_ssp="ssp126",
    reference_mode="Equity_DP",
    strict_fixed_mapping=True,
    force_rebuild_mapping=True,
):
    """
    Map optimized group templates back to real buildings and output one file per original county.

    New fixed-mapping logic:
    1) For each county, build the real-building -> template-building mapping only once.
    2) The mapping is created using the reference result: ssp126 + Equity_DP.
    3) The reference mapping preserves the previous income-bin, county/group fallback,
       strategy-distribution, and template-capacity constraints.
    4) All other SSPs and modes reuse the saved fixed mapping and only merge their own
       template-level optimized results onto the fixed Template_Key.

    Requirement: in the main workflow, run ssp126 before ssp245/ssp585 so the fixed
    mapping is created before later SSPs are mapped. If strict_fixed_mapping=True and
    the fixed mapping is unavailable, later SSPs/modes are skipped rather than building
    an inconsistent fallback mapping. If force_rebuild_mapping=True, the reference
    SSP/mode always overwrites the fixed mapping and its diagnostic reports.
    """
    group_members = [str(f).zfill(5) for f in group_members]
    state_map_clean = _prepare_state_map(state_map, group_members, feature_cols=feature_cols)
    group_map = state_map_clean[state_map_clean["county_fips"].isin(group_members)].copy()

    df_opt = final_combined_df.copy()

    if "Building_ID_Original" not in df_opt.columns:
        df_opt["Building_ID_Original"] = df_opt["Building_ID"].map(extract_original_building_id)
    else:
        df_opt["Building_ID_Original"] = df_opt["Building_ID_Original"].map(extract_original_building_id)

    df_opt["Original_FIPS"] = df_opt["Original_FIPS"].astype(str).str.zfill(5)
    df_opt["Building_ID_Original"] = df_opt["Building_ID_Original"].map(normalize_id).astype("string")
    df_opt["Template_Key"] = df_opt["Original_FIPS"].astype(str) + "|" + df_opt["Building_ID_Original"].astype(str)

    mode_list = df_opt["Optimization_Mode"].dropna().unique().tolist()

    for fips in group_members:
        print(f"\n>>> Fixed mapping with output by original county: FIPS {fips} | {ssp}")
        county_map = group_map[group_map["county_fips"] == fips].copy()
        if county_map.empty:
            print(f"  [!] FIPS {fips}: Real-building mapping table is empty,Skipping.")
            continue

        output_dir = os.path.join(output_root, f"FIPS_{fips}")
        os.makedirs(output_dir, exist_ok=True)
        fixed_path = _fixed_mapping_path(output_dir, fips)

        fixed_mapping = None

        # In the reference SSP/mode, always rebuild and overwrite the canonical mapping
        # when requested. This prevents accidentally reusing old mapping files from
        # previous runs. Later SSPs still reuse this newly rebuilt mapping.
        if force_rebuild_mapping and str(ssp) == reference_ssp:
            for old_path in [fixed_path, _fixed_mapping_report_path(output_dir, fips), _distribution_report_path(output_dir, fips)]:
                if os.path.exists(old_path):
                    os.remove(old_path)

        # Build and save the canonical mapping only from ssp126 + Equity_DP.
        if str(ssp) == reference_ssp and reference_mode in mode_list:
            ref_opt = df_opt[df_opt["Optimization_Mode"] == reference_mode].copy()
            _, ref_template_pool = _prepare_template_options(ref_opt, feature_cols=feature_cols)
            ref_template_pool = ref_template_pool[ref_template_pool["Original_FIPS"].isin(group_members)].copy()

            if ref_template_pool.empty:
                print(f"  [!] FIPS {fips}: {reference_ssp}/{reference_mode} Template pool is empty; fixed mapping cannot be built.")
                if strict_fixed_mapping:
                    continue
            else:
                fixed_mapping = _build_fixed_mapping_from_reference(
                    county_map=county_map,
                    reference_template_pool=ref_template_pool,
                    original_fips=fips,
                    group_members=group_members,
                    feature_cols=feature_cols,
                    reuse_factor=reuse_factor,
                    k_neighbors=k_neighbors,
                    reference_ssp=reference_ssp,
                    reference_mode=reference_mode,
                )
                if fixed_mapping.empty:
                    print(f"  [!] FIPS {fips}: Fixed mapping is empty,Skipping.")
                    continue

                fixed_mapping.to_parquet(fixed_path, engine="pyarrow", compression="brotli", index=False)
                _write_fixed_mapping_reports(output_dir, fips, fixed_mapping, ref_template_pool, group_members)
                print(f"  >>> Fixed mapping created and saved: {fixed_path}")

        # For non-reference SSPs, or if already created, load the saved fixed mapping.
        if fixed_mapping is None:
            if os.path.exists(fixed_path):
                fixed_mapping = pd.read_parquet(fixed_path)
                print(f"  >>> Fixed mapping loaded: {fixed_path}")
            else:
                msg = (
                    f"  [!] FIPS {fips}: Fixed mapping not found {fixed_path}."
                    f"Run first {reference_ssp}/{reference_mode}."
                )
                print(msg)
                if strict_fixed_mapping:
                    continue
                # Non-strict fallback: use the first available mode to avoid crash, but this is not recommended.
                fallback_mode = mode_list[0] if mode_list else None
                if fallback_mode is None:
                    continue
                fallback_opt = df_opt[df_opt["Optimization_Mode"] == fallback_mode].copy()
                _, fallback_pool = _prepare_template_options(fallback_opt, feature_cols=feature_cols)
                fallback_pool = fallback_pool[fallback_pool["Original_FIPS"].isin(group_members)].copy()
                fixed_mapping = _build_fixed_mapping_from_reference(
                    county_map=county_map,
                    reference_template_pool=fallback_pool,
                    original_fips=fips,
                    group_members=group_members,
                    feature_cols=feature_cols,
                    reuse_factor=reuse_factor,
                    k_neighbors=k_neighbors,
                    reference_ssp=str(ssp),
                    reference_mode=fallback_mode,
                )

        if fixed_mapping is None or fixed_mapping.empty:
            print(f"  [!] FIPS {fips}: Fixed mapping is invalid,Skipping.")
            continue

        mapping_rows = []
        wrote_any_mode = False

        for mode in mode_list:
            mode_opt = df_opt[df_opt["Optimization_Mode"] == mode].copy()
            mode_opt_clean = _prepare_mode_template_for_fixed_merge(mode_opt, feature_cols=feature_cols)
            mode_opt_clean = mode_opt_clean[mode_opt_clean["Original_FIPS"].isin(group_members)].copy()

            if mode_opt_clean.empty:
                print(f"  [!] mode={mode}: Template results are empty,Skipping.")
                continue

            merged, stats = _merge_fixed_mapping_with_mode_result(
                fixed_mapping=fixed_mapping,
                mode_opt_clean=mode_opt_clean,
                mode=mode,
                ssp=ssp,
                fips=fips,
                output_dir=output_dir,
                optimizer=optimizer,
                feature_cols=feature_cols,
                decades=decades,
            )

            if stats.get("mapped_buildings", 0) == 0:
                print(f"  [!] FIPS {fips} | {ssp} | {mode}: Fixed-mapping merge is empty.")
                mapping_rows.append(stats)
                continue

            wrote_any_mode = True
            mapping_rows.append(stats)

            print(
                f"  Finish fixed real mapping: {mode} | "
                f"real-building coverage {stats['real_coverage']:.2%} "
                f"({stats['mapped_buildings']}/{stats['real_buildings']}); "
                f"fixed templates {stats['fixed_templates']}; "
                f"fixed templates missing in current mode {stats['missing_fixed_templates_in_mode']}; "
                f"mean reuse {stats['avg_template_reuse']:.1f}; "
                f"output {stats.get('output_path', '')}"
            )

            gc.collect()

        if not wrote_any_mode:
            print(f"  [!] FIPS {fips}: No real-building results were generated.")
            continue

        mapping_report = pd.DataFrame(mapping_rows)
        mapping_report.to_csv(os.path.join(output_dir, f"{fips}_{ssp}_mapping_report.csv"), index=False)

        print(f"--- FIPS {fips} | {ssp} fixed mapping and per-mode outputs completed: {output_dir}")
