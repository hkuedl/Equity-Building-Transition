
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import os
import sys
import glob
import re
import time
import pyarrow.parquet as pq
from collections import defaultdict
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from annualized_retrofit_cost import add_annualized_capex_columns
try:
    from annualized_retrofit_cost import capex_event_cost_for_decade
except ImportError:
    capex_event_cost_for_decade = None


def assign_income_bin(income):
    """
    Assign household income to fixed national income brackets.
    Income bins:
    0: <$25k
    25000: $25k-$50k
    50000: $50k-$75k
    75000: $75k-$100k
    100000: $100k-$150k
    150000: $150k-$200k
    200000: >$200k
    """
    if pd.isna(income):
        return np.nan

    if income < 25000:
        return 0
    elif income < 50000:
        return 25000
    elif income < 75000:
        return 50000
    elif income < 100000:
        return 75000
    elif income < 150000:
        return 100000
    elif income < 200000:
        return 150000
    else:
        return 200000

INCOME_BIN_LABELS = {
    0: '<$25k',
    25000: '$25k-$50k',
    50000: '$50k-$75k',
    75000: '$75k-$100k',
    100000: '$100k-$150k',
    150000: '$150k-$200k',
    200000: '>$200k'
}


INCOME_RETROFIT_SUBSIDY_RATE = {
    0: 0.60,
    25000: 0.40,
    50000: 0.15,
    75000: 0.10,
    100000: 0.00,
    150000: 0.00,
    200000: 0.00,
}

UPFRONT_CAPEX_BURDEN_YEARS = 5.0  # upfront CAPEX burden is evaluated as CAPEX / 5-year income
UPFRONT_CAPEX_BURDEN_MODES = {'Equity_DP', 'Equity_RSP', 'Equity_RSBAP'}

INCOME_BILL_AID_RATE = {
    0: 0.40,
    25000: 0.20,
    50000: 0.10,
    75000: 0.05,
    100000: 0.00,
    150000: 0.00,
    200000: 0.00,
}


ONE_SIZE_STRATEGY_BY_SSP = {
    'ssp126': 'up16',
    'ssp245': 'up16',
    'ssp585': 'up03',
}

def get_resident_capex_share(income_bin):
    """Return the resident share of upfront CAPEX for RSP and RSBAP."""
    if pd.isna(income_bin):
        return 1.0
    return 1.0 - INCOME_RETROFIT_SUBSIDY_RATE.get(int(income_bin), 0.0)

def get_resident_energy_share(income_bin):
    """Return the resident share of energy costs under RSBAP."""
    if pd.isna(income_bin):
        return 1.0
    return 1.0 - INCOME_BILL_AID_RATE.get(int(income_bin), 0.0)


class JusticePathwayOptimizer:
    def __init__(self, replacement_cost_factor, state_name, base_dir, fips, ssp, active_decades=['2020s', '2030s', '2040s', '2050s'], use_all_dir=False):
        self.base_dir = base_dir
        self.fips = fips
        self.ssp = ssp
        self.active_decades = active_decades

        self.baseline_strategy = 'up04' if ssp == 'ssp585' else 'up17'
        state_dir_name = f"{state_name}_all" if use_all_dir else state_name
        self.results_path = os.path.join(base_dir, f"#R2/{state_dir_name}/FIPS_{fips}")
        self.replacement_cost_factor = replacement_cost_factor

    def load_strategy_data(self):
        """Load and preprocess data using the metric-script column conventions"""
        pattern = os.path.join(self.results_path, f"{self.ssp}_up*.parquet")
        strategy_files = glob.glob(pattern)

        if not strategy_files:
            print(f"  [!] No files found in {self.results_path} ")
            return None

        all_options = []
        for file_path in strategy_files:

            strategy_name = re.search(r'up\d+', os.path.basename(file_path)).group()
            df = pd.read_parquet(file_path)
            df['Strategy'] = strategy_name


            df['Total_Carbon'] = 0.0
            df['Avg_Comfort'] = 0.0
            for d in self.active_decades:

                carbon_col = f'Carbon(kg)_{d}'
                comfort_col1 = f'Comfort_Extreme_{d}'
                comfort_col2 = f'Comfort_Outage_{d}'

                df['Total_Carbon'] += df[carbon_col]
                df[f'Total_Comfort_{d}'] = df[comfort_col1] + df[comfort_col2]
                df['Avg_Comfort'] += df[f'Total_Comfort_{d}'] / len(self.active_decades)

            all_options.append(df)

        full_df = pd.concat(all_options, ignore_index=True)

        full_df['Income_Bin'] = full_df['Income'].apply(assign_income_bin)
        full_df['Income_Bin_Label'] = full_df['Income_Bin'].map(INCOME_BIN_LABELS)


        full_df = full_df.dropna(subset=['Income_Bin']).copy()


        full_df['Income_Bin'] = full_df['Income_Bin'].astype(int)
        return full_df

    def _build_valid_data(self, df, comfort_limit):
        """Build a feasible strategy pool for each building.

        Strategies meeting the comfort limit in every decade are retained.
        If none qualify, the strategy with the lowest average discomfort is
        retained as a deterministic fallback.
        """
        valid_rows = []
        comfort_cols = [f'Total_Comfort_{d}' for d in self.active_decades]

        for _, group in df.groupby('Building_ID'):
            mask = (group[comfort_cols] <= comfort_limit).all(axis=1)
            qualified = group[mask]
            if not qualified.empty:
                valid_rows.append(qualified)
            else:
                valid_rows.append(group.sort_values('Avg_Comfort').head(1))

        return pd.concat(valid_rows, ignore_index=True)

    def solve_single_building_mode(self, df, mode):
        """Select the best feasible strategy independently for each building.

        Carbon mode minimizes total carbon, while resilience mode minimizes
        average discomfort.
        """
        comfort_limit = 40
        valid_rows = []
        comfort_cols = [f'Total_Comfort_{d}' for d in self.active_decades]

        for b_id, group in df.groupby('Building_ID'):
            mask = (group[comfort_cols] <= comfort_limit).all(axis=1)
            qualified = group[mask]

            if qualified.empty:
                qualified = group.sort_values('Avg_Comfort').head(1)

            if mode == 'carbon':
                best = qualified.sort_values('Total_Carbon').head(1)
            elif mode == 'resilience':
                best = qualified.sort_values('Avg_Comfort').head(1)
            else:
                raise ValueError(f"Unsupported single-building mode: {mode}")

            valid_rows.append(best)

        optimal_pathway_df = pd.concat(valid_rows, ignore_index=True)
        optimal_pathway_df['Optimal_Strategy'] = optimal_pathway_df['Strategy']
        optimal_pathway_df = optimal_pathway_df.drop(columns=['Strategy'])

        return optimal_pathway_df

    def _decade_start_offset(self, decade):
        decade_start_year_offset = {
            "2020s": 0,
            "2030s": 10,
            "2040s": 20,
            "2050s": 30,
        }
        if str(decade) in decade_start_year_offset:
            return decade_start_year_offset[str(decade)]
        default_decade_order = {d: i for i, d in enumerate(self.active_decades)}
        return 10 * default_decade_order[str(decade)]

    def _annual_cost_pv_factor(self, decade, discount_rate=0.02):
        """Present-value factor for one representative annual cost in a decade."""
        start = self._decade_start_offset(decade)
        decade_years = 10
        if float(discount_rate) == 0.0:
            return float(decade_years)
        return sum(
            1.0 / ((1.0 + float(discount_rate)) ** t)
            for t in range(start + 1, start + decade_years + 1)
        )

    def _decade_lump_sum_pv_factor(self, decade, discount_rate=0.02):
        """Discount a decade-level retrofit/replacement cash cost to 2020.

        RetrofitReplacement_CAPEX_{decade} is a decade-level cash-flow column.
        Here it is discounted at the beginning of that decade: 2020s=t0,
        2030s=t10, 2040s=t20, 2050s=t30.
        """
        start = self._decade_start_offset(decade)
        if float(discount_rate) == 0.0:
            return 1.0
        return 1.0 / ((1.0 + float(discount_rate)) ** start)

    def _get_capex_event_cost(self, df, decade):
        """Return decade-level cash CAPEX: initial retrofit in 2020s plus later replacements."""
        col = f"RetrofitReplacement_CAPEX_{decade}"
        if col not in df.columns:
            raise KeyError(
                f"Missing {col}. Please use the updated annualized_retrofit_cost.py, "
                "where add_annualized_capex_columns also creates RetrofitReplacement_CAPEX_* columns."
            )
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    def _cost_mode_objective_series(self, df, discount_rate=0.02):
        """Cost-mode objective: NPV energy cost + NPV retrofit/replacement cash CAPEX."""
        energy_npv = pd.Series(0.0, index=df.index)
        capex_npv = pd.Series(0.0, index=df.index)
        for d in self.active_decades:
            energy_col = f"Energy_cost_{d}"
            if energy_col not in df.columns:
                raise KeyError(f"Missing {energy_col}")
            energy_npv += pd.to_numeric(df[energy_col], errors="coerce").fillna(0.0) * self._annual_cost_pv_factor(d, discount_rate)
            capex_npv += self._get_capex_event_cost(df, d) * self._decade_lump_sum_pv_factor(d, discount_rate)
        return energy_npv + capex_npv
    def solve_cost_mode(self, df):
        """
        Cost-optimal mode.

        For each template building, select the non-baseline strategy with the minimum
        discounted total cost:
            40-year NPV energy cost + discounted retrofit/replacement cash CAPEX.

        Baseline/current strategy is excluded by default:
            ssp126/ssp245: up17
            ssp585:        up04

        If a building has no non-baseline candidate after the comfort filter, the
        original candidate pool is used as a fallback to avoid dropping that building.
        """
        valid_data = self._build_valid_data(df, comfort_limit=40)
        valid_data = self.prepare_cost_columns(valid_data, strategy_col="Strategy")
        valid_data["Cost_Objective_Total"] = self._cost_mode_objective_series(valid_data, discount_rate=0.02)
        valid_data = self._drop_invalid_model_rows(valid_data, mode="cost")

        baseline_strategy = self.baseline_strategy

        # Prefer non-baseline strategies when selecting the least-cost option.
        non_baseline_data = valid_data[valid_data["Strategy"].astype(str) != str(baseline_strategy)].copy()

        selected_parts = []
        fallback_buildings = []

        for b_id, group in valid_data.groupby("Building_ID", sort=False):
            group_non_base = non_baseline_data[non_baseline_data["Building_ID"] == b_id]

            if not group_non_base.empty:
                best_idx = group_non_base["Cost_Objective_Total"].idxmin()
            else:
                # Fallback only when all available candidates are baseline/current.
                best_idx = group["Cost_Objective_Total"].idxmin()
                fallback_buildings.append(b_id)

            selected_parts.append(valid_data.loc[[best_idx]])

        if fallback_buildings:
            print(
                f"  [!] cost mode fallback: {len(fallback_buildings)} buildings have no "
                f"non-baseline candidate after comfort filtering; baseline={baseline_strategy} was used for them."
            )

        optimal_pathway_df = pd.concat(selected_parts, ignore_index=True)
        optimal_pathway_df["Optimal_Strategy"] = optimal_pathway_df["Strategy"]
        optimal_pathway_df = optimal_pathway_df.drop(columns=["Strategy"])

        return optimal_pathway_df
    def prepare_cost_columns(self, df, strategy_col="Strategy", keep_component_columns=False):
        """Add annualized and cash-event retrofit cost columns.

        ``Annualized_CAPEX_*`` accounts for component lives and replacements.
        ``RetrofitReplacement_CAPEX_*`` records cash events by decade.
        """
        out = add_annualized_capex_columns(
            df.copy(),
            ssp=self.ssp,
            strategy_col=strategy_col,
            install_cost_cols=[f'Install_cost_{d}' for d in self.active_decades],
            total_capex_mode="sum",
            decades=self.active_decades,
            r=0.02,
            install_year=2020,
            horizon_end=2060,
            replacement_cost_factor=self.replacement_cost_factor,
            keep_component_columns=keep_component_columns,
        )

        missing_event_cols = [f"RetrofitReplacement_CAPEX_{d}" for d in self.active_decades if f"RetrofitReplacement_CAPEX_{d}" not in out.columns]
        if missing_event_cols:
            if capex_event_cost_for_decade is None:
                raise KeyError(
                    "Missing RetrofitReplacement_CAPEX_* columns and capex_event_cost_for_decade() is unavailable. "
                    "Please replace annualized_retrofit_cost.py with the updated version."
                )
            for d in self.active_decades:
                out[f"RetrofitReplacement_CAPEX_{d}"] = out.apply(
                    lambda row: capex_event_cost_for_decade(
                        total_capex=row["Total_CAPEX_for_annualization"],
                        strategy=row[strategy_col],
                        decade=d,
                        ssp=self.ssp,
                        r=0.02,
                        install_year=2020,
                        horizon_end=2060,
                        replacement_cost_factor=self.replacement_cost_factor,
                    ),
                    axis=1,
                )
            out["Total_RetrofitReplacement_CAPEX"] = out[[f"RetrofitReplacement_CAPEX_{d}" for d in self.active_decades]].sum(axis=1)
        return out

    def get_paid_capex(self, df, decade, mode):
        """Return resident-paid retrofit and replacement costs by mode.

        LIFP uses annualized CAPEX. Other modes use decade cash events, with
        income-based retrofit subsidies applied in RSP and RSBAP.
        """
        if mode == "Equity_LIFP":
            return pd.to_numeric(df[f"Annualized_CAPEX_{decade}"], errors="coerce").fillna(0.0)

        capex_cash = self._get_capex_event_cost(df, decade)

        if mode in ["Equity_RSP", "Equity_RSBAP"]:
            payer_share = df["Income_Bin"].map(lambda x: get_resident_capex_share(x))
            return capex_cash * payer_share.astype(float)

        return capex_cash

    def get_paid_energy_cost(self, df, decade, mode):
        """Return resident-paid energy cost after any RSBAP bill assistance."""
        energy = df[f"Energy_cost_{decade}"]
        if mode != "Equity_RSBAP":
            return energy
        payer_share = df["Income_Bin"].map(lambda x: get_resident_energy_share(x))
        return energy * payer_share

    def get_resident_total_cost(self, df, decade, mode):
        """Return resident-paid CAPEX plus energy cost in dollars.

        This amount is not the normalized energy-burden metric.
        """
        return self.get_paid_capex(df, decade, mode) + self.get_paid_energy_cost(df, decade, mode)

    def get_resident_burden_equivalent_cost(self, df, decade, mode):
        """Return the cost numerator used in burden and equity calculations.

        Upfront cash CAPEX in DP, RSP, and RSBAP is spread across
        ``UPFRONT_CAPEX_BURDEN_YEARS`` for comparison with annual income.
        """
        capex = self.get_paid_capex(df, decade, mode)
        if mode in UPFRONT_CAPEX_BURDEN_MODES:
            capex = capex / UPFRONT_CAPEX_BURDEN_YEARS
        return capex + self.get_paid_energy_cost(df, decade, mode)

    def build_cost_dict(self, df, mode):
        """Build the burden-equivalent cost dictionary consumed by Pyomo."""
        cost_dict = {}

        for d in self.active_decades:
            decade_burden = self.get_resident_burden_equivalent_cost(df, d, mode)

            temp_map = {
                (d, b, s): val
                for (b, s), val in zip(
                    zip(df["Building_ID"], df["Strategy"]),
                    decade_burden
                )
            }
            cost_dict.update(temp_map)

        return cost_dict


    def _safe_positive_scalar(self, value, fallback=1.0, name="scalar"):
        """Return a finite positive scalar for normalization denominators."""
        try:
            value = float(value)
        except Exception:
            value = np.nan
        if not np.isfinite(value) or value <= 0:
            print(f"  [!] {name} is invalid ({value}); use fallback={fallback}.")
            return float(fallback)
        return float(value)

    def _build_safe_bin_total_income(self, bldg_income_map, bldg_bin_map, income_bins, min_income_per_building=1000.0):
        """
        Build income-bin denominators for burden calculation.

        Some counties/groups can contain an income bin whose recorded total income is
        zero or non-finite. Dividing by such a value creates inf/nan coefficients in
        Pyomo/Gurobi and can make Gurobi report the model as unbounded. Here we keep
        the bin but use a small positive denominator fallback proportional to the
        number of buildings in that bin.
        """
        bin_total_income = {}
        for k in income_bins:
            mask = (bldg_bin_map == k)
            n_buildings = int(mask.sum())
            total_income = pd.to_numeric(bldg_income_map[mask], errors="coerce").sum()
            if (not np.isfinite(total_income)) or total_income <= 0:
                fallback = max(float(n_buildings) * float(min_income_per_building), 1.0)
                print(
                    f"  [!] Income denominator fixed | FIPS/group={self.fips}, "
                    f"ssp={self.ssp}, income_bin={k}, n_buildings={n_buildings}, "
                    f"raw_total_income={total_income}, fallback_total_income={fallback}"
                )
                total_income = fallback
            bin_total_income[k] = float(total_income)
        return bin_total_income

    def _drop_invalid_model_rows(self, valid_data, mode):
        """Remove rows with non-finite coefficients before building the Pyomo model."""
        check_cols = ["Income", "Income_Bin", "Total_Carbon", "Avg_Comfort"]
        for d in self.active_decades:
            check_cols.extend([
                f"Carbon(kg)_{d}",
                f"Energy_cost_{d}",
                f"Install_cost_{d}",
                f"Total_Comfort_{d}",
            ])
            if f"Annualized_CAPEX_{d}" in valid_data.columns:
                check_cols.append(f"Annualized_CAPEX_{d}")
            if f"RetrofitReplacement_CAPEX_{d}" in valid_data.columns:
                check_cols.append(f"RetrofitReplacement_CAPEX_{d}")
        if "Total_CAPEX_for_annualization" in valid_data.columns:
            check_cols.append("Total_CAPEX_for_annualization")

        check_cols = [c for c in check_cols if c in valid_data.columns]
        tmp = valid_data[check_cols].replace([np.inf, -np.inf], np.nan)
        invalid_mask = tmp.isna().any(axis=1)

        if invalid_mask.any():
            n_bad_rows = int(invalid_mask.sum())
            n_bad_buildings = int(valid_data.loc[invalid_mask, "Building_ID"].nunique())
            print(
                f"  [!] Drop invalid optimization rows | mode={mode}, "
                f"rows={n_bad_rows}, buildings={n_bad_buildings}"
            )
            valid_data = valid_data.loc[~invalid_mask].copy()

        # After row dropping, keep only buildings that still have at least one candidate.
        valid_buildings = valid_data["Building_ID"].dropna().unique().tolist()
        valid_data = valid_data[valid_data["Building_ID"].isin(valid_buildings)].copy()
        if valid_data.empty:
            raise ValueError(f"No valid optimization rows remain after finite-value filtering: mode={mode}")
        return valid_data

    def compute_baseline_terms(self, df_all, building_ids=None, mode="Equity_LIFP"):
        """
        :
        base_total_carbon, base_total_comfort, base_total_mad

        Robustness fix:
        - guard against zero/non-finite income denominators;
        - guard against nan/inf normalization terms.
        """
        base_df = df_all[df_all["Strategy"] == self.baseline_strategy].copy()

        if building_ids is not None:
            base_df = base_df[base_df["Building_ID"].isin(building_ids)].copy()

        if base_df.empty:
            raise ValueError(
                f"Baseline dataframe is empty for fips={self.fips}, "
                f"ssp={self.ssp}, baseline={self.baseline_strategy}"
            )

        base_df = self.prepare_cost_columns(base_df, strategy_col="Strategy")
        base_df = base_df.replace([np.inf, -np.inf], np.nan)

        # Keep only finite baseline rows for stable normalization.
        required_cols = ["Building_ID", "Income", "Income_Bin", "Total_Carbon", "Avg_Comfort"]
        for d in self.active_decades:
            required_cols.extend([f"Energy_cost_{d}", f"Install_cost_{d}"])
        required_cols = [c for c in required_cols if c in base_df.columns]
        invalid = base_df[required_cols].isna().any(axis=1)
        if invalid.any():
            print(f"  [!] Drop invalid baseline rows for normalization: {int(invalid.sum())}")
            base_df = base_df.loc[~invalid].copy()
        if base_df.empty:
            raise ValueError(
                f"No valid baseline rows remain for fips={self.fips}, "
                f"ssp={self.ssp}, baseline={self.baseline_strategy}"
            )

        base_total_carbon = self._safe_positive_scalar(base_df["Total_Carbon"].sum(), 1.0, "base_total_carbon")
        base_total_comfort = self._safe_positive_scalar(base_df["Avg_Comfort"].sum(), 1.0, "base_total_comfort")

        income_bins = [
            b for b in [0, 25000, 50000, 75000, 100000, 150000, 200000]
            if b in base_df["Income_Bin"].unique()
        ]

        bldg_income_map = base_df.drop_duplicates("Building_ID").set_index("Building_ID")["Income"]
        bldg_bin_map = base_df.drop_duplicates("Building_ID").set_index("Building_ID")["Income_Bin"]
        bin_total_income = self._build_safe_bin_total_income(
            bldg_income_map=bldg_income_map,
            bldg_bin_map=bldg_bin_map,
            income_bins=income_bins,
            min_income_per_building=1000.0,
        )

        base_total_mad = 0.0

        for d in self.active_decades:
            tmp = base_df.copy()
            tmp["_burden_cost"] = self.get_resident_burden_equivalent_cost(tmp, d, mode)
            tmp["_burden_cost"] = pd.to_numeric(tmp["_burden_cost"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

            b_costs = tmp.groupby("Income_Bin")["_burden_cost"].sum()
            b_burdens = b_costs / pd.Series(bin_total_income)
            b_burdens = b_burdens.replace([np.inf, -np.inf], np.nan).dropna()
            if b_burdens.empty:
                continue
            base_total_mad += ((b_burdens - b_burdens.mean()) ** 2).sum()

        base_total_mad = self._safe_positive_scalar(base_total_mad, 0.001, "base_total_mad")
        base_total_mad = max(base_total_mad, 0.001)

        return base_total_carbon, base_total_comfort, base_total_mad, bin_total_income, income_bins

    def evaluate_pathway_terms(self, optimal_pathway_df, df_all, mode):
        """Evaluate normalized equity, carbon, and comfort pathway terms."""
        eval_df = optimal_pathway_df.copy()
        strategy_col = "Optimal_Strategy" if "Optimal_Strategy" in eval_df.columns else "Strategy"

        eval_df = self.prepare_cost_columns(eval_df, strategy_col=strategy_col)

        building_ids = eval_df["Building_ID"].unique().tolist()

        base_total_carbon, base_total_comfort, base_total_mad, bin_total_income, income_bins = (
            self.compute_baseline_terms(
                df_all=df_all,
                building_ids=building_ids,
                mode=mode
            )
        )

        total_dev_sq = 0.0

        for d in self.active_decades:
            tmp = eval_df.copy()
            tmp["_burden_cost"] = self.get_resident_burden_equivalent_cost(tmp, d, mode)

            costs = tmp.groupby("Income_Bin")["_burden_cost"].sum()
            burdens = costs / pd.Series(bin_total_income)
            avg_burden = burdens.mean()

            total_dev_sq += ((burdens - avg_burden) ** 2).sum()

        v_eq = total_dev_sq / base_total_mad
        v_carb = eval_df["Total_Carbon"].sum() / base_total_carbon
        v_comf = eval_df["Avg_Comfort"].sum() / base_total_comfort

        return v_eq, v_carb, v_comf


    def _model_has_complete_selection(self, model):
        """Return True if the model currently contains a usable one-strategy-per-building incumbent."""
        bldg_to_strats = getattr(model, "_bldg_to_strats", None)
        if not bldg_to_strats:
            return False
        for b, strats in bldg_to_strats.items():
            chosen = 0
            for s in strats:
                val = pyo.value(model.x[b, s], exception=False)
                if val is None or not np.isfinite(float(val)):
                    return False
                if float(val) > 0.5:
                    chosen += 1
            if chosen != 1:
                return False
        return True

    def _set_heuristic_feasible_solution(self, model, valid_data, mode, reason=""):
        """
        Build a deterministic fallback solution when Gurobi returns no usable incumbent.
        The candidate pool has already passed the comfort filter, so this gives a practical
        feasible pathway for downstream mapping by selecting the lowest resident-cost option
        for each template building under the current policy mode.
        """
        tmp = valid_data.copy()
        score = pd.Series(0.0, index=tmp.index)
        for d in self.active_decades:
            score_component = self.get_resident_burden_equivalent_cost(tmp, d, mode) if mode in ["Equity_DP", "Equity_LIFP", "Equity_RSP", "Equity_RSBAP"] else self.get_resident_total_cost(tmp, d, mode)
            score = score + pd.to_numeric(score_component, errors="coerce").fillna(np.inf)
        tmp["_fallback_score"] = score + 1e-9 * pd.to_numeric(tmp["Total_Carbon"], errors="coerce").fillna(0.0)
        best_idx = tmp.groupby("Building_ID")["_fallback_score"].idxmin()
        selected = set(zip(tmp.loc[best_idx, "Building_ID"], tmp.loc[best_idx, "Strategy"]))
        for b, s in model.BS:
            model.x[b, s].value = 1.0 if (b, s) in selected else 0.0
        for v in list(model.carbon_slack.values()) + list(model.eb_slack.values()):
            v.value = 0.0
        for v in list(model.bin_burden.values()) + list(model.avg_bin_burden.values()) + list(model.dev.values()):
            v.value = 0.0
        msg = f" because {reason}" if reason else ""
        print(f">>> Gurobi returned no usable incumbent; use deterministic fallback solution{msg}.")

    def _keep_incumbent_or_fallback(self, model, valid_data, mode, reason=""):
        if self._model_has_complete_selection(model):
            msg = f" ({reason})" if reason else ""
            print(f">>> Use best available Gurobi incumbent as final solution{msg}.")
            return "incumbent"
        self._set_heuristic_feasible_solution(model, valid_data, mode, reason=reason)
        return "heuristic"

    @staticmethod
    def _fmt_optional_value(value):
        if value is None:
            return "NA"
        try:
            if not np.isfinite(float(value)):
                return "NA"
            return f"{float(value):.4f}"
        except Exception:
            return "NA"


    def solve_optimization(self, df, county_carbon_budgets, mode, solver='gurobi'):
        eb_limit = 0.5
        valid_data = self._build_valid_data(df, comfort_limit=4)


        valid_data = self.prepare_cost_columns(valid_data, strategy_col="Strategy")


        bldg_income_map = valid_data.drop_duplicates('Building_ID').set_index('Building_ID')['Income']
        bldg_bin_map = valid_data.drop_duplicates('Building_ID').set_index('Building_ID')['Income_Bin']

        income_bins = [b for b in [0, 25000, 50000, 75000, 100000, 150000, 200000] if b in valid_data['Income_Bin'].unique()]

        bin_total_income = self._build_safe_bin_total_income(
            bldg_income_map=bldg_income_map,
            bldg_bin_map=bldg_bin_map,
            income_bins=income_bins,
            min_income_per_building=1000.0,
        )


        base_total_carbon, base_total_comfort, base_total_mad, _, _ = self.compute_baseline_terms(
            df_all=df,
            building_ids=valid_data['Building_ID'].unique().tolist(),
            mode=mode
        )


        model = pyo.ConcreteModel()
        indices = list(zip(valid_data['Building_ID'], valid_data['Strategy']))

        model.BS = pyo.Set(initialize=indices, dimen=2)
        model.Bins = pyo.Set(initialize=income_bins)
        model.Decades = pyo.Set(initialize=self.active_decades)
        model.x = pyo.Var(model.BS, domain=pyo.Binary)
        model.carbon_slack = pyo.Var(model.Decades, domain=pyo.NonNegativeReals)
        model.eb_slack = pyo.Var(model.Bins, domain=pyo.NonNegativeReals)


        carbon_dict = {
            (d, b, s): val
            for d in self.active_decades
            for (b, s), val in valid_data.set_index(['Building_ID', 'Strategy'])[f'Carbon(kg)_{d}'].to_dict().items()
        }

        cost_dict = self.build_cost_dict(valid_data, mode=mode)
        comfort_total_map = valid_data.set_index(['Building_ID', 'Strategy'])['Avg_Comfort'].to_dict()
        carbon_total_map = valid_data.set_index(['Building_ID', 'Strategy'])['Total_Carbon'].to_dict()

        bin_to_bs = defaultdict(list)
        for b, s in indices:
            bin_to_bs[bldg_bin_map[b]].append((b, s))


        model.bin_burden = pyo.Var(model.Bins, model.Decades, domain=pyo.NonNegativeReals)
        model.avg_bin_burden = pyo.Var(model.Decades)
        model.dev = pyo.Var(model.Bins, model.Decades, domain=pyo.NonNegativeReals)

        def bin_burden_rule(m, k, d):
            num = sum(cost_dict[d, b, s] * m.x[b, s] for (b, s) in bin_to_bs[k])
            return m.bin_burden[k, d] == num / bin_total_income[k]

        model.c_bin_burden = pyo.Constraint(model.Bins, model.Decades, rule=bin_burden_rule)

        def avg_burden_rule(m, d):
            return m.avg_bin_burden[d] == sum(m.bin_burden[k, d] for k in m.Bins) / len(income_bins)

        model.c_avg_burden = pyo.Constraint(model.Decades, rule=avg_burden_rule)

        def dev_rule_pos(m, k, d):
            return m.dev[k, d] >= m.bin_burden[k, d] - m.avg_bin_burden[d]

        def dev_rule_neg(m, k, d):
            return m.dev[k, d] >= m.avg_bin_burden[d] - m.bin_burden[k, d]

        model.c_dev_pos = pyo.Constraint(model.Bins, model.Decades, rule=dev_rule_pos)
        model.c_dev_neg = pyo.Constraint(model.Bins, model.Decades, rule=dev_rule_neg)


        def carbon_budget_rule(m, d):
            total_carb_d = sum(carbon_dict[d, b, s] * m.x[b, s] for b, s in m.BS)
            return total_carb_d <= county_carbon_budgets.get(d, 1e18) + m.carbon_slack[d]

        model.c_carb_limit = pyo.Constraint(model.Decades, rule=carbon_budget_rule)


        def bin_total_limit_rule(m, k):
            total_bin_burden = sum(m.bin_burden[k, d] for d in m.Decades) / len(self.active_decades)
            return total_bin_burden <= eb_limit + m.eb_slack[k]

        model.c_eb_limit = pyo.Constraint(model.Bins, rule=bin_total_limit_rule)


        bldg_ids = valid_data['Building_ID'].unique().tolist()
        bldg_to_strats = defaultdict(list)

        for b, s in indices:
            bldg_to_strats[b].append(s)
        model._bldg_to_strats = bldg_to_strats

        def selection_rule(m, b):
            return sum(m.x[b, s] for s in bldg_to_strats[b]) == 1

        model.c_select = pyo.Constraint(bldg_ids, rule=selection_rule)


        base_total_carbon = self._safe_positive_scalar(base_total_carbon, 1.0, "base_total_carbon")
        base_total_comfort = self._safe_positive_scalar(base_total_comfort, 1.0, "base_total_comfort")
        base_total_mad = self._safe_positive_scalar(base_total_mad, 0.001, "base_total_mad")
        base_total_mad = max(base_total_mad, 0.001)

        term_equity = sum(model.dev[k, d] ** 2 for k in model.Bins for d in model.Decades) / base_total_mad
        term_carbon = sum(carbon_total_map[b, s] * model.x[b, s] for b, s in model.BS) / base_total_carbon
        term_comfort = sum(comfort_total_map[b, s] * model.x[b, s] for b, s in model.BS) / base_total_comfort

        PENALTY = 1e6
        penalty_terms_1 = PENALTY * sum(model.carbon_slack[d] for d in model.Decades)
        penalty_terms_2 = PENALTY * sum(model.eb_slack[k] for k in model.Bins)
        penalty_terms = penalty_terms_1 + penalty_terms_2

        alphas = [1.0, 0.0, 0.0]
        if mode in ['Equity_DP', 'Equity_LIFP', 'Equity_RSP', 'Equity_RSBAP']:
            alphas = [1.0, 0.0, 0.0]

        model.obj = pyo.Objective(
            expr=alphas[0] * term_equity + alphas[1] * term_carbon + alphas[2] * term_comfort + penalty_terms,
            sense=pyo.minimize
        )


        opt = pyo.SolverFactory(solver)
        res = None
        term_cond = None
        status = None
        solve_t0 = time.time()

        try:
            # Stage 1: strict solve, at most 2 minutes.
            opt.options.clear()
            opt.options['TimeLimit'] = 120
            res = opt.solve(model, tee=False)
            term_cond = res.solver.termination_condition
            status = res.solver.status
            print(f">>> Gurobi stage 1 status: {status}, termination: {term_cond}")
        except Exception as e:
            self._keep_incumbent_or_fallback(model, valid_data, mode, reason=f"stage 1 error: {e}")
            return model, res

        if term_cond != pyo.TerminationCondition.optimal:
            print(">>> Stage 1 did not prove optimal within 2 min; relax MIPGap and continue.")
            try:
                # Stage 2: relaxed solve. If it reaches 4 minutes or errors, keep incumbent.
                opt.options.clear()
                opt.options['TimeLimit'] = 240
                opt.options['MIPGap'] = 0.05
                opt.options['MIPFocus'] = 1
                opt.options['Heuristics'] = 0.8
                opt.options['Cuts'] = 0
                opt.options['Presolve'] = 2
                res = opt.solve(model, tee=False)
                term_cond = res.solver.termination_condition
                status = res.solver.status
                print(f">>> Gurobi stage 2 status: {status}, termination: {term_cond}")
            except Exception as e:
                self._keep_incumbent_or_fallback(model, valid_data, mode, reason=f"stage 2 error after relaxation: {e}")
                return model, res

        elapsed = time.time() - solve_t0
        if term_cond == pyo.TerminationCondition.optimal:
            print(f">>> Gurobi proved optimal in {elapsed:.1f}s.")
        elif term_cond in [pyo.TerminationCondition.feasible, pyo.TerminationCondition.maxTimeLimit]:
            self._keep_incumbent_or_fallback(model, valid_data, mode, reason=f"termination={term_cond}, elapsed={elapsed:.1f}s")
        else:
            self._keep_incumbent_or_fallback(model, valid_data, mode, reason=f"termination={term_cond}, elapsed={elapsed:.1f}s")

        v_pen_carbon = pyo.value(penalty_terms_1, exception=False)
        v_pen_burden = pyo.value(penalty_terms_2, exception=False)
        print(f">>> Penalty_Carbon: {self._fmt_optional_value(v_pen_carbon)} | Penalty_Burden: {self._fmt_optional_value(v_pen_burden)}")

        return model, res


# 2. Robust multi-state/group workflow


BASE_DIR = os.environ.get(
    "PROJECT_DATA_DIR",
    os.path.join(os.path.dirname(CURRENT_DIR), "data"),
)
SSPS = ['ssp126', 'ssp245', 'ssp585']
DECADES = ['2020s', '2030s', '2040s', '2050s']
MODES = ['baseline', 'one', 'carbon', 'resilience', 'cost', 'Equity_DP', 'Equity_LIFP', 'Equity_RSP', 'Equity_RSBAP']

FEATURE_COLS = ['sqft', 'stories', 'bldg_assigned_income']
FORCE_REBUILD_REAL_MAPPING = False

# Only read columns required by the real-building mapping module.
# This avoids loading the full county/state real-building parquet into memory.
STATE_MAP_REQUIRED_COLS = [
    'BUILD_ID',
    'county_fips',
    'matched_bldg_id',
    'sqft',
    'stories',
    'bldg_assigned_income',
    'building_type',
    'template_sqft',
]


def read_parquet_existing_columns(path, requested_cols):
    """Read only requested columns that exist in a parquet file.

    This is memory-safer than pd.read_parquet(path) for large real-building
    and strategy files.
    """
    pf = pq.ParquetFile(path)
    available = set(pf.schema.names)
    usecols = [c for c in requested_cols if c in available]
    if not usecols:
        raise ValueError(f"No requested columns found in parquet: {path}")
    return pd.read_parquet(path, columns=usecols)


def get_required_strategy_cols(decades=DECADES):
    """Columns required for optimization + template-side mapping features."""
    cols = [
        'Building_ID',
        'Income',
        # Optional template physical features used by robust_real_mapping.py
        'sqft',
        'stories',
    ]
    for d in decades:
        cols.extend([
            f'Carbon(kg)_{d}',
            f'Comfort_Extreme_{d}',
            f'Comfort_Outage_{d}',
            f'Energy_cost_{d}',
            f'Install_cost_{d}',
        ])
    return cols


TARGET_STATE_ABBRS = [
    value.strip().upper()
    for value in os.environ.get("TARGET_STATES", "").split(",")
    if value.strip()
] or None
ALL_STATE_DATASETS = {
    value.strip().upper()
    for value in os.environ.get("ALL_STATE_DATASETS", "").split(",")
    if value.strip()
}

def use_all_dir_for_state(state_abbr):
    """Return whether the state uses the optional ``{state}_all`` input folder."""
    return state_abbr in ALL_STATE_DATASETS

def get_state_r2_root(base_dir, state_abbr):
    state_dir_name = f"{state_abbr}_all" if use_all_dir_for_state(state_abbr) else state_abbr
    return os.path.join(base_dir, f"#R2/{state_dir_name}")

def get_fips_results_path(base_dir, state_abbr, fips):
    return os.path.join(get_state_r2_root(base_dir, state_abbr), f"FIPS_{fips}")


MIN_TEMPLATE_BUILDINGS = 50
MIN_INCOME_BINS = 3


MODERATE_ROW_MISSING_RATE = 0.05
HEAVY_ROW_MISSING_RATE = 0.20
HEAVY_BUILDING_LOSS_RATE = 0.15


COUNTY_ADJACENCY_FILE = None

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


def load_state_fips_from_upgrade0(res_path, state_name, chunksize=1_000_000):
    usecols = ['in.state', 'in.county']
    fips_set = set()
    for chunk in pd.read_csv(res_path, usecols=usecols, chunksize=chunksize, dtype={'in.state': 'string', 'in.county': 'string'}):
        sub = chunk[chunk['in.state'] == state_name].copy()
        if sub.empty:
            continue
        fips = sub['in.county'].str[1:3] + sub['in.county'].str[4:7]
        fips_set.update(fips.dropna().astype(str).str.zfill(5).unique().tolist())
    return sorted(fips_set)


def get_hard_metric_cols(decades=DECADES):
    cols = ['Building_ID', 'Strategy', 'Income']
    for d in decades:
        cols.extend([
            f'Carbon(kg)_{d}',
            f'Comfort_Extreme_{d}',
            f'Comfort_Outage_{d}',
            f'Energy_cost_{d}',
            f'Install_cost_{d}',
        ])
    return cols


def safe_median(series, fallback=np.nan):
    series = pd.to_numeric(series, errors='coerce')
    med = series.median(skipna=True)
    return fallback if pd.isna(med) else med


def load_raw_fips_strategy_data(base_dir, state_name, fips, ssp):
    """Load one row per building-strategy pair without imputing values."""
    results_path = get_fips_results_path(base_dir, state_name, fips)
    pattern = os.path.join(results_path, f"{ssp}_up*.parquet")
    strategy_files = sorted(glob.glob(pattern))
    if not strategy_files:
        print(f"  [!] Parquet file not found: {results_path} | {ssp}")
        return None

    all_options = []
    for file_path in strategy_files:
        m = re.search(r'up\d+', os.path.basename(file_path))
        if not m:
            continue
        strategy_name = m.group()
        # Read only columns needed by optimization and later template mapping.
        df = read_parquet_existing_columns(file_path, get_required_strategy_cols(DECADES))
        df['Strategy'] = strategy_name
        df['Original_FIPS'] = str(fips).zfill(5)
        all_options.append(df)

    if not all_options:
        return None
    return pd.concat(all_options, ignore_index=True)


def clean_strategy_data(raw_df, fips, ssp, state_income_median=None):
    """Drop rows missing core metrics and impute only missing income values."""
    if raw_df is None or raw_df.empty:
        return None, {
            'fips': fips, 'ssp': ssp, 'raw_rows': 0, 'raw_buildings': 0,
            'row_missing_rate': 1.0, 'building_loss_rate': 1.0,
            'effective_buildings': 0, 'income_bins': 0,
            'valid_county': False, 'missing_baseline': True,
            'quality_class': 'D_no_data'
        }

    df = raw_df.copy()
    df['Original_FIPS'] = str(fips).zfill(5)
    baseline_strategy = 'up04' if ssp == 'ssp585' else 'up17'

    raw_rows = len(df)
    raw_buildings = df['Building_ID'].nunique() if 'Building_ID' in df.columns else 0


    required = get_hard_metric_cols(DECADES)
    missing_cols = [c for c in required if c not in df.columns]
    for c in missing_cols:
        df[c] = np.nan


    hard_result_cols = [c for c in required if c not in ['Income']]
    row_invalid = df[hard_result_cols].isna().any(axis=1)
    row_missing_rate = float(row_invalid.mean()) if raw_rows else 1.0
    df = df.loc[~row_invalid].copy()


    df['income_imputed_flag'] = df['Income'].isna()
    county_income_median = safe_median(df['Income'], fallback=state_income_median)
    if pd.isna(county_income_median):
        county_income_median = 50000.0
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce').fillna(county_income_median)


    df['Total_Carbon'] = 0.0
    df['Avg_Comfort'] = 0.0
    for d in DECADES:
        carbon_col = f'Carbon(kg)_{d}'
        comfort_col1 = f'Comfort_Extreme_{d}'
        comfort_col2 = f'Comfort_Outage_{d}'
        df['Total_Carbon'] += df[carbon_col]
        df[f'Total_Comfort_{d}'] = df[comfort_col1] + df[comfort_col2]
        df['Avg_Comfort'] += df[f'Total_Comfort_{d}'] / len(DECADES)


    df['Income_Bin'] = df['Income'].apply(assign_income_bin)
    df = df.dropna(subset=['Income_Bin']).copy()
    df['Income_Bin'] = df['Income_Bin'].astype(int)
    df['Income_Bin_Label'] = df['Income_Bin'].map(INCOME_BIN_LABELS)


    baseline_ids = set(df.loc[df['Strategy'] == baseline_strategy, 'Building_ID'].unique())
    all_ids_after_row_clean = set(df['Building_ID'].unique())
    df = df[df['Building_ID'].isin(baseline_ids)].copy()


    strategy_counts = df.groupby('Building_ID')['Strategy'].nunique()
    no_retrofit_ids = set(strategy_counts[strategy_counts <= 1].index)
    df['no_retrofit_option_flag'] = df['Building_ID'].isin(no_retrofit_ids)

    effective_buildings = df['Building_ID'].nunique()
    income_bins = df.drop_duplicates('Building_ID')['Income_Bin'].nunique()
    building_loss_rate = 1.0 - effective_buildings / raw_buildings if raw_buildings else 1.0
    missing_baseline = len(baseline_ids) == 0
    valid_county = (effective_buildings >= MIN_TEMPLATE_BUILDINGS) and (income_bins >= MIN_INCOME_BINS) and (not missing_baseline)

    if raw_rows == 0 or effective_buildings == 0 or missing_baseline:
        quality_class = 'D_no_valid_baseline_or_data'
    elif row_missing_rate > HEAVY_ROW_MISSING_RATE or building_loss_rate > HEAVY_BUILDING_LOSS_RATE:
        quality_class = 'C_heavy_missing_merge_recommended'
    elif (effective_buildings < MIN_TEMPLATE_BUILDINGS) or (income_bins < MIN_INCOME_BINS):
        quality_class = 'C_small_or_low_income_bins'
    elif row_missing_rate > MODERATE_ROW_MISSING_RATE:
        quality_class = 'B_moderate_missing'
    else:
        quality_class = 'A_good'

    report = {
        'fips': str(fips).zfill(5),
        'ssp': ssp,
        'raw_rows': int(raw_rows),
        'raw_buildings': int(raw_buildings),
        'row_missing_rate': row_missing_rate,
        'building_loss_rate': building_loss_rate,
        'effective_buildings': int(effective_buildings),
        'income_bins': int(income_bins),
        'missing_baseline': bool(missing_baseline),
        'no_retrofit_buildings': int(len(no_retrofit_ids)),
        'valid_county': bool(valid_county),
        'quality_class': quality_class,
        'missing_required_cols': ';'.join(missing_cols),
    }
    return df, report


def load_optional_adjacency(path):
    if not path or not os.path.exists(path):
        return {}
    adj = pd.read_csv(path, dtype={'fips': str, 'neighbor_fips': str})
    adj['fips'] = adj['fips'].str.zfill(5)
    adj['neighbor_fips'] = adj['neighbor_fips'].str.zfill(5)
    out = defaultdict(set)
    for _, r in adj.iterrows():
        out[r['fips']].add(r['neighbor_fips'])
        out[r['neighbor_fips']].add(r['fips'])
    return {k: sorted(v) for k, v in out.items()}


def summarize_group(group_fips, data_cache):
    frames = [data_cache[f] for f in group_fips if data_cache.get(f) is not None and not data_cache[f].empty]
    if not frames:
        return {'effective_buildings': 0, 'income_bins': 0, 'valid_group': False, 'median_income': np.nan}
    g = pd.concat(frames, ignore_index=True)
    b = g.drop_duplicates(['Original_FIPS', 'Building_ID'])
    eff = b.shape[0]
    bins = b['Income_Bin'].nunique()
    med_income = safe_median(b['Income'], fallback=np.nan)
    return {
        'effective_buildings': int(eff),
        'income_bins': int(bins),
        'valid_group': bool(eff >= MIN_TEMPLATE_BUILDINGS and bins >= MIN_INCOME_BINS),
        'median_income': med_income,
    }


def group_similarity_score(source_group, target_group, data_cache, adjacency_map):
    s_fips = sorted(source_group)
    t_fips = sorted(target_group)
    s_sum = summarize_group(s_fips, data_cache)
    t_sum = summarize_group(t_fips, data_cache)


    adjacent = False
    if adjacency_map:
        for f in s_fips:
            if set(adjacency_map.get(f, [])) & set(t_fips):
                adjacent = True
                break
        spatial_penalty = 0 if adjacent else 100000
    else:
        spatial_penalty = min(abs(int(a) - int(b)) for a in s_fips for b in t_fips)

    income_penalty = 0
    if not pd.isna(s_sum['median_income']) and not pd.isna(t_sum['median_income']):
        income_penalty = abs(s_sum['median_income'] - t_sum['median_income']) / 1000.0


    size_bonus = -0.01 * t_sum['effective_buildings']
    return spatial_penalty + income_penalty + size_bonus


def build_fips_groups(fips_list, data_cache, quality_reports, adjacency_map):
    """Merge small counties or counties with too few income bins into nearby or similar counties."""
    groups = [{f} for f in fips_list if data_cache.get(f) is not None and not data_cache[f].empty]
    if not groups:
        return []

    changed = True
    max_iter = len(groups) * 3
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        invalid_idx = None
        for i, g in enumerate(groups):
            s = summarize_group(sorted(g), data_cache)
            if not s['valid_group']:
                invalid_idx = i
                break
        if invalid_idx is None:
            break
        if len(groups) == 1:
            break

        source = groups[invalid_idx]
        candidates = [(j, group_similarity_score(source, groups[j], data_cache, adjacency_map)) for j in range(len(groups)) if j != invalid_idx]
        if not candidates:
            break
        best_j = min(candidates, key=lambda x: x[1])[0]
        groups[best_j] = groups[best_j] | source
        del groups[invalid_idx]
        changed = True

    group_records = []
    for g in groups:
        members = sorted(g)
        group_id = 'GROUP_' + '_'.join(members)
        s = summarize_group(members, data_cache)
        group_records.append({
            'group_id': group_id,
            'members': members,
            'effective_buildings': s['effective_buildings'],
            'income_bins': s['income_bins'],
            'valid_group': s['valid_group'],
        })
    return group_records


def combine_group_data(group_members, data_cache, group_id):
    frames = []
    for f in group_members:
        df = data_cache.get(f)
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp['FIPS_Group'] = group_id

            tmp['Building_ID_Original'] = tmp['Building_ID'].astype(str)
            tmp['Building_ID'] = tmp['Original_FIPS'].astype(str) + '|' + tmp['Building_ID_Original'].astype(str)
            frames.append(tmp)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def get_reduction_factor(ssp):
    if ssp == 'ssp126':
        return  0.8
    if ssp == 'ssp245':
        return 0.9
    return 1.1

def compute_group_budgets(df_all, ssp):
    baseline_strat = 'up04' if ssp == 'ssp585' else 'up17'
    budgets = {}
    base_df_for_budget = df_all[df_all['Strategy'] == baseline_strat]
    for d in DECADES:
        carbon_col = f'Carbon(kg)_{d}'
        base_carb = base_df_for_budget[carbon_col].sum() if carbon_col in base_df_for_budget.columns else 1e18
        budgets[d] = base_carb * get_reduction_factor(ssp)
    return budgets


def solve_all_modes_for_group(df_all, optimizer, ssp, group_id):
    baseline_strat = 'up04' if ssp == 'ssp585' else 'up17'
    base_df_for_budget = df_all[df_all['Strategy'] == baseline_strat].copy()
    if base_df_for_budget.empty:
        print(f"  [!] {group_id}: Missing baseline {baseline_strat},Skipping.")
        return None

    bldg_ids = df_all['Building_ID'].unique().tolist()
    budgets = compute_group_budgets(df_all, ssp)
    all_strategy_results = []

    for mode in MODES:
        print(f"  - Optimization mode: {mode}")
        if mode == 'cost':
            optimal_pathway_df = optimizer.solve_cost_mode(df_all)

        elif mode == 'baseline':
            optimal_pathway_df = base_df_for_budget.copy()
            optimal_pathway_df['Optimal_Strategy'] = baseline_strat
            optimal_pathway_df = optimal_pathway_df.drop(columns=['Strategy'])

        elif mode == 'one':
            one_strat = ONE_SIZE_STRATEGY_BY_SSP[ssp]
            one_df = df_all[df_all['Strategy'] == one_strat].copy()
            missing_ids = set(bldg_ids) - set(one_df['Building_ID'])
            if missing_ids:
                print(f"    [!] one mode: {len(missing_ids)} buildings missing {one_strat}, fallback to baseline.")
                fallback = base_df_for_budget[base_df_for_budget['Building_ID'].isin(missing_ids)].copy()
                fallback['Optimal_Strategy'] = baseline_strat
                fallback = fallback.drop(columns=['Strategy'])
                one_df = one_df[~one_df['Building_ID'].isin(missing_ids)].copy()
                one_df['Optimal_Strategy'] = one_strat
                one_df = one_df.drop(columns=['Strategy'])
                optimal_pathway_df = pd.concat([one_df, fallback], ignore_index=True)
            else:
                optimal_pathway_df = one_df.copy()
                optimal_pathway_df['Optimal_Strategy'] = one_strat
                optimal_pathway_df = optimal_pathway_df.drop(columns=['Strategy'])

        elif mode in ['carbon', 'resilience']:
            optimal_pathway_df = optimizer.solve_single_building_mode(df_all, mode=mode)

        elif mode in ['Equity_DP', 'Equity_LIFP', 'Equity_RSP', 'Equity_RSBAP']:
            model, res = optimizer.solve_optimization(df_all, budgets, mode=mode, solver='gurobi')
            selected = [(b, s) for (b, s) in model.BS if pyo.value(model.x[b, s]) > 0.5]
            strategy_map = pd.DataFrame(selected, columns=['Building_ID', 'Optimal_Strategy'])
            optimal_pathway_df = pd.merge(
                strategy_map,
                df_all,
                left_on=['Building_ID', 'Optimal_Strategy'],
                right_on=['Building_ID', 'Strategy'],
                how='left'
            ).drop(columns=['Strategy'])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        v_eq, v_carb, v_comf = optimizer.evaluate_pathway_terms(optimal_pathway_df=optimal_pathway_df, df_all=df_all, mode=mode)
        if mode == 'cost':
            tmp_cost = optimizer.prepare_cost_columns(optimal_pathway_df, strategy_col="Optimal_Strategy")
            total_cost = optimizer._cost_mode_objective_series(tmp_cost, discount_rate=0.02).sum()
            print(f">>> Mode: {mode:12} | Discounted total cost objective: {total_cost:.4f} | Eq: {v_eq:.4f} | Carb: {v_carb:.4f} | Comf: {v_comf:.4f}")
        else:
            print(f">>> Mode: {mode:12} | Eq: {v_eq:.4f} | Carb: {v_carb:.4f} | Comf: {v_comf:.4f}")
        optimal_pathway_df['Optimization_Mode'] = mode
        optimal_pathway_df['FIPS_Group'] = group_id
        all_strategy_results.append(optimal_pathway_df)

    final_combined_df = pd.concat(all_strategy_results, ignore_index=True)
    final_combined_df = optimizer.prepare_cost_columns(
        final_combined_df,
        strategy_col="Optimal_Strategy",
        keep_component_columns=True,
    )
    return final_combined_df


def clear_previous_fixed_mapping_files(decision_root):
    """Remove saved fixed real-to-template mapping files so this run rebuilds mappings from scratch."""
    if not os.path.exists(decision_root):
        return
    patterns = [
        os.path.join(decision_root, "FIPS_*", "*_fixed_real_to_template_mapping.parquet"),
        os.path.join(decision_root, "FIPS_*", "*_fixed_mapping_reference_report.csv"),
        os.path.join(decision_root, "FIPS_*", "*_fixed_mapping_distribution_check.csv"),
    ]
    removed = 0
    for pat in patterns:
        for fp in glob.glob(pat):
            try:
                os.remove(fp)
                removed += 1
            except FileNotFoundError:
                pass
    if removed:
        print(f">>> Removed old fixed-mapping files {removed} files; rebuilding mappings.")


# Real-building mapping is implemented in a separate module.
from Robust_real_mapping import map_group_results_to_original_counties


# 3. Main


res_path = os.environ.get(
    "RESSTOCK_METADATA_FILE",
    os.path.join(BASE_DIR, "upgrade0.csv"),
)
replacement_cost_factor = 0.7

adjacency_map = load_optional_adjacency(COUNTY_ADJACENCY_FILE)

for _, state_name, _ in US_STATES:
    if TARGET_STATE_ABBRS is not None and state_name not in TARGET_STATE_ABBRS:
        continue

    print("\n" + "#" * 80)
    print(f"Processing state: {state_name}")
    print("#" * 80)

    MAPPING_FILE = os.path.join(BASE_DIR, "Building footprint household", state_name, f"{state_name}_ResStock.parquet")
    if not os.path.exists(MAPPING_FILE):
        print(f"[!] Real-building mapping file not found: {MAPPING_FILE},Skipping {state_name}")
        continue

    FIPS_LIST = load_state_fips_from_upgrade0(res_path, state_name)
    if not FIPS_LIST:
        print(f"[!] upgrade0.csv not found in {state_name} FIPS values,Skipping.")
        continue


    state_map = read_parquet_existing_columns(MAPPING_FILE, STATE_MAP_REQUIRED_COLS)
    if 'county_fips' not in state_map.columns:
        raise ValueError(f"Real-building mapping file is missing county_fips: {MAPPING_FILE}")
    state_map['county_fips'] = state_map['county_fips'].astype(str).str.zfill(5)

    state_decision_root = os.path.join(get_state_r2_root(BASE_DIR, state_name), "Decision_Robust")
    if FORCE_REBUILD_REAL_MAPPING:
        clear_previous_fixed_mapping_files(state_decision_root)

    for ssp in SSPS:
        print("\n" + "=" * 80)
        print(f"{state_name} | {ssp}: starting quality control, grouping, optimization, and real-building mapping")
        print("=" * 80)


        state_income_values = []
        raw_cache = {}
        for fips in FIPS_LIST:
            raw = load_raw_fips_strategy_data(BASE_DIR, state_name, fips, ssp)
            raw_cache[fips] = raw
            if raw is not None and 'Income' in raw.columns:
                state_income_values.append(pd.to_numeric(raw['Income'], errors='coerce'))
        state_income_median = safe_median(pd.concat(state_income_values, ignore_index=True) if state_income_values else pd.Series(dtype=float), fallback=50000.0)


        data_cache = {}
        quality_reports = []
        for fips in FIPS_LIST:
            cleaned, report = clean_strategy_data(raw_cache.get(fips), fips, ssp, state_income_median=state_income_median)
            data_cache[fips] = cleaned
            quality_reports.append(report)

        decision_root = os.path.join(get_state_r2_root(BASE_DIR, state_name), "Decision_Robust")
        os.makedirs(decision_root, exist_ok=True)
        qr_df = pd.DataFrame(quality_reports)
        qr_path = os.path.join(decision_root, f"{state_name}_{ssp}_quality_report.csv")
        qr_df.to_csv(qr_path, index=False)
        print(f">>> Quality report saved: {qr_path}")


        fips_groups = build_fips_groups(FIPS_LIST, data_cache, quality_reports, adjacency_map)
        group_map_records = []
        for gi, g in enumerate(fips_groups, start=1):
            for f in g['members']:
                q = qr_df[qr_df['fips'] == f].iloc[0].to_dict() if (qr_df['fips'] == f).any() else {}
                group_map_records.append({
                    'ssp': ssp,
                    'original_fips': f,
                    'fips_group_id': g['group_id'],
                    'group_members': ';'.join(g['members']),
                    'group_effective_buildings': g['effective_buildings'],
                    'group_income_bins': g['income_bins'],
                    'valid_group': g['valid_group'],
                    'county_quality_class': q.get('quality_class', ''),
                    'county_effective_buildings': q.get('effective_buildings', np.nan),
                    'county_income_bins': q.get('income_bins', np.nan),
                })
        fg_df = pd.DataFrame(group_map_records)
        fg_path = os.path.join(decision_root, f"{state_name}_{ssp}_fips_group_map.csv")
        fg_df.to_csv(fg_path, index=False)
        print(f">>> FIPS_GROUP map saved: {fg_path}")


        for gi, g in enumerate(fips_groups, start=1):
            group_id = g['group_id']
            members = g['members']
            print("\n" + "-" * 80)
            print(f">>> Process {group_id} | members={members} | valid_group={g['valid_group']}")
            print("-" * 80)

            df_group = combine_group_data(members, data_cache, group_id)
            if df_group is None or df_group.empty:
                print(f"  [!] {group_id}: No valid template data,Skipping.")
                continue
            if not g['valid_group']:
                print(f"  [!] {group_id}: Merged group remains below the quality threshold; optimization will still be attempted.")

            optimizer = JusticePathwayOptimizer(
                replacement_cost_factor=replacement_cost_factor,
                state_name=state_name,
                base_dir=BASE_DIR,
                fips=group_id,
                ssp=ssp,
                active_decades=DECADES,
                use_all_dir=use_all_dir_for_state(state_name)
            )

            final_combined_df = solve_all_modes_for_group(df_group, optimizer, ssp, group_id)
            if final_combined_df is None or final_combined_df.empty:
                continue


            group_id_safe = f"GROUP_{gi:03d}"
            group_output_dir = os.path.join(decision_root, "Group_Template", group_id_safe)
            os.makedirs(group_output_dir, exist_ok=True)
            group_output_path = os.path.join(
                group_output_dir,
                f"{group_id_safe}_{ssp}_optimal_pathway.parquet"
            )
            final_combined_df.to_parquet(group_output_path, engine='pyarrow', compression='brotli', index=False)
            print(f">>> Group-level template optimization results saved: {group_output_path}")

            map_group_results_to_original_counties(
                final_combined_df=final_combined_df,
                state_map=state_map,
                group_members=members,
                output_root=decision_root,
                ssp=ssp,
                optimizer=optimizer,
                force_rebuild_mapping=FORCE_REBUILD_REAL_MAPPING
            )

print("\n" + "#" * 70)
print("Cost-optimal pathway + fixed real-building mapping Workflow completed.")
print("#" * 70)
