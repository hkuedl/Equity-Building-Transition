
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import os
import sys
import glob
import re
import gc
import time
import shutil
import pyarrow.parquet as pq
from collections import defaultdict
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from annualized_retrofit_cost_Prediction import (
    add_annualized_capex_columns,
    COST_TREND_CASES,
    COST_TREND_CASE_METADATA,
    STRATEGY_COMPONENTS_BY_SSP,
    split_capex_by_component,
    component_cost_class,
    cost_multiplier_for_component,
)


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


UPFRONT_CAPEX_BURDEN_YEARS = 5.0  # burden only: upfront CAPEX is evaluated against 5 years of annual income
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
    def __init__(self, replacement_cost_factor, state_name, base_dir, fips, ssp, active_decades=['2020s', '2030s', '2040s', '2050s'], use_all_dir=False, cost_trend_case='active_down_passive_down'):
        self.base_dir = base_dir
        self.fips = fips
        self.ssp = ssp
        self.active_decades = active_decades

        self.baseline_strategy = 'up04' if ssp == 'ssp585' else 'up17'
        state_dir_name = f"{state_name}_all" if use_all_dir else state_name
        self.results_path = os.path.join(base_dir, f"#R2/{state_dir_name}/FIPS_{fips}")
        self.replacement_cost_factor = replacement_cost_factor
        self.cost_trend_case = cost_trend_case

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

    def solve_cost_mode(self, df):
        """
        Cost-optimal mode.
        For each template building, select the strategy with the minimum total cost:
        sum of Energy_cost over all active decades + one-time total retrofit CAPEX.
        No retrofit subsidy, bill aid, or financing incentive is applied.
        The same comfort feasibility filter used by other pathways is retained: if a
        building has strategies satisfying the comfort limit in all decades, only those
        strategies are considered; otherwise the minimum-comfort fallback is used.
        """
        valid_data = self._build_valid_data(df, comfort_limit=40)
        valid_data = self.prepare_cost_columns(valid_data, strategy_col="Strategy")
        energy_cols = [f"Energy_cost_{d}" for d in self.active_decades]
        valid_data["Cost_Objective_Total"] = (
            valid_data[energy_cols].sum(axis=1)
            + valid_data["Total_CAPEX_for_annualization"]
        )
        valid_data = self._drop_invalid_model_rows(valid_data, mode="cost")
        best_idx = valid_data.groupby("Building_ID")["Cost_Objective_Total"].idxmin()
        optimal_pathway_df = valid_data.loc[best_idx].copy().reset_index(drop=True)
        optimal_pathway_df["Optimal_Strategy"] = optimal_pathway_df["Strategy"]
        optimal_pathway_df = optimal_pathway_df.drop(columns=["Strategy"])
        return optimal_pathway_df


    def prepare_cost_columns(self, df, strategy_col="Strategy"):
        """Add cost columns using strategy timing and the active cost trend.

        ``Install_Decade`` controls the timing of adjusted and annualized
        CAPEX. Rows without it are treated as 2020s installations.
        """
        return add_annualized_capex_columns(
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
            install_decade_col="Install_Decade" if "Install_Decade" in df.columns else None,
            cost_trend_case=self.cost_trend_case,
            adjusted_total_capex_col="Adjusted_Total_CAPEX_for_annualization",
            keep_component_columns=False
        )

    def get_paid_capex(self, df, decade, mode):
        """Return resident-paid CAPEX for a candidate installation decade.

        DP and cost modes pay adjusted upfront CAPEX in the installation
        decade. LIFP uses annualized CAPEX. RSP and RSBAP apply subsidies to
        upfront CAPEX in the installation decade.
        """
        install_decade = df["Install_Decade"] if "Install_Decade" in df.columns else pd.Series("2020s", index=df.index)
        installed_this_decade = install_decade.astype(str).eq(str(decade))
        capex_col = "Adjusted_Total_CAPEX_for_annualization" if "Adjusted_Total_CAPEX_for_annualization" in df.columns else "Total_CAPEX_for_annualization"
        upfront_capex = pd.to_numeric(df.get(capex_col, 0.0), errors="coerce").fillna(0.0)

        if mode in ["Equity_DP", "cost"]:
            return upfront_capex.where(installed_this_decade, 0.0)

        if mode == "Equity_LIFP":
            return pd.to_numeric(df.get(f"Annualized_CAPEX_{decade}", 0.0), errors="coerce").fillna(0.0)

        if mode in ["Equity_RSP", "Equity_RSBAP"]:
            payer_share = df["Income_Bin"].map(lambda x: get_resident_capex_share(x))
            return (upfront_capex * payer_share).where(installed_this_decade, 0.0)

        return pd.to_numeric(df.get(f"Annualized_CAPEX_{decade}", 0.0), errors="coerce").fillna(0.0)

    def _installed_by_decade_mask(self, df, decade):
        """Return True for rows whose retrofit has already been installed by this decade."""
        if "Install_Decade" not in df.columns:
            # Static case: treat non-baseline retrofit strategies as installed from 2020s.
            strategy_col = "Optimal_Strategy" if "Optimal_Strategy" in df.columns else "Strategy" if "Strategy" in df.columns else None
            if strategy_col is None:
                return pd.Series(True, index=df.index)
            return df[strategy_col].astype(str).ne(str(self.baseline_strategy))
        install_year = df["Install_Decade"].map(lambda x: self._decade_start(str(x)))
        current_year = self._decade_start(decade)
        strategy_col = "Optimal_Strategy" if "Optimal_Strategy" in df.columns else "Strategy" if "Strategy" in df.columns else None
        not_baseline = True if strategy_col is None else df[strategy_col].astype(str).ne(str(self.baseline_strategy))
        return (install_year <= current_year) & not_baseline

    def get_paid_energy_cost(self, df, decade, mode):
        """Return resident-paid energy cost after any RSBAP bill assistance.

        Dynamic candidates already contain baseline energy cost before
        installation and retrofit energy cost from the installation decade
        onward.
        """
        energy = pd.to_numeric(df[f"Energy_cost_{decade}"], errors="coerce").fillna(0.0)

        if mode != "Equity_RSBAP":
            return energy

        payer_share = df["Income_Bin"].map(lambda x: get_resident_energy_share(x)).astype(float)
        return energy * payer_share

    def get_resident_total_cost(self, df, decade, mode):
        """Return resident-paid CAPEX plus energy cost in dollars."""
        total = self.get_paid_capex(df, decade, mode) + self.get_paid_energy_cost(df, decade, mode)
        return pd.to_numeric(total, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def get_resident_burden_equivalent_cost(self, df, decade, mode):
        """Return the numerator used in burden and equity calculations.

        In DP, RSP, and RSBAP, installation-decade cash CAPEX is spread over
        ``UPFRONT_CAPEX_BURDEN_YEARS`` for comparison with annual income.
        LIFP CAPEX is already annualized.
        """
        capex = self.get_paid_capex(df, decade, mode)
        if mode in UPFRONT_CAPEX_BURDEN_MODES:
            install_decade = df["Install_Decade"] if "Install_Decade" in df.columns else pd.Series("2020s", index=df.index)
            installed_this_decade = install_decade.astype(str).eq(str(decade))
            capex = capex.where(~installed_this_decade, capex / UPFRONT_CAPEX_BURDEN_YEARS)
        burden_cost = capex + self.get_paid_energy_cost(df, decade, mode)
        # Burden variables are nonnegative; avoid negative burden from net-export credits.
        return pd.to_numeric(burden_cost, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    def build_fallback_solution(self, df, county_carbon_budgets=None, mode="Equity_RSP", reference_df=None, reason=""):
        """Build a usable one-candidate-per-building fallback solution.

        Priority 1: reuse a previous successful equity solution, usually Equity_DP.
        This is physically feasible for RSP/RSBAP because subsidies only change the
        resident payment, not the selected retrofit strategy or install decade.
        Priority 2: choose a low-burden/low-carbon heuristic candidate per building.
        Carbon and burden constraints remain practically feasible because the main
        optimization model already contains nonnegative slack variables.
        """
        if reference_df is not None and (not reference_df.empty) and {"Building_ID", "Strategy"}.issubset(reference_df.columns):
            out = reference_df.copy().drop_duplicates("Building_ID", keep="first")
            out["Fallback_Flag"] = True
            out["Fallback_Method"] = "reuse_previous_equity_solution"
            out["Fallback_Reason"] = reason
            print(f"  [!] Fallback used for {mode}: reuse previous equity solution, buildings={out['Building_ID'].nunique()}")
            return out

        valid_data = self._build_dynamic_valid_data(df, comfort_limit=4)
        valid_data = self.prepare_cost_columns(valid_data, strategy_col="Strategy")
        valid_data = self._drop_invalid_model_rows(valid_data, mode=f"{mode}_fallback")
        for d in self.active_decades:
            valid_data[f"_resident_cost_{d}"] = self.get_resident_burden_equivalent_cost(valid_data, d, mode)
        cost_cols = [f"_resident_cost_{d}" for d in self.active_decades]
        valid_data["_cost_sum"] = valid_data[cost_cols].sum(axis=1)
        carbon_scale = max(float(pd.to_numeric(valid_data["Total_Carbon"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()), 1.0)
        comfort_scale = max(float(pd.to_numeric(valid_data["Avg_Comfort"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()), 1.0)
        valid_data["_fallback_score"] = valid_data["_cost_sum"] + 0.01 * valid_data["Total_Carbon"] / carbon_scale + 0.01 * valid_data["Avg_Comfort"] / comfort_scale
        idx = valid_data.groupby("Building_ID")["_fallback_score"].idxmin()
        out = valid_data.loc[idx].copy().reset_index(drop=True)
        out["Fallback_Flag"] = True
        out["Fallback_Method"] = "min_resident_cost_with_small_carbon_comfort_tiebreak"
        out["Fallback_Reason"] = reason
        print(f"  [!] Fallback used for {mode}: heuristic solution, buildings={out['Building_ID'].nunique()}")
        return out

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
        if "Total_CAPEX_for_annualization" in valid_data.columns:
            check_cols.append("Total_CAPEX_for_annualization")
        if "Adjusted_Total_CAPEX_for_annualization" in valid_data.columns:
            check_cols.append("Adjusted_Total_CAPEX_for_annualization")

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
        """
         mode pathresults Eq / Carbon / Comfort.
        """
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


    def _decade_start(self, decade):
        return {"2020s": 2020, "2030s": 2030, "2040s": 2040, "2050s": 2050}.get(str(decade), 2020)

    def _build_dynamic_valid_data(self, df, comfort_limit=4):
        """Build building-strategy-installation-decade candidates.

        Metrics use the baseline before installation and the retrofit strategy
        from installation onward. The baseline keeps one 2020s candidate.
        """
        rows = []
        metric_prefixes = ["Carbon(kg)", "Comfort_Extreme", "Comfort_Outage", "Energy_cost"]
        install_cost_cols = [f"Install_cost_{d}" for d in self.active_decades]

        for b_id, group in df.groupby("Building_ID", sort=False):
            base = group[group["Strategy"] == self.baseline_strategy]
            if base.empty:
                continue
            base = base.iloc[0]

            for _, strat in group.iterrows():
                strategy = strat["Strategy"]
                install_decades = ["2020s"] if strategy == self.baseline_strategy else list(self.active_decades)
                for install_decade in install_decades:
                    out = strat.copy()
                    out["Install_Decade"] = install_decade
                    out["Install_Year"] = self._decade_start(install_decade)
                    if strategy == self.baseline_strategy:
                        out["Install_Decade"] = "2020s"
                        out["Install_Year"] = 2020

                    out["Total_Carbon"] = 0.0
                    out["Avg_Comfort"] = 0.0
                    for d in self.active_decades:
                        use_base = (strategy == self.baseline_strategy) or (self._decade_start(d) < self._decade_start(install_decade))
                        src = base if use_base else strat
                        for pref in metric_prefixes:
                            col = f"{pref}_{d}"
                            if col in out.index and col in src.index:
                                out[col] = src[col]
                        out[f"Total_Comfort_{d}"] = out[f"Comfort_Extreme_{d}"] + out[f"Comfort_Outage_{d}"]
                        out["Total_Carbon"] += out[f"Carbon(kg)_{d}"]
                        out["Avg_Comfort"] += out[f"Total_Comfort_{d}"] / len(self.active_decades)


                    if strategy == self.baseline_strategy:
                        for c in install_cost_cols:
                            if c in out.index:
                                out[c] = 0.0
                    rows.append(out)

        if not rows:
            raise ValueError(f"No dynamic candidates built for fips={self.fips}, ssp={self.ssp}")
        dynamic = pd.DataFrame(rows).reset_index(drop=True)
        dynamic["Strategy_Install"] = dynamic["Strategy"].astype(str) + "@" + dynamic["Install_Decade"].astype(str)

        valid_rows = []
        comfort_cols = [f"Total_Comfort_{d}" for d in self.active_decades]
        for _, group in dynamic.groupby("Building_ID", sort=False):
            mask = (group[comfort_cols] <= comfort_limit).all(axis=1)
            qualified = group[mask]
            if not qualified.empty:
                valid_rows.append(qualified)
            else:
                valid_rows.append(group.sort_values("Avg_Comfort").head(1))
        return pd.concat(valid_rows, ignore_index=True)


    def _select_incumbent_or_heuristic_candidates(self, model, valid_data, mode, bldg_to_c, reason=""):
        """Return selected candidate dataframe from incumbent when possible; otherwise heuristic fallback."""
        selected_cids = []
        incumbent_complete = True
        for b, cands in bldg_to_c.items():
            chosen = []
            for c in cands:
                val = pyo.value(model.x[c], exception=False)
                if val is None:
                    incumbent_complete = False
                    break
                try:
                    if np.isfinite(float(val)) and float(val) > 0.5:
                        chosen.append(int(c))
                except Exception:
                    incumbent_complete = False
                    break
            if len(chosen) != 1:
                incumbent_complete = False
                break
            selected_cids.extend(chosen)

        if incumbent_complete:
            msg = f" ({reason})" if reason else ""
            print(f">>> Use best available Gurobi incumbent as final solution{msg}.")
            out = valid_data[valid_data["Candidate_ID"].isin(selected_cids)].copy()
            out["Fallback_Flag"] = False
            out["Fallback_Method"] = "gurobi_incumbent"
            out["Fallback_Reason"] = reason
            return out

        tmp = valid_data.copy()
        score = pd.Series(0.0, index=tmp.index)
        for d in self.active_decades:
            score = score + self.get_resident_burden_equivalent_cost(tmp, d, mode)
        carbon_scale = max(float(pd.to_numeric(tmp["Total_Carbon"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()), 1.0)
        comfort_scale = max(float(pd.to_numeric(tmp["Avg_Comfort"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()), 1.0)
        tmp["_fallback_score"] = score + 0.01 * tmp["Total_Carbon"] / carbon_scale + 0.01 * tmp["Avg_Comfort"] / comfort_scale
        best_idx = tmp.groupby("Building_ID")["_fallback_score"].idxmin()
        out = tmp.loc[best_idx].copy().reset_index(drop=True)
        out["Fallback_Flag"] = True
        out["Fallback_Method"] = "min_burden_equivalent_cost_with_small_carbon_comfort_tiebreak"
        out["Fallback_Reason"] = reason
        print(f">>> Gurobi returned no complete incumbent; use deterministic heuristic fallback because {reason}.")
        return out

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
        valid_data = self._build_dynamic_valid_data(df, comfort_limit=40)
        valid_data = self.prepare_cost_columns(valid_data, strategy_col="Strategy")
        valid_data = self._drop_invalid_model_rows(valid_data, mode=mode)

        bldg_income_map = valid_data.drop_duplicates('Building_ID').set_index('Building_ID')['Income']
        bldg_bin_map = valid_data.drop_duplicates('Building_ID').set_index('Building_ID')['Income_Bin']
        income_bins = [b for b in [0, 25000, 50000, 75000, 100000, 150000, 200000] if b in valid_data['Income_Bin'].unique()]
        bin_total_income = self._build_safe_bin_total_income(
            bldg_income_map=bldg_income_map, bldg_bin_map=bldg_bin_map,
            income_bins=income_bins, min_income_per_building=1000.0,
        )

        base_total_carbon, base_total_comfort, base_total_mad, _, _ = self.compute_baseline_terms(
            df_all=df, building_ids=valid_data['Building_ID'].unique().tolist(), mode=mode
        )

        model = pyo.ConcreteModel()
        valid_data = valid_data.reset_index(drop=True)
        valid_data["Candidate_ID"] = valid_data.index.astype(int)
        model.C = pyo.Set(initialize=valid_data["Candidate_ID"].tolist())
        model.Bins = pyo.Set(initialize=income_bins)
        model.Decades = pyo.Set(initialize=self.active_decades)
        model.x = pyo.Var(model.C, domain=pyo.Binary)
        model.carbon_slack = pyo.Var(model.Decades, domain=pyo.NonNegativeReals)

        model.eb_slack = pyo.Var(model.Bins, domain=pyo.NonNegativeReals)

        cid_to_building = valid_data.set_index("Candidate_ID")["Building_ID"].to_dict()
        cid_to_bin = valid_data.set_index("Candidate_ID")["Income_Bin"].to_dict()
        carbon_dict = {(d, c): float(valid_data.loc[valid_data["Candidate_ID"] == c, f"Carbon(kg)_{d}"].iloc[0]) for d in self.active_decades for c in model.C}
        cost_df = valid_data.copy()
        cost_dict = {}
        for d in self.active_decades:
            vals = self.get_resident_burden_equivalent_cost(cost_df, d, mode)
            cost_dict.update({(d, int(c)): float(v) for c, v in zip(cost_df["Candidate_ID"], vals)})
        comfort_total_map = valid_data.set_index("Candidate_ID")["Avg_Comfort"].to_dict()
        carbon_total_map = valid_data.set_index("Candidate_ID")["Total_Carbon"].to_dict()

        bin_to_c = defaultdict(list)
        bldg_to_c = defaultdict(list)
        for c in model.C:
            b = cid_to_building[c]
            bin_to_c[cid_to_bin[c]].append(c)
            bldg_to_c[b].append(c)

        raw_cost_vals = np.array(list(cost_dict.values()), dtype=float)
        if (not np.isfinite(raw_cost_vals).all()):
            raise RuntimeError(f"Non-finite resident cost coefficients detected before Pyomo build: mode={mode}")
        if (raw_cost_vals < -1e-9).any():
            print(f"  [!] Negative resident costs detected and should have been clipped: mode={mode}, n={(raw_cost_vals < -1e-9).sum()}")
        burden_ub = 1.0
        for k in income_bins:
            for d in self.active_decades:
                denom = max(float(bin_total_income[k]), 1.0)
                burden_ub = max(burden_ub, sum(max(cost_dict[d, c], 0.0) for c in bin_to_c[k]) / denom)
        burden_ub = min(max(10.0 * burden_ub, 1.0), 1e6)

        model.bin_burden = pyo.Var(model.Bins, model.Decades, bounds=(0.0, burden_ub))
        model.avg_bin_burden = pyo.Var(model.Decades, bounds=(0.0, burden_ub))
        model.dev = pyo.Var(model.Bins, model.Decades, bounds=(0.0, burden_ub))

        def bin_burden_rule(m, k, d):
            return m.bin_burden[k, d] == sum(cost_dict[d, c] * m.x[c] for c in bin_to_c[k]) / bin_total_income[k]
        model.c_bin_burden = pyo.Constraint(model.Bins, model.Decades, rule=bin_burden_rule)

        def avg_burden_rule(m, d):
            return m.avg_bin_burden[d] == sum(m.bin_burden[k, d] for k in m.Bins) / len(income_bins)
        model.c_avg_burden = pyo.Constraint(model.Decades, rule=avg_burden_rule)

        model.c_dev_pos = pyo.Constraint(model.Bins, model.Decades, rule=lambda m, k, d: m.dev[k, d] >= m.bin_burden[k, d] - m.avg_bin_burden[d])
        model.c_dev_neg = pyo.Constraint(model.Bins, model.Decades, rule=lambda m, k, d: m.dev[k, d] >= m.avg_bin_burden[d] - m.bin_burden[k, d])

        def carbon_budget_rule(m, d):
            return sum(carbon_dict[d, c] * m.x[c] for c in m.C) <= county_carbon_budgets.get(d, 1e18) + m.carbon_slack[d]
        model.c_carb_limit = pyo.Constraint(model.Decades, rule=carbon_budget_rule)

        def bin_total_limit_rule(m, k):
            return sum(m.bin_burden[k, d] for d in m.Decades) / len(self.active_decades) <= eb_limit + m.eb_slack[k]
        model.c_eb_limit = pyo.Constraint(model.Bins, rule=bin_total_limit_rule)

        def selection_rule(m, b):
            return sum(m.x[c] for c in bldg_to_c[b]) == 1
        model.c_select = pyo.Constraint(list(bldg_to_c.keys()), rule=selection_rule)

        base_total_carbon = self._safe_positive_scalar(base_total_carbon, 1.0, "base_total_carbon")
        base_total_comfort = self._safe_positive_scalar(base_total_comfort, 1.0, "base_total_comfort")
        base_total_mad = max(self._safe_positive_scalar(base_total_mad, 0.001, "base_total_mad"), 0.001)
        term_equity = sum(model.dev[k, d] ** 2 for k in model.Bins for d in model.Decades) / base_total_mad
        term_carbon = sum(carbon_total_map[c] * model.x[c] for c in model.C) / base_total_carbon
        term_comfort = sum(comfort_total_map[c] * model.x[c] for c in model.C) / base_total_comfort
        penalty_terms_1 = 1e6 * sum(model.carbon_slack[d] for d in model.Decades)

        penalty_terms_2 = 1e6 * sum(model.eb_slack[k] for k in model.Bins)
        model.obj = pyo.Objective(expr=term_equity + term_carbon * 0.0 + term_comfort * 0.0 + 0*penalty_terms_1 + 0*penalty_terms_2, sense=pyo.minimize)

        opt = pyo.SolverFactory(solver)
        res = None
        term_cond = None
        status = None
        solve_t0 = time.time()

        try:
            # Stage 1: strict solve, at most 2 minutes.
            opt.options.clear()
            opt.options['TimeLimit'] = 120
            opt.options['DualReductions'] = 0
            opt.options['NumericFocus'] = 2
            res = opt.solve(model, tee=False)
            term_cond = res.solver.termination_condition
            status = res.solver.status
            print(f">>> Gurobi stage 1 status: {status}, termination: {term_cond}")
        except Exception as e:
            selected_df = self._select_incumbent_or_heuristic_candidates(
                model, valid_data, mode, bldg_to_c, reason=f"stage 1 error: {repr(e)}"
            )
            return model, res, selected_df

        if term_cond != pyo.TerminationCondition.optimal:
            print(">>> Stage 1 did not prove optimal within 2 min; relax MIPGap and continue.")
            try:
                # Stage 2: relaxed solve, at most 4 minutes.
                opt.options.clear()
                opt.options['TimeLimit'] = 240
                opt.options['MIPGap'] = 0.05
                opt.options['MIPFocus'] = 1
                opt.options['Heuristics'] = 0.8
                opt.options['Cuts'] = 0
                opt.options['Presolve'] = 2
                opt.options['DualReductions'] = 0
                opt.options['NumericFocus'] = 2
                res = opt.solve(model, tee=False)
                term_cond = res.solver.termination_condition
                status = res.solver.status
                print(f">>> Gurobi stage 2 status: {status}, termination: {term_cond}")
            except Exception as e:
                selected_df = self._select_incumbent_or_heuristic_candidates(
                    model, valid_data, mode, bldg_to_c, reason=f"stage 2 error after relaxation: {repr(e)}"
                )
                return model, res, selected_df

        elapsed = time.time() - solve_t0
        if term_cond == pyo.TerminationCondition.optimal:
            selected_cids = [int(c) for c in model.C if (pyo.value(model.x[c], exception=False) is not None and pyo.value(model.x[c], exception=False) > 0.5)]
            selected_df = valid_data[valid_data["Candidate_ID"].isin(selected_cids)].copy()
            selected_df["Fallback_Flag"] = False
            selected_df["Fallback_Method"] = "gurobi_optimal"
            selected_df["Fallback_Reason"] = ""
            print(f">>> Gurobi proved optimal in {elapsed:.1f}s.")
        elif term_cond in [pyo.TerminationCondition.feasible, pyo.TerminationCondition.maxTimeLimit]:
            selected_df = self._select_incumbent_or_heuristic_candidates(
                model, valid_data, mode, bldg_to_c, reason=f"termination={term_cond}, elapsed={elapsed:.1f}s"
            )
        else:
            selected_df = self._select_incumbent_or_heuristic_candidates(
                model, valid_data, mode, bldg_to_c, reason=f"termination={term_cond}, elapsed={elapsed:.1f}s"
            )

        v_pen_carbon = pyo.value(penalty_terms_1, exception=False)
        v_pen_burden = pyo.value(penalty_terms_2, exception=False)
        print(f">>> Penalty_Carbon: {self._fmt_optional_value(v_pen_carbon)} | Penalty_Burden: {self._fmt_optional_value(v_pen_burden)}")
        return model, res, selected_df


def clear_directory_contents(dir_path):
    """directoryfile/directory,delete."""
    if not os.path.exists(dir_path):
        return

    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"  [!] Deletion failed: {path} | {repr(e)}")


# 2. Robust multi-state/group workflow


BASE_DIR = os.environ.get(
    "PROJECT_DATA_DIR",
    os.path.join(os.path.dirname(CURRENT_DIR), "data"),
)
SSPS = ['ssp126']  #, 'ssp245', 'ssp585']
DECADES = ['2020s', '2030s', '2040s', '2050s']
MODES = ['Equity_DP']
COST_TREND_CASES_TO_RUN = list(COST_TREND_CASES.keys())  # 40 reproducible sampled scenarios
FEATURE_COLS = ['sqft', 'stories', 'bldg_assigned_income']

DECISION_OUTPUT_FOLDER = "Decision_Pre"
FIXED_MAPPING_SOURCE_FOLDER = "Decision_Robust"

# Runtime controls

# of solving the same group/scenario again. This is important when running 49

SKIP_EXISTING_GROUP_TEMPLATE_RESULTS = True

# Set to an integer for quick debugging, e.g. 4. Keep None for all 40 scenarios.
MAX_COST_TREND_CASES_TO_RUN = None
if MAX_COST_TREND_CASES_TO_RUN is not None:
    COST_TREND_CASES_TO_RUN = COST_TREND_CASES_TO_RUN[:int(MAX_COST_TREND_CASES_TO_RUN)]

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
        return 0.8
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
    fallback_refs = {}

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
            try:
                model, res, selected_df = optimizer.solve_optimization(df_all, budgets, mode=mode, solver='gurobi')
                selected_df['Fallback_Flag'] = False
                fallback_refs[mode] = selected_df.copy()
            except Exception as e:
                print(f"  [!] {mode} optimization failed for {group_id}: {repr(e)}")
                ref = None
                if mode in ['Equity_RSP', 'Equity_RSBAP'] and 'Equity_DP' in fallback_refs:
                    ref = fallback_refs['Equity_DP']
                elif fallback_refs:
                    ref = list(fallback_refs.values())[-1]
                selected_df = optimizer.build_fallback_solution(df_all, budgets, mode=mode, reference_df=ref, reason=repr(e))
                fallback_refs[mode] = selected_df.copy()
            optimal_pathway_df = selected_df.copy()
            optimal_pathway_df['Optimal_Strategy'] = optimal_pathway_df['Strategy']
            optimal_pathway_df = optimal_pathway_df.drop(columns=['Strategy', 'Candidate_ID'], errors='ignore')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        v_eq, v_carb, v_comf = optimizer.evaluate_pathway_terms(optimal_pathway_df=optimal_pathway_df, df_all=df_all, mode=mode)
        if mode == 'cost':
            tmp_cost = optimizer.prepare_cost_columns(optimal_pathway_df, strategy_col="Optimal_Strategy")
            energy_cols = [f"Energy_cost_{d}" for d in DECADES]
            total_cost = tmp_cost[energy_cols].sum(axis=1).sum() + tmp_cost["Total_CAPEX_for_annualization"].sum()
            print(f">>> Mode: {mode:12} | Total cost objective: {total_cost:.4f} | Eq: {v_eq:.4f} | Carb: {v_carb:.4f} | Comf: {v_comf:.4f}")
        else:
            print(f">>> Mode: {mode:12} | Eq: {v_eq:.4f} | Carb: {v_carb:.4f} | Comf: {v_comf:.4f}")
        optimal_pathway_df['Optimization_Mode'] = mode
        optimal_pathway_df['FIPS_Group'] = group_id
        all_strategy_results.append(optimal_pathway_df)

    final_combined_df = pd.concat(all_strategy_results, ignore_index=True)
    final_combined_df = add_annualized_capex_columns(
        final_combined_df,
        ssp=ssp,
        strategy_col="Optimal_Strategy",
        install_cost_cols=[f'Install_cost_{d}' for d in DECADES],
        total_capex_mode="sum",
        decades=DECADES,
        r=0.02,
        install_year=2020,
        horizon_end=2060,
        replacement_cost_factor=replacement_cost_factor,
        install_decade_col="Install_Decade" if "Install_Decade" in final_combined_df.columns else None,
        cost_trend_case=optimizer.cost_trend_case,
        adjusted_total_capex_col="Adjusted_Total_CAPEX_for_annualization",
        keep_component_columns=True
    )
    return final_combined_df


# Real-building mapping is implemented in a separate module.
from Robust_real_mapping import map_group_results_to_original_counties


# 2.6 State-level real-building retrofit timing summary

def normalize_up(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r"up\s*0*(\d+)", s)
    return f"up{int(m.group(1)):02d}" if m else s
def infer_mode_from_real_path(path):
    low = os.path.basename(path).lower().replace("-", "_").replace(" ", "_")
    if "equity_rsbap" in low or "rsbap" in low:
        return "Equity_RSBAP"
    if "equity_rsp" in low or "_rsp_" in low:
        return "Equity_RSP"
    if "equity_lifp" in low or "lifp" in low:
        return "Equity_LIFP"
    if "equity_dp" in low or "_dp_" in low:
        return "Equity_DP"
    if "cost" in low:
        return "cost"
    return "unknown"


def iter_real_pathway_light_chunks(path, batch_size=250000):
    wanted = [
        "BUILD_ID", "Building_ID", "Income", "Income_Bin", "bldg_assigned_income",
        "Optimal_Strategy", "Strategy", "up_code", "Install_Decade", "Optimization_Mode"
    ]
    pf = pq.ParquetFile(path)
    cols = [c for c in wanted if c in set(pf.schema.names)]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch.to_pandas(split_blocks=True, self_destruct=True)


def summarize_state_real_retrofit_timing(decision_root, state_name, ssp, cost_cases):
    """statisticsbuilding,incomedecadeshare;mergeall county count,share."""
    baseline_strategy = "up04" if ssp == "ssp585" else "up17"


    denom_counts = defaultdict(int)


    retrofit_counts = defaultdict(int)

    for cost_case in cost_cases:
        case_root = os.path.join(decision_root, f"Noncons_CostTrend_{cost_case}")
        pattern = os.path.join(case_root, "FIPS_*", f"*_{ssp}_*_optimal_pathway_real.parquet")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"  [!] retrofit timing summary: no real files for {state_name} | {ssp} | {cost_case}")
            continue

        print(f"\n>>> Computing real-building retrofit shares: {state_name} | {ssp} | {cost_case} | files={len(files)}")

        for fp in files:
            mode_from_file = infer_mode_from_real_path(fp)

            for df in iter_real_pathway_light_chunks(fp):
                if df.empty:
                    continue


                if "Income_Bin" not in df.columns:
                    if "Income" in df.columns:
                        df["Income_Bin"] = pd.to_numeric(df["Income"], errors="coerce").apply(assign_income_bin)
                    elif "bldg_assigned_income" in df.columns:
                        df["Income_Bin"] = pd.to_numeric(df["bldg_assigned_income"], errors="coerce").apply(assign_income_bin)
                    else:
                        continue

                df = df.dropna(subset=["Income_Bin"]).copy()
                df["Income_Bin"] = df["Income_Bin"].astype(int)
                df = df[df["Income_Bin"].isin(INCOME_BIN_LABELS.keys())].copy()

                if df.empty:
                    continue

                # mode
                if "Optimization_Mode" in df.columns:
                    df["_mode"] = df["Optimization_Mode"].fillna(mode_from_file).astype(str)
                    df.loc[df["_mode"].isin(["", "nan", "None"]), "_mode"] = mode_from_file
                else:
                    df["_mode"] = mode_from_file


                if "up_code" in df.columns:
                    df["_strategy"] = df["up_code"].apply(normalize_up)
                elif "Optimal_Strategy" in df.columns:
                    df["_strategy"] = df["Optimal_Strategy"].apply(normalize_up)
                elif "Strategy" in df.columns:
                    df["_strategy"] = df["Strategy"].apply(normalize_up)
                else:
                    df["_strategy"] = baseline_strategy


                if "Install_Decade" not in df.columns:
                    df["Install_Decade"] = "2020s"
                df["Install_Decade"] = df["Install_Decade"].astype(str)

                for mode_value, sub in df.groupby("_mode", sort=False):

                    den = sub.groupby("Income_Bin").size()
                    for ib, n in den.items():
                        denom_counts[(cost_case, mode_value, int(ib))] += int(n)


                    retro = sub[
                        (sub["_strategy"] != baseline_strategy) &
                        (sub["Install_Decade"].isin(DECADES))
                    ].copy()

                    if not retro.empty:
                        cnt = retro.groupby(["Income_Bin", "Install_Decade"]).size()
                        for (ib, d), n in cnt.items():
                            retrofit_counts[(cost_case, mode_value, int(ib), str(d))] += int(n)

                del df
                gc.collect()


    rows = []
    for (cost_case, mode, ib), total_n in denom_counts.items():
        for d in DECADES:
            n_retro = retrofit_counts.get((cost_case, mode, ib, d), 0)
            rows.append({
                "state": state_name,
                "ssp": ssp,
                "cost_case": cost_case,
                "mode": mode,
                "income_bin": ib,
                "income_label": INCOME_BIN_LABELS.get(ib, str(ib)),
                "install_decade": d,
                "total_real_buildings": total_n,
                "retrofitted_buildings": n_retro,
                "retrofit_share": n_retro / total_n if total_n > 0 else np.nan,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        print(f"  [!] {state_name} | {ssp}: No real-building retrofit-share results are available.")
        return out

    income_order_labels = [INCOME_BIN_LABELS[i] for i in sorted(INCOME_BIN_LABELS.keys())]

    out["income_label"] = pd.Categorical(
        out["income_label"],
        categories=income_order_labels,
        ordered=True
    )
    out["install_decade"] = pd.Categorical(
        out["install_decade"],
        categories=DECADES,
        ordered=True
    )

    out = out.sort_values(
        ["cost_case", "mode", "income_bin", "install_decade"]
    ).reset_index(drop=True)

    out_path = os.path.join(
        decision_root,
        f"{state_name}_{ssp}_real_retrofit_timing_by_income.csv"
    )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f">>> Income- and decade-specific retrofit shares saved: {out_path}")


    print("\n" + "=" * 100)
    print(f"[STATE-MERGED REAL BUILDING RETROFIT TIMING] {state_name} | {ssp}")
    print("Share = retrofitted buildings in this install decade / all real buildings in the income group")
    print("=" * 100)

    for (cost_case, mode), g in out.groupby(["cost_case", "mode"], observed=False):
        print("\n" + "-" * 100)
        print(f"Cost case: {cost_case} | Mode: {mode}")
        print("-" * 100)

        share_tbl = (
            g.pivot_table(
                index="income_label",
                columns="install_decade",
                values="retrofit_share",
                aggfunc="first",
                observed=False
            )
            .reindex(index=income_order_labels, columns=DECADES)
            .fillna(0.0)
        )

        count_tbl = (
            g.pivot_table(
                index="income_label",
                columns="install_decade",
                values="retrofitted_buildings",
                aggfunc="first",
                observed=False
            )
            .reindex(index=income_order_labels, columns=DECADES)
            .fillna(0)
            .astype(int)
        )

        denom = (
            g[["income_label", "total_real_buildings"]]
            .drop_duplicates("income_label")
            .set_index("income_label")
            .reindex(income_order_labels)["total_real_buildings"]
            .fillna(0)
            .astype(int)
        )

        count_ratio_tbl = count_tbl.copy().astype(object)
        for inc in count_ratio_tbl.index:
            total_n = int(denom.loc[inc]) if inc in denom.index else 0
            for d in DECADES:
                count_ratio_tbl.loc[inc, d] = f"{int(count_tbl.loc[inc, d])}/{total_n}"

        print("\nRetrofit share by income group and install decade:")
        print((share_tbl * 100).round(2).astype(str) + "%")

        print("\nRetrofitted buildings / total real buildings:")
        print(count_ratio_tbl)

        cumulative_share = share_tbl.sum(axis=1).rename("cumulative_retrofit_share")
        current_share = (1.0 - cumulative_share).clip(lower=0.0).rename("not_retrofitted_share")
        summary_tbl = pd.concat([cumulative_share, current_share], axis=1)

        print("\nCumulative retrofit share across 2020s-2050s and remaining current share:")
        print((summary_tbl * 100).round(2).astype(str) + "%")
    return out

def copy_saved_fixed_mappings_for_cost_case(source_root, target_root, group_members):
    """Copy saved real-building -> template mappings into the cost-case output folder.

    This keeps ``Robust_real_mapping.py`` unchanged and prevents rebuilding
    mappings for each cost-trend case. If a saved mapping is missing, this function
    returns False and the caller skips real-building mapping for that group.
    """
    ok = True
    suffixes = [
        "fixed_real_to_template_mapping.parquet",
        "fixed_mapping_reference_report.csv",
        "fixed_mapping_distribution_check.csv",
    ]
    for fips in [str(x).zfill(5) for x in group_members]:
        src_dir = os.path.join(source_root, f"FIPS_{fips}")
        dst_dir = os.path.join(target_root, f"FIPS_{fips}")
        os.makedirs(dst_dir, exist_ok=True)
        src_mapping = os.path.join(src_dir, f"{fips}_fixed_real_to_template_mapping.parquet")
        if not os.path.exists(src_mapping):
            print(f"  [!] Saved fixed mapping not found; skipping real-building mapping: {src_mapping}")
            ok = False
            continue
        for suffix in suffixes:
            src = os.path.join(src_dir, f"{fips}_{suffix}")
            dst = os.path.join(dst_dir, f"{fips}_{suffix}")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
    return ok


def save_cost_trend_scenario_metadata(output_root):
    """Save the sampled cost-trend scenario definitions for reproducibility."""
    os.makedirs(output_root, exist_ok=True)
    meta = COST_TREND_CASE_METADATA.copy()
    if not meta.empty:
        meta = meta[meta["Cost_Trend_Case"].isin(COST_TREND_CASES_TO_RUN)].copy()
    meta_path = os.path.join(output_root, "sampled_40_cost_trend_scenarios.csv")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f">>> Cost-trend scenario definitions saved: {meta_path}")
    return meta_path


# 2.5 Cost-trend visualization for all retrofit strategies


def build_strategy_cost_trend_table(ssps=SSPS, decades=DECADES, cost_cases=COST_TREND_CASES_TO_RUN):
    """Build cost-trend multiplier table for all non-baseline retrofit strategies.

    Each strategy starts with unit CAPEX, is split into active and passive
    components using the annualized-cost module, and then receives the
    installation-decade multipliers for the selected cost-trend case.
    The default strategy set contains 35 retrofit options: 16 under SSP1-2.6,
    16 under SSP2-4.5, and 3 under SSP5-8.5.
    """
    records = []
    strategy_order = []
    for ssp in ssps:
        ssp_key = str(ssp).lower()
        strategy_map = STRATEGY_COMPONENTS_BY_SSP.get(ssp_key, {})
        baseline = 'up04' if ssp_key == 'ssp585' else 'up17'
        for strategy in sorted(strategy_map.keys(), key=lambda x: int(str(x).replace('up', '')) if str(x).replace('up', '').isdigit() else 999):
            components = [c for c in strategy_map.get(strategy, []) if c]

            if strategy == baseline or len(components) == 0:
                continue
            strategy_id = f"{ssp_key}_{strategy}"
            strategy_order.append(strategy_id)
            base_split = split_capex_by_component(1.0, components, ssp=ssp_key)
            active_share = sum(v for c, v in base_split.items() if component_cost_class(c) == 'active')
            passive_share = sum(v for c, v in base_split.items() if component_cost_class(c) == 'passive')
            for cost_case in cost_cases:
                for decade in decades:
                    multiplier = sum(
                        float(v) * cost_multiplier_for_component(c, install_decade=decade, cost_trend_case=cost_case)
                        for c, v in base_split.items()
                    )
                    records.append({
                        'SSP': ssp_key,
                        'Strategy': strategy,
                        'Strategy_ID': strategy_id,
                        'Install_Decade': decade,
                        'Cost_Trend_Case': cost_case,
                        'CAPEX_Multiplier_vs_2020s': multiplier,
                        'Active_CAPEX_Share': active_share,
                        'Passive_CAPEX_Share': passive_share,
                        'Num_Components': len(components),
                        'Components': ';'.join(components),
                    })
    out = pd.DataFrame(records)
    out['Strategy_ID'] = pd.Categorical(out['Strategy_ID'], categories=strategy_order, ordered=True)
    out['Install_Decade'] = pd.Categorical(out['Install_Decade'], categories=list(decades), ordered=True)
    out['Cost_Trend_Case'] = pd.Categorical(out['Cost_Trend_Case'], categories=list(cost_cases), ordered=True)
    return out.sort_values(['Cost_Trend_Case', 'SSP', 'Strategy_ID', 'Install_Decade']).reset_index(drop=True)


def plot_strategy_cost_trend_changes(output_dir=None, ssps=SSPS, decades=DECADES, cost_cases=COST_TREND_CASES_TO_RUN):
    """Save CSV + heatmaps showing future CAPEX changes for all 35 retrofit strategies."""
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, '#R2', 'Fig costtrend')
    os.makedirs(output_dir, exist_ok=True)

    trend_df = build_strategy_cost_trend_table(ssps=ssps, decades=decades, cost_cases=cost_cases)
    csv_path = os.path.join(output_dir, 'all_35_strategy_cost_trend_multipliers.csv')
    trend_df.to_csv(csv_path, index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        strategies = list(trend_df['Strategy_ID'].cat.categories)
        case_titles = {
            'active_up_passive_up': 'Active up / Passive up',
            'active_down_passive_up': 'Active down / Passive up',
            'active_up_passive_down': 'Active up / Passive down',
            'active_down_passive_down': 'Active down / Passive down',
        }

        # Combined dynamic heatmap grid. With the new design this is 40 cases,
        # so use a compact multi-row layout instead of the old fixed 2x2 panel.
        n_cases = len(list(cost_cases))
        ncols = 5 if n_cases > 4 else 2
        nrows = int(np.ceil(n_cases / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows), constrained_layout=True)
        axes = np.asarray(axes).ravel()
        vmin = trend_df['CAPEX_Multiplier_vs_2020s'].min()
        vmax = trend_df['CAPEX_Multiplier_vs_2020s'].max()
        im = None
        for ax, cost_case in zip(axes, cost_cases):
            sub = trend_df[trend_df['Cost_Trend_Case'].astype(str) == cost_case]
            mat = (
                sub.pivot(index='Strategy_ID', columns='Install_Decade', values='CAPEX_Multiplier_vs_2020s')
                .reindex(index=strategies, columns=decades)
            )
            im = ax.imshow(mat.values, aspect='auto', vmin=vmin, vmax=vmax, cmap='RdBu_r')
            ax.set_title(case_titles.get(cost_case, cost_case), fontsize=13)
            ax.set_xticks(range(len(decades)))
            ax.set_xticklabels(decades, rotation=0)
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels(strategies, fontsize=7)
            ax.set_xlabel('Installation decade')
            ax.set_ylabel('Strategy')
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat.values[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5)
        for ax in axes[len(list(cost_cases)):]:
            ax.axis('off')
        fig.colorbar(im, ax=axes.tolist(), shrink=0.65, label='CAPEX multiplier vs 2020s')
        combined_path = os.path.join(output_dir, 'all_35_strategy_cost_trend_heatmaps.png')
        fig.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # One larger heatmap per case, easier to inspect in papers/reports.
        for cost_case in cost_cases:
            sub = trend_df[trend_df['Cost_Trend_Case'].astype(str) == cost_case]
            mat = (
                sub.pivot(index='Strategy_ID', columns='Install_Decade', values='CAPEX_Multiplier_vs_2020s')
                .reindex(index=strategies, columns=decades)
            )
            fig, ax = plt.subplots(figsize=(8, 12), constrained_layout=True)
            im = ax.imshow(mat.values, aspect='auto', vmin=vmin, vmax=vmax, cmap='RdBu_r')
            ax.set_title(case_titles.get(cost_case, cost_case), fontsize=13)
            ax.set_xticks(range(len(decades)))
            ax.set_xticklabels(decades)
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels(strategies, fontsize=7)
            ax.set_xlabel('Installation decade')
            ax.set_ylabel('Strategy')
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat.values[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5)
            fig.colorbar(im, ax=ax, shrink=0.8, label='CAPEX multiplier vs 2020s')
            fig.savefig(os.path.join(output_dir, f'all_35_strategy_cost_trend_heatmap_{cost_case}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f">>> 35strategy cost-trend table saved: {csv_path}")
        print(f">>> 35strategy cost-trend heatmap saved: {combined_path}")
    except Exception as e:
        print(f"  [!] Cost-trend plotting failed, but the CSV was saved: {csv_path}. Error: {e}")

    return trend_df


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

        state_r2_root = get_state_r2_root(BASE_DIR, state_name)

        # All new prediction outputs are saved here.
        decision_root = os.path.join(state_r2_root, DECISION_OUTPUT_FOLDER)

        # Fixed real-building mappings are read from the original Decision_Robust folder.
        # They were created by the baseline/main mapping workflow and should not be
        # expected to already exist under Decision_Pre.
        fixed_mapping_source_root = os.path.join(state_r2_root, FIXED_MAPPING_SOURCE_FOLDER)

        os.makedirs(decision_root, exist_ok=True)
        save_cost_trend_scenario_metadata(decision_root)

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


        for cost_case in COST_TREND_CASES_TO_RUN:
            decision_case_root = os.path.join(decision_root, f"Noncons_CostTrend_{cost_case}")


            os.makedirs(decision_case_root, exist_ok=True)

            print("\n" + "=" * 80)
            print(f">>> cost case: {cost_case}")
            print("=" * 80)

            for gi, g in enumerate(fips_groups, start=1):
                group_id = g['group_id']
                members = g['members']
                print("\n" + "-" * 80)
                print(f">>> Process {group_id} | members={members} | valid_group={g['valid_group']} | cost_case={cost_case}")
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
                    use_all_dir=use_all_dir_for_state(state_name),
                    cost_trend_case=cost_case
                )

                group_id_safe = f"GROUP_{gi:03d}"
                group_output_dir = os.path.join(decision_case_root, "Group_Template", group_id_safe)
                os.makedirs(group_output_dir, exist_ok=True)
                group_output_path = os.path.join(
                    group_output_dir,
                    f"{group_id_safe}_{ssp}_{cost_case}_optimal_pathway.parquet"
                )

                if SKIP_EXISTING_GROUP_TEMPLATE_RESULTS and os.path.exists(group_output_path):
                    print(f">>> Reusing existing group-level results: {group_output_path}")
                    final_combined_df = pd.read_parquet(group_output_path)
                else:
                    final_combined_df = solve_all_modes_for_group(df_group, optimizer, ssp, group_id)
                    if final_combined_df is None or final_combined_df.empty:
                        continue
                    final_combined_df['Cost_Trend_Case'] = cost_case
                    final_combined_df.to_parquet(group_output_path, engine='pyarrow', compression='brotli', index=False)
                    print(f">>> Group-level template optimization results saved: {group_output_path}")

                if copy_saved_fixed_mappings_for_cost_case(fixed_mapping_source_root, decision_case_root, members):
                    map_group_results_to_original_counties(
                        final_combined_df=final_combined_df,
                        state_map=state_map,
                        group_members=members,
                        output_root=decision_case_root,
                        ssp=ssp,
                        optimizer=optimizer
                    )
                else:
                    print("  [!] Real-building mapping was not run for this group;Create the fixed mapping from the original Decision_Robust results first.")
        summarize_state_real_retrofit_timing(
            decision_root=decision_root,
            state_name=state_name,
            ssp=ssp,
            cost_cases=COST_TREND_CASES_TO_RUN
        )
print("\n" + "#" * 70)
print("Equity-oriented dynamic retrofit timing + cost-trend fixed-mapping Workflow completed.")
print("#" * 70)
