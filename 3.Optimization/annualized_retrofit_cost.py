"""
Reusable annualized retrofit-cost module for 2020-2060 residential transition analysis.

Core idea:
- Split one strategy's total retrofit CAPEX into physical components by approximate cost weights.
- Assign each component its own service life.
- If a component's lifetime expires before 2060, add replacement cycles.
- Return decade-specific average annualized CAPEX, which can be added to annual energy cost.
- Also return decade-specific cash CAPEX events: initial retrofit CAPEX in the install decade and replacement CAPEX in later decades.

Typical usage:
    from annualized_retrofit_cost import add_annualized_capex_columns

    # In this project, Install_cost_2020s ... Install_cost_2050s
    # are four portions of the one-time retrofit CAPEX implemented in the 2020s.
    # The default behavior therefore sums all Install_cost_* columns.
    df = add_annualized_capex_columns(
        df,
        ssp="ssp245",
        strategy_col="Strategy",
        r=0.02,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DECADE_START: Dict[str, int] = {
    "2020s": 2020,
    "2030s": 2030,
    "2040s": 2040,
    "2050s": 2050,
}

DEFAULT_HORIZON_START = 2020
DEFAULT_HORIZON_END = 2060


@dataclass(frozen=True)
class ComponentSpec:
    life: int
    weight: float


# Approximate service lives. These are intended as a transparent baseline,
# not as immutable engineering constants.
COMPONENT_LIFE: Dict[str, int] = {
    "roof_upgrade_plus_1": 40, "roof_upgrade_plus_2": 40, "roof_upgrade_iecc": 40,
    "pkg_wall_up1": 40, "pkg_wall_up2": 40, "pkg_wall_iecc": 40,
    "pkg_windows_best_no_infil": 30, "pkg_windows_energystar_standard": 30, "pkg_windows_energystar_standard_non_airtightness": 30,
    "pkg_final_infil_reduce_13": 20, "pkg_final_infil_reduce_20": 20, "pkg_final_infil_reduce_30": 20, "pkg_final_infil_iecc_5": 20,
    "pkg_min_eff_hvac": 15, "pkg_hvac_common_elec": 15, "pkg_hvac_furnace_high_eff": 15, "pkg_hvac_deep_cchp": 15,
    "rooftop PV 5Kw": 25, "rooftop PV 11Kw": 25,
    "Battery 10 kWh": 10, "Battery 20 kWh": 10,
}

# Component weights are assumed relative shares used only to split
# strategy-level total CAPEX into component-level costs.
# The total CAPEX still comes from the simulation/Excel data and is not changed.
# Because component-level costs are unavailable, these shares should be treated
# as transparent modeling assumptions and tested in sensitivity analysis.
COMPONENT_WEIGHT_BY_SSP = {
    "ssp585": {
        "roof_upgrade_plus_1": 0.594314405,
        "pkg_wall_up1": 0.523752534,
        "pkg_final_infil_reduce_13": 0.145401832,
        "pkg_min_eff_hvac": 2.020103941,
    },

    "ssp245": {
        "pkg_windows_best_no_infil": 1.654813102,
        "roof_upgrade_plus_2": 1.084682142,
        "pkg_wall_up2": 0.623386798,
        "pkg_final_infil_reduce_13": 0.145401832,
        "pkg_final_infil_reduce_20": 0.171604272,
        "pkg_min_eff_hvac": 2.020103941,
        "pkg_hvac_common_elec": 4.292095212,
        "rooftop PV 5Kw": 5.87368708,
        "Battery 10 kWh": 4.964847847,
    },

    "ssp126": {
        "pkg_windows_energystar_standard": 5.825044486,
        "pkg_windows_energystar_standard_non_airtightness": 5.825044486,
        "roof_upgrade_iecc": 1.581492394,
        "pkg_wall_iecc": 1.363828859,
        "pkg_final_infil_reduce_30": 0.209038878,
        "pkg_final_infil_iecc_5": 0.399889263,
        "pkg_hvac_furnace_high_eff": 1,
        "pkg_hvac_deep_cchp": 4.902923171,
        "rooftop PV 11Kw": 11.05374907,
        "Battery 20 kWh": 9.359612002,
    },
}
STRATEGY_COMPONENTS_BY_SSP: Dict[str, Dict[str, List[str]]] = {
    "ssp585": {
        "up00": [],
        "up01": ["pkg_wall_up1"],
        "up02": ["pkg_wall_up1", "pkg_min_eff_hvac"],
        "up03": ["roof_upgrade_plus_1", "pkg_wall_up1", "pkg_final_infil_reduce_13", "pkg_min_eff_hvac"],
        "up04": [],
    },
    "ssp245": {
        "up00": [],
        "up01": ["roof_upgrade_plus_2", "pkg_final_infil_reduce_13", "pkg_min_eff_hvac"],
        "up02": ["pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_min_eff_hvac"],
        "up03": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_min_eff_hvac"],
        "up04": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_windows_best_no_infil", "pkg_final_infil_reduce_20", "pkg_min_eff_hvac"],
        "up05": ["roof_upgrade_plus_2", "pkg_final_infil_reduce_13", "pkg_hvac_common_elec"],
        "up06": ["pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec"],
        "up07": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec"],
        "up08": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_windows_best_no_infil", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec"],
        "up09": ["roof_upgrade_plus_2", "pkg_final_infil_reduce_13", "pkg_hvac_common_elec", "rooftop PV 5Kw"],
        "up10": ["pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec", "rooftop PV 5Kw"],
        "up11": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec", "rooftop PV 5Kw"],
        "up12": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_windows_best_no_infil", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec", "rooftop PV 5Kw"],
        "up13": ["roof_upgrade_plus_2", "pkg_final_infil_reduce_13", "pkg_hvac_common_elec", "rooftop PV 5Kw", "Battery 10 kWh"],
        "up14": ["pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec", "rooftop PV 5Kw", "Battery 10 kWh"],
        "up15": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec", "rooftop PV 5Kw", "Battery 10 kWh"],
        "up16": ["roof_upgrade_plus_2", "pkg_wall_up2", "pkg_windows_best_no_infil", "pkg_final_infil_reduce_20", "pkg_hvac_common_elec", "rooftop PV 5Kw", "Battery 10 kWh"],
        "up17": [],
    },
    "ssp126": {
        "up00": [],
        "up01": ["pkg_windows_energystar_standard_non_airtightness", "pkg_hvac_furnace_high_eff"],
        "up02": ["roof_upgrade_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_furnace_high_eff"],
        "up03": ["pkg_wall_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_furnace_high_eff"],
        "up04": ["roof_upgrade_iecc", "pkg_wall_iecc", "pkg_windows_energystar_standard_non_airtightness", "pkg_final_infil_iecc_5", "pkg_hvac_furnace_high_eff"],
        "up05": ["pkg_windows_energystar_standard", "pkg_hvac_deep_cchp"],
        "up06": ["roof_upgrade_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_deep_cchp"],
        "up07": ["pkg_wall_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_deep_cchp"],
        "up08": ["roof_upgrade_iecc", "pkg_wall_iecc", "pkg_windows_energystar_standard_non_airtightness", "pkg_final_infil_iecc_5", "pkg_hvac_deep_cchp"],
        "up09": ["pkg_windows_energystar_standard", "pkg_hvac_deep_cchp", "rooftop PV 11Kw"],
        "up10": ["roof_upgrade_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_deep_cchp", "rooftop PV 11Kw"],
        "up11": ["pkg_wall_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_deep_cchp", "rooftop PV 11Kw"],
        "up12": ["roof_upgrade_iecc", "pkg_wall_iecc", "pkg_windows_energystar_standard_non_airtightness", "pkg_final_infil_iecc_5", "pkg_hvac_deep_cchp", "rooftop PV 11Kw"],
        "up13": ["pkg_windows_energystar_standard", "pkg_hvac_deep_cchp", "rooftop PV 11Kw", "Battery 20 kWh"],
        "up14": ["roof_upgrade_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_deep_cchp", "rooftop PV 11Kw", "Battery 20 kWh"],
        "up15": ["pkg_wall_iecc", "pkg_final_infil_reduce_30", "pkg_hvac_deep_cchp", "rooftop PV 11Kw", "Battery 20 kWh"],
        "up16": ["roof_upgrade_iecc", "pkg_wall_iecc", "pkg_windows_energystar_standard_non_airtightness", "pkg_final_infil_iecc_5", "pkg_hvac_deep_cchp", "rooftop PV 11Kw", "Battery 20 kWh"],
        "up17": [],
    },
}


def capital_recovery_factor(r: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if abs(r) < 1e-12:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def normalize_strategy_name(strategy: object) -> str:
    s = str(strategy).strip().lower()
    return s.replace("upgrade", "up").replace(" ", "")


def infer_components(strategy: object, ssp: Optional[str] = None) -> List[str]:
    s = normalize_strategy_name(strategy)
    ssp_key = str(ssp).lower() if ssp is not None else None

    if ssp_key in STRATEGY_COMPONENTS_BY_SSP and s in STRATEGY_COMPONENTS_BY_SSP[ssp_key]:
        return list(STRATEGY_COMPONENTS_BY_SSP[ssp_key][s])

    name = s
    components: List[str] = []
    if "roof" in name:
        components.append("roof")
    if "wall" in name:
        components.append("wall")
    if "win" in name:
        components.append("window")
    if "envelope" in name or "full" in name:
        components.extend(["roof", "wall", "window", "infiltration"])
    if "cchp" in name:
        components.append("cchp_hvac")
    elif "elec" in name:
        components.append("elec_hvac")
    elif "hvac" in name or "fossil" in name:
        components.append("fossil_hvac")
    if "pv11" in name:
        components.append("pv11")
    elif "pv5" in name or "pv" in name:
        components.append("pv5")
    if "b20" in name:
        components.append("battery20")
    elif "b10" in name or "battery" in name:
        components.append("battery10")

    return sorted(set(components))


def split_capex_by_component(
    total_capex: float,
    components: Sequence[str],
    ssp: Optional[str] = None,
    custom_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    total_capex = 0.0 if pd.isna(total_capex) else float(total_capex)
    components = [c for c in components if c]
    if total_capex == 0.0 or not components:
        return {}

    ssp_key = str(ssp).lower() if ssp is not None else "ssp245"
    weights = dict(COMPONENT_WEIGHT_BY_SSP.get(ssp_key, COMPONENT_WEIGHT_BY_SSP["ssp245"]))
    if custom_weights is not None:
        weights.update({k: float(v) for k, v in custom_weights.items()})

    selected_weights = {c: max(float(weights.get(c, 1.0)), 0.0) for c in components}
    weight_sum = sum(selected_weights.values())
    if weight_sum <= 0:
        equal = total_capex / len(components)
        return {c: equal for c in components}

    return {c: total_capex * w / weight_sum for c, w in selected_weights.items()}


def component_annualized_payment(
    component_capex: float,
    component: str,
    r: float = 0.02,
    custom_life: Optional[Mapping[str, int]] = None,
) -> float:
    life_map = dict(COMPONENT_LIFE)
    if custom_life is not None:
        life_map.update({k: int(v) for k, v in custom_life.items()})
    life = life_map.get(component, 20)
    return float(component_capex) * capital_recovery_factor(r, life)


def annualized_capex_for_year(
    total_capex: float,
    strategy: object,
    year: int,
    ssp: Optional[str] = None,
    r: float = 0.02,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
) -> float:
    if year < install_year or year >= horizon_end:
        return 0.0

    components = infer_components(strategy, ssp=ssp)
    capex_by_component = split_capex_by_component(
        total_capex=total_capex,
        components=components,
        ssp=ssp,
        custom_weights=custom_weights,
    )
    if not capex_by_component:
        return 0.0

    life_map = dict(COMPONENT_LIFE)
    if custom_life is not None:
        life_map.update({k: int(v) for k, v in custom_life.items()})

    total = 0.0
    for component, component_capex in capex_by_component.items():
        life = life_map.get(component, 20)
        cycle_index = int((year - install_year) // life)
        cycle_start = install_year + cycle_index * life
        if cycle_start <= year < min(cycle_start + life, horizon_end):
            cycle_capex = component_capex * (replacement_cost_factor ** cycle_index)
            total += cycle_capex * capital_recovery_factor(r, life)
    return float(total)


def annualized_capex_for_decade(
    total_capex: float,
    strategy: object,
    decade: str,
    ssp: Optional[str] = None,
    r: float = 0.02,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
) -> float:
    if decade not in DECADE_START:
        raise ValueError(f"Unknown decade: {decade}. Expected one of {list(DECADE_START)}")

    start = DECADE_START[decade]
    years = range(start, start + 10)
    values = [
        annualized_capex_for_year(
            total_capex=total_capex,
            strategy=strategy,
            year=y,
            ssp=ssp,
            r=r,
            install_year=install_year,
            horizon_end=horizon_end,
            replacement_cost_factor=replacement_cost_factor,
            custom_weights=custom_weights,
            custom_life=custom_life,
        )
        for y in years
        if install_year <= y < horizon_end
    ]
    return float(np.mean(values)) if values else 0.0


def annualized_capex_series_by_decade(
    total_capex: float,
    strategy: object,
    decades: Iterable[str] = ("2020s", "2030s", "2040s", "2050s"),
    ssp: Optional[str] = None,
    r: float = 0.02,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
) -> Dict[str, float]:
    return {
        d: annualized_capex_for_decade(
            total_capex=total_capex,
            strategy=strategy,
            decade=d,
            ssp=ssp,
            r=r,
            install_year=install_year,
            horizon_end=horizon_end,
            replacement_cost_factor=replacement_cost_factor,
            custom_weights=custom_weights,
            custom_life=custom_life,
        )
        for d in decades
    }


def capex_event_cost_for_year(
    total_capex: float,
    strategy: object,
    year: int,
    ssp: Optional[str] = None,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
) -> float:
    """Return cash CAPEX occurring in a specific year.

    The install year records the initial upfront retrofit CAPEX. Later years record
    replacement CAPEX only when a component reaches the end of its service life.
    This is a cash-flow measure, not an annualized payment.
    """
    if year < install_year or year >= horizon_end:
        return 0.0

    components = infer_components(strategy, ssp=ssp)
    capex_by_component = split_capex_by_component(
        total_capex=total_capex,
        components=components,
        ssp=ssp,
        custom_weights=custom_weights,
    )
    if not capex_by_component:
        return 0.0

    life_map = dict(COMPONENT_LIFE)
    if custom_life is not None:
        life_map.update({k: int(v) for k, v in custom_life.items()})

    total = 0.0
    for component, component_capex in capex_by_component.items():
        life = life_map.get(component, 20)
        if life <= 0:
            continue
        if (year - install_year) % life != 0:
            continue
        cycle_index = int((year - install_year) // life)
        cycle_capex = float(component_capex) * (float(replacement_cost_factor) ** cycle_index)
        total += cycle_capex
    return float(total)


def capex_event_cost_for_decade(
    total_capex: float,
    strategy: object,
    decade: str,
    ssp: Optional[str] = None,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
) -> float:
    """Return total cash retrofit/replacement CAPEX occurring within one decade.

    Example with default install_year=2020:
    - 2020s includes the initial upfront retrofit CAPEX.
    - 2030s, 2040s, and 2050s include only component replacements whose service
      lives expire in those decades.
    """
    if decade not in DECADE_START:
        raise ValueError(f"Unknown decade: {decade}. Expected one of {list(DECADE_START)}")

    start = DECADE_START[decade]
    years = range(start, start + 10)
    return float(
        sum(
            capex_event_cost_for_year(
                total_capex=total_capex,
                strategy=strategy,
                year=y,
                ssp=ssp,
                install_year=install_year,
                horizon_end=horizon_end,
                replacement_cost_factor=replacement_cost_factor,
                custom_weights=custom_weights,
                custom_life=custom_life,
            )
            for y in years
            if install_year <= y < horizon_end
        )
    )


def capex_event_series_by_decade(
    total_capex: float,
    strategy: object,
    decades: Iterable[str] = ("2020s", "2030s", "2040s", "2050s"),
    ssp: Optional[str] = None,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
) -> Dict[str, float]:
    return {
        d: capex_event_cost_for_decade(
            total_capex=total_capex,
            strategy=strategy,
            decade=d,
            ssp=ssp,
            install_year=install_year,
            horizon_end=horizon_end,
            replacement_cost_factor=replacement_cost_factor,
            custom_weights=custom_weights,
            custom_life=custom_life,
        )
        for d in decades
    }


def choose_total_capex(
    row: pd.Series,
    total_capex_col: Optional[str] = None,
    install_cost_cols: Optional[Sequence[str]] = None,
    mode: str = "sum",
) -> float:
    """Return the upfront CAPEX used for annualization.

    Project convention:
    - Install_cost_2020s, Install_cost_2030s, Install_cost_2040s,
      and Install_cost_2050s are four stored portions of one retrofit
      implemented in the 2020s.
    - Therefore the default behavior is to sum all Install_cost_* columns.

    If total_capex_col is supplied, that column is treated as an already
    reconstructed total upfront CAPEX and takes priority.
    """
    if total_capex_col and total_capex_col in row.index and pd.notna(row[total_capex_col]):
        return float(row[total_capex_col])

    if install_cost_cols is None:
        install_cost_cols = [c for c in row.index if str(c).startswith("Install_cost_")]

    vals = [float(row[c]) for c in install_cost_cols if c in row.index and pd.notna(row[c])]
    if not vals:
        return 0.0
    if mode == "sum":
        return float(sum(vals))
    if mode == "max":
        return float(max(vals))
    if mode == "first_available":
        return float(vals[0])
    raise ValueError("mode must be one of: 'sum', 'max', 'first_available'")


def add_annualized_capex_columns(
    df: pd.DataFrame,
    ssp: str,
    strategy_col: str = "Strategy",
    total_capex_col: Optional[str] = None,
    install_cost_cols: Optional[Sequence[str]] = None,
    total_capex_mode: str = "sum",
    decades: Sequence[str] = ("2020s", "2030s", "2040s", "2050s"),
    r: float = 0.02,
    install_year: int = DEFAULT_HORIZON_START,
    horizon_end: int = DEFAULT_HORIZON_END,
    replacement_cost_factor: float = 0.8,
    custom_weights: Optional[Mapping[str, float]] = None,
    custom_life: Optional[Mapping[str, int]] = None,
    output_prefix: str = "Annualized_CAPEX",
    capex_event_prefix: str = "RetrofitReplacement_CAPEX",
    keep_component_columns: bool = False,
) -> pd.DataFrame:
    if strategy_col not in df.columns:
        raise KeyError(f"strategy_col not found: {strategy_col}")

    out = df.copy()
    out["Total_CAPEX_for_annualization"] = out.apply(
        lambda row: choose_total_capex(
            row,
            total_capex_col=total_capex_col,
            install_cost_cols=install_cost_cols,
            mode=total_capex_mode,
        ),
        axis=1,
    )

    if keep_component_columns:
        out["Retrofit_Components"] = out[strategy_col].apply(lambda x: ",".join(infer_components(x, ssp=ssp)))

    for d in decades:
        out[f"{output_prefix}_{d}"] = out.apply(
            lambda row: annualized_capex_for_decade(
                total_capex=row["Total_CAPEX_for_annualization"],
                strategy=row[strategy_col],
                decade=d,
                ssp=ssp,
                r=r,
                install_year=install_year,
                horizon_end=horizon_end,
                replacement_cost_factor=replacement_cost_factor,
                custom_weights=custom_weights,
                custom_life=custom_life,
            ),
            axis=1,
        )
        out[f"{capex_event_prefix}_{d}"] = out.apply(
            lambda row: capex_event_cost_for_decade(
                total_capex=row["Total_CAPEX_for_annualization"],
                strategy=row[strategy_col],
                decade=d,
                ssp=ssp,
                install_year=install_year,
                horizon_end=horizon_end,
                replacement_cost_factor=replacement_cost_factor,
                custom_weights=custom_weights,
                custom_life=custom_life,
            ),
            axis=1,
        )

    out[f"Total_{capex_event_prefix}"] = out[[f"{capex_event_prefix}_{d}" for d in decades]].sum(axis=1)

    return out


def add_total_annual_cost_columns(
    df: pd.DataFrame,
    decades: Sequence[str] = ("2020s", "2030s", "2040s", "2050s"),
    annualized_capex_prefix: str = "Annualized_CAPEX",
    energy_cost_prefix: str = "Energy_cost",
    output_prefix: str = "Annual_Total_Cost",
) -> pd.DataFrame:
    out = df.copy()
    for d in decades:
        capex_col = f"{annualized_capex_prefix}_{d}"
        energy_col = f"{energy_cost_prefix}_{d}"
        if capex_col not in out.columns:
            raise KeyError(f"Missing column: {capex_col}")
        if energy_col not in out.columns:
            raise KeyError(f"Missing column: {energy_col}")
        out[f"{output_prefix}_{d}"] = out[capex_col] + out[energy_col]
    return out


if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "Building_ID": [1, 2, 3],
            "Strategy": ["up01", "up09", "up16"],
            "Install_cost_2020s": [2500, 5000, 8750],
            "Install_cost_2030s": [2500, 5000, 8750],
            "Install_cost_2040s": [2500, 5000, 8750],
            "Install_cost_2050s": [2500, 5000, 8750],
            "Energy_cost_2020s": [1800, 1500, 1200],
            "Energy_cost_2030s": [1700, 1400, 1100],
            "Energy_cost_2040s": [1600, 1300, 1000],
            "Energy_cost_2050s": [1500, 1200, 900],
        }
    )
    demo = add_annualized_capex_columns(demo, ssp="ssp245", keep_component_columns=True)
    demo = add_total_annual_cost_columns(demo)
    print(demo)
