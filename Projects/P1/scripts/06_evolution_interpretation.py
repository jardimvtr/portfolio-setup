from __future__ import annotations

"""
SCRIPT 06 v1.1 — CLUSTER PROFILES, FUZZY TRANSITION, TEMPORAL CONVERGENCE,
COUNTRY REFERENCES AND GLOBAL-vs-LOCAL INTERPRETATION

Purpose
-------
Interpret the frozen Script-05 M-Exp-FCMd solution without re-estimating the
clustering model.

Frozen P1 design expected by this script
----------------------------------------
- Analytical period: 2015–2019.
- Official sample: 512 regions in 28 countries.
- Dynamic-only M-Exp-FCMd clustering.
- Frozen global solution: C=2 and m=1.5.
- Structural/static variables remain excluded.

Main outputs
------------
1) Membership-weighted profiles of C1 and C2 using the five final clustering
   features and their annual trajectories.
2) Fuzzy-transition diagnostics based on membership margin.
3) Annual region-to-medoid distances and 2015→2019 convergence toward the
   alternative global profile.
4) Exact feature/family decomposition of the convergence index.
5) Region-to-country-reference comparisons in the same dynamic analytical
   space.
6) Ex-post comparison between global and viable intracountry partitions.

Important interpretation rule
-----------------------------
A region keeps one global cluster assignment for the full 2015–2019 period.
C1→C2 or C2→C1 in the temporal-convergence outputs means movement toward the
alternative medoid profile, not formal cluster reassignment.
"""

from pathlib import Path
import json
import math
import time
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
SCRIPT04_DIR = INTERIM_DIR / "script04"
SCRIPT05_DIR = INTERIM_DIR / "script05"
SCRIPT06_DIR = INTERIM_DIR / "script06"
OUTPUTS_DIR = ROOT / "outputs"

DYNAMIC_INPUT = SCRIPT04_DIR / "04_final_dynamic_model_panel_2015_2019.csv"
FEATURE_WEIGHTS_INPUT = SCRIPT04_DIR / "04_final_feature_weights.csv"
COUNTRY_REFERENCE_INPUT = SCRIPT04_DIR / "04_country_reference_dynamic_final.csv"

GLOBAL_PARTITION_INPUT = SCRIPT05_DIR / "05_global_partition.csv"
GLOBAL_MEDOIDS_INPUT = SCRIPT05_DIR / "05_global_medoids.csv"
COUNTRY_PARTITION_INPUT = SCRIPT05_DIR / "05_country_partition.csv"
COUNTRY_SUMMARY_INPUT = SCRIPT05_DIR / "05_country_cluster_summary.csv"
SELECTED_CONFIG_INPUT = SCRIPT05_DIR / "05_selected_configuration.json"

EXPECTED_REGIONS = 512
EXPECTED_COUNTRIES = 28
EXPECTED_YEARS = [2015, 2016, 2017, 2018, 2019]
EXPECTED_C = 2
EXPECTED_M = 1.5
TRANSITION_MARGIN_THRESHOLD = 0.20
CONVERGENCE_NUMERIC_TOL = 1e-12
EPS = 1e-15

# Script 06 outputs
PROFILE_SUMMARY_OUTPUT = SCRIPT06_DIR / "06_cluster_profile_summary.csv"
PROFILE_ANNUAL_OUTPUT = SCRIPT06_DIR / "06_cluster_profile_annual.csv"
PROFILE_COMPARISON_OUTPUT = SCRIPT06_DIR / "06_cluster_profile_comparison.csv"
TRANSITION_REGIONS_OUTPUT = SCRIPT06_DIR / "06_fuzzy_transition_regions.csv"
TRANSITION_SUMMARY_OUTPUT = SCRIPT06_DIR / "06_fuzzy_transition_summary.csv"
ANNUAL_MEDOID_DISTANCE_OUTPUT = SCRIPT06_DIR / "06_region_annual_medoid_distances.csv"
CONVERGENCE_REGION_OUTPUT = SCRIPT06_DIR / "06_region_convergence_2015_2019.csv"
CONVERGENCE_FLOW_OUTPUT = SCRIPT06_DIR / "06_convergence_flow_summary.csv"
CONTRIB_FEATURE_OUTPUT = SCRIPT06_DIR / "06_convergence_feature_contributions.csv"
CONTRIB_FAMILY_OUTPUT = SCRIPT06_DIR / "06_convergence_family_contributions.csv"
COUNTRY_DISTANCE_OUTPUT = SCRIPT06_DIR / "06_region_country_reference_distances.csv"
COUNTRY_NEAREST_OUTPUT = SCRIPT06_DIR / "06_region_country_reference_nearest.csv"
GLOBAL_LOCAL_REGION_OUTPUT = SCRIPT06_DIR / "06_global_local_region_comparison.csv"
GLOBAL_LOCAL_COUNTRY_OUTPUT = SCRIPT06_DIR / "06_global_local_country_summary.csv"
METADATA_OUTPUT = SCRIPT06_DIR / "06_analysis_metadata.csv"
RESULTS_WORKBOOK_OUTPUT = OUTPUTS_DIR / "06_evolution_interpretation.xlsx"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ensure_directories() -> None:
    SCRIPT06_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / total
    return float(np.interp(quantile, cumulative, values))


def weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    if not np.isfinite(mu):
        return float("nan")
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[mask]
    weights = weights[mask]
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(weights * (values - mu) ** 2) / total))


def adjusted_rand_index(labels_a: Iterable, labels_b: Iterable) -> float:
    a = np.asarray(list(labels_a))
    b = np.asarray(list(labels_b))
    if len(a) != len(b):
        raise ValueError("ARI label vectors must have equal length.")
    if len(a) < 2:
        return float("nan")

    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    contingency = np.zeros((len(a_vals), len(b_vals)), dtype=int)
    np.add.at(contingency, (a_inv, b_inv), 1)

    def comb2(x: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(x) * (np.asarray(x) - 1.0) / 2.0

    sum_comb = float(np.sum(comb2(contingency)))
    row_comb = float(np.sum(comb2(contingency.sum(axis=1))))
    col_comb = float(np.sum(comb2(contingency.sum(axis=0))))
    total_comb = float(comb2(len(a)))
    if total_comb == 0:
        return float("nan")
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) <= EPS:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((sum_comb - expected) / denom)


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_required_inputs() -> None:
    required = {
        "DYNAMIC_PANEL": DYNAMIC_INPUT,
        "FEATURE_WEIGHTS": FEATURE_WEIGHTS_INPUT,
        "COUNTRY_REFERENCE": COUNTRY_REFERENCE_INPUT,
        "GLOBAL_PARTITION": GLOBAL_PARTITION_INPUT,
        "GLOBAL_MEDOIDS": GLOBAL_MEDOIDS_INPUT,
        "COUNTRY_PARTITION": COUNTRY_PARTITION_INPUT,
        "COUNTRY_SUMMARY": COUNTRY_SUMMARY_INPUT,
        "SELECTED_CONFIG": SELECTED_CONFIG_INPUT,
    }
    missing = []
    for label, path in required.items():
        if path.exists():
            print(f"[FOUND]   {label:<20} {path.name}")
        else:
            print(f"[MISSING] {label:<20} {path}")
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Script 06 cannot start because required handoff files are missing: "
            + "; ".join(missing)
        )


def load_and_validate_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    list[str],
]:
    dynamic = read_csv(DYNAMIC_INPUT)
    weights_all = read_csv(FEATURE_WEIGHTS_INPUT)
    country_ref = read_csv(COUNTRY_REFERENCE_INPUT)
    global_partition = read_csv(GLOBAL_PARTITION_INPUT)
    global_medoids = read_csv(GLOBAL_MEDOIDS_INPUT)
    country_partition = read_csv(COUNTRY_PARTITION_INPUT)
    country_summary = read_csv(COUNTRY_SUMMARY_INPUT)
    with open(SELECTED_CONFIG_INPUT, "r", encoding="utf-8") as f:
        config = json.load(f)

    if int(config.get("selected_c", -1)) != EXPECTED_C:
        raise ValueError(
            f"Script 06 expects frozen C={EXPECTED_C}, but selected config contains "
            f"C={config.get('selected_c')}."
        )
    if not math.isclose(float(config.get("selected_m", np.nan)), EXPECTED_M, abs_tol=1e-12):
        raise ValueError(
            f"Script 06 expects frozen m={EXPECTED_M}, but selected config contains "
            f"m={config.get('selected_m')}."
        )

    required_dyn_ids = {"region_id", "code", "geo_code", "geo_name", "year"}
    if not required_dyn_ids.issubset(dynamic.columns):
        raise ValueError(
            "Dynamic panel is missing identifiers: "
            f"{sorted(required_dyn_ids - set(dynamic.columns))}"
        )

    dynamic["year"] = pd.to_numeric(dynamic["year"], errors="raise").astype(int)
    if dynamic["region_id"].nunique() != EXPECTED_REGIONS:
        raise ValueError("Dynamic panel does not contain exactly 512 regions.")
    if dynamic["code"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError("Dynamic panel does not contain exactly 28 countries.")
    if sorted(dynamic["year"].unique().tolist()) != EXPECTED_YEARS:
        raise ValueError("Dynamic panel years differ from 2015–2019.")
    if len(dynamic) != EXPECTED_REGIONS * len(EXPECTED_YEARS):
        raise ValueError("Dynamic panel does not contain 2,560 region-year rows.")
    if dynamic.duplicated(["region_id", "year"]).any():
        raise ValueError("Dynamic panel has duplicate region-year keys.")

    weights = weights_all[
        weights_all["block"].astype(str).str.upper().eq("DYNAMIC")
    ].copy()
    required_weight_cols = {
        "feature", "family", "within_family_distance_weight", "family_distance_weight"
    }
    if not required_weight_cols.issubset(weights.columns):
        raise ValueError(
            "Dynamic feature-weight table is missing columns: "
            f"{sorted(required_weight_cols - set(weights.columns))}"
        )
    features = weights["feature"].astype(str).tolist()
    if len(features) != 5:
        raise ValueError(f"Expected exactly 5 dynamic model features, found {len(features)}.")
    missing_features = [f for f in features if f not in dynamic.columns]
    if missing_features:
        raise ValueError(f"Dynamic panel is missing model features: {missing_features}")
    if dynamic[features].isna().any().any():
        raise ValueError("Dynamic model features contain missing values.")

    required_partition = {
        "region_id", "code", "geo_code", "geo_name", "hard_cluster",
        "max_membership", "second_membership", "membership_margin",
        "membership_entropy_normalized", "u_cluster_1", "u_cluster_2",
    }
    if not required_partition.issubset(global_partition.columns):
        raise ValueError(
            "Global partition is missing columns: "
            f"{sorted(required_partition - set(global_partition.columns))}"
        )
    if len(global_partition) != EXPECTED_REGIONS:
        raise ValueError("Global partition does not contain exactly 512 rows.")
    if global_partition["region_id"].nunique() != EXPECTED_REGIONS:
        raise ValueError("Global partition region_id is not unique.")
    if set(global_partition["hard_cluster"].astype(int).unique()) != {1, 2}:
        raise ValueError("Global hard partition is not exactly C1/C2.")
    u = global_partition[["u_cluster_1", "u_cluster_2"]].to_numpy(dtype=float)
    if not np.allclose(u.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Global memberships do not sum to one.")

    if len(global_medoids) != EXPECTED_C:
        raise ValueError("Global medoid table does not contain exactly two medoids.")
    if not {"cluster", "region_id"}.issubset(global_medoids.columns):
        raise ValueError("Global medoid table is missing cluster/region_id.")
    global_medoids["cluster"] = pd.to_numeric(
        global_medoids["cluster"], errors="raise"
    ).astype(int)
    if set(global_medoids["cluster"]) != {1, 2}:
        raise ValueError("Global medoid cluster labels are not {1,2}.")

    config_medoids = [str(x) for x in config.get("medoid_region_ids", [])]
    table_medoids = (
        global_medoids.sort_values("cluster")["region_id"].astype(str).tolist()
    )
    if config_medoids and config_medoids != table_medoids:
        raise ValueError(
            "Selected-configuration medoids differ from 05_global_medoids.csv."
        )

    country_ref["year"] = pd.to_numeric(country_ref["year"], errors="raise").astype(int)
    if not {"code", "year"}.issubset(country_ref.columns):
        raise ValueError("Country reference table is missing code/year.")
    missing_country_features = [f for f in features if f not in country_ref.columns]
    if missing_country_features:
        raise ValueError(
            "Country reference table is missing model features: "
            f"{missing_country_features}"
        )
    if country_ref[features].isna().any().any():
        raise ValueError("Country reference model features contain missing values.")
    if sorted(country_ref["year"].unique().tolist()) != EXPECTED_YEARS:
        raise ValueError("Country-reference years differ from 2015–2019.")

    return (
        dynamic,
        weights,
        country_ref,
        global_partition,
        global_medoids,
        country_partition,
        country_summary,
        config,
        features,
    )


# =============================================================================
# DISTANCE WEIGHTS
# =============================================================================

def build_feature_distance_weights(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights.copy()
    out["within_family_distance_weight"] = pd.to_numeric(
        out["within_family_distance_weight"], errors="raise"
    )
    out["family_distance_weight"] = pd.to_numeric(
        out["family_distance_weight"], errors="raise"
    )

    family_table = (
        out[["family", "family_distance_weight"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if family_table["family"].duplicated().any():
        raise ValueError("A family has inconsistent family-level weights.")
    total_family_weight = float(family_table["family_distance_weight"].sum())
    if total_family_weight <= 0:
        raise ValueError("Total dynamic family weight must be positive.")

    out["effective_annual_distance_weight"] = (
        out["family_distance_weight"] / total_family_weight
        * out["within_family_distance_weight"]
    )
    if not np.isclose(
        out["effective_annual_distance_weight"].sum(), 1.0, atol=1e-10
    ):
        raise ValueError("Effective annual feature weights do not sum to one.")
    return out


# =============================================================================
# CLUSTER PROFILES
# =============================================================================

def build_cluster_profiles(
    dynamic: pd.DataFrame,
    global_partition: pd.DataFrame,
    global_medoids: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    membership_cols = {1: "u_cluster_1", 2: "u_cluster_2"}
    meta_membership = global_partition[
        ["region_id", "hard_cluster", "u_cluster_1", "u_cluster_2"]
    ].copy()

    panel = dynamic.merge(
        meta_membership,
        on="region_id",
        how="left",
        validate="many_to_one",
    )

    region_means = (
        panel.groupby("region_id", as_index=False)[features].mean()
        .merge(meta_membership, on="region_id", how="left", validate="one_to_one")
    )

    pooled_region_means = region_means[features].mean()
    pooled_region_sd = region_means[features].std(ddof=0).replace(0, np.nan)

    summary_rows: list[dict] = []
    annual_rows: list[dict] = []

    medoid_lookup = (
        global_medoids.set_index("cluster")["region_id"].astype(str).to_dict()
    )

    for cluster in [1, 2]:
        w_col = membership_cols[cluster]
        w_region = region_means[w_col].to_numpy(dtype=float)
        medoid_id = medoid_lookup[cluster]
        hard_count = int((global_partition["hard_cluster"].astype(int) == cluster).sum())
        effective_size = float(global_partition[w_col].sum())

        for feature in features:
            values = region_means[feature].to_numpy(dtype=float)
            q25 = weighted_quantile(values, w_region, 0.25)
            q50 = weighted_quantile(values, w_region, 0.50)
            q75 = weighted_quantile(values, w_region, 0.75)
            wmean = weighted_mean(values, w_region)
            wsd = weighted_sd(values, w_region)
            medoid_mean = float(
                region_means.loc[region_means["region_id"].astype(str).eq(medoid_id), feature].iloc[0]
            )
            summary_rows.append({
                "cluster": cluster,
                "feature": feature,
                "medoid_region_id": medoid_id,
                "hard_cluster_size": hard_count,
                "effective_fuzzy_size": effective_size,
                "weighted_mean_2015_2019": wmean,
                "weighted_median_2015_2019": q50,
                "weighted_q25_2015_2019": q25,
                "weighted_q75_2015_2019": q75,
                "weighted_iqr_2015_2019": q75 - q25,
                "weighted_sd_2015_2019": wsd,
                "pooled_region_mean_2015_2019": float(pooled_region_means[feature]),
                "difference_from_pooled_mean": wmean - float(pooled_region_means[feature]),
                "standardized_difference_from_pooled_mean": (
                    (wmean - float(pooled_region_means[feature])) / float(pooled_region_sd[feature])
                    if np.isfinite(pooled_region_sd[feature]) else np.nan
                ),
                "medoid_temporal_mean": medoid_mean,
            })

            for year in EXPECTED_YEARS:
                sub = panel[panel["year"].eq(year)]
                vals = sub[feature].to_numpy(dtype=float)
                w = sub[w_col].to_numpy(dtype=float)
                annual_rows.append({
                    "cluster": cluster,
                    "year": year,
                    "feature": feature,
                    "weighted_mean": weighted_mean(vals, w),
                    "weighted_median": weighted_quantile(vals, w, 0.50),
                    "weighted_q25": weighted_quantile(vals, w, 0.25),
                    "weighted_q75": weighted_quantile(vals, w, 0.75),
                    "weighted_sd": weighted_sd(vals, w),
                })

    summary = pd.DataFrame(summary_rows)
    annual = pd.DataFrame(annual_rows)

    comparison_rows: list[dict] = []
    for feature in features:
        s = summary[summary["feature"].eq(feature)].set_index("cluster")
        a = annual[annual["feature"].eq(feature)]
        c1_2015 = float(a[(a["cluster"].eq(1)) & (a["year"].eq(2015))]["weighted_mean"].iloc[0])
        c1_2019 = float(a[(a["cluster"].eq(1)) & (a["year"].eq(2019))]["weighted_mean"].iloc[0])
        c2_2015 = float(a[(a["cluster"].eq(2)) & (a["year"].eq(2015))]["weighted_mean"].iloc[0])
        c2_2019 = float(a[(a["cluster"].eq(2)) & (a["year"].eq(2019))]["weighted_mean"].iloc[0])
        mean1 = float(s.loc[1, "weighted_mean_2015_2019"])
        mean2 = float(s.loc[2, "weighted_mean_2015_2019"])
        comparison_rows.append({
            "feature": feature,
            "c1_weighted_mean": mean1,
            "c2_weighted_mean": mean2,
            "c1_minus_c2": mean1 - mean2,
            "higher_profile": "C1" if mean1 > mean2 else ("C2" if mean2 > mean1 else "EQUAL"),
            "c1_change_2015_2019": c1_2019 - c1_2015,
            "c2_change_2015_2019": c2_2019 - c2_2015,
            "change_difference_c1_minus_c2": (c1_2019 - c1_2015) - (c2_2019 - c2_2015),
        })

    comparison = pd.DataFrame(comparison_rows)
    return summary, annual, comparison


# =============================================================================
# FUZZY TRANSITION
# =============================================================================

def build_fuzzy_transition(
    global_partition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = global_partition.copy()
    out["hard_cluster"] = out["hard_cluster"].astype(int)
    out["alternative_cluster"] = 3 - out["hard_cluster"]
    out["transition_zone"] = out["membership_margin"] < TRANSITION_MARGIN_THRESHOLD
    out["fuzzy_direction"] = (
        "C" + out["hard_cluster"].astype(str)
        + "->C" + out["alternative_cluster"].astype(str)
    )
    out["origin_membership"] = np.where(
        out["hard_cluster"].eq(1), out["u_cluster_1"], out["u_cluster_2"]
    )
    out["alternative_membership"] = np.where(
        out["hard_cluster"].eq(1), out["u_cluster_2"], out["u_cluster_1"]
    )

    transition_regions = out[out["transition_zone"]].copy()

    rows = []
    for direction, group in out.groupby("fuzzy_direction", sort=True):
        trans = group[group["transition_zone"]]
        rows.append({
            "fuzzy_direction": direction,
            "hard_origin_regions": len(group),
            "transition_regions": len(trans),
            "transition_share_within_origin": len(trans) / len(group) if len(group) else np.nan,
            "mean_origin_membership_all": float(group["origin_membership"].mean()),
            "mean_alternative_membership_all": float(group["alternative_membership"].mean()),
            "mean_membership_margin_all": float(group["membership_margin"].mean()),
            "median_membership_margin_all": float(group["membership_margin"].median()),
            "mean_entropy_all": float(group["membership_entropy_normalized"].mean()),
            "mean_origin_membership_transition": float(trans["origin_membership"].mean()) if len(trans) else np.nan,
            "mean_alternative_membership_transition": float(trans["alternative_membership"].mean()) if len(trans) else np.nan,
        })

    rows.append({
        "fuzzy_direction": "ALL",
        "hard_origin_regions": len(out),
        "transition_regions": int(out["transition_zone"].sum()),
        "transition_share_within_origin": float(out["transition_zone"].mean()),
        "mean_origin_membership_all": float(out["origin_membership"].mean()),
        "mean_alternative_membership_all": float(out["alternative_membership"].mean()),
        "mean_membership_margin_all": float(out["membership_margin"].mean()),
        "median_membership_margin_all": float(out["membership_margin"].median()),
        "mean_entropy_all": float(out["membership_entropy_normalized"].mean()),
        "mean_origin_membership_transition": float(transition_regions["origin_membership"].mean()),
        "mean_alternative_membership_transition": float(transition_regions["alternative_membership"].mean()),
    })

    return transition_regions, pd.DataFrame(rows)


# =============================================================================
# ANNUAL MEDOID DISTANCES + CONVERGENCE
# =============================================================================

def build_medoid_panel(
    dynamic: pd.DataFrame,
    global_medoids: pd.DataFrame,
    features: list[str],
) -> dict[int, pd.DataFrame]:
    lookup: dict[int, pd.DataFrame] = {}
    for _, row in global_medoids.sort_values("cluster").iterrows():
        cluster = int(row["cluster"])
        region_id = str(row["region_id"])
        med = dynamic[dynamic["region_id"].astype(str).eq(region_id)][
            ["year"] + features
        ].copy()
        if len(med) != len(EXPECTED_YEARS):
            raise ValueError(
                f"Medoid {region_id} does not have exactly five annual observations."
            )
        lookup[cluster] = med.set_index("year").sort_index()
    return lookup


def build_annual_medoid_distances(
    dynamic: pd.DataFrame,
    global_partition: pd.DataFrame,
    global_medoids: pd.DataFrame,
    weight_table: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_weight = (
        weight_table.set_index("feature")["effective_annual_distance_weight"].to_dict()
    )
    feature_family = weight_table.set_index("feature")["family"].astype(str).to_dict()
    medoid_panel = build_medoid_panel(dynamic, global_medoids, features)

    part = global_partition[
        ["region_id", "code", "geo_code", "geo_name", "hard_cluster",
         "u_cluster_1", "u_cluster_2", "membership_margin"]
    ].copy()
    panel = dynamic.merge(part, on=["region_id", "code", "geo_code", "geo_name"], how="left", validate="many_to_one")
    panel["hard_cluster"] = panel["hard_cluster"].astype(int)
    panel["destination_cluster"] = 3 - panel["hard_cluster"]
    panel["flow"] = (
        "C" + panel["hard_cluster"].astype(str)
        + "->C" + panel["destination_cluster"].astype(str)
    )

    annual_rows: list[dict] = []
    contrib_rows: list[dict] = []

    for row in panel.itertuples(index=False):
        year = int(row.year)
        origin = int(row.hard_cluster)
        destination = int(row.destination_cluster)

        origin_total = 0.0
        destination_total = 0.0
        feature_values = {}
        for feature in features:
            x = float(getattr(row, feature))
            origin_med = float(medoid_panel[origin].loc[year, feature])
            dest_med = float(medoid_panel[destination].loc[year, feature])
            w = float(feature_weight[feature])
            d_origin = w * (x - origin_med) ** 2
            d_destination = w * (x - dest_med) ** 2
            origin_total += d_origin
            destination_total += d_destination
            feature_values[feature] = (d_origin, d_destination)

        denom = origin_total + destination_total + EPS
        relative_gap = (destination_total - origin_total) / denom

        annual_rows.append({
            "region_id": row.region_id,
            "code": row.code,
            "geo_code": row.geo_code,
            "geo_name": row.geo_name,
            "year": year,
            "origin_cluster": origin,
            "destination_cluster": destination,
            "flow": row.flow,
            "annual_distance_to_origin_medoid": origin_total,
            "annual_distance_to_destination_medoid": destination_total,
            "relative_destination_origin_gap": relative_gap,
        })

        for feature in features:
            d_origin, d_destination = feature_values[feature]
            normalized_gap_component = (d_destination - d_origin) / denom
            contrib_rows.append({
                "region_id": row.region_id,
                "code": row.code,
                "geo_code": row.geo_code,
                "geo_name": row.geo_name,
                "year": year,
                "origin_cluster": origin,
                "destination_cluster": destination,
                "flow": row.flow,
                "feature": feature,
                "family": feature_family[feature],
                "feature_distance_to_origin": d_origin,
                "feature_distance_to_destination": d_destination,
                "normalized_gap_component": normalized_gap_component,
            })

    annual = pd.DataFrame(annual_rows)
    contrib_annual = pd.DataFrame(contrib_rows)

    # Exact QA: feature normalized components must sum to the total relative gap.
    qa = (
        contrib_annual.groupby(["region_id", "year"], as_index=False)
        .agg(component_sum=("normalized_gap_component", "sum"))
        .merge(
            annual[["region_id", "year", "relative_destination_origin_gap"]],
            on=["region_id", "year"], how="left", validate="one_to_one"
        )
    )
    max_err = float(
        np.max(np.abs(qa["component_sum"] - qa["relative_destination_origin_gap"]))
    )
    if max_err > 1e-10:
        raise RuntimeError(
            f"Feature contribution decomposition failed annual QA; max error={max_err}."
        )

    # Region-level 2015 -> 2019 convergence.
    gap_wide = annual.pivot(index="region_id", columns="year", values="relative_destination_origin_gap")
    if not {2015, 2019}.issubset(gap_wide.columns):
        raise RuntimeError("Annual medoid distances do not contain 2015 and 2019.")
    conv = (
        part.copy()
        .rename(columns={"hard_cluster": "origin_cluster"})
    )
    conv["destination_cluster"] = 3 - conv["origin_cluster"].astype(int)
    conv["flow"] = (
        "C" + conv["origin_cluster"].astype(int).astype(str)
        + "->C" + conv["destination_cluster"].astype(int).astype(str)
    )
    conv["relative_gap_2015"] = conv["region_id"].map(gap_wide[2015])
    conv["relative_gap_2019"] = conv["region_id"].map(gap_wide[2019])
    conv["convergence_index"] = conv["relative_gap_2015"] - conv["relative_gap_2019"]
    conv["no_direction_within_numeric_tolerance"] = (
        conv["convergence_index"].abs() <= CONVERGENCE_NUMERIC_TOL
    )
    conv["converged_toward_destination"] = (
        conv["convergence_index"] > CONVERGENCE_NUMERIC_TOL
    )
    conv["moved_away_from_destination"] = (
        conv["convergence_index"] < -CONVERGENCE_NUMERIC_TOL
    )

    direction_count = (
        conv[
            [
                "converged_toward_destination",
                "moved_away_from_destination",
                "no_direction_within_numeric_tolerance",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    if not direction_count.eq(1).all():
        raise RuntimeError(
            "Convergence direction flags are not mutually exclusive/exhaustive."
        )

    # Feature contribution to 2015 -> 2019 convergence index.
    component_wide = contrib_annual.pivot_table(
        index=["region_id", "feature", "family", "flow"],
        columns="year",
        values="normalized_gap_component",
        aggfunc="first",
    ).reset_index()
    component_wide["convergence_contribution"] = component_wide[2015] - component_wide[2019]
    component_wide = component_wide.merge(
        conv[["region_id", "convergence_index", "converged_toward_destination"]],
        on="region_id", how="left", validate="many_to_one"
    )

    # Exact QA: feature contributions sum to convergence index.
    contrib_qa = (
        component_wide.groupby("region_id", as_index=False)
        .agg(contribution_sum=("convergence_contribution", "sum"))
        .merge(conv[["region_id", "convergence_index"]], on="region_id", how="left", validate="one_to_one")
    )
    max_conv_err = float(
        np.max(np.abs(contrib_qa["contribution_sum"] - contrib_qa["convergence_index"]))
    )
    if max_conv_err > 1e-10:
        raise RuntimeError(
            f"Feature contribution decomposition failed convergence QA; max error={max_conv_err}."
        )

    flow_rows = []
    for flow, g in conv.groupby("flow", sort=True):
        pos = g[g["converged_toward_destination"]]
        flow_rows.append({
            "flow": flow,
            "origin_regions": len(g),
            "regions_converging_toward_destination": len(pos),
            "share_converging_toward_destination": len(pos) / len(g) if len(g) else np.nan,
            "median_convergence_index_all": float(g["convergence_index"].median()),
            "mean_convergence_index_all": float(g["convergence_index"].mean()),
            "median_convergence_index_convergers": float(pos["convergence_index"].median()) if len(pos) else np.nan,
            "mean_convergence_index_convergers": float(pos["convergence_index"].mean()) if len(pos) else np.nan,
        })
    flow_summary = pd.DataFrame(flow_rows)

    # Aggregate contributions only among regions that actually converged toward destination.
    converger_components = component_wide[
        component_wide["converged_toward_destination"].fillna(False)
    ].copy()

    feature_rows = []
    for (flow, feature, family), g in converger_components.groupby(
        ["flow", "feature", "family"], sort=True
    ):
        positive = g["convergence_contribution"] > 0
        feature_rows.append({
            "flow": flow,
            "feature": feature,
            "family": family,
            "converging_regions_in_flow": int(g["region_id"].nunique()),
            "median_contribution": float(g["convergence_contribution"].median()),
            "mean_contribution": float(g["convergence_contribution"].mean()),
            "regions_positive_contribution": int(positive.sum()),
            "consistency_pct": float(100.0 * positive.mean()),
        })
    feature_summary = pd.DataFrame(feature_rows)

    family_region = (
        converger_components.groupby(["region_id", "flow", "family"], as_index=False)
        .agg(convergence_contribution=("convergence_contribution", "sum"))
    )
    family_rows = []
    for (flow, family), g in family_region.groupby(["flow", "family"], sort=True):
        positive = g["convergence_contribution"] > 0
        family_rows.append({
            "flow": flow,
            "family": family,
            "converging_regions_in_flow": int(g["region_id"].nunique()),
            "median_contribution": float(g["convergence_contribution"].median()),
            "mean_contribution": float(g["convergence_contribution"].mean()),
            "regions_positive_contribution": int(positive.sum()),
            "consistency_pct": float(100.0 * positive.mean()),
        })
    family_summary = pd.DataFrame(family_rows)

    return annual, conv, flow_summary, feature_summary, family_summary


# =============================================================================
# COUNTRY-REFERENCE COMPARISONS
# =============================================================================

def build_country_reference_comparison(
    dynamic: pd.DataFrame,
    country_ref: pd.DataFrame,
    weight_table: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each region with national reference trajectories.

    For the region's own country, use a leave-one-out (LOO) equal-weight
    country mean so the focal region does not mechanically pull its own
    national reference closer to itself. For all other countries, use the
    full equal-weight country reference produced by Script 04.
    """
    feature_weight = (
        weight_table.set_index("feature")["effective_annual_distance_weight"].to_dict()
    )
    country_codes = sorted(country_ref["code"].astype(str).unique().tolist())

    ref_by_country_year = (
        country_ref.set_index(["code", "year"])[features].sort_index()
    )
    dyn_sorted = dynamic.sort_values(["region_id", "year"], kind="stable").copy()

    country_year_sum = (
        dyn_sorted.groupby(["code", "year"], sort=False)[features].sum()
    )
    country_year_n = (
        dyn_sorted.groupby(["code", "year"], sort=False)["region_id"].nunique()
    )

    rows: list[dict] = []
    for region_id, rg in dyn_sorted.groupby("region_id", sort=False):
        rg = rg.sort_values("year")
        home_code = str(rg["code"].iloc[0])
        meta = rg.iloc[0]
        region_matrix = rg.set_index("year")[features]

        for candidate_code in country_codes:
            annual_distances = []
            reference_method = None
            contributing_regions = None

            for year in EXPECTED_YEARS:
                region_vector = region_matrix.loc[year, features].astype(float)

                if candidate_code == home_code:
                    n_home = int(country_year_n.loc[(home_code, year)])
                    if n_home <= 1:
                        raise RuntimeError(
                            f"Cannot build leave-one-out country reference for "
                            f"{region_id} / {home_code} / {year}: n={n_home}."
                        )
                    ref_vector = (
                        country_year_sum.loc[(home_code, year), features].astype(float)
                        - region_vector
                    ) / float(n_home - 1)
                    reference_method = "LEAVE_ONE_OUT_COUNTRY_MEAN"
                    contributing_regions = n_home - 1
                else:
                    try:
                        ref_vector = (
                            ref_by_country_year.loc[(candidate_code, year), features]
                            .astype(float)
                        )
                    except KeyError:
                        annual_distances = []
                        break
                    reference_method = "FULL_COUNTRY_MEAN"
                    contributing_regions = int(
                        country_year_n.loc[(candidate_code, year)]
                    )

                d = 0.0
                for feature in features:
                    diff = float(region_vector[feature]) - float(ref_vector[feature])
                    d += float(feature_weight[feature]) * diff * diff
                annual_distances.append(d)

            if len(annual_distances) != len(EXPECTED_YEARS):
                continue

            rows.append({
                "region_id": region_id,
                "code": home_code,
                "geo_code": meta["geo_code"],
                "geo_name": meta["geo_name"],
                "reference_country_code": candidate_code,
                "is_own_country_reference": candidate_code == home_code,
                "reference_method": reference_method,
                "reference_contributing_regions": contributing_regions,
                "mean_annual_distance_2015_2019": float(np.mean(annual_distances)),
                "distance_2015": float(annual_distances[0]),
                "distance_2019": float(annual_distances[-1]),
                "distance_change_2015_2019": float(
                    annual_distances[-1] - annual_distances[0]
                ),
            })

    distances = pd.DataFrame(rows)
    if distances.empty:
        raise RuntimeError("Country-reference comparison produced no rows.")

    expected_comparisons = EXPECTED_REGIONS * EXPECTED_COUNTRIES
    if len(distances) != expected_comparisons:
        raise RuntimeError(
            f"Expected {expected_comparisons:,} region-country comparisons, "
            f"found {len(distances):,}."
        )

    own_method_ok = distances.loc[
        distances["is_own_country_reference"], "reference_method"
    ].eq("LEAVE_ONE_OUT_COUNTRY_MEAN").all()
    other_method_ok = distances.loc[
        ~distances["is_own_country_reference"], "reference_method"
    ].eq("FULL_COUNTRY_MEAN").all()
    if not own_method_ok or not other_method_ok:
        raise RuntimeError("Country-reference method assignment failed QA.")

    distances["overall_reference_rank"] = (
        distances.groupby("region_id")["mean_annual_distance_2015_2019"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    nearest_rows = []
    for region_id, g in distances.groupby("region_id", sort=False):
        g = g.sort_values("mean_annual_distance_2015_2019", kind="stable")
        own = g[g["is_own_country_reference"]]
        other = g[~g["is_own_country_reference"]]
        if len(own) != 1:
            raise RuntimeError(
                f"Region {region_id} does not have exactly one own-country reference."
            )
        if other.empty:
            raise RuntimeError(
                f"Region {region_id} has no alternative country reference."
            )

        own_row = own.iloc[0]
        other_row = other.iloc[0]
        nearest_row = g.iloc[0]

        nearest_rows.append({
            "region_id": region_id,
            "code": own_row["code"],
            "geo_code": own_row["geo_code"],
            "geo_name": own_row["geo_name"],
            "own_country_distance": own_row["mean_annual_distance_2015_2019"],
            "own_country_rank": int(own_row["overall_reference_rank"]),
            "own_country_reference_method": own_row["reference_method"],
            "own_country_reference_regions": int(
                own_row["reference_contributing_regions"]
            ),
            "nearest_other_country_code": other_row["reference_country_code"],
            "nearest_other_country_distance": other_row[
                "mean_annual_distance_2015_2019"
            ],
            "nearest_reference_country_code": nearest_row[
                "reference_country_code"
            ],
            "nearest_reference_is_own_country": bool(
                nearest_row["is_own_country_reference"]
            ),
            "nearest_reference_distance": nearest_row[
                "mean_annual_distance_2015_2019"
            ],
            "own_minus_nearest_other_distance": (
                own_row["mean_annual_distance_2015_2019"]
                - other_row["mean_annual_distance_2015_2019"]
            ),
        })

    nearest = pd.DataFrame(nearest_rows)
    if len(nearest) != EXPECTED_REGIONS:
        raise RuntimeError("Country-reference nearest summary lost regions.")

    return distances, nearest


# =============================================================================
# GLOBAL-vs-LOCAL COMPARISON
# =============================================================================

def build_global_local_comparison(
    global_partition: pd.DataFrame,
    country_partition: pd.DataFrame,
    country_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_cols = [
        "region_id", "code", "geo_code", "geo_name", "hard_cluster",
        "max_membership", "membership_margin", "membership_entropy_normalized",
        "u_cluster_1", "u_cluster_2",
    ]
    g = global_partition[global_cols].copy().rename(columns={
        "hard_cluster": "global_cluster",
        "max_membership": "global_max_membership",
        "membership_margin": "global_membership_margin",
        "membership_entropy_normalized": "global_membership_entropy",
    })

    cp = country_partition.copy()
    local_keep = [
        c for c in [
            "region_id", "local_status", "local_c", "local_hard_cluster",
            "max_membership", "membership_margin", "membership_entropy_normalized"
        ] if c in cp.columns
    ]
    if "region_id" in local_keep:
        cp = cp[local_keep].drop_duplicates("region_id").rename(columns={
            "max_membership": "local_max_membership",
            "membership_margin": "local_membership_margin",
            "membership_entropy_normalized": "local_membership_entropy",
        })
    else:
        cp = pd.DataFrame(columns=["region_id"])

    region_compare = g.merge(cp, on="region_id", how="left", validate="one_to_one")
    status_map = country_summary.set_index("code")["status"].astype(str).to_dict()
    selected_c_map = pd.to_numeric(
        country_summary.set_index("code")["selected_c"], errors="coerce"
    ).to_dict()
    region_compare["country_model_status"] = region_compare["code"].map(status_map)
    region_compare["country_selected_c"] = region_compare["code"].map(selected_c_map)
    region_compare["global_transition_zone"] = (
        region_compare["global_membership_margin"] < TRANSITION_MARGIN_THRESHOLD
    )
    if "local_membership_margin" in region_compare.columns:
        region_compare["local_transition_zone"] = (
            region_compare["local_membership_margin"] < TRANSITION_MARGIN_THRESHOLD
        )
    else:
        region_compare["local_transition_zone"] = np.nan

    summary_rows = []
    for code, gg in region_compare.groupby("code", sort=True):
        status = str(gg["country_model_status"].iloc[0])
        row = {
            "code": code,
            "n_regions": len(gg),
            "country_model_status": status,
            "country_selected_c": gg["country_selected_c"].iloc[0],
            "global_c": EXPECTED_C,
            "global_mean_max_membership": float(gg["global_max_membership"].mean()),
            "global_mean_membership_margin": float(gg["global_membership_margin"].mean()),
            "global_transition_regions": int(gg["global_transition_zone"].sum()),
        }
        success = status == "SUCCESS"
        if success and "local_hard_cluster" in gg.columns:
            valid = gg[gg["local_hard_cluster"].notna()].copy()
            if len(valid) >= 2:
                row.update({
                    "regions_with_local_partition": len(valid),
                    "ari_global_vs_local": adjusted_rand_index(
                        valid["global_cluster"].astype(int),
                        valid["local_hard_cluster"].astype(int),
                    ),
                    "local_mean_max_membership": float(valid["local_max_membership"].mean()),
                    "local_mean_membership_margin": float(valid["local_membership_margin"].mean()),
                    "local_transition_regions": int(valid["local_transition_zone"].sum()),
                })
        summary_rows.append(row)

    country_compare = pd.DataFrame(summary_rows)
    return region_compare, country_compare


# =============================================================================
# EXCEL + METADATA
# =============================================================================

def autofit_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            max_length = 0
            letter = column_cells[0].column_letter
            for cell in column_cells[:2500]:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            ws.column_dimensions[letter].width = min(max(max_length + 2, 10), 55)
    wb.save(path)


def write_results_workbook(
    metadata: pd.DataFrame,
    profile_summary: pd.DataFrame,
    profile_annual: pd.DataFrame,
    profile_comparison: pd.DataFrame,
    transition_summary: pd.DataFrame,
    transition_regions: pd.DataFrame,
    convergence_flow: pd.DataFrame,
    convergence_region: pd.DataFrame,
    contrib_feature: pd.DataFrame,
    contrib_family: pd.DataFrame,
    country_nearest: pd.DataFrame,
    global_local_country: pd.DataFrame,
    global_local_region: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(RESULTS_WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
        profile_summary.to_excel(writer, sheet_name="Cluster_Profile", index=False)
        profile_annual.to_excel(writer, sheet_name="Profile_Annual", index=False)
        profile_comparison.to_excel(writer, sheet_name="Profile_Compare", index=False)
        transition_summary.to_excel(writer, sheet_name="Fuzzy_Summary", index=False)
        transition_regions.to_excel(writer, sheet_name="Fuzzy_Regions", index=False)
        convergence_flow.to_excel(writer, sheet_name="Convergence_Flow", index=False)
        convergence_region.to_excel(writer, sheet_name="Convergence_Region", index=False)
        contrib_feature.to_excel(writer, sheet_name="Conv_Feature", index=False)
        contrib_family.to_excel(writer, sheet_name="Conv_Family", index=False)
        country_nearest.to_excel(writer, sheet_name="Country_Refs", index=False)
        global_local_country.to_excel(writer, sheet_name="Global_Local_Country", index=False)
        global_local_region.to_excel(writer, sheet_name="Global_Local_Region", index=False)
    autofit_workbook(RESULTS_WORKBOOK_OUTPUT)


def build_metadata(config: dict, features: list[str]) -> pd.DataFrame:
    rows = [
        ("stage", "06 — Evolution + interpretation"),
        ("analytical_period", "2015-2019"),
        ("regions", EXPECTED_REGIONS),
        ("countries", EXPECTED_COUNTRIES),
        ("frozen_global_c", EXPECTED_C),
        ("frozen_global_m", EXPECTED_M),
        ("script05_fuzzy_silhouette", config.get("fuzzy_silhouette")),
        ("script05_objective", config.get("objective")),
        ("features", ",".join(features)),
        ("fuzzy_transition_rule", f"membership_margin < {TRANSITION_MARGIN_THRESHOLD}"),
        (
            "annual_distance_definition",
            "family-balanced weighted squared distance from each region-year to the corresponding medoid-year",
        ),
        (
            "relative_gap_definition",
            "(distance_destination - distance_origin) / (distance_destination + distance_origin)",
        ),
        (
            "convergence_index_definition",
            "relative_gap_2015 - relative_gap_2019; positive means movement toward alternative profile",
        ),
        (
            "contribution_definition",
            "exact feature decomposition of the change in the normalized destination-origin distance gap",
        ),
        (
            "cluster_assignment_rule",
            "global cluster remains fixed for 2015-2019; convergence is not cluster reassignment",
        ),
        (
            "convergence_numeric_tolerance",
            CONVERGENCE_NUMERIC_TOL,
        ),
        (
            "country_reference_definition",
            "own country = leave-one-out equal-weight regional mean; other countries = full equal-weight regional mean",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    ensure_directories()

    print_header("SCRIPT 06 v1.1 — EVOLUTION + INTERPRETATION")
    print(f"Project root: {ROOT}")
    print(f"Frozen global configuration: C={EXPECTED_C}, m={EXPECTED_M}")
    print(f"Analytical period: {EXPECTED_YEARS[0]}–{EXPECTED_YEARS[-1]}")

    print_header("1/8 — VALIDATE SCRIPT 04 + SCRIPT 05 HANDOFF")
    validate_required_inputs()
    (
        dynamic,
        weights,
        country_ref,
        global_partition,
        global_medoids,
        country_partition,
        country_summary,
        config,
        features,
    ) = load_and_validate_inputs()
    weight_table = build_feature_distance_weights(weights)
    print(f"Dynamic panel: {len(dynamic):,} rows | {dynamic['region_id'].nunique()} regions")
    print(f"Features: {features}")
    print("Script-05 frozen configuration validated.")

    print_header("2/8 — CLUSTER PROFILES")
    profile_summary, profile_annual, profile_comparison = build_cluster_profiles(
        dynamic, global_partition, global_medoids, features
    )
    print(profile_comparison.to_string(index=False))

    print_header("3/8 — FUZZY TRANSITION DIAGNOSTIC")
    transition_regions, transition_summary = build_fuzzy_transition(global_partition)
    print(transition_summary.to_string(index=False))

    print_header("4/8 — TEMPORAL CONVERGENCE TO ALTERNATIVE GLOBAL PROFILE")
    (
        annual_medoid_distance,
        convergence_region,
        convergence_flow,
        contrib_feature,
        contrib_family,
    ) = build_annual_medoid_distances(
        dynamic,
        global_partition,
        global_medoids,
        weight_table,
        features,
    )
    print(convergence_flow.to_string(index=False))

    print_header("5/8 — VARIABLE/FAMILY CONTRIBUTIONS TO CONVERGENCE")
    if not contrib_feature.empty:
        print(
            contrib_feature.sort_values(
                ["flow", "median_contribution"], ascending=[True, False]
            ).to_string(index=False)
        )
    else:
        print("No regions showed positive convergence toward the alternative profile.")

    print_header("6/8 — REGION × COUNTRY-REFERENCE COMPARISON")
    country_distance, country_nearest = build_country_reference_comparison(
        dynamic, country_ref, weight_table, features
    )
    print(
        f"Country-reference comparisons: {len(country_distance):,} rows | "
        f"{country_nearest['region_id'].nunique()} regions"
    )
    print(
        "Regions whose leave-one-out own-country reference is the nearest reference: "
        f"{int(country_nearest['nearest_reference_is_own_country'].sum())}/"
        f"{len(country_nearest)}"
    )

    print_header("7/8 — GLOBAL × INTRACOUNTRY COMPARISON")
    global_local_region, global_local_country = build_global_local_comparison(
        global_partition, country_partition, country_summary
    )
    print(global_local_country.to_string(index=False))

    print_header("8/8 — WRITE OUTPUTS + FINAL QA")
    metadata = build_metadata(config, features)

    write_csv(profile_summary, PROFILE_SUMMARY_OUTPUT)
    write_csv(profile_annual, PROFILE_ANNUAL_OUTPUT)
    write_csv(profile_comparison, PROFILE_COMPARISON_OUTPUT)
    write_csv(transition_regions, TRANSITION_REGIONS_OUTPUT)
    write_csv(transition_summary, TRANSITION_SUMMARY_OUTPUT)
    write_csv(annual_medoid_distance, ANNUAL_MEDOID_DISTANCE_OUTPUT)
    write_csv(convergence_region, CONVERGENCE_REGION_OUTPUT)
    write_csv(convergence_flow, CONVERGENCE_FLOW_OUTPUT)
    write_csv(contrib_feature, CONTRIB_FEATURE_OUTPUT)
    write_csv(contrib_family, CONTRIB_FAMILY_OUTPUT)
    write_csv(country_distance, COUNTRY_DISTANCE_OUTPUT)
    write_csv(country_nearest, COUNTRY_NEAREST_OUTPUT)
    write_csv(global_local_region, GLOBAL_LOCAL_REGION_OUTPUT)
    write_csv(global_local_country, GLOBAL_LOCAL_COUNTRY_OUTPUT)
    write_csv(metadata, METADATA_OUTPUT)

    write_results_workbook(
        metadata=metadata,
        profile_summary=profile_summary,
        profile_annual=profile_annual,
        profile_comparison=profile_comparison,
        transition_summary=transition_summary,
        transition_regions=transition_regions,
        convergence_flow=convergence_flow,
        convergence_region=convergence_region,
        contrib_feature=contrib_feature,
        contrib_family=contrib_family,
        country_nearest=country_nearest,
        global_local_country=global_local_country,
        global_local_region=global_local_region,
    )

    # Final QA
    if len(convergence_region) != EXPECTED_REGIONS:
        raise RuntimeError("Convergence output lost regions.")
    direction_count = (
        convergence_region[
            [
                "converged_toward_destination",
                "moved_away_from_destination",
                "no_direction_within_numeric_tolerance",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    if not direction_count.eq(1).all():
        raise RuntimeError(
            "Final convergence direction flags are not mutually exclusive/exhaustive."
        )
    if len(country_nearest) != EXPECTED_REGIONS:
        raise RuntimeError("Country-reference nearest output lost regions.")
    if global_local_region["region_id"].nunique() != EXPECTED_REGIONS:
        raise RuntimeError("Global-local comparison lost regions.")
    if int(transition_summary.loc[
        transition_summary["fuzzy_direction"].eq("ALL"), "transition_regions"
    ].iloc[0]) != len(transition_regions):
        raise RuntimeError("Fuzzy-transition summary and region detail disagree.")

    print("\nFINAL SCRIPT 06 SUMMARY")
    print("-" * 100)
    print(f"Frozen global C:                 {EXPECTED_C}")
    print(f"Frozen global m:                 {EXPECTED_M}")
    print(f"Regions profiled:                {EXPECTED_REGIONS}")
    print(f"Model features profiled:         {len(features)}")
    print(f"Fuzzy transition regions:        {len(transition_regions)}")
    print(
        "Regions converging to alternative profile: "
        f"{int(convergence_region['converged_toward_destination'].sum())}"
    )
    print(f"Country-reference comparisons:   {len(country_distance):,}")
    print(
        "Countries with viable local model: "
        f"{int(country_summary['status'].astype(str).eq('SUCCESS').sum())}"
    )
    print(f"Results workbook:                {RESULTS_WORKBOOK_OUTPUT}")
    print(f"Script-06 interim directory:     {SCRIPT06_DIR}")

    elapsed = time.perf_counter() - t0
    print("\nSCRIPT 06 COMPLETE — READY FOR REVIEW BEFORE SCRIPT 07")
    print(f"Script execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
