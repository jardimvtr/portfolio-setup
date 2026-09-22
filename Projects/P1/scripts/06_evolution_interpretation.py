from __future__ import annotations

"""
SCRIPT 06 v2.0 — CLUSTER PROFILES, BoD DECOMPOSITION, FUZZY TRANSITION,
TEMPORAL CONVERGENCE, COUNTRY REFERENCES AND GLOBAL-vs-LOCAL INTERPRETATION

Purpose
-------
Interpret the frozen Script-05 M-Exp-FCMd solution without re-estimating the
clustering model, while preserving the two analytical layers produced by
Script 04:

1) common-weight family scores used as the comparable dynamic clustering space;
2) DMU × year × underlying variable × common-weight decomposition used for
   audit and interpretation.

Frozen P1 design expected by this script
----------------------------------------
- Analytical period: 2015–2019.
- Official sample: 512 regions in 28 countries.
- Dynamic-only M-Exp-FCMd clustering.
- Frozen global solution from Script 05: C=2 and m=2.5.
- Final clustering features:
    * bod_demographic_productive_potential
    * bod_socioeconomic_deprivation
    * ntl_per_capita_norm
- Structural/static variables remain excluded from the clustering and temporal
  convergence distance.
- Script-04 common BoD weights are never re-estimated here.

Main outputs
------------
1) Membership-weighted profiles of C1 and C2 using the three final clustering
   features and their annual trajectories.
2) Membership-weighted profiles of the five underlying dynamic BoD indicators,
   retaining raw values, normalized values, common weights and weighted
   contributions.
3) Fuzzy-transition diagnostics based on membership margin.
4) Annual region-to-medoid distances and 2015→2019 convergence toward the
   alternative global profile.
5) Exact additive decomposition of the convergence index at three levels:
   model feature, analytical family and underlying variable.
6) Region-to-country-reference comparisons in the same dynamic analytical
   space.
7) Ex-post comparison between global and viable intracountry partitions.

Exact variable-level distance decomposition
-------------------------------------------
For a BoD family score S = sum_j c_j, where c_j = z_j * w_j is the Script-04
weighted contribution of underlying indicator j, the squared gap to a medoid
can be decomposed exactly as:

    (S_i - S_m)^2 = sum_j [(c_ij - c_mj) * (S_i - S_m)]

Therefore each underlying variable receives the additive component

    phi_j = family_distance_weight * (c_ij - c_mj) * (S_i - S_m)

which sums exactly to the family-level squared distance. This is the exact
Shapley allocation for the quadratic squared-gap function and naturally
retains cross-variable interaction terms without double counting. For the
single-indicator NTL family, the same expression reduces to its ordinary
squared distance.

Important interpretation rule
-----------------------------
A region keeps one global cluster assignment for the full 2015–2019 period.
C1->C2 or C2->C1 in temporal-convergence outputs means movement toward the
alternative fixed medoid profile, not formal cluster reassignment.
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
BOD_CONTRIBUTIONS_INPUT = SCRIPT04_DIR / "04_bod_weighted_contributions_long.csv"

GLOBAL_PARTITION_INPUT = SCRIPT05_DIR / "05_global_partition.csv"
GLOBAL_MEDOIDS_INPUT = SCRIPT05_DIR / "05_global_medoids.csv"
COUNTRY_PARTITION_INPUT = SCRIPT05_DIR / "05_country_partition.csv"
COUNTRY_SUMMARY_INPUT = SCRIPT05_DIR / "05_country_cluster_summary.csv"
SELECTED_CONFIG_INPUT = SCRIPT05_DIR / "05_selected_configuration.json"

EXPECTED_REGIONS = 512
EXPECTED_COUNTRIES = 28
EXPECTED_YEARS = [2015, 2016, 2017, 2018, 2019]
EXPECTED_C = 2
EXPECTED_M = 2.5
EXPECTED_DYNAMIC_FEATURES = (
    "bod_demographic_productive_potential",
    "bod_socioeconomic_deprivation",
    "ntl_per_capita_norm",
)
EXPECTED_DYNAMIC_FAMILIES = (
    "demographic_productive_potential",
    "socioeconomic_deprivation",
    "regional_economic_activity_proxy",
)
EXPECTED_BOD_VARIABLES = {
    "demographic_productive_potential": ("share_15_64", "share_65_plus"),
    "socioeconomic_deprivation": ("poor420", "gini", "prosgap2021"),
}
TRANSITION_MARGIN_THRESHOLD = 0.20
CONVERGENCE_NUMERIC_TOL = 1e-12
EPS = 1e-15

# Script 06 outputs
PROFILE_SUMMARY_OUTPUT = SCRIPT06_DIR / "06_cluster_profile_summary.csv"
PROFILE_ANNUAL_OUTPUT = SCRIPT06_DIR / "06_cluster_profile_annual.csv"
PROFILE_COMPARISON_OUTPUT = SCRIPT06_DIR / "06_cluster_profile_comparison.csv"
BOD_VARIABLE_PROFILE_OUTPUT = SCRIPT06_DIR / "06_bod_variable_profile_summary.csv"
BOD_VARIABLE_PROFILE_ANNUAL_OUTPUT = SCRIPT06_DIR / "06_bod_variable_profile_annual.csv"
TRANSITION_REGIONS_OUTPUT = SCRIPT06_DIR / "06_fuzzy_transition_regions.csv"
TRANSITION_SUMMARY_OUTPUT = SCRIPT06_DIR / "06_fuzzy_transition_summary.csv"
ANNUAL_MEDOID_DISTANCE_OUTPUT = SCRIPT06_DIR / "06_region_annual_medoid_distances.csv"
CONVERGENCE_REGION_OUTPUT = SCRIPT06_DIR / "06_region_convergence_2015_2019.csv"
CONVERGENCE_FLOW_OUTPUT = SCRIPT06_DIR / "06_convergence_flow_summary.csv"
CONTRIB_FEATURE_OUTPUT = SCRIPT06_DIR / "06_convergence_feature_contributions.csv"
CONTRIB_FAMILY_OUTPUT = SCRIPT06_DIR / "06_convergence_family_contributions.csv"
CONTRIB_VARIABLE_OUTPUT = SCRIPT06_DIR / "06_convergence_variable_contributions.csv"
CONTRIB_VARIABLE_REGION_OUTPUT = SCRIPT06_DIR / "06_convergence_variable_region_detail.csv"
CONTRIB_VARIABLE_ANNUAL_OUTPUT = SCRIPT06_DIR / "06_convergence_variable_annual_components.csv"
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
        "BOD_CONTRIBUTIONS": BOD_CONTRIBUTIONS_INPUT,
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
    pd.DataFrame,
    dict,
    list[str],
]:
    dynamic = read_csv(DYNAMIC_INPUT)
    weights_all = read_csv(FEATURE_WEIGHTS_INPUT)
    country_ref = read_csv(COUNTRY_REFERENCE_INPUT)
    bod_contrib_all = read_csv(BOD_CONTRIBUTIONS_INPUT)
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
    if not math.isclose(
        float(config.get("selected_m", np.nan)), EXPECTED_M, abs_tol=1e-12
    ):
        raise ValueError(
            f"Script 06 expects frozen m={EXPECTED_M}, but selected config contains "
            f"m={config.get('selected_m')}."
        )

    config_features = tuple(str(x) for x in config.get("dynamic_features", []))
    if config_features and set(config_features) != set(EXPECTED_DYNAMIC_FEATURES):
        raise ValueError(
            "Script-05 selected configuration contains an unexpected dynamic feature set: "
            f"{config_features}."
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
    if len(features) != 3 or set(features) != set(EXPECTED_DYNAMIC_FEATURES):
        raise ValueError(
            "Expected exactly the three revised dynamic model features; found "
            f"{features}."
        )
    missing_features = [f for f in features if f not in dynamic.columns]
    if missing_features:
        raise ValueError(f"Dynamic panel is missing model features: {missing_features}")
    if dynamic[features].isna().any().any():
        raise ValueError("Dynamic model features contain missing values.")

    families = tuple(weights["family"].astype(str).tolist())
    if set(families) != set(EXPECTED_DYNAMIC_FAMILIES):
        raise ValueError(
            "Unexpected final dynamic families in feature-weight handoff: "
            f"{families}."
        )

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

    if not {"code", "year"}.issubset(country_ref.columns):
        raise ValueError("Country reference table is missing code/year.")
    country_ref["year"] = pd.to_numeric(country_ref["year"], errors="raise").astype(int)
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
    if country_ref[["code", "year"]].duplicated().any():
        raise ValueError("Country-reference table has duplicate country-year keys.")

    # Dynamic BoD decomposition: keep only the five underlying variables that
    # build the two dynamic common-weight family scores used in clustering.
    required_bod_cols = {
        "region_id", "code", "block", "family", "variable", "year",
        "value_raw", "value_normalized", "common_weight",
        "weighted_contribution", "family_score",
        "contribution_share_of_family_score",
    }
    if not required_bod_cols.issubset(bod_contrib_all.columns):
        raise ValueError(
            "BoD weighted-contribution table is missing columns: "
            f"{sorted(required_bod_cols - set(bod_contrib_all.columns))}"
        )
    bod_contrib = bod_contrib_all[
        bod_contrib_all["block"].astype(str).str.upper().eq("DYNAMIC")
    ].copy()
    bod_contrib["year"] = pd.to_numeric(bod_contrib["year"], errors="raise").astype(int)

    expected_pairs = {
        (family, variable)
        for family, variables in EXPECTED_BOD_VARIABLES.items()
        for variable in variables
    }
    observed_pairs = set(
        zip(bod_contrib["family"].astype(str), bod_contrib["variable"].astype(str))
    )
    if observed_pairs != expected_pairs:
        raise ValueError(
            "Dynamic BoD decomposition has unexpected family-variable pairs. "
            f"Expected {sorted(expected_pairs)}, found {sorted(observed_pairs)}."
        )

    expected_bod_rows = EXPECTED_REGIONS * len(EXPECTED_YEARS) * sum(
        len(v) for v in EXPECTED_BOD_VARIABLES.values()
    )
    if len(bod_contrib) != expected_bod_rows:
        raise ValueError(
            f"Expected {expected_bod_rows:,} dynamic BoD decomposition rows, "
            f"found {len(bod_contrib):,}."
        )
    if bod_contrib.duplicated(["region_id", "year", "family", "variable"]).any():
        raise ValueError("Dynamic BoD decomposition has duplicate region-year-family-variable keys.")
    if bod_contrib[[
        "value_raw", "value_normalized", "common_weight", "weighted_contribution", "family_score"
    ]].isna().any().any():
        raise ValueError("Dynamic BoD decomposition contains missing numeric values.")

    # Common weights must truly be common across all regions and all years.
    weight_nunique = (
        bod_contrib.groupby(["family", "variable"])["common_weight"].nunique(dropna=False)
    )
    if not weight_nunique.eq(1).all():
        raise ValueError("At least one dynamic BoD indicator does not have one common weight.")
    if "weight_scope" in bod_contrib.columns:
        scopes = set(bod_contrib["weight_scope"].astype(str).unique())
        if scopes != {"COMMON_ACROSS_REGIONS_AND_2015_2019"}:
            raise ValueError(f"Unexpected dynamic BoD weight scope: {sorted(scopes)}")

    # Exact reconciliation against the actual clustering family scores.
    family_feature = {
        "demographic_productive_potential": "bod_demographic_productive_potential",
        "socioeconomic_deprivation": "bod_socioeconomic_deprivation",
    }
    recon = (
        bod_contrib.groupby(["region_id", "year", "family"], as_index=False)
        .agg(
            reconstructed_score=("weighted_contribution", "sum"),
            stored_family_score=("family_score", "first"),
        )
    )
    dyn_long = dynamic.melt(
        id_vars=["region_id", "year"],
        value_vars=list(family_feature.values()),
        var_name="feature",
        value_name="dynamic_family_score",
    )
    dyn_long["family"] = dyn_long["feature"].map(
        {v: k for k, v in family_feature.items()}
    )
    recon = recon.merge(
        dyn_long[["region_id", "year", "family", "dynamic_family_score"]],
        on=["region_id", "year", "family"],
        how="left",
        validate="one_to_one",
    )
    err1 = np.max(np.abs(recon["reconstructed_score"] - recon["stored_family_score"]))
    err2 = np.max(np.abs(recon["reconstructed_score"] - recon["dynamic_family_score"]))
    if max(float(err1), float(err2)) > 1e-10:
        raise ValueError(
            "Dynamic BoD contribution decomposition does not reconcile exactly with "
            f"the clustering family scores; max error={max(float(err1), float(err2)):.3e}."
        )

    return (
        dynamic,
        weights,
        country_ref,
        bod_contrib,
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
# UNDERLYING BoD VARIABLE PROFILES
# =============================================================================

def build_bod_variable_profiles(
    bod_contrib: pd.DataFrame,
    global_partition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile the five underlying BoD variables without changing the cluster model."""
    membership_cols = {1: "u_cluster_1", 2: "u_cluster_2"}
    memberships = global_partition[
        ["region_id", "hard_cluster", "u_cluster_1", "u_cluster_2"]
    ].copy()
    panel = bod_contrib.merge(
        memberships,
        on="region_id",
        how="left",
        validate="many_to_one",
    )

    common_weight_lookup = (
        panel[["family", "variable", "common_weight"]]
        .drop_duplicates()
        .set_index(["family", "variable"])["common_weight"]
        .to_dict()
    )
    if len(common_weight_lookup) != sum(len(v) for v in EXPECTED_BOD_VARIABLES.values()):
        raise RuntimeError("Unexpected number of common weights in BoD variable profile input.")

    # Temporal summary first at region level so each region contributes one
    # membership weight to 2015–2019 summary statistics.
    metric_cols = [
        "value_raw",
        "value_normalized",
        "weighted_contribution",
        "contribution_share_of_family_score",
    ]
    region_mean = (
        panel.groupby(["region_id", "family", "variable"], as_index=False)[metric_cols]
        .mean()
        .merge(memberships, on="region_id", how="left", validate="many_to_one")
    )

    summary_rows: list[dict] = []
    annual_rows: list[dict] = []

    for cluster in [1, 2]:
        w_col = membership_cols[cluster]
        hard_count = int((global_partition["hard_cluster"].astype(int) == cluster).sum())
        effective_size = float(global_partition[w_col].sum())

        for (family, variable), group in region_mean.groupby(
            ["family", "variable"], sort=True
        ):
            w = group[w_col].to_numpy(dtype=float)
            row = {
                "cluster": cluster,
                "family": family,
                "variable": variable,
                "common_weight": float(common_weight_lookup[(family, variable)]),
                "hard_cluster_size": hard_count,
                "effective_fuzzy_size": effective_size,
            }
            for metric in metric_cols:
                values = group[metric].to_numpy(dtype=float)
                prefix = metric.replace("contribution_share_of_family_score", "share_of_family_score")
                row[f"weighted_mean_{prefix}_2015_2019"] = weighted_mean(values, w)
                row[f"weighted_median_{prefix}_2015_2019"] = weighted_quantile(values, w, 0.50)
                row[f"weighted_q25_{prefix}_2015_2019"] = weighted_quantile(values, w, 0.25)
                row[f"weighted_q75_{prefix}_2015_2019"] = weighted_quantile(values, w, 0.75)
            summary_rows.append(row)

        for year in EXPECTED_YEARS:
            py = panel[panel["year"].eq(year)]
            for (family, variable), group in py.groupby(
                ["family", "variable"], sort=True
            ):
                w = group[w_col].to_numpy(dtype=float)
                row = {
                    "cluster": cluster,
                    "year": year,
                    "family": family,
                    "variable": variable,
                    "common_weight": float(common_weight_lookup[(family, variable)]),
                }
                for metric in metric_cols:
                    values = group[metric].to_numpy(dtype=float)
                    prefix = metric.replace("contribution_share_of_family_score", "share_of_family_score")
                    row[f"weighted_mean_{prefix}"] = weighted_mean(values, w)
                    row[f"weighted_median_{prefix}"] = weighted_quantile(values, w, 0.50)
                annual_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    annual = pd.DataFrame(annual_rows)
    return summary, annual


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


def build_component_panel(
    dynamic: pd.DataFrame,
    bod_contrib: pd.DataFrame,
    weight_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build underlying components whose sums reproduce each model feature.

    For BoD families, component_value is the Script-04 weighted contribution
    z_ijt * w_j. For the NTL single-indicator family, component_value is simply
    ntl_per_capita_norm.
    """
    bod = bod_contrib[[
        "region_id", "code", "year", "family", "variable",
        "value_raw", "value_normalized", "common_weight",
        "weighted_contribution", "family_score",
        "contribution_share_of_family_score",
    ]].copy()
    bod["feature"] = "bod_" + bod["family"].astype(str)
    bod["component_value"] = pd.to_numeric(
        bod["weighted_contribution"], errors="raise"
    )
    bod["component_kind"] = "BOD_WEIGHTED_CONTRIBUTION"

    ntl = dynamic[[
        "region_id", "code", "year", "ntl_per_capita_norm"
    ]].copy()
    ntl["family"] = "regional_economic_activity_proxy"
    ntl["variable"] = "ntl_per_capita_norm"
    ntl["value_raw"] = np.nan
    ntl["value_normalized"] = ntl["ntl_per_capita_norm"].astype(float)
    ntl["common_weight"] = 1.0
    ntl["weighted_contribution"] = ntl["ntl_per_capita_norm"].astype(float)
    ntl["family_score"] = ntl["ntl_per_capita_norm"].astype(float)
    ntl["contribution_share_of_family_score"] = np.where(
        ntl["family_score"].abs() > EPS, 1.0, np.nan
    )
    ntl["feature"] = "ntl_per_capita_norm"
    ntl["component_value"] = ntl["ntl_per_capita_norm"].astype(float)
    ntl["component_kind"] = "SINGLE_INDICATOR"
    ntl = ntl.drop(columns=["ntl_per_capita_norm"])

    components = pd.concat([bod, ntl], ignore_index=True, sort=False)

    feature_family = weight_table.set_index("feature")["family"].astype(str).to_dict()
    family_weight = (
        weight_table[["family", "family_distance_weight"]]
        .drop_duplicates()
        .set_index("family")["family_distance_weight"]
        .astype(float)
        .to_dict()
    )
    total_family_weight = float(sum(family_weight.values()))
    if total_family_weight <= 0:
        raise ValueError("Total family distance weight must be positive.")
    components["family_distance_weight_normalized"] = components["family"].map(
        {k: v / total_family_weight for k, v in family_weight.items()}
    )
    if components["family_distance_weight_normalized"].isna().any():
        raise ValueError("Underlying component could not be linked to a clustering family weight.")

    # Every model feature must be exactly reconstructible from its components.
    recon = (
        components.groupby(["region_id", "year", "feature"], as_index=False)
        .agg(reconstructed_feature=("component_value", "sum"))
    )
    dyn_long = dynamic.melt(
        id_vars=["region_id", "year"],
        value_vars=list(feature_family.keys()),
        var_name="feature",
        value_name="feature_value",
    )
    recon = recon.merge(
        dyn_long,
        on=["region_id", "year", "feature"],
        how="left",
        validate="one_to_one",
    )
    max_err = float(np.max(np.abs(recon["reconstructed_feature"] - recon["feature_value"])))
    if max_err > 1e-10:
        raise RuntimeError(
            "Underlying component panel does not reconstruct the three model features; "
            f"max error={max_err:.3e}."
        )

    return components.sort_values(
        ["region_id", "year", "family", "variable"], kind="stable"
    ).reset_index(drop=True)


def build_annual_medoid_distances(
    dynamic: pd.DataFrame,
    bod_contrib: pd.DataFrame,
    global_partition: pd.DataFrame,
    global_medoids: pd.DataFrame,
    weight_table: pd.DataFrame,
    features: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    feature_weight = (
        weight_table.set_index("feature")["effective_annual_distance_weight"].to_dict()
    )
    feature_family = weight_table.set_index("feature")["family"].astype(str).to_dict()
    medoid_panel = build_medoid_panel(dynamic, global_medoids, features)
    components = build_component_panel(dynamic, bod_contrib, weight_table)

    component_lookup = {
        (str(r.region_id), int(r.year), str(r.feature), str(r.variable)): float(r.component_value)
        for r in components.itertuples(index=False)
    }
    component_meta = (
        components[[
            "feature", "family", "variable", "component_kind", "common_weight"
        ]]
        .drop_duplicates()
        .sort_values(["feature", "variable"], kind="stable")
    )
    variables_by_feature = {
        feature: grp["variable"].astype(str).tolist()
        for feature, grp in component_meta.groupby("feature", sort=False)
    }
    common_weight_lookup = {
        (str(r.feature), str(r.variable)): float(r.common_weight)
        for r in component_meta.itertuples(index=False)
    }
    component_kind_lookup = {
        (str(r.feature), str(r.variable)): str(r.component_kind)
        for r in component_meta.itertuples(index=False)
    }

    medoid_region = (
        global_medoids.set_index("cluster")["region_id"].astype(str).to_dict()
    )

    part = global_partition[
        ["region_id", "code", "geo_code", "geo_name", "hard_cluster",
         "u_cluster_1", "u_cluster_2", "membership_margin"]
    ].copy()
    panel = dynamic.merge(
        part,
        on=["region_id", "code", "geo_code", "geo_name"],
        how="left",
        validate="many_to_one",
    )
    panel["hard_cluster"] = panel["hard_cluster"].astype(int)
    panel["destination_cluster"] = 3 - panel["hard_cluster"]
    panel["flow"] = (
        "C" + panel["hard_cluster"].astype(str)
        + "->C" + panel["destination_cluster"].astype(str)
    )

    annual_rows: list[dict] = []
    feature_rows_annual: list[dict] = []
    variable_rows_annual: list[dict] = []

    for row in panel.itertuples(index=False):
        region_id = str(row.region_id)
        year = int(row.year)
        origin = int(row.hard_cluster)
        destination = int(row.destination_cluster)
        origin_medoid_id = medoid_region[origin]
        destination_medoid_id = medoid_region[destination]

        origin_total = 0.0
        destination_total = 0.0
        feature_values: dict[str, tuple[float, float]] = {}

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
            feature_rows_annual.append({
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
                "normalized_gap_component": (d_destination - d_origin) / denom,
            })

            x_total = float(getattr(row, feature))
            o_total = float(medoid_panel[origin].loc[year, feature])
            d_total = float(medoid_panel[destination].loc[year, feature])
            delta_origin_total = x_total - o_total
            delta_destination_total = x_total - d_total
            w = float(feature_weight[feature])

            for variable in variables_by_feature[feature]:
                x_component = component_lookup[(region_id, year, feature, variable)]
                o_component = component_lookup[(origin_medoid_id, year, feature, variable)]
                d_component = component_lookup[(destination_medoid_id, year, feature, variable)]

                # Exact additive/Shapley decomposition of squared family gap.
                origin_component = w * (x_component - o_component) * delta_origin_total
                destination_component = (
                    w * (x_component - d_component) * delta_destination_total
                )
                variable_rows_annual.append({
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
                    "variable": variable,
                    "component_kind": component_kind_lookup[(feature, variable)],
                    "common_weight": common_weight_lookup[(feature, variable)],
                    "region_component_value": x_component,
                    "origin_medoid_component_value": o_component,
                    "destination_medoid_component_value": d_component,
                    "variable_distance_component_to_origin": origin_component,
                    "variable_distance_component_to_destination": destination_component,
                    "normalized_gap_component": (
                        destination_component - origin_component
                    ) / denom,
                })

    annual = pd.DataFrame(annual_rows)
    feature_annual = pd.DataFrame(feature_rows_annual)
    variable_annual = pd.DataFrame(variable_rows_annual)

    # Exact QA 1: feature components must sum to total relative gap.
    qa_feature = (
        feature_annual.groupby(["region_id", "year"], as_index=False)
        .agg(component_sum=("normalized_gap_component", "sum"))
        .merge(
            annual[["region_id", "year", "relative_destination_origin_gap"]],
            on=["region_id", "year"],
            how="left",
            validate="one_to_one",
        )
    )
    max_feature_err = float(np.max(np.abs(
        qa_feature["component_sum"] - qa_feature["relative_destination_origin_gap"]
    )))
    if max_feature_err > 1e-10:
        raise RuntimeError(
            "Feature contribution decomposition failed annual QA; "
            f"max error={max_feature_err:.3e}."
        )

    # Exact QA 2: underlying-variable components must also sum to total relative gap.
    qa_variable = (
        variable_annual.groupby(["region_id", "year"], as_index=False)
        .agg(component_sum=("normalized_gap_component", "sum"))
        .merge(
            annual[["region_id", "year", "relative_destination_origin_gap"]],
            on=["region_id", "year"],
            how="left",
            validate="one_to_one",
        )
    )
    max_variable_err = float(np.max(np.abs(
        qa_variable["component_sum"] - qa_variable["relative_destination_origin_gap"]
    )))
    if max_variable_err > 1e-10:
        raise RuntimeError(
            "Underlying-variable contribution decomposition failed annual QA; "
            f"max error={max_variable_err:.3e}."
        )

    # Region-level 2015 -> 2019 convergence.
    gap_wide = annual.pivot(
        index="region_id", columns="year", values="relative_destination_origin_gap"
    )
    if not {2015, 2019}.issubset(gap_wide.columns):
        raise RuntimeError("Annual medoid distances do not contain 2015 and 2019.")
    conv = part.copy().rename(columns={"hard_cluster": "origin_cluster"})
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
        conv[[
            "converged_toward_destination",
            "moved_away_from_destination",
            "no_direction_within_numeric_tolerance",
        ]]
        .astype(int)
        .sum(axis=1)
    )
    if not direction_count.eq(1).all():
        raise RuntimeError(
            "Convergence direction flags are not mutually exclusive/exhaustive."
        )

    def _region_level_contributions(
        annual_components: pd.DataFrame,
        grouping_cols: list[str],
    ) -> pd.DataFrame:
        wide = annual_components.pivot_table(
            index=["region_id", "flow"] + grouping_cols,
            columns="year",
            values="normalized_gap_component",
            aggfunc="first",
        ).reset_index()
        if 2015 not in wide.columns or 2019 not in wide.columns:
            raise RuntimeError("Contribution table is missing 2015 or 2019.")
        wide["convergence_contribution"] = wide[2015] - wide[2019]
        wide = wide.merge(
            conv[["region_id", "convergence_index", "converged_toward_destination"]],
            on="region_id",
            how="left",
            validate="many_to_one",
        )
        return wide

    feature_region = _region_level_contributions(
        feature_annual, ["feature", "family"]
    )
    variable_region = _region_level_contributions(
        variable_annual, ["feature", "family", "variable", "component_kind", "common_weight"]
    )

    # Exact QA: both feature and variable contributions sum to convergence index.
    for label, frame in [("feature", feature_region), ("variable", variable_region)]:
        qa = (
            frame.groupby("region_id", as_index=False)
            .agg(contribution_sum=("convergence_contribution", "sum"))
            .merge(
                conv[["region_id", "convergence_index"]],
                on="region_id",
                how="left",
                validate="one_to_one",
            )
        )
        max_err = float(np.max(np.abs(qa["contribution_sum"] - qa["convergence_index"])))
        if max_err > 1e-10:
            raise RuntimeError(
                f"{label.title()} contribution decomposition failed convergence QA; "
                f"max error={max_err:.3e}."
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

    # Dashboard-facing aggregation uses only regions whose overall convergence
    # index is positive for the selected directional flow.
    feature_convergers = feature_region[
        feature_region["converged_toward_destination"].fillna(False)
    ].copy()
    variable_convergers = variable_region[
        variable_region["converged_toward_destination"].fillna(False)
    ].copy()

    def _aggregate_contributions(
        frame: pd.DataFrame,
        keys: list[str],
        total_name: str,
    ) -> pd.DataFrame:
        rows = []
        if frame.empty:
            return pd.DataFrame()
        for key_values, g in frame.groupby(["flow"] + keys, sort=True):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            flow = key_values[0]
            key_vals = key_values[1:]
            positive = g["convergence_contribution"] > CONVERGENCE_NUMERIC_TOL
            row = {
                "flow": flow,
                "converging_regions_in_flow": int(g["region_id"].nunique()),
                "median_contribution": float(g["convergence_contribution"].median()),
                "mean_contribution": float(g["convergence_contribution"].mean()),
                "regions_positive_contribution": int(positive.sum()),
                "consistency_pct": float(100.0 * positive.mean()),
            }
            for k, v in zip(keys, key_vals):
                row[k] = v
            rows.append(row)
        out = pd.DataFrame(rows)
        if not out.empty:
            out["aggregation_scope"] = total_name
        return out

    feature_summary = _aggregate_contributions(
        feature_convergers,
        ["feature", "family"],
        "REGIONS_WITH_POSITIVE_OVERALL_CONVERGENCE",
    )
    variable_summary = _aggregate_contributions(
        variable_convergers,
        ["feature", "family", "variable", "component_kind", "common_weight"],
        "REGIONS_WITH_POSITIVE_OVERALL_CONVERGENCE",
    )

    family_region = (
        variable_region.groupby(["region_id", "flow", "family"], as_index=False)
        .agg(
            convergence_contribution=("convergence_contribution", "sum"),
            convergence_index=("convergence_index", "first"),
            converged_toward_destination=("converged_toward_destination", "first"),
        )
    )
    family_convergers = family_region[
        family_region["converged_toward_destination"].fillna(False)
    ].copy()
    family_summary = _aggregate_contributions(
        family_convergers,
        ["family"],
        "REGIONS_WITH_POSITIVE_OVERALL_CONVERGENCE",
    )

    return (
        annual,
        conv,
        flow_summary,
        feature_summary,
        family_summary,
        variable_region,
        variable_summary,
        variable_annual,
    )


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
    bod_variable_profile: pd.DataFrame,
    bod_variable_profile_annual: pd.DataFrame,
    transition_summary: pd.DataFrame,
    transition_regions: pd.DataFrame,
    convergence_flow: pd.DataFrame,
    convergence_region: pd.DataFrame,
    contrib_feature: pd.DataFrame,
    contrib_family: pd.DataFrame,
    contrib_variable: pd.DataFrame,
    contrib_variable_region: pd.DataFrame,
    country_nearest: pd.DataFrame,
    global_local_country: pd.DataFrame,
    global_local_region: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(RESULTS_WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
        profile_summary.to_excel(writer, sheet_name="Cluster_Profile", index=False)
        profile_annual.to_excel(writer, sheet_name="Profile_Annual", index=False)
        profile_comparison.to_excel(writer, sheet_name="Profile_Compare", index=False)
        bod_variable_profile.to_excel(writer, sheet_name="BoD_Var_Profile", index=False)
        bod_variable_profile_annual.to_excel(writer, sheet_name="BoD_Var_Annual", index=False)
        transition_summary.to_excel(writer, sheet_name="Fuzzy_Summary", index=False)
        transition_regions.to_excel(writer, sheet_name="Fuzzy_Regions", index=False)
        convergence_flow.to_excel(writer, sheet_name="Convergence_Flow", index=False)
        convergence_region.to_excel(writer, sheet_name="Convergence_Region", index=False)
        contrib_feature.to_excel(writer, sheet_name="Conv_Feature", index=False)
        contrib_family.to_excel(writer, sheet_name="Conv_Family", index=False)
        contrib_variable.to_excel(writer, sheet_name="Conv_Variable", index=False)
        contrib_variable_region.to_excel(writer, sheet_name="Conv_Var_Region", index=False)
        country_nearest.to_excel(writer, sheet_name="Country_Refs", index=False)
        global_local_country.to_excel(writer, sheet_name="Global_Local_Country", index=False)
        global_local_region.to_excel(writer, sheet_name="Global_Local_Region", index=False)
    autofit_workbook(RESULTS_WORKBOOK_OUTPUT)


def build_metadata(config: dict, features: list[str]) -> pd.DataFrame:
    rows = [
        ("stage", "06 — Evolution + interpretation"),
        ("script_version", "2.0"),
        ("analytical_period", "2015-2019"),
        ("regions", EXPECTED_REGIONS),
        ("countries", EXPECTED_COUNTRIES),
        ("frozen_global_c", EXPECTED_C),
        ("frozen_global_m", EXPECTED_M),
        ("script05_fuzzy_silhouette", config.get("fuzzy_silhouette")),
        ("script05_objective", config.get("objective")),
        ("features", ",".join(features)),
        ("dynamic_families", ",".join(EXPECTED_DYNAMIC_FAMILIES)),
        (
            "bod_common_weight_rule",
            "Script-04 weights are common across all 512 regions and all years 2015-2019 and are not re-estimated in Script 06",
        ),
        (
            "bod_decomposition_role",
            "DMU-year-variable common-weight contributions are retained for audit, cluster profiling and exact variable-level convergence attribution; they are not extra clustering dimensions",
        ),
        ("fuzzy_transition_rule", f"membership_margin < {TRANSITION_MARGIN_THRESHOLD}"),
        (
            "annual_distance_definition",
            "family-balanced weighted squared distance from each region-year to the corresponding fixed medoid-year",
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
            "feature_contribution_definition",
            "exact additive feature decomposition of change in normalized destination-origin distance gap",
        ),
        (
            "variable_contribution_definition",
            "exact Shapley allocation within BoD family squared gaps: phi_j = W_f*(c_ij-c_mj)*(S_i-S_m); single-indicator NTL reduces to ordinary squared-distance component",
        ),
        (
            "cluster_assignment_rule",
            "global cluster remains fixed for 2015-2019; convergence is not cluster reassignment",
        ),
        ("convergence_numeric_tolerance", CONVERGENCE_NUMERIC_TOL),
        (
            "contribution_aggregation_scope",
            "flow-level median (primary) and mean (complementary) among regions with positive overall convergence; consistency is the share with positive variable contribution",
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

    print_header("SCRIPT 06 v2.0 — EVOLUTION + INTERPRETATION")
    print(f"Project root: {ROOT}")
    print(f"Frozen global configuration: C={EXPECTED_C}, m={EXPECTED_M}")
    print(f"Analytical period: {EXPECTED_YEARS[0]}–{EXPECTED_YEARS[-1]}")
    print(f"Expected final dynamic features: {EXPECTED_DYNAMIC_FEATURES}")

    print_header("1/9 — VALIDATE SCRIPT 04 + SCRIPT 05 HANDOFF")
    validate_required_inputs()
    (
        dynamic,
        weights,
        country_ref,
        bod_contrib,
        global_partition,
        global_medoids,
        country_partition,
        country_summary,
        config,
        features,
    ) = load_and_validate_inputs()
    weight_table = build_feature_distance_weights(weights)
    print(f"Dynamic panel: {len(dynamic):,} rows | {dynamic['region_id'].nunique()} regions")
    print(f"Model features: {features}")
    print(f"Dynamic BoD variable contributions: {len(bod_contrib):,} rows")
    print("Script-05 frozen configuration and Script-04 common-weight decomposition validated.")

    print_header("2/9 — CLUSTER PROFILES — MODEL FEATURES")
    profile_summary, profile_annual, profile_comparison = build_cluster_profiles(
        dynamic, global_partition, global_medoids, features
    )
    print(profile_comparison.to_string(index=False))

    print_header("3/9 — CLUSTER PROFILES — UNDERLYING BoD VARIABLES")
    bod_variable_profile, bod_variable_profile_annual = build_bod_variable_profiles(
        bod_contrib, global_partition
    )
    print(
        bod_variable_profile[[
            "cluster", "family", "variable", "common_weight",
            "weighted_mean_value_normalized_2015_2019",
            "weighted_mean_weighted_contribution_2015_2019",
        ]].to_string(index=False)
    )

    print_header("4/9 — FUZZY TRANSITION DIAGNOSTIC")
    transition_regions, transition_summary = build_fuzzy_transition(global_partition)
    print(transition_summary.to_string(index=False))

    print_header("5/9 — TEMPORAL CONVERGENCE TO ALTERNATIVE GLOBAL PROFILE")
    (
        annual_medoid_distance,
        convergence_region,
        convergence_flow,
        contrib_feature,
        contrib_family,
        contrib_variable_region,
        contrib_variable,
        contrib_variable_annual,
    ) = build_annual_medoid_distances(
        dynamic,
        bod_contrib,
        global_partition,
        global_medoids,
        weight_table,
        features,
    )
    print(convergence_flow.to_string(index=False))

    print_header("6/9 — FEATURE/FAMILY/VARIABLE CONTRIBUTIONS TO CONVERGENCE")
    if not contrib_variable.empty:
        print(
            contrib_variable.sort_values(
                ["flow", "median_contribution"], ascending=[True, False]
            )[[
                "flow", "family", "variable", "common_weight",
                "converging_regions_in_flow", "median_contribution",
                "mean_contribution", "regions_positive_contribution", "consistency_pct",
            ]].to_string(index=False)
        )
    else:
        print("No regions showed positive convergence toward the alternative profile.")

    print_header("7/9 — REGION × COUNTRY-REFERENCE COMPARISON")
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

    print_header("8/9 — GLOBAL × INTRACOUNTRY COMPARISON")
    global_local_region, global_local_country = build_global_local_comparison(
        global_partition, country_partition, country_summary
    )
    print(global_local_country.to_string(index=False))

    print_header("9/9 — WRITE OUTPUTS + FINAL QA")
    metadata = build_metadata(config, features)

    write_csv(profile_summary, PROFILE_SUMMARY_OUTPUT)
    write_csv(profile_annual, PROFILE_ANNUAL_OUTPUT)
    write_csv(profile_comparison, PROFILE_COMPARISON_OUTPUT)
    write_csv(bod_variable_profile, BOD_VARIABLE_PROFILE_OUTPUT)
    write_csv(bod_variable_profile_annual, BOD_VARIABLE_PROFILE_ANNUAL_OUTPUT)
    write_csv(transition_regions, TRANSITION_REGIONS_OUTPUT)
    write_csv(transition_summary, TRANSITION_SUMMARY_OUTPUT)
    write_csv(annual_medoid_distance, ANNUAL_MEDOID_DISTANCE_OUTPUT)
    write_csv(convergence_region, CONVERGENCE_REGION_OUTPUT)
    write_csv(convergence_flow, CONVERGENCE_FLOW_OUTPUT)
    write_csv(contrib_feature, CONTRIB_FEATURE_OUTPUT)
    write_csv(contrib_family, CONTRIB_FAMILY_OUTPUT)
    write_csv(contrib_variable, CONTRIB_VARIABLE_OUTPUT)
    write_csv(contrib_variable_region, CONTRIB_VARIABLE_REGION_OUTPUT)
    write_csv(contrib_variable_annual, CONTRIB_VARIABLE_ANNUAL_OUTPUT)
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
        bod_variable_profile=bod_variable_profile,
        bod_variable_profile_annual=bod_variable_profile_annual,
        transition_summary=transition_summary,
        transition_regions=transition_regions,
        convergence_flow=convergence_flow,
        convergence_region=convergence_region,
        contrib_feature=contrib_feature,
        contrib_family=contrib_family,
        contrib_variable=contrib_variable,
        contrib_variable_region=contrib_variable_region,
        country_nearest=country_nearest,
        global_local_country=global_local_country,
        global_local_region=global_local_region,
    )

    # Final QA
    if len(convergence_region) != EXPECTED_REGIONS:
        raise RuntimeError("Convergence output lost regions.")
    direction_count = (
        convergence_region[[
            "converged_toward_destination",
            "moved_away_from_destination",
            "no_direction_within_numeric_tolerance",
        ]]
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

    # There must be one variable-level convergence row per region and each of
    # the six underlying analytical variables (5 BoD + 1 NTL).
    expected_variable_region_rows = EXPECTED_REGIONS * 6
    if len(contrib_variable_region) != expected_variable_region_rows:
        raise RuntimeError(
            f"Expected {expected_variable_region_rows:,} region-variable convergence rows, "
            f"found {len(contrib_variable_region):,}."
        )

    print("\nFINAL SCRIPT 06 SUMMARY")
    print("-" * 100)
    print(f"Frozen global C:                 {EXPECTED_C}")
    print(f"Frozen global m:                 {EXPECTED_M}")
    print(f"Regions profiled:                {EXPECTED_REGIONS}")
    print(f"Model features profiled:         {len(features)}")
    print(f"Underlying BoD variables:        {sum(len(v) for v in EXPECTED_BOD_VARIABLES.values())}")
    print(f"Fuzzy transition regions:        {len(transition_regions)}")
    print(
        "Regions converging to alternative profile: "
        f"{int(convergence_region['converged_toward_destination'].sum())}"
    )
    print(f"Variable convergence rows:       {len(contrib_variable_region):,}")
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
