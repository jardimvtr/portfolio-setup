from __future__ import annotations

"""
SCRIPT 05 v1.3 — DYNAMIC-ONLY M-Exp-FCMd GLOBAL + INTRACOUNTRY CLUSTERING

Purpose
-------
Run the complete P1 clustering stage using only the robust Mixed Fuzzy
C-Medoids model with exponential dissimilarity (M-Exp-FCMd), based exclusively
on the dynamic Script-04 analytical handoff.

Official P1 design implemented here
-----------------------------------
- Analytical period: 2015–2019.
- Official common sample: 512 subnational regions in 28 countries.
- Only dynamic variables/families enter clustering.
- Structural/static variables are excluded from Script 05 and from the final
  clustering representation.
- M-Exp-FCMd is the only clustering method.
- No K-means, DTW-FCMd, HMM, GMM, annual clustering, Xie-Beni,
  Partition Coefficient, Partition Entropy, or competing algorithms.
- Number of clusters C is selected with Fuzzy Silhouette only.
- C candidates: 2..6.
- Fuzziness m is examined over a small interpretable grid.
- Medoids are observed regions.
- Exponential transformation provides metric robustness to outlying trajectories.
- Bootstrap is used only as a stability diagnostic after the configuration is
  selected; it is NOT a competing cluster-validity criterion.
- Independent intracountry clustering is performed with the same dynamic
  variables and distance definition. The local number of clusters is selected
  independently by Fuzzy Silhouette. The global partition never constrains
  the country-specific partitions.

Core references
---------------
D'Urso, P., De Giovanni, L., & Massari, R. (2018).
Robust fuzzy clustering of multivariate time trajectories.
International Journal of Approximate Reasoning, 99, 12–38.
https://doi.org/10.1016/j.ijar.2018.05.002

Campello, R. J. G. B., & Hruschka, E. R. (2006).
A fuzzy extension of the silhouette width criterion for cluster analysis.
Fuzzy Sets and Systems, 157(21), 2858–2875.
https://doi.org/10.1016/j.fss.2006.07.006

Dynamic dissimilarity
---------------------
For dynamic family f:
    D_CS,f(i,h) = mean_t sum_j w_j (x_itj - x_htj)^2
    D_L,f(i,h)  = mean_t sum_j w_j [(x_itj-x_i,t-1,j)
                                   -(x_htj-x_h,t-1,j)]^2
    D_M,f       = 0.5 D_CS,f + 0.5 D_L,f

Because years are equally spaced annually, the first difference is the
one-year velocity term.

Overall base squared dissimilarity:
    D_base = weighted mean of all dynamic-family dissimilarities

Within-family weights and family weights are supplied by Script 04.

Robust exponential transformation:
    D_exp = 1 - exp(-beta * D_base)

beta is the inverse of the mean dissimilarity from the most central observed
trajectory q:
    q = argmin_q sum_i D_base(i,q)
    beta = [mean_i D_base(i,q)]^(-1)

Fuzzy C-medoids objective:
    J = sum_i sum_c u_ic^m D_exp(i, medoid_c)

Minimum-search strategy
-----------------------
- deterministic multi-start initialization;
- k-medoids++ style seeding;
- exact medoid update under the current memberships using a linear assignment
  step so medoids are distinct;
- best restart is the one with the smallest objective.

Cluster-number selection
------------------------
Fuzzy Silhouette is the sole principal criterion.
To avoid selecting C from one idiosyncratic fuzziness value:
1) evaluate C=2..6 for all m values;
2) calculate mean Fuzzy Silhouette across viable m values for each C;
3) select C with the highest mean Fuzzy Silhouette;
4) within selected C, select m with the highest Fuzzy Silhouette.

No other internal validity index is used to choose C.

Bootstrap stability
-------------------
A multinomial nonparametric bootstrap of regions is used. Multiplicity is
represented as observation weights so duplicate sampled regions do not become
duplicate medoid identities. Each bootstrap fit:
- uses the selected C and m;
- re-estimates beta in the bootstrap sample;
- predicts memberships for all original regions from the bootstrap medoids;
- aligns cluster labels to the full-sample medoids;
- reports Adjusted Rand Index for hard assignments and a fuzzy-membership
  similarity diagnostic.

The bootstrap measures stability; it does not alter the selected C.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
SCRIPT04_DIR = INTERIM_DIR / "script04"
SCRIPT05_DIR = INTERIM_DIR / "script05"
OUTPUTS_DIR = ROOT / "outputs"

SCRIPT04_DYNAMIC_INPUT = SCRIPT04_DIR / "04_final_dynamic_model_panel_2015_2019.csv"
SCRIPT04_FEATURE_WEIGHTS_INPUT = SCRIPT04_DIR / "04_final_feature_weights.csv"

EXPECTED_REGIONS = 512
EXPECTED_COUNTRIES = 28
EXPECTED_YEARS = [2015, 2016, 2017, 2018, 2019]

C_GRID = tuple(range(2, 7))
M_GRID = (1.5, 1.8, 2.0, 2.2, 2.5)

# Equal emphasis on instantaneous position and one-year evolution
# inside each dynamic family.
MIX_LEVEL_WEIGHT = 0.50
MIX_CHANGE_WEIGHT = 0.50

FUZZY_SILHOUETTE_ALPHA = 1.0

RANDOM_SEED = 20260919

# Multi-start search effort
# -------------------------
# The global C x m grid uses the SAME search effort for every configuration,
# which is important for a fair comparison of Fuzzy Silhouette values.
#
# 10,000 restarts per (C,m) provides a deep and uniform multi-start search
# across all 25 global C x m configurations.
GRID_N_STARTS = 10000
#
# After C and m are selected, the final global solution is refit with the same
# deep 10,000-restart budget, ensuring that the retained medoids are not based
# on a shallower search than the grid configurations.
FINAL_N_STARTS = 10000
#
# Country models are secondary/ex-post analyses. They receive enough search
# effort for reliable local fits without inheriting the much larger global
# final-search budget.
LOCAL_N_STARTS = 25
LOCAL_FINAL_N_STARTS = 100
#
# Bootstrap refits are repeated many times; keeping 4 starts per bootstrap
# replicate controls computational cost while the bootstrap itself supplies
# repeated perturbations of the sample.
BOOTSTRAP_N_STARTS = 4

GLOBAL_BOOTSTRAP_REPS = 500
LOCAL_BOOTSTRAP_REPS = 25

MAX_ITER = 100
OBJECTIVE_TOL = 1e-10

# Technical non-degeneration only; NOT a competing selection criterion.
MIN_CLUSTER_FRACTION = 0.01
MIN_CLUSTER_ABSOLUTE = 2

# Country-specific clusterization is skipped if a meaningful two-cluster
# partition cannot be assessed with at least 2 hard members per cluster.
MIN_COUNTRY_REGIONS = 6

# Require at least this fraction of the m-grid to yield technically viable
# partitions for a C value to enter the C-selection aggregation.
MIN_M_VIABILITY_FRACTION = 0.60

MODEL_LABEL = "M-Exp-FCMd — dynamic trajectories only"

# Outputs
DISTANCE_COMPONENTS_OUTPUT = SCRIPT05_DIR / "05_distance_components_summary.csv"
GLOBAL_GRID_OUTPUT = SCRIPT05_DIR / "05_global_grid_search.csv"
GLOBAL_C_SUMMARY_OUTPUT = SCRIPT05_DIR / "05_global_c_selection_summary.csv"
GLOBAL_PARTITION_OUTPUT = SCRIPT05_DIR / "05_global_partition.csv"
GLOBAL_MEMBERSHIPS_LONG_OUTPUT = SCRIPT05_DIR / "05_global_memberships_long.csv"
GLOBAL_MEDOIDS_OUTPUT = SCRIPT05_DIR / "05_global_medoids.csv"
GLOBAL_BOOTSTRAP_OUTPUT = SCRIPT05_DIR / "05_global_bootstrap_stability.csv"
GLOBAL_BOOTSTRAP_SUMMARY_OUTPUT = SCRIPT05_DIR / "05_global_bootstrap_summary.csv"
COUNTRY_SUMMARY_OUTPUT = SCRIPT05_DIR / "05_country_cluster_summary.csv"
COUNTRY_PARTITION_OUTPUT = SCRIPT05_DIR / "05_country_partition.csv"
COUNTRY_MEMBERSHIPS_LONG_OUTPUT = SCRIPT05_DIR / "05_country_memberships_long.csv"
COUNTRY_MEDOIDS_OUTPUT = SCRIPT05_DIR / "05_country_medoids.csv"
COUNTRY_BOOTSTRAP_OUTPUT = SCRIPT05_DIR / "05_country_bootstrap_stability.csv"
COUNTRY_BOOTSTRAP_SUMMARY_OUTPUT = SCRIPT05_DIR / "05_country_bootstrap_summary.csv"
REGION_TO_MEDOIDS_OUTPUT = SCRIPT05_DIR / "05_region_to_global_medoids_distances.csv"
MODEL_METADATA_OUTPUT = SCRIPT05_DIR / "05_model_metadata.csv"
SELECTED_CONFIGURATION_OUTPUT = SCRIPT05_DIR / "05_selected_configuration.json"
DISTANCE_MATRICES_OUTPUT = SCRIPT05_DIR / "05_distance_matrices.npz"
RESULTS_WORKBOOK_OUTPUT = OUTPUTS_DIR / "05_mexp_fcmd_results.xlsx"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FitResult:
    c: int
    m: float
    medoids: np.ndarray
    memberships: np.ndarray
    objective: float
    iterations: int
    converged: bool
    hard_labels: np.ndarray
    hard_sizes: np.ndarray
    effective_sizes: np.ndarray
    fuzzy_silhouette: float
    min_hard_cluster_size: int
    viable: bool
    seed: int


# =============================================================================
# BASIC IO / VALIDATION
# =============================================================================

def ensure_directories() -> None:
    SCRIPT05_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def validate_required_inputs() -> None:
    required = {
        "FINAL_DYNAMIC_MODEL": SCRIPT04_DYNAMIC_INPUT,
        "FINAL_FEATURE_WEIGHTS": SCRIPT04_FEATURE_WEIGHTS_INPUT,
    }
    missing = []
    for label, path in required.items():
        if path.exists():
            print(f"[FOUND]   {label:<24} {path.name}")
        else:
            print(f"[MISSING] {label:<24} {path}")
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Script 05 cannot start because required Script-04 handoff files "
            f"are missing: {missing}"
        )


def validate_and_prepare_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, list[str]
]:
    """Validate the frozen dynamic-only Script-04 handoff."""
    dynamic = read_csv(SCRIPT04_DYNAMIC_INPUT)
    weights_all = read_csv(SCRIPT04_FEATURE_WEIGHTS_INPUT)

    dyn_id = ["region_id", "code", "geo_code", "geo_name", "year"]
    missing_dyn_id = [c for c in dyn_id if c not in dynamic.columns]

    required_weight_cols = {
        "block", "feature", "family", "within_family_distance_weight",
        "family_distance_weight",
    }

    if missing_dyn_id:
        raise ValueError(
            f"Dynamic handoff missing identifier columns: {missing_dyn_id}"
        )
    if not required_weight_cols.issubset(weights_all.columns):
        raise ValueError(
            "Feature-weight metadata is missing columns: "
            f"{sorted(required_weight_cols - set(weights_all.columns))}"
        )

    dynamic["year"] = pd.to_numeric(
        dynamic["year"], errors="raise"
    ).astype(int)

    if dynamic["region_id"].nunique() != EXPECTED_REGIONS:
        raise ValueError(
            f"Expected {EXPECTED_REGIONS} dynamic regions, found "
            f"{dynamic['region_id'].nunique()}."
        )
    if dynamic["code"].nunique() != EXPECTED_COUNTRIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTRIES} countries, found "
            f"{dynamic['code'].nunique()}."
        )

    years = sorted(dynamic["year"].unique().tolist())
    if years != EXPECTED_YEARS:
        raise ValueError(
            f"Expected analytical years {EXPECTED_YEARS}, found {years}."
        )

    expected_rows = EXPECTED_REGIONS * len(EXPECTED_YEARS)
    if len(dynamic) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} dynamic rows, found {len(dynamic)}."
        )

    if dynamic.duplicated(["region_id", "year"]).any():
        raise ValueError("Dynamic handoff has duplicate region-year keys.")

    # Structural/static rows in Script-04 metadata are intentionally ignored.
    weights = weights_all[
        weights_all["block"].astype(str).str.upper().eq("DYNAMIC")
    ].copy()

    if weights.empty:
        raise ValueError(
            "No DYNAMIC rows found in Script-04 feature-weight metadata."
        )

    dyn_features = weights["feature"].astype(str).tolist()
    missing_dyn_features = [
        f for f in dyn_features if f not in dynamic.columns
    ]
    if missing_dyn_features:
        raise ValueError(
            "Dynamic model features missing from handoff: "
            f"{missing_dyn_features}"
        )

    if dynamic[dyn_features].isna().any().any():
        bad = dynamic[dyn_features].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(
            f"Dynamic analytical features contain missing values: {bad}"
        )

    for feature in dyn_features:
        vals = pd.to_numeric(
            dynamic[feature], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(
                f"Dynamic feature {feature} contains non-finite values."
            )

    for family, group in weights.groupby("family", dropna=False):
        w = pd.to_numeric(
            group["within_family_distance_weight"], errors="raise"
        ).to_numpy(dtype=float)
        if np.any(w < 0):
            raise ValueError(
                f"Negative within-family weight in DYNAMIC/{family}."
            )
        if not np.isclose(w.sum(), 1.0, atol=1e-10):
            raise ValueError(
                f"Within-family weights for DYNAMIC/{family} "
                f"sum to {w.sum()}, not 1."
            )

        fam_w = pd.to_numeric(
            group["family_distance_weight"], errors="raise"
        ).unique()
        if (
            len(fam_w) != 1
            or not np.isfinite(fam_w[0])
            or fam_w[0] < 0
        ):
            raise ValueError(
                f"Invalid family-level weight for DYNAMIC/{family}."
            )

    return dynamic, weights, dyn_features


# =============================================================================
# FAMILY-BALANCED MIXED DISSIMILITIES
# =============================================================================

def pairwise_weighted_squared_euclidean(
    values: np.ndarray,
    feature_weights: np.ndarray,
) -> np.ndarray:
    """Pairwise weighted squared Euclidean distance for an N x P matrix."""
    if values.ndim != 2:
        raise ValueError("values must be 2-D.")
    if values.shape[1] != len(feature_weights):
        raise ValueError("Feature weight length mismatch.")
    scaled = values * np.sqrt(feature_weights)[None, :]
    sq = np.sum(scaled * scaled, axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * (scaled @ scaled.T)
    dist = np.maximum(dist, 0.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def validate_distance_matrix(matrix: np.ndarray, label: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} is not square.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values.")
    if np.min(matrix) < -1e-10:
        raise ValueError(f"{label} contains materially negative values.")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{label} is not symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-10):
        raise ValueError(f"{label} diagonal is not zero.")


def build_distance_matrices(
    dynamic: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    float,
    int,
]:
    """
    Build the dynamic-only mixed trajectory dissimilarity and its robust
    exponential transformation.
    """
    meta = (
        dynamic[["region_id", "code", "geo_code", "geo_name"]]
        .drop_duplicates("region_id")
        .sort_values(
            ["code", "geo_code", "region_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    region_order = meta["region_id"].astype(str).tolist()
    region_to_pos = {
        region_id: pos
        for pos, region_id in enumerate(region_order)
    }

    dyn = dynamic.copy()
    dyn["region_id"] = dyn["region_id"].astype(str)
    dyn["_region_pos"] = dyn["region_id"].map(region_to_pos)
    dyn = dyn.sort_values(
        ["_region_pos", "year"],
        kind="stable",
    )

    n = len(meta)
    t = len(EXPECTED_YEARS)
    if len(dyn) != n * t:
        raise ValueError(
            "Dynamic panel cannot be reshaped to N x T."
        )

    components: dict[str, np.ndarray] = {}
    family_weights: dict[str, float] = {}

    for family, group in weights.groupby(
        "family",
        sort=False,
    ):
        family = str(family)
        features = group["feature"].astype(str).tolist()

        within_weights = pd.to_numeric(
            group["within_family_distance_weight"],
            errors="raise",
        ).to_numpy(dtype=float)

        family_weight_unique = pd.to_numeric(
            group["family_distance_weight"],
            errors="raise",
        ).unique()

        if len(family_weight_unique) != 1:
            raise ValueError(
                f"Family {family} has inconsistent family weights."
            )

        family_weight = float(family_weight_unique[0])
        family_weights[family] = family_weight

        x = (
            dyn[features]
            .to_numpy(dtype=float)
            .reshape(n, t, len(features))
        )

        d_cs = np.zeros((n, n), dtype=float)
        for tt in range(t):
            d_cs += pairwise_weighted_squared_euclidean(
                x[:, tt, :],
                within_weights,
            )
        d_cs /= t

        dx = np.diff(x, axis=1)
        d_l = np.zeros((n, n), dtype=float)
        for tt in range(t - 1):
            d_l += pairwise_weighted_squared_euclidean(
                dx[:, tt, :],
                within_weights,
            )
        d_l /= (t - 1)

        d_mixed = (
            MIX_LEVEL_WEIGHT * d_cs
            + MIX_CHANGE_WEIGHT * d_l
        )

        components[
            f"DYNAMIC::{family}::cross_sectional"
        ] = d_cs
        components[
            f"DYNAMIC::{family}::longitudinal"
        ] = d_l
        components[
            f"DYNAMIC::{family}::mixed"
        ] = d_mixed

    family_matrices: list[np.ndarray] = []
    family_weight_values: list[float] = []

    for family in (
        weights["family"]
        .drop_duplicates()
        .astype(str)
    ):
        family_matrices.append(
            components[f"DYNAMIC::{family}::mixed"]
        )
        family_weight_values.append(
            family_weights[family]
        )

    fam_w = np.asarray(
        family_weight_values,
        dtype=float,
    )

    if (
        np.any(fam_w < 0)
        or not np.isfinite(fam_w).all()
        or fam_w.sum() <= 0
    ):
        raise ValueError(
            "Invalid dynamic family-level distance weights."
        )

    d_base = np.zeros((n, n), dtype=float)
    for family_weight, matrix in zip(
        fam_w,
        family_matrices,
    ):
        d_base += family_weight * matrix
    d_base /= fam_w.sum()

    validate_distance_matrix(
        d_base,
        "D_base_dynamic_only",
    )

    central_index = int(
        np.argmin(
            d_base.sum(axis=1)
        )
    )
    central_mean = float(
        np.mean(
            d_base[:, central_index]
        )
    )

    if (
        not np.isfinite(central_mean)
        or central_mean <= 0
    ):
        raise ValueError(
            "Cannot compute exponential beta because "
            "central mean dissimilarity is invalid: "
            f"{central_mean}"
        )

    beta = 1.0 / central_mean

    d_exp = 1.0 - np.exp(
        -beta * d_base
    )
    d_exp = np.clip(
        d_exp,
        0.0,
        1.0,
    )
    np.fill_diagonal(
        d_exp,
        0.0,
    )

    validate_distance_matrix(
        d_exp,
        "D_exp_dynamic_only",
    )

    return (
        meta,
        d_base,
        d_exp,
        components,
        beta,
        central_index,
    )


def beta_for_weighted_bootstrap(
    d_base: np.ndarray,
    observation_weights: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[float, int]:
    weights = np.asarray(observation_weights, dtype=float)
    candidates = np.asarray(candidate_indices, dtype=int)
    if weights.shape != (d_base.shape[0],):
        raise ValueError("Bootstrap weight vector has invalid shape.")
    if weights.sum() <= 0:
        raise ValueError("Bootstrap weights sum to zero.")

    # Weighted analogue of q = argmin_q sum_i d^2(i,q), restricted to
    # candidate units actually present in the bootstrap sample.
    costs = (weights[:, None] * d_base[:, candidates]).sum(axis=0)
    q = int(candidates[int(np.argmin(costs))])
    central_mean = float(np.sum(weights * d_base[:, q]) / weights.sum())
    if not np.isfinite(central_mean) or central_mean <= 0:
        raise ValueError("Invalid weighted central mean dissimilarity.")
    return 1.0 / central_mean, q


# =============================================================================
# FUZZY C-MEDOIDS
# =============================================================================

def memberships_from_medoids(
    distance: np.ndarray,
    medoids: np.ndarray,
    m: float,
) -> np.ndarray:
    if m <= 1.0:
        raise ValueError("Fuzziness m must be > 1.")
    d = np.asarray(distance[:, medoids], dtype=float)
    n, c = d.shape
    u = np.zeros((n, c), dtype=float)
    power = 1.0 / (m - 1.0)
    eps = 1e-15

    zero_mask = d <= eps
    rows_with_zero = zero_mask.any(axis=1)

    if np.any(rows_with_zero):
        z = zero_mask[rows_with_zero]
        counts = z.sum(axis=1, keepdims=True)
        u[rows_with_zero] = z / counts

    rows_positive = ~rows_with_zero
    if np.any(rows_positive):
        dp = np.maximum(d[rows_positive], eps)
        # Stable inverse-distance form equivalent to the standard FCMd update:
        # u_ic ∝ d_ic^{-1/(m-1)} when 'distance' is the objective dissimilarity.
        inv = dp ** (-power)
        inv_sum = inv.sum(axis=1, keepdims=True)
        u[rows_positive] = inv / inv_sum

    if not np.allclose(u.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("Membership rows do not sum to one.")
    return u


def objective_value(
    distance: np.ndarray,
    medoids: np.ndarray,
    memberships: np.ndarray,
    m: float,
    observation_weights: np.ndarray | None = None,
) -> float:
    n = distance.shape[0]
    obs_w = (
        np.ones(n, dtype=float)
        if observation_weights is None
        else np.asarray(observation_weights, dtype=float)
    )
    d = distance[:, medoids]
    value = np.sum(obs_w[:, None] * (memberships ** m) * d)
    return float(value)


def kmedoids_plus_plus_init(
    distance: np.ndarray,
    c: int,
    rng: np.random.Generator,
    candidate_indices: np.ndarray | None = None,
    observation_weights: np.ndarray | None = None,
) -> np.ndarray:
    n = distance.shape[0]
    candidates = (
        np.arange(n, dtype=int)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int)
    )
    if len(candidates) < c:
        raise ValueError("Fewer medoid candidates than requested clusters.")

    obs_w = (
        np.ones(n, dtype=float)
        if observation_weights is None
        else np.asarray(observation_weights, dtype=float)
    )
    cand_w = np.maximum(obs_w[candidates], 0.0)
    if cand_w.sum() <= 0:
        cand_w = np.ones(len(candidates), dtype=float)

    first = int(rng.choice(candidates, p=cand_w / cand_w.sum()))
    selected = [first]

    while len(selected) < c:
        remaining = np.array(
            [idx for idx in candidates if idx not in selected], dtype=int
        )
        min_d = np.min(distance[np.ix_(remaining, np.asarray(selected))], axis=1)
        probs = np.maximum(min_d, 0.0) * np.maximum(obs_w[remaining], 0.0)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            next_medoid = int(rng.choice(remaining))
        else:
            next_medoid = int(rng.choice(remaining, p=probs / probs.sum()))
        selected.append(next_medoid)

    return np.asarray(selected, dtype=int)


def update_medoids_exact(
    distance: np.ndarray,
    memberships: np.ndarray,
    m: float,
    observation_weights: np.ndarray | None = None,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    n, c = memberships.shape
    obs_w = (
        np.ones(n, dtype=float)
        if observation_weights is None
        else np.asarray(observation_weights, dtype=float)
    )
    candidates = (
        np.arange(n, dtype=int)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int)
    )

    # cost[candidate cluster, candidate medoid]
    cluster_unit_weights = (memberships ** m) * obs_w[:, None]
    cost = cluster_unit_weights.T @ distance[:, candidates]

    if not np.isfinite(cost).all():
        raise RuntimeError("Non-finite medoid update costs.")

    # Linear assignment enforces distinct medoids and gives the globally best
    # set of distinct medoids for fixed memberships.
    row_ind, col_ind = linear_sum_assignment(cost)
    if len(row_ind) != c:
        raise RuntimeError("Could not assign one distinct medoid to each cluster.")

    medoids = np.empty(c, dtype=int)
    medoids[row_ind] = candidates[col_ind]
    if len(np.unique(medoids)) != c:
        raise RuntimeError("Medoid update produced duplicate medoids.")
    return medoids


def canonicalize_cluster_order(
    medoids: np.ndarray,
    memberships: np.ndarray,
    region_ids: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if region_ids is None:
        order = np.argsort(medoids)
    else:
        ids = np.asarray(list(region_ids), dtype=str)
        medoid_ids = ids[medoids]
        order = np.argsort(medoid_ids, kind="stable")
    return medoids[order], memberships[:, order]


def fit_fuzzy_cmedoids_once(
    distance: np.ndarray,
    c: int,
    m: float,
    seed: int,
    observation_weights: np.ndarray | None = None,
    candidate_indices: np.ndarray | None = None,
    region_ids: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    rng = np.random.default_rng(seed)
    medoids = kmedoids_plus_plus_init(
        distance,
        c,
        rng,
        candidate_indices=candidate_indices,
        observation_weights=observation_weights,
    )

    previous_objective = math.inf
    converged = False

    for iteration in range(1, MAX_ITER + 1):
        memberships = memberships_from_medoids(distance, medoids, m)
        new_medoids = update_medoids_exact(
            distance,
            memberships,
            m,
            observation_weights=observation_weights,
            candidate_indices=candidate_indices,
        )
        new_memberships = memberships_from_medoids(distance, new_medoids, m)
        obj = objective_value(
            distance,
            new_medoids,
            new_memberships,
            m,
            observation_weights=observation_weights,
        )

        if not np.isfinite(obj):
            raise RuntimeError("Non-finite fuzzy C-medoids objective.")

        medoids_unchanged = np.array_equal(
            np.sort(new_medoids), np.sort(medoids)
        )
        rel_change = (
            abs(previous_objective - obj) / max(abs(previous_objective), 1.0)
            if np.isfinite(previous_objective)
            else math.inf
        )

        medoids = new_medoids
        memberships = new_memberships

        if medoids_unchanged or rel_change <= OBJECTIVE_TOL:
            converged = True
            break
        previous_objective = obj

    medoids, memberships = canonicalize_cluster_order(
        medoids, memberships, region_ids=region_ids
    )
    obj = objective_value(
        distance,
        medoids,
        memberships,
        m,
        observation_weights=observation_weights,
    )
    return medoids, memberships, obj, iteration, converged


def minimum_allowed_cluster_size(n: int) -> int:
    return max(
        MIN_CLUSTER_ABSOLUTE,
        int(math.ceil(MIN_CLUSTER_FRACTION * n)),
    )


def fuzzy_silhouette(
    distance: np.ndarray,
    memberships: np.ndarray,
    alpha: float = FUZZY_SILHOUETTE_ALPHA,
) -> float:
    """Campello-Hruschka Fuzzy Silhouette using the induced hard partition."""
    n, c = memberships.shape
    hard = np.argmax(memberships, axis=1)
    sizes = np.bincount(hard, minlength=c)

    s = np.zeros(n, dtype=float)
    for i in range(n):
        p = int(hard[i])
        same = np.where(hard == p)[0]
        same = same[same != i]

        a = float(np.mean(distance[i, same])) if len(same) > 0 else 0.0

        other_means = []
        for q in range(c):
            if q == p:
                continue
            idx = np.where(hard == q)[0]
            if len(idx) == 0:
                continue
            other_means.append(float(np.mean(distance[i, idx])))

        if not other_means:
            s[i] = 0.0
            continue

        b = min(other_means)
        denom = max(a, b)
        s[i] = 0.0 if denom <= 0 else (b - a) / denom

    sorted_u = np.sort(memberships, axis=1)
    top = sorted_u[:, -1]
    second = sorted_u[:, -2] if c >= 2 else np.zeros(n)
    weights = np.maximum(top - second, 0.0) ** alpha
    denom = float(weights.sum())
    if denom <= 0:
        return float("nan")
    return float(np.sum(weights * s) / denom)


def fit_multistart(
    distance: np.ndarray,
    c: int,
    m: float,
    n_starts: int,
    base_seed: int,
    observation_weights: np.ndarray | None = None,
    candidate_indices: np.ndarray | None = None,
    region_ids: Iterable[str] | None = None,
) -> FitResult:
    best: tuple[np.ndarray, np.ndarray, float, int, bool, int] | None = None

    for start in range(n_starts):
        seed = int(base_seed + 104729 * start + 1009 * c + round(1000 * m))
        try:
            medoids, u, obj, iters, converged = fit_fuzzy_cmedoids_once(
                distance=distance,
                c=c,
                m=m,
                seed=seed,
                observation_weights=observation_weights,
                candidate_indices=candidate_indices,
                region_ids=region_ids,
            )
        except Exception:
            continue

        if best is None or obj < best[2] - 1e-12:
            best = (medoids, u, obj, iters, converged, seed)

    if best is None:
        raise RuntimeError(f"All {n_starts} starts failed for c={c}, m={m}.")

    medoids, u, obj, iters, converged, seed = best
    hard = np.argmax(u, axis=1)
    hard_sizes = np.bincount(hard, minlength=c)
    effective = u.sum(axis=0)
    min_size = int(hard_sizes.min())

    min_allowed = minimum_allowed_cluster_size(distance.shape[0])
    viable = bool(
        converged
        and len(np.unique(medoids)) == c
        and min_size >= min_allowed
        and np.all(effective > 1.0)
        and np.isfinite(obj)
    )
    fs = fuzzy_silhouette(distance, u)

    return FitResult(
        c=c,
        m=m,
        medoids=medoids,
        memberships=u,
        objective=obj,
        iterations=iters,
        converged=converged,
        hard_labels=hard,
        hard_sizes=hard_sizes,
        effective_sizes=effective,
        fuzzy_silhouette=fs,
        min_hard_cluster_size=min_size,
        viable=viable,
        seed=seed,
    )


# =============================================================================
# GLOBAL MODEL SELECTION
# =============================================================================

def run_global_grid(
    d_exp: np.ndarray,
    region_ids: list[str],
) -> tuple[pd.DataFrame, dict[tuple[int, float], FitResult]]:
    rows: list[dict[str, Any]] = []
    fits: dict[tuple[int, float], FitResult] = {}

    config_counter = 0
    for c in C_GRID:
        for m in M_GRID:
            config_counter += 1
            print(
                f"  [{config_counter:02d}/{len(C_GRID) * len(M_GRID)}] "
                f"c={c}, m={m:.1f}"
            )
            fit = fit_multistart(
                distance=d_exp,
                c=c,
                m=m,
                n_starts=GRID_N_STARTS,
                base_seed=RANDOM_SEED + 100000 * c + int(m * 1000),
                region_ids=region_ids,
            )
            fits[(c, m)] = fit

            rows.append({
                "c": c,
                "m": m,
                "objective": fit.objective,
                "fuzzy_silhouette": fit.fuzzy_silhouette,
                "iterations": fit.iterations,
                "converged": fit.converged,
                "viable": fit.viable,
                "min_hard_cluster_size": fit.min_hard_cluster_size,
                "hard_cluster_sizes": ",".join(map(str, fit.hard_sizes.tolist())),
                "effective_cluster_sizes": ",".join(
                    f"{x:.6f}" for x in fit.effective_sizes.tolist()
                ),
                "medoid_indices": ",".join(map(str, fit.medoids.tolist())),
                "selected_restart_seed": fit.seed,
                "multistarts": GRID_N_STARTS,
            })

    return pd.DataFrame(rows), fits


def select_global_configuration(
    grid: pd.DataFrame,
) -> tuple[int, float, pd.DataFrame]:
    viable = grid[
        grid["viable"].fillna(False)
        & np.isfinite(pd.to_numeric(grid["fuzzy_silhouette"], errors="coerce"))
    ].copy()

    if viable.empty:
        raise RuntimeError("No technically viable global M-Exp-FCMd configuration.")

    min_m_count = max(
        1, int(math.ceil(MIN_M_VIABILITY_FRACTION * len(M_GRID)))
    )

    c_summary = (
        viable.groupby("c", as_index=False)
        .agg(
            mean_fuzzy_silhouette=("fuzzy_silhouette", "mean"),
            min_fuzzy_silhouette=("fuzzy_silhouette", "min"),
            max_fuzzy_silhouette=("fuzzy_silhouette", "max"),
            sd_fuzzy_silhouette=("fuzzy_silhouette", "std"),
            viable_m_count=("m", "nunique"),
        )
    )
    c_summary["eligible_for_c_selection"] = (
        c_summary["viable_m_count"] >= min_m_count
    )

    eligible_c = c_summary[c_summary["eligible_for_c_selection"]].copy()
    if eligible_c.empty:
        raise RuntimeError(
            "No C value is viable for enough fuzziness settings to select C robustly."
        )

    # Sole principal criterion: highest mean Fuzzy Silhouette across m.
    # Smaller C is used only as a deterministic tie-break.
    eligible_c = eligible_c.sort_values(
        ["mean_fuzzy_silhouette", "c"],
        ascending=[False, True],
        kind="stable",
    )
    selected_c = int(eligible_c.iloc[0]["c"])

    within_c = viable[viable["c"].eq(selected_c)].copy()
    within_c["distance_from_m2"] = (within_c["m"] - 2.0).abs()
    within_c = within_c.sort_values(
        ["fuzzy_silhouette", "distance_from_m2", "m"],
        ascending=[False, True, True],
        kind="stable",
    )
    selected_m = float(within_c.iloc[0]["m"])

    c_summary["selected_c"] = c_summary["c"].eq(selected_c)
    return selected_c, selected_m, c_summary


# =============================================================================
# PARTITION OUTPUTS
# =============================================================================

def partition_dataframe(
    meta: pd.DataFrame,
    fit: FitResult,
    prefix: str = "u_cluster_",
) -> pd.DataFrame:
    out = meta.copy()
    hard = np.argmax(fit.memberships, axis=1)
    sorted_u = np.sort(fit.memberships, axis=1)

    out["hard_cluster"] = hard + 1
    out["max_membership"] = sorted_u[:, -1]
    out["second_membership"] = sorted_u[:, -2]
    out["membership_margin"] = sorted_u[:, -1] - sorted_u[:, -2]

    u_safe = np.clip(
        fit.memberships,
        1e-15,
        1.0,
    )
    entropy = -np.sum(
        fit.memberships * np.log(u_safe),
        axis=1,
    )
    out["membership_entropy_normalized"] = entropy / np.log(fit.c)

    for c in range(fit.c):
        out[f"{prefix}{c + 1}"] = fit.memberships[:, c]

    return out


def memberships_long_dataframe(
    meta: pd.DataFrame,
    memberships: np.ndarray,
    cluster_scope: str,
    country_code: str | None = None,
) -> pd.DataFrame:
    parts = []
    for c in range(memberships.shape[1]):
        part = meta.copy()
        part["cluster"] = c + 1
        part["membership"] = memberships[:, c]
        part["cluster_scope"] = cluster_scope
        if country_code is not None:
            part["country_cluster_code"] = country_code
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def medoids_dataframe(
    meta: pd.DataFrame,
    fit: FitResult,
    scope: str,
    country_code: str | None = None,
) -> pd.DataFrame:
    rows = []
    for c, medoid_idx in enumerate(fit.medoids):
        row = meta.iloc[int(medoid_idx)].to_dict()
        row.update({
            "cluster": c + 1,
            "scope": scope,
            "hard_cluster_size": int(fit.hard_sizes[c]),
            "effective_cluster_size": float(fit.effective_sizes[c]),
            "medoid_membership": float(fit.memberships[int(medoid_idx), c]),
        })
        if country_code is not None:
            row["country_cluster_code"] = country_code
        rows.append(row)
    return pd.DataFrame(rows)


def region_to_medoids_dataframe(
    meta: pd.DataFrame,
    fit: FitResult,
    d_base: np.ndarray,
    d_exp: np.ndarray,
) -> pd.DataFrame:
    parts = []
    for c, medoid in enumerate(fit.medoids):
        part = meta.copy()
        part["cluster"] = c + 1
        part["medoid_region_id"] = str(meta.iloc[int(medoid)]["region_id"])
        part["base_dissimilarity_to_medoid"] = d_base[:, medoid]
        part["exponential_dissimilarity_to_medoid"] = d_exp[:, medoid]
        part["membership"] = fit.memberships[:, c]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# STABILITY
# =============================================================================

def comb2(x: np.ndarray | int | float) -> np.ndarray | float:
    return np.asarray(x) * (np.asarray(x) - 1.0) / 2.0


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if len(a) != len(b):
        raise ValueError("ARI label vectors must have equal length.")

    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    contingency = np.zeros((len(a_vals), len(b_vals)), dtype=int)
    np.add.at(contingency, (a_inv, b_inv), 1)

    sum_comb = float(np.sum(comb2(contingency)))
    row_comb = float(np.sum(comb2(contingency.sum(axis=1))))
    col_comb = float(np.sum(comb2(contingency.sum(axis=0))))
    total_comb = float(comb2(len(a)))

    if total_comb == 0:
        return 1.0

    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) <= 1e-15:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((sum_comb - expected) / denom)


def align_bootstrap_to_reference(
    d_base: np.ndarray,
    reference_medoids: np.ndarray,
    bootstrap_medoids: np.ndarray,
    bootstrap_memberships: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    cost = d_base[np.ix_(reference_medoids, bootstrap_medoids)]
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.empty(len(reference_medoids), dtype=int)
    order[row_ind] = col_ind

    aligned_medoids = bootstrap_medoids[order]
    aligned_u = bootstrap_memberships[:, order]
    mean_medoid_distance = float(
        np.mean(d_base[reference_medoids, aligned_medoids])
    )
    return aligned_medoids, aligned_u, mean_medoid_distance


def bootstrap_stability(
    d_base: np.ndarray,
    full_fit: FitResult,
    region_ids: list[str],
    n_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = d_base.shape[0]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for b in range(1, n_reps + 1):
        counts = rng.multinomial(n, np.full(n, 1.0 / n))
        candidates = np.where(counts > 0)[0]

        if len(candidates) < full_fit.c:
            rows.append({
                "bootstrap_rep": b,
                "status": "FAILED_TOO_FEW_UNIQUE_UNITS",
            })
            continue

        try:
            beta_b, q_b = beta_for_weighted_bootstrap(
                d_base, counts, candidates
            )
            d_exp_b = 1.0 - np.exp(-beta_b * d_base)
            d_exp_b = np.clip(d_exp_b, 0.0, 1.0)
            np.fill_diagonal(d_exp_b, 0.0)

            boot_fit = fit_multistart(
                distance=d_exp_b,
                c=full_fit.c,
                m=full_fit.m,
                n_starts=BOOTSTRAP_N_STARTS,
                base_seed=seed + 1000003 * b,
                observation_weights=counts.astype(float),
                candidate_indices=candidates,
                region_ids=region_ids,
            )

            _, u_aligned, medoid_distance = align_bootstrap_to_reference(
                d_base,
                full_fit.medoids,
                boot_fit.medoids,
                boot_fit.memberships,
            )

            hard_full = np.argmax(full_fit.memberships, axis=1)
            hard_boot = np.argmax(u_aligned, axis=1)

            ari = adjusted_rand_index(hard_full, hard_boot)
            fuzzy_similarity = float(
                1.0 - np.mean(np.abs(full_fit.memberships - u_aligned))
            )
            fuzzy_similarity = float(np.clip(fuzzy_similarity, 0.0, 1.0))

            rows.append({
                "bootstrap_rep": b,
                "status": "SUCCESS",
                "unique_regions_in_bootstrap": int(len(candidates)),
                "beta_bootstrap": beta_b,
                "central_index_bootstrap": q_b,
                "adjusted_rand_index": ari,
                "fuzzy_membership_similarity": fuzzy_similarity,
                "mean_aligned_medoid_base_dissimilarity": medoid_distance,
                "bootstrap_objective": boot_fit.objective,
                "bootstrap_min_hard_cluster_size": boot_fit.min_hard_cluster_size,
                "bootstrap_viable": boot_fit.viable,
            })
        except Exception as exc:
            rows.append({
                "bootstrap_rep": b,
                "status": f"FAILED::{type(exc).__name__}",
                "error": str(exc),
            })

    detail = pd.DataFrame(rows)
    success = detail[detail["status"].eq("SUCCESS")].copy()

    if len(success) < max(5, int(math.ceil(0.80 * n_reps))):
        raise RuntimeError(
            f"Bootstrap stability had only {len(success)}/{n_reps} successful runs."
        )

    summary = pd.DataFrame([{
        "requested_reps": n_reps,
        "successful_reps": len(success),
        "success_rate": len(success) / n_reps,
        "ari_mean": float(success["adjusted_rand_index"].mean()),
        "ari_median": float(success["adjusted_rand_index"].median()),
        "ari_p05": float(success["adjusted_rand_index"].quantile(0.05)),
        "ari_p95": float(success["adjusted_rand_index"].quantile(0.95)),
        "fuzzy_similarity_mean": float(
            success["fuzzy_membership_similarity"].mean()
        ),
        "fuzzy_similarity_median": float(
            success["fuzzy_membership_similarity"].median()
        ),
        "fuzzy_similarity_p05": float(
            success["fuzzy_membership_similarity"].quantile(0.05)
        ),
        "fuzzy_similarity_p95": float(
            success["fuzzy_membership_similarity"].quantile(0.95)
        ),
        "mean_aligned_medoid_distance": float(
            success["mean_aligned_medoid_base_dissimilarity"].mean()
        ),
    }])
    return detail, summary


# =============================================================================
# COUNTRY-SPECIFIC MODELS
# =============================================================================

def fit_country_models(
    meta: pd.DataFrame,
    d_base: np.ndarray,
    selected_m: float,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame
]:
    summary_rows: list[dict[str, Any]] = []
    partition_parts: list[pd.DataFrame] = []
    membership_parts: list[pd.DataFrame] = []
    medoid_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    bootstrap_summary_parts: list[pd.DataFrame] = []

    for country_idx, (code, country_meta) in enumerate(
        meta.groupby("code", sort=True)
    ):
        idx = country_meta.index.to_numpy(dtype=int)
        n = len(idx)

        print(f"  {code}: {n} regions")

        if n < MIN_COUNTRY_REGIONS:
            summary_rows.append({
                "code": code,
                "n_regions": n,
                "status": "SKIPPED_TOO_SMALL",
                "selected_c": np.nan,
                "selected_m": selected_m,
            })
            part = country_meta.copy()
            part["local_status"] = "SKIPPED_TOO_SMALL"
            part["local_c"] = np.nan
            part["local_hard_cluster"] = np.nan
            partition_parts.append(part)
            continue

        d_country_base = d_base[np.ix_(idx, idx)]
        central_idx_local = int(np.argmin(d_country_base.sum(axis=1)))
        central_mean = float(np.mean(d_country_base[:, central_idx_local]))
        if not np.isfinite(central_mean) or central_mean <= 0:
            summary_rows.append({
                "code": code,
                "n_regions": n,
                "status": "SKIPPED_DEGENERATE_DISTANCE",
                "selected_c": np.nan,
                "selected_m": selected_m,
            })
            continue

        beta_country = 1.0 / central_mean
        d_country_exp = 1.0 - np.exp(-beta_country * d_country_base)
        d_country_exp = np.clip(d_country_exp, 0.0, 1.0)
        np.fill_diagonal(d_country_exp, 0.0)

        country_region_ids = country_meta["region_id"].astype(str).tolist()
        max_c = min(max(C_GRID), n - 1)
        candidate_c = [c for c in C_GRID if c <= max_c]

        grid_rows = []
        fit_by_c: dict[int, FitResult] = {}

        for c in candidate_c:
            try:
                fit = fit_multistart(
                    distance=d_country_exp,
                    c=c,
                    m=selected_m,
                    n_starts=LOCAL_N_STARTS,
                    base_seed=RANDOM_SEED + 10_000_000 + country_idx * 10000 + c * 100,
                    region_ids=country_region_ids,
                )
                fit_by_c[c] = fit
                grid_rows.append({
                    "c": c,
                    "fs": fit.fuzzy_silhouette,
                    "viable": fit.viable,
                    "objective": fit.objective,
                    "min_hard_cluster_size": fit.min_hard_cluster_size,
                })
            except Exception as exc:
                grid_rows.append({
                    "c": c,
                    "fs": np.nan,
                    "viable": False,
                    "objective": np.nan,
                    "min_hard_cluster_size": np.nan,
                    "error": str(exc),
                })

        country_grid = pd.DataFrame(grid_rows)
        viable = country_grid[
            country_grid["viable"].fillna(False)
            & np.isfinite(pd.to_numeric(country_grid["fs"], errors="coerce"))
        ].copy()

        if viable.empty:
            summary_rows.append({
                "code": code,
                "n_regions": n,
                "status": "SKIPPED_NO_VIABLE_PARTITION",
                "selected_c": np.nan,
                "selected_m": selected_m,
                "beta_country": beta_country,
            })
            continue

        viable = viable.sort_values(
            ["fs", "c"], ascending=[False, True], kind="stable"
        )
        selected_c = int(viable.iloc[0]["c"])

        # Refit selected local configuration with more starts.
        final_fit = fit_multistart(
            distance=d_country_exp,
            c=selected_c,
            m=selected_m,
            n_starts=LOCAL_FINAL_N_STARTS,
            base_seed=RANDOM_SEED + 20_000_000 + country_idx * 10000,
            region_ids=country_region_ids,
        )

        summary_rows.append({
            "code": code,
            "n_regions": n,
            "status": "SUCCESS",
            "selected_c": selected_c,
            "selected_m": selected_m,
            "beta_country": beta_country,
            "fuzzy_silhouette": final_fit.fuzzy_silhouette,
            "objective": final_fit.objective,
            "hard_cluster_sizes": ",".join(map(str, final_fit.hard_sizes.tolist())),
            "effective_cluster_sizes": ",".join(
                f"{x:.6f}" for x in final_fit.effective_sizes.tolist()
            ),
            "min_hard_cluster_size": final_fit.min_hard_cluster_size,
        })

        part = partition_dataframe(country_meta.reset_index(drop=True), final_fit, prefix="u_local_")
        part["local_status"] = "SUCCESS"
        part["local_c"] = selected_c
        part = part.rename(columns={"hard_cluster": "local_hard_cluster"})
        partition_parts.append(part)

        memberships = memberships_long_dataframe(
            country_meta.reset_index(drop=True),
            final_fit.memberships,
            cluster_scope="COUNTRY",
            country_code=str(code),
        )
        membership_parts.append(memberships)

        medoids = medoids_dataframe(
            country_meta.reset_index(drop=True),
            final_fit,
            scope="COUNTRY",
            country_code=str(code),
        )
        medoid_parts.append(medoids)

        # Local bootstrap is diagnostic only.
        try:
            b_detail, b_summary = bootstrap_stability(
                d_base=d_country_base,
                full_fit=final_fit,
                region_ids=country_region_ids,
                n_reps=LOCAL_BOOTSTRAP_REPS,
                seed=RANDOM_SEED + 30_000_000 + country_idx * 10000,
            )
            b_detail["code"] = code
            b_summary["code"] = code
            b_summary["selected_c"] = selected_c
            bootstrap_parts.append(b_detail)
            bootstrap_summary_parts.append(b_summary)
        except Exception as exc:
            bootstrap_summary_parts.append(pd.DataFrame([{
                "code": code,
                "selected_c": selected_c,
                "status": f"BOOTSTRAP_FAILED::{type(exc).__name__}",
                "error": str(exc),
            }]))

    summary = pd.DataFrame(summary_rows)
    partition = (
        pd.concat(partition_parts, ignore_index=True, sort=False)
        if partition_parts else pd.DataFrame()
    )
    memberships = (
        pd.concat(membership_parts, ignore_index=True, sort=False)
        if membership_parts else pd.DataFrame()
    )
    medoids = (
        pd.concat(medoid_parts, ignore_index=True, sort=False)
        if medoid_parts else pd.DataFrame()
    )
    bootstrap = (
        pd.concat(bootstrap_parts, ignore_index=True, sort=False)
        if bootstrap_parts else pd.DataFrame()
    )
    bootstrap_summary = (
        pd.concat(bootstrap_summary_parts, ignore_index=True, sort=False)
        if bootstrap_summary_parts else pd.DataFrame()
    )

    return summary, partition, memberships, medoids, bootstrap, bootstrap_summary


# =============================================================================
# EXCEL / METADATA
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
            for cell in column_cells[:2000]:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            ws.column_dimensions[letter].width = min(max(max_length + 2, 10), 60)
    wb.save(path)


def write_results_workbook(
    global_grid: pd.DataFrame,
    c_summary: pd.DataFrame,
    global_partition: pd.DataFrame,
    global_medoids: pd.DataFrame,
    global_bootstrap_summary: pd.DataFrame,
    country_summary: pd.DataFrame,
    country_partition: pd.DataFrame,
    country_medoids: pd.DataFrame,
    country_bootstrap_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(RESULTS_WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="Model_Metadata", index=False)
        distance_summary.to_excel(writer, sheet_name="Distance_Components", index=False)
        global_grid.to_excel(writer, sheet_name="Global_Grid", index=False)
        c_summary.to_excel(writer, sheet_name="Global_C_Selection", index=False)
        global_partition.to_excel(writer, sheet_name="Global_Partition", index=False)
        global_medoids.to_excel(writer, sheet_name="Global_Medoids", index=False)
        global_bootstrap_summary.to_excel(
            writer, sheet_name="Global_Bootstrap", index=False
        )
        country_summary.to_excel(writer, sheet_name="Country_Summary", index=False)
        country_partition.to_excel(
            writer, sheet_name="Country_Partition", index=False
        )
        country_medoids.to_excel(writer, sheet_name="Country_Medoids", index=False)
        country_bootstrap_summary.to_excel(
            writer, sheet_name="Country_Bootstrap", index=False
        )
    autofit_workbook(RESULTS_WORKBOOK_OUTPUT)


def build_model_metadata(
    selected_c: int,
    selected_m: float,
    beta: float,
    central_region_id: str,
    global_fs: float,
) -> pd.DataFrame:
    rows = [
        ("model", MODEL_LABEL),
        ("analytical_period", "2015-2019"),
        ("regions", EXPECTED_REGIONS),
        ("countries", EXPECTED_COUNTRIES),
        ("cluster_method", "M-Exp-FCMd only"),
        ("cluster_number_grid", ",".join(map(str, C_GRID))),
        ("fuzziness_grid", ",".join(map(str, M_GRID))),
        ("selected_c", selected_c),
        ("selected_m", selected_m),
        ("selected_fuzzy_silhouette", global_fs),
        ("fuzzy_silhouette_alpha", FUZZY_SILHOUETTE_ALPHA),
        ("c_selection_criterion", "Fuzzy Silhouette only"),
        (
            "c_selection_rule",
            "highest mean Fuzzy Silhouette across viable m; then highest FS m within selected c",
        ),
        ("dynamic_level_weight", MIX_LEVEL_WEIGHT),
        ("dynamic_change_weight", MIX_CHANGE_WEIGHT),
        (
            "dynamic_longitudinal_definition",
            "one-year first differences; annual spacing is constant",
        ),
        (
            "variable_scope",
            "dynamic trajectories only; all structural/static variables excluded from clustering",
        ),
        ("family_weighting", "Script-04 DYNAMIC family weights; currently equal across dynamic families"),
        ("within_family_weighting", "Script-04 within-family distance weights"),
        ("robust_transform", "1-exp(-beta*D_base)"),
        ("beta_global", beta),
        ("beta_central_region_id", central_region_id),
        ("grid_multistarts", GRID_N_STARTS),
        ("final_multistarts", FINAL_N_STARTS),
        ("local_grid_multistarts", LOCAL_N_STARTS),
        ("local_final_multistarts", LOCAL_FINAL_N_STARTS),
        ("global_bootstrap_reps", GLOBAL_BOOTSTRAP_REPS),
        ("local_bootstrap_reps", LOCAL_BOOTSTRAP_REPS),
        (
            "bootstrap_role",
            "stability diagnostic only; does not select c",
        ),
        (
            "local_clustering",
            "independent by country; same metric and selected global m; c selected independently by FS",
        ),
        (
            "main_reference",
            "D'Urso, De Giovanni & Massari (2018), DOI 10.1016/j.ijar.2018.05.002",
        ),
        (
            "fuzzy_silhouette_reference",
            "Campello & Hruschka (2006), DOI 10.1016/j.fss.2006.07.006",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    ensure_directories()

    print_header("SCRIPT 05 v1.3 — DYNAMIC-ONLY M-Exp-FCMd")
    print(f"Project root: {ROOT}")
    print(f"Analytical period: {EXPECTED_YEARS[0]}–{EXPECTED_YEARS[-1]}")
    print(f"Official scope: {EXPECTED_REGIONS} regions | {EXPECTED_COUNTRIES} countries")
    print(f"C grid: {C_GRID}")
    print(f"m grid: {M_GRID}")
    print(f"Global grid restarts per (C,m): {GRID_N_STARTS}")
    print(f"Final global restarts: {FINAL_N_STARTS}")
    print(f"Model: {MODEL_LABEL}")

    # ------------------------------------------------------------------
    # 1. Input handoff
    # ------------------------------------------------------------------
    print_header("1/9 — VALIDATE SCRIPT 04 HANDOFF")
    validate_required_inputs()
    dynamic, weights, dyn_features = validate_and_prepare_inputs()
    print(
        f"Dynamic handoff: {len(dynamic):,} rows | "
        f"{dynamic['region_id'].nunique()} regions | "
        f"{len(dyn_features)} model features"
    )
    # ------------------------------------------------------------------
    # 2. Dissimilarity
    # ------------------------------------------------------------------
    print_header("2/9 — BUILD DYNAMIC MIXED DISSIMILARITY")
    meta, d_base, d_exp, components, beta, central_index = build_distance_matrices(
        dynamic, weights
    )
    region_ids = meta["region_id"].astype(str).tolist()
    central_region_id = str(meta.iloc[central_index]["region_id"])
    distance_summary = []

    # Rebuild summary directly from matrices to include the final distance.
    tri = np.triu_indices(len(meta), 1)
    for key, matrix in components.items():
        block, family, component = key.split("::")
        distance_summary.append({
            "block": block,
            "family": family,
            "component": component,
            "mean_pairwise_dissimilarity": float(np.mean(matrix[tri])),
            "median_pairwise_dissimilarity": float(np.median(matrix[tri])),
            "max_pairwise_dissimilarity": float(np.max(matrix[tri])),
        })
    distance_summary.extend([
        {
            "block": "TOTAL",
            "family": "ALL_FAMILIES",
            "component": "base_family_balanced",
            "mean_pairwise_dissimilarity": float(np.mean(d_base[tri])),
            "median_pairwise_dissimilarity": float(np.median(d_base[tri])),
            "max_pairwise_dissimilarity": float(np.max(d_base[tri])),
        },
        {
            "block": "TOTAL",
            "family": "ALL_FAMILIES",
            "component": "robust_exponential",
            "mean_pairwise_dissimilarity": float(np.mean(d_exp[tri])),
            "median_pairwise_dissimilarity": float(np.median(d_exp[tri])),
            "max_pairwise_dissimilarity": float(np.max(d_exp[tri])),
        },
    ])
    distance_summary_df = pd.DataFrame(distance_summary)
    print(f"Global beta: {beta:.8f}")
    print(f"Central trajectory for beta: {central_region_id}")
    print(
        "Only dynamic trajectories enter the clustering distance. "
        "Structural/static variables are excluded by design."
    )

    # Persist compact matrices for exact reproducibility.
    np.savez_compressed(
        DISTANCE_MATRICES_OUTPUT,
        d_base=d_base,
        d_exp=d_exp,
        region_ids=np.asarray(region_ids, dtype=str),
        beta=np.asarray([beta], dtype=float),
    )

    # ------------------------------------------------------------------
    # 3. Global grid
    # ------------------------------------------------------------------
    print_header("3/9 — GLOBAL M-Exp-FCMd GRID: C × m")
    global_grid, grid_fits = run_global_grid(d_exp, region_ids)
    selected_c, selected_m, c_summary = select_global_configuration(global_grid)

    print("\nC SELECTION SUMMARY")
    print(c_summary.to_string(index=False))
    print(
        f"\nSelected by Fuzzy Silhouette only: "
        f"C={selected_c}, m={selected_m:.1f}"
    )

    # ------------------------------------------------------------------
    # 4. Final global fit
    # ------------------------------------------------------------------
    print_header("4/9 — REFIT SELECTED GLOBAL CONFIGURATION")
    global_fit = fit_multistart(
        distance=d_exp,
        c=selected_c,
        m=selected_m,
        n_starts=FINAL_N_STARTS,
        base_seed=RANDOM_SEED + 40_000_000,
        region_ids=region_ids,
    )
    if not global_fit.viable:
        raise RuntimeError(
            "Selected global configuration became technically non-viable "
            "during the higher-restart final fit."
        )

    print(f"Final objective: {global_fit.objective:.10f}")
    print(f"Fuzzy Silhouette: {global_fit.fuzzy_silhouette:.6f}")
    print(f"Hard cluster sizes: {global_fit.hard_sizes.tolist()}")
    print(
        "Effective fuzzy cluster sizes: "
        f"{[round(x, 3) for x in global_fit.effective_sizes.tolist()]}"
    )

    global_partition = partition_dataframe(meta, global_fit)
    global_memberships_long = memberships_long_dataframe(
        meta, global_fit.memberships, cluster_scope="GLOBAL"
    )
    global_medoids = medoids_dataframe(meta, global_fit, scope="GLOBAL")
    region_to_medoids = region_to_medoids_dataframe(
        meta, global_fit, d_base, d_exp
    )

    # ------------------------------------------------------------------
    # 5. Global stability
    # ------------------------------------------------------------------
    print_header("5/9 — GLOBAL BOOTSTRAP STABILITY")
    global_bootstrap, global_bootstrap_summary = bootstrap_stability(
        d_base=d_base,
        full_fit=global_fit,
        region_ids=region_ids,
        n_reps=GLOBAL_BOOTSTRAP_REPS,
        seed=RANDOM_SEED + 50_000_000,
    )
    print(global_bootstrap_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Independent country-specific models
    # ------------------------------------------------------------------
    print_header("6/9 — INDEPENDENT INTRACOUNTRY M-Exp-FCMd")
    (
        country_summary,
        country_partition,
        country_memberships,
        country_medoids,
        country_bootstrap,
        country_bootstrap_summary,
    ) = fit_country_models(
        meta=meta,
        d_base=d_base,
        selected_m=selected_m,
    )

    print("\nCOUNTRY CLUSTERING STATUS")
    if not country_summary.empty:
        print(
            country_summary[
                ["code", "n_regions", "status", "selected_c", "fuzzy_silhouette"]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 7. Final metadata / selected config
    # ------------------------------------------------------------------
    print_header("7/9 — FINAL MODEL METADATA")
    metadata = build_model_metadata(
        selected_c=selected_c,
        selected_m=selected_m,
        beta=beta,
        central_region_id=central_region_id,
        global_fs=global_fit.fuzzy_silhouette,
    )
    print(metadata.to_string(index=False))

    selected_config = {
        "model": MODEL_LABEL,
        "analytical_period": [EXPECTED_YEARS[0], EXPECTED_YEARS[-1]],
        "regions": EXPECTED_REGIONS,
        "countries": EXPECTED_COUNTRIES,
        "selected_c": selected_c,
        "selected_m": selected_m,
        "fuzzy_silhouette": global_fit.fuzzy_silhouette,
        "objective": global_fit.objective,
        "beta": beta,
        "beta_central_region_id": central_region_id,
        "medoid_region_ids": [
            str(meta.iloc[int(i)]["region_id"]) for i in global_fit.medoids
        ],
        "hard_cluster_sizes": global_fit.hard_sizes.tolist(),
        "effective_cluster_sizes": global_fit.effective_sizes.tolist(),
        "dynamic_level_weight": MIX_LEVEL_WEIGHT,
        "dynamic_change_weight": MIX_CHANGE_WEIGHT,
        "c_selection_criterion": "Fuzzy Silhouette only",
        "variable_scope": "dynamic trajectories only",
        "structural_static_variables_included": False,
    }
    with open(SELECTED_CONFIGURATION_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(selected_config, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 8. Persist
    # ------------------------------------------------------------------
    print_header("8/9 — WRITE SCRIPT 05 OUTPUTS")
    write_csv(distance_summary_df, DISTANCE_COMPONENTS_OUTPUT)
    write_csv(global_grid, GLOBAL_GRID_OUTPUT)
    write_csv(c_summary, GLOBAL_C_SUMMARY_OUTPUT)
    write_csv(global_partition, GLOBAL_PARTITION_OUTPUT)
    write_csv(global_memberships_long, GLOBAL_MEMBERSHIPS_LONG_OUTPUT)
    write_csv(global_medoids, GLOBAL_MEDOIDS_OUTPUT)
    write_csv(global_bootstrap, GLOBAL_BOOTSTRAP_OUTPUT)
    write_csv(global_bootstrap_summary, GLOBAL_BOOTSTRAP_SUMMARY_OUTPUT)
    write_csv(country_summary, COUNTRY_SUMMARY_OUTPUT)
    write_csv(country_partition, COUNTRY_PARTITION_OUTPUT)
    write_csv(country_memberships, COUNTRY_MEMBERSHIPS_LONG_OUTPUT)
    write_csv(country_medoids, COUNTRY_MEDOIDS_OUTPUT)
    write_csv(country_bootstrap, COUNTRY_BOOTSTRAP_OUTPUT)
    write_csv(country_bootstrap_summary, COUNTRY_BOOTSTRAP_SUMMARY_OUTPUT)
    write_csv(region_to_medoids, REGION_TO_MEDOIDS_OUTPUT)
    write_csv(metadata, MODEL_METADATA_OUTPUT)

    write_results_workbook(
        global_grid=global_grid,
        c_summary=c_summary,
        global_partition=global_partition,
        global_medoids=global_medoids,
        global_bootstrap_summary=global_bootstrap_summary,
        country_summary=country_summary,
        country_partition=country_partition,
        country_medoids=country_medoids,
        country_bootstrap_summary=country_bootstrap_summary,
        distance_summary=distance_summary_df,
        metadata=metadata,
    )

    # ------------------------------------------------------------------
    # 9. Final QA
    # ------------------------------------------------------------------
    print_header("9/9 — FINAL QA / SCRIPT 06 HANDOFF")
    if global_partition["region_id"].nunique() != EXPECTED_REGIONS:
        raise RuntimeError("Global partition lost regions.")
    membership_cols = [
        c for c in global_partition.columns if c.startswith("u_cluster_")
    ]
    if not np.allclose(
        global_partition[membership_cols].sum(axis=1).to_numpy(),
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError("Final global memberships do not sum to one.")

    if len(global_medoids) != selected_c:
        raise RuntimeError("Unexpected number of final global medoids.")

    print("\nFINAL SCRIPT 05 SUMMARY")
    print("-" * 100)
    print(f"Model:                         {MODEL_LABEL}")
    print(f"Regions:                       {EXPECTED_REGIONS}")
    print(f"Countries:                     {EXPECTED_COUNTRIES}")
    print(f"Selected C:                    {selected_c}")
    print(f"Selected m:                    {selected_m:.1f}")
    print(f"Grid restarts per (C,m):       {GRID_N_STARTS}")
    print(f"Final global restarts:         {FINAL_N_STARTS}")
    print(f"Fuzzy Silhouette:              {global_fit.fuzzy_silhouette:.6f}")
    print(f"Hard cluster sizes:             {global_fit.hard_sizes.tolist()}")
    print(f"Global bootstrap reps:          {GLOBAL_BOOTSTRAP_REPS}")
    print(
        "Global bootstrap ARI mean:     "
        f"{float(global_bootstrap_summary.iloc[0]['ari_mean']):.6f}"
    )
    print(
        "Global fuzzy stability mean:   "
        f"{float(global_bootstrap_summary.iloc[0]['fuzzy_similarity_mean']):.6f}"
    )
    successful_countries = int(
        country_summary["status"].eq("SUCCESS").sum()
    ) if not country_summary.empty else 0
    print(f"Countries locally clustered:    {successful_countries}")
    print(f"Global partition:               {GLOBAL_PARTITION_OUTPUT}")
    print(f"Global medoids:                 {GLOBAL_MEDOIDS_OUTPUT}")
    print(f"Country summary:                {COUNTRY_SUMMARY_OUTPUT}")
    print(f"Region-to-medoid distances:     {REGION_TO_MEDOIDS_OUTPUT}")
    print(f"Results workbook:               {RESULTS_WORKBOOK_OUTPUT}")
    print(f"Selected configuration JSON:    {SELECTED_CONFIGURATION_OUTPUT}")

    elapsed = time.perf_counter() - t0
    print("\nSCRIPT 05 COMPLETE — READY FOR SCRIPT 06")
    print(f"Script execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
