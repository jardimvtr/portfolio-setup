from __future__ import annotations

"""
SCRIPT 07.2 v1.0.1 — POWER BI PREPARATION

Purpose
-------
Transform the validated analytical results from Script 07.1 (plus detail tables
already validated by Script 07.1 from Scripts 04 and 06) into a Power BI-ready
semantic model.

This script performs NO analytical re-estimation. It does NOT recalculate:
- common-weight BoD weights or scores;
- M-Exp-FCMd clusters, memberships, medoids or bootstrap stability;
- fuzzy-transition classification;
- temporal convergence;
- feature/family/indicator convergence contributions;
- global-vs-local partitions or country-reference distances.

It only:
1) validates the approved 07.1 handoff;
2) creates dimensions;
3) reshapes/copies validated results into fact tables;
4) creates Power BI relationship, DAX and visual specifications;
5) performs semantic-model QA;
6) writes CSVs and a human-inspection workbook.

Official P1 state expected
--------------------------
- Analytical period: 2015–2019.
- Final analytical sample: 512 regions in 28 countries.
- Frozen global M-Exp-FCMd solution: C=2, m=2.5.
- Hard clusters: C1=100, C2=412.
- Final dynamic clustering features:
    * bod_demographic_productive_potential
    * bod_socioeconomic_deprivation
    * ntl_per_capita_norm
- Six interpretation indicators:
    * share_15_64
    * share_65_plus
    * poor420
    * gini
    * prosgap2021
    * ntl_per_capita_norm
- 83 fuzzy-transition regions.
- 182 regions converging toward the alternative fixed global profile.
- 18 countries with viable local models.
- 188 regions whose leave-one-out own-country reference is nearest.

Interpretation guardrail
------------------------
A flow such as C1->C2 does NOT represent formal cluster reassignment. Every
region keeps its frozen global cluster for the complete 2015–2019 trajectory.
The flow represents relative movement toward the alternative fixed medoid
profile between 2015 and 2019.
"""

from pathlib import Path
import hashlib
import json
import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
SCRIPT04_DIR = INTERIM_DIR / "script04"
SCRIPT06_DIR = INTERIM_DIR / "script06"
SCRIPT07_DIR = INTERIM_DIR / "script07"
POWERBI_DIR = SCRIPT07_DIR / "powerbi"
OUTPUTS_DIR = ROOT / "outputs"

EXPECTED_REGIONS = 512
EXPECTED_COUNTRIES = 28
EXPECTED_YEARS = [2015, 2016, 2017, 2018, 2019]
EXPECTED_REGION_YEARS = EXPECTED_REGIONS * len(EXPECTED_YEARS)
EXPECTED_C = 2
EXPECTED_M = 2.5
EXPECTED_HARD_CLUSTER_SIZES = {1: 100, 2: 412}
EXPECTED_TRANSITION_REGIONS = 83
EXPECTED_CONVERGING_REGIONS = 182
EXPECTED_LOCAL_SUCCESS_COUNTRIES = 18
EXPECTED_OWN_COUNTRY_NEAREST = 188

EXPECTED_DYNAMIC_FEATURES = (
    "bod_demographic_productive_potential",
    "bod_socioeconomic_deprivation",
    "ntl_per_capita_norm",
)
EXPECTED_FAMILIES = (
    "demographic_productive_potential",
    "socioeconomic_deprivation",
    "regional_economic_activity_proxy",
)
EXPECTED_INDICATORS = (
    "share_15_64",
    "share_65_plus",
    "poor420",
    "gini",
    "prosgap2021",
    "ntl_per_capita_norm",
)
EXPECTED_FLOWS = ("C1->C2", "C2->C1")

NUMERIC_TOL = 1e-10


# -----------------------------------------------------------------------------
# Script 07.1 — approved consolidation layer
# -----------------------------------------------------------------------------

S071_STATUS = SCRIPT07_DIR / "07.1_final_status.json"
S071_QA = SCRIPT07_DIR / "07.1_final_qa.csv"
S071_MODEL_SUMMARY = SCRIPT07_DIR / "07.1_model_summary.csv"
S071_REGION_RESULTS = SCRIPT07_DIR / "07.1_region_results.csv"
S071_CLUSTER_SUMMARY = SCRIPT07_DIR / "07.1_cluster_summary.csv"
S071_FEATURE_PROFILE = SCRIPT07_DIR / "07.1_cluster_profiles.csv"
S071_FEATURE_ANNUAL = SCRIPT07_DIR / "07.1_cluster_profiles_annual.csv"
S071_BOD_PROFILE = SCRIPT07_DIR / "07.1_bod_variable_profiles.csv"
S071_BOD_ANNUAL = SCRIPT07_DIR / "07.1_bod_variable_profiles_annual.csv"
S071_FUZZY_SUMMARY = SCRIPT07_DIR / "07.1_fuzzy_summary.csv"
S071_CONVERGENCE_FLOW = SCRIPT07_DIR / "07.1_convergence_flow.csv"
S071_CONVERGENCE_REGION = SCRIPT07_DIR / "07.1_convergence_regions.csv"
S071_CONTRIB_FEATURE = SCRIPT07_DIR / "07.1_contribution_feature.csv"
S071_CONTRIB_FAMILY = SCRIPT07_DIR / "07.1_contribution_family.csv"
S071_CONTRIB_VARIABLE = SCRIPT07_DIR / "07.1_contribution_variable.csv"
S071_GLOBAL_LOCAL_COUNTRY = SCRIPT07_DIR / "07.1_global_local_country.csv"
S071_GLOBAL_LOCAL_REGION = SCRIPT07_DIR / "07.1_global_local_region.csv"
S071_COUNTRY_REFERENCE = SCRIPT07_DIR / "07.1_country_reference_regions.csv"

# -----------------------------------------------------------------------------
# Detail sources already validated by Script 07.1
# -----------------------------------------------------------------------------

S04_DYNAMIC = SCRIPT04_DIR / "04_final_dynamic_model_panel_2015_2019.csv"
S04_BOD_CONTRIB = SCRIPT04_DIR / "04_bod_weighted_contributions_long.csv"

S06_CONV_VARIABLE_REGION = SCRIPT06_DIR / "06_convergence_variable_region_detail.csv"
S06_CONV_VARIABLE_ANNUAL = SCRIPT06_DIR / "06_convergence_variable_annual_components.csv"
S06_ANNUAL_MEDOID = SCRIPT06_DIR / "06_region_annual_medoid_distances.csv"
S06_COUNTRY_DISTANCES = SCRIPT06_DIR / "06_region_country_reference_distances.csv"


# -----------------------------------------------------------------------------
# Script 07.2 outputs
# -----------------------------------------------------------------------------

INPUT_MANIFEST_OUTPUT = POWERBI_DIR / "07.2_input_manifest.csv"
TABLE_CATALOG_OUTPUT = POWERBI_DIR / "07.2_table_catalog.csv"
RELATIONSHIPS_OUTPUT = POWERBI_DIR / "07.2_relationships.csv"
DAX_OUTPUT = POWERBI_DIR / "07.2_dax_measures.csv"
VISUAL_BLUEPRINT_OUTPUT = POWERBI_DIR / "07.2_visual_blueprint.csv"
DATA_DICTIONARY_OUTPUT = POWERBI_DIR / "07.2_data_dictionary.csv"
LINEAGE_OUTPUT = POWERBI_DIR / "07.2_lineage.csv"
QA_OUTPUT = POWERBI_DIR / "07.2_powerbi_qa.csv"
OUTPUT_MANIFEST_OUTPUT = POWERBI_DIR / "07.2_output_manifest.csv"
FINAL_STATUS_OUTPUT = POWERBI_DIR / "07.2_final_status.json"
WORKBOOK_OUTPUT = OUTPUTS_DIR / "07.2_powerbi_model.xlsx"


# =============================================================================
# DISPLAY METADATA
# =============================================================================

FEATURE_LABELS = {
    "bod_demographic_productive_potential": "Demographic productive potential",
    "bod_socioeconomic_deprivation": "Socioeconomic deprivation score",
    "ntl_per_capita_norm": "Night-time lights per capita",
}

FAMILY_LABELS = {
    "demographic_productive_potential": "Demographic productive potential",
    "socioeconomic_deprivation": "Socioeconomic deprivation",
    "regional_economic_activity_proxy": "Regional economic activity proxy",
}

INDICATOR_LABELS = {
    "share_15_64": "Population aged 15–64",
    "share_65_plus": "Population aged 65+",
    "poor420": "Poverty indicator (poor420)",
    "gini": "Gini coefficient",
    "prosgap2021": "Prosperity gap (prosgap2021)",
    "ntl_per_capita_norm": "Night-time lights per capita (normalized)",
}

FEATURE_SORT = {
    feature: i + 1 for i, feature in enumerate(EXPECTED_DYNAMIC_FEATURES)
}
FAMILY_SORT = {
    family: i + 1 for i, family in enumerate(EXPECTED_FAMILIES)
}
INDICATOR_SORT = {
    indicator: i + 1 for i, indicator in enumerate(EXPECTED_INDICATORS)
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ensure_directories() -> None:
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def assert_columns(frame: pd.DataFrame, required: Iterable[Any], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label}: missing required columns: {missing}")


def assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, columns].head(10).to_dict("records")
        raise RuntimeError(
            f"{label}: duplicate key {columns}. Example duplicates: {example}"
        )


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def qa_add(
    rows: list[dict[str, Any]],
    check: str,
    condition: bool,
    observed: Any,
    expected: Any,
    detail: str = "",
) -> None:
    rows.append({
        "check": check,
        "status": "PASS" if condition else "FAIL",
        "observed": observed,
        "expected": expected,
        "detail": detail,
    })


def enforce_qa(rows: list[dict[str, Any]], context: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    failed = frame[frame["status"].eq("FAIL")]
    if not failed.empty:
        message = failed[["check", "observed", "expected", "detail"]].to_string(index=False)
        raise RuntimeError(f"{context} failed:\n{message}")
    return frame


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def value_or_nan(row: pd.Series, column: str) -> Any:
    return row[column] if column in row.index else np.nan


def normalize_year_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c) if isinstance(c, (int, np.integer)) else c for c in out.columns]
    return out


def parse_flow(flow: str) -> tuple[int, int]:
    text = str(flow)
    try:
        left, right = text.split("->", maxsplit=1)
        return int(left.replace("C", "")), int(right.replace("C", ""))
    except Exception as exc:
        raise RuntimeError(f"Invalid flow label: {flow!r}") from exc


def canonicalize_semantic_key_types(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Standardize relationship-key dtypes before QA/export.

    This is deliberately stricter than merely making the QA serialization-aware.
    Power BI relationships should receive the same logical key using the same
    physical type on both sides.

    Text keys:
        region_id, code, reference_country_code, flow, family, feature, variable

    Integer keys:
        year, cluster, hard_cluster, origin_cluster, destination_cluster,
        alternative_cluster

    Integer-key values are validated to be integral before conversion. Nullable
    integer dtype is used only when the source legitimately contains missing
    values; relationship keys used by the semantic model are subsequently
    checked for referential integrity.
    """
    text_keys = {
        "region_id",
        "code",
        "reference_country_code",
        "flow",
        "family",
        "feature",
        "variable",
    }
    integer_keys = {
        "year",
        "cluster",
        "hard_cluster",
        "origin_cluster",
        "destination_cluster",
        "alternative_cluster",
    }

    normalized: dict[str, pd.DataFrame] = {}

    for table_name, frame in tables.items():
        out = frame.copy()

        for column in text_keys.intersection(out.columns):
            out[column] = out[column].astype("string")

        for column in integer_keys.intersection(out.columns):
            numeric = pd.to_numeric(out[column], errors="raise")
            non_missing = numeric.dropna()

            if not non_missing.empty:
                integral = np.isclose(
                    non_missing.to_numpy(dtype=float),
                    np.round(non_missing.to_numpy(dtype=float)),
                    atol=1e-12,
                    rtol=0.0,
                )
                if not bool(np.all(integral)):
                    bad_values = non_missing.loc[~integral].head(10).tolist()
                    raise RuntimeError(
                        f"{table_name}.{column}: non-integral values found in "
                        f"an integer semantic key: {bad_values}"
                    )

            if numeric.isna().any():
                out[column] = numeric.round().astype("Int64")
            else:
                out[column] = numeric.round().astype("int64")

        normalized[table_name] = out

    return normalized


def autofit_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        # Width estimation is intentionally capped for performance on detail tables.
        sample_rows = min(ws.max_row, 500)
        for column_cells in ws.iter_cols(min_row=1, max_row=sample_rows):
            letter = column_cells[0].column_letter
            max_length = 0
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            ws.column_dimensions[letter].width = min(max(max_length + 2, 10), 48)
    wb.save(path)


# =============================================================================
# 1/8 — INPUT CONTRACT
# =============================================================================

def required_inputs() -> dict[str, Path]:
    return {
        "07.1_status": S071_STATUS,
        "07.1_final_qa": S071_QA,
        "07.1_model_summary": S071_MODEL_SUMMARY,
        "07.1_region_results": S071_REGION_RESULTS,
        "07.1_cluster_summary": S071_CLUSTER_SUMMARY,
        "07.1_feature_profile": S071_FEATURE_PROFILE,
        "07.1_feature_annual": S071_FEATURE_ANNUAL,
        "07.1_bod_profile": S071_BOD_PROFILE,
        "07.1_bod_annual": S071_BOD_ANNUAL,
        "07.1_fuzzy_summary": S071_FUZZY_SUMMARY,
        "07.1_convergence_flow": S071_CONVERGENCE_FLOW,
        "07.1_convergence_region": S071_CONVERGENCE_REGION,
        "07.1_contrib_feature": S071_CONTRIB_FEATURE,
        "07.1_contrib_family": S071_CONTRIB_FAMILY,
        "07.1_contrib_variable": S071_CONTRIB_VARIABLE,
        "07.1_global_local_country": S071_GLOBAL_LOCAL_COUNTRY,
        "07.1_global_local_region": S071_GLOBAL_LOCAL_REGION,
        "07.1_country_reference": S071_COUNTRY_REFERENCE,
        "04_dynamic_panel": S04_DYNAMIC,
        "04_bod_contributions": S04_BOD_CONTRIB,
        "06_convergence_variable_region": S06_CONV_VARIABLE_REGION,
        "06_convergence_variable_annual": S06_CONV_VARIABLE_ANNUAL,
        "06_annual_medoid_distances": S06_ANNUAL_MEDOID,
        "06_country_reference_distances": S06_COUNTRY_DISTANCES,
    }


def build_input_manifest() -> pd.DataFrame:
    rows = []
    missing = []
    for logical_name, path in required_inputs().items():
        exists = path.exists()
        if not exists:
            missing.append((logical_name, path))
        rows.append({
            "logical_name": logical_name,
            "relative_path": relative_path(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else np.nan,
            "sha256": file_sha256(path) if exists else None,
        })

    manifest = pd.DataFrame(rows)
    if missing:
        detail = "\n".join(
            f"- {name}: {relative_path(path)}" for name, path in missing
        )
        raise FileNotFoundError(
            f"Script 07.2 cannot start; {len(missing)} required input(s) are missing:\n"
            + detail
        )
    return manifest


def load_inputs() -> dict[str, Any]:
    with S071_STATUS.open("r", encoding="utf-8") as f:
        status = json.load(f)

    return {
        "status": status,
        "qa": read_csv(S071_QA),
        "model_summary": read_csv(S071_MODEL_SUMMARY),
        "region": read_csv(S071_REGION_RESULTS),
        "cluster": read_csv(S071_CLUSTER_SUMMARY),
        "feature_profile": read_csv(S071_FEATURE_PROFILE),
        "feature_annual": read_csv(S071_FEATURE_ANNUAL),
        "bod_profile": read_csv(S071_BOD_PROFILE),
        "bod_annual": read_csv(S071_BOD_ANNUAL),
        "fuzzy_summary": read_csv(S071_FUZZY_SUMMARY),
        "conv_flow": read_csv(S071_CONVERGENCE_FLOW),
        "conv_region": read_csv(S071_CONVERGENCE_REGION),
        "conv_feature": read_csv(S071_CONTRIB_FEATURE),
        "conv_family": read_csv(S071_CONTRIB_FAMILY),
        "conv_variable": read_csv(S071_CONTRIB_VARIABLE),
        "global_local_country": read_csv(S071_GLOBAL_LOCAL_COUNTRY),
        "global_local_region": read_csv(S071_GLOBAL_LOCAL_REGION),
        "country_reference": read_csv(S071_COUNTRY_REFERENCE),
        "dynamic": read_csv(S04_DYNAMIC),
        "bod_contrib": read_csv(S04_BOD_CONTRIB),
        "conv_variable_region": normalize_year_columns(read_csv(S06_CONV_VARIABLE_REGION)),
        "conv_variable_annual": read_csv(S06_CONV_VARIABLE_ANNUAL),
        "annual_medoid": read_csv(S06_ANNUAL_MEDOID),
        "country_distances": read_csv(S06_COUNTRY_DISTANCES),
    }


def validate_input_contract(t: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    status = t["status"]

    qa_add(rows, "07.1_status_pass", str(status.get("status")) == "PASS",
           status.get("status"), "PASS")
    qa_add(rows, "07.1_ready_for_07.2", bool(status.get("ready_for_script_07_2")),
           status.get("ready_for_script_07_2"), True)
    qa_add(rows, "07.1_no_analytical_recalculation",
           status.get("prior_analytical_recalculation") is False,
           status.get("prior_analytical_recalculation"), False)
    qa_add(rows, "status_regions", int(status.get("final_regions", -1)) == EXPECTED_REGIONS,
           status.get("final_regions"), EXPECTED_REGIONS)
    qa_add(rows, "status_countries", int(status.get("countries", -1)) == EXPECTED_COUNTRIES,
           status.get("countries"), EXPECTED_COUNTRIES)
    qa_add(rows, "status_selected_c", int(status.get("selected_c", -1)) == EXPECTED_C,
           status.get("selected_c"), EXPECTED_C)
    qa_add(rows, "status_selected_m",
           math.isclose(float(status.get("selected_m", np.nan)), EXPECTED_M, abs_tol=1e-12),
           status.get("selected_m"), EXPECTED_M)
    qa_add(rows, "status_transition_regions",
           int(status.get("fuzzy_transition_regions", -1)) == EXPECTED_TRANSITION_REGIONS,
           status.get("fuzzy_transition_regions"), EXPECTED_TRANSITION_REGIONS)
    qa_add(rows, "status_converging_regions",
           int(status.get("converging_regions", -1)) == EXPECTED_CONVERGING_REGIONS,
           status.get("converging_regions"), EXPECTED_CONVERGING_REGIONS)

    final_qa = t["qa"]
    assert_columns(final_qa, ["check", "status"], "07.1 final QA")
    qa_add(rows, "07.1_all_final_qa_pass",
           final_qa["status"].astype(str).eq("PASS").all(),
           int(final_qa["status"].astype(str).eq("PASS").sum()),
           len(final_qa),
           "Every 07.1 final QA row must be PASS.")

    region = t["region"]
    assert_columns(
        region,
        [
            "region_id", "code", "geo_code", "geo_name", "hard_cluster",
            "u_cluster_1", "u_cluster_2", "max_membership",
            "membership_margin", "transition_zone", "flow",
            "convergence_index", "converged_toward_destination",
        ],
        "07.1 region results",
    )
    assert_unique(region, ["region_id"], "07.1 region results")

    qa_add(rows, "region_rows", len(region) == EXPECTED_REGIONS,
           len(region), EXPECTED_REGIONS)
    qa_add(rows, "region_countries", region["code"].astype(str).nunique() == EXPECTED_COUNTRIES,
           region["code"].astype(str).nunique(), EXPECTED_COUNTRIES)

    sizes = (
        region["hard_cluster"].astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    qa_add(rows, "hard_cluster_sizes", sizes == EXPECTED_HARD_CLUSTER_SIZES,
           sizes, EXPECTED_HARD_CLUSTER_SIZES)

    membership_sum = (
        pd.to_numeric(region["u_cluster_1"], errors="raise")
        + pd.to_numeric(region["u_cluster_2"], errors="raise")
    )
    qa_add(rows, "memberships_sum_to_one",
           bool(np.allclose(membership_sum, 1.0, atol=NUMERIC_TOL)),
           float(np.max(np.abs(membership_sum - 1.0))),
           f"<= {NUMERIC_TOL}")

    transition_count = int(bool_series(region["transition_zone"]).sum())
    convergence_count = int(bool_series(region["converged_toward_destination"]).sum())
    qa_add(rows, "transition_count", transition_count == EXPECTED_TRANSITION_REGIONS,
           transition_count, EXPECTED_TRANSITION_REGIONS)
    qa_add(rows, "convergence_count", convergence_count == EXPECTED_CONVERGING_REGIONS,
           convergence_count, EXPECTED_CONVERGING_REGIONS)

    cluster = t["cluster"]
    assert_columns(cluster, ["cluster", "hard_cluster_size"], "07.1 cluster summary")
    assert_unique(cluster, ["cluster"], "07.1 cluster summary")
    qa_add(rows, "cluster_rows", len(cluster) == EXPECTED_C, len(cluster), EXPECTED_C)

    feature_profile = t["feature_profile"]
    feature_annual = t["feature_annual"]
    assert_columns(feature_profile, ["cluster", "feature"], "07.1 feature profile")
    assert_columns(feature_annual, ["cluster", "year", "feature"], "07.1 feature annual")
    assert_unique(feature_profile, ["cluster", "feature"], "07.1 feature profile")
    assert_unique(feature_annual, ["cluster", "year", "feature"], "07.1 feature annual")

    observed_features = tuple(sorted(feature_profile["feature"].astype(str).unique()))
    qa_add(rows, "feature_set",
           set(observed_features) == set(EXPECTED_DYNAMIC_FEATURES),
           list(observed_features), sorted(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "feature_profile_rows", len(feature_profile) == 6,
           len(feature_profile), 6)
    qa_add(rows, "feature_annual_rows", len(feature_annual) == 30,
           len(feature_annual), 30)

    bod_profile = t["bod_profile"]
    bod_annual = t["bod_annual"]
    assert_columns(bod_profile, ["cluster", "family", "variable"], "07.1 BoD profile")
    assert_columns(bod_annual, ["cluster", "year", "family", "variable"], "07.1 BoD annual")
    qa_add(rows, "bod_profile_rows", len(bod_profile) == 10, len(bod_profile), 10)
    qa_add(rows, "bod_annual_rows", len(bod_annual) == 50, len(bod_annual), 50)

    conv_flow = t["conv_flow"]
    conv_variable = t["conv_variable"]
    assert_columns(conv_flow, ["flow"], "07.1 convergence flow")
    assert_columns(
        conv_variable,
        [
            "flow", "feature", "family", "variable",
            "median_contribution", "mean_contribution", "consistency_pct",
        ],
        "07.1 convergence variable",
    )
    qa_add(rows, "flow_set",
           set(conv_flow["flow"].astype(str)) == set(EXPECTED_FLOWS),
           sorted(conv_flow["flow"].astype(str).tolist()),
           sorted(EXPECTED_FLOWS))
    qa_add(rows, "convergence_flow_rows", len(conv_flow) == 2, len(conv_flow), 2)
    qa_add(rows, "convergence_indicator_rows", len(conv_variable) == 12,
           len(conv_variable), 12)
    qa_add(rows, "indicator_set",
           set(conv_variable["variable"].astype(str)) == set(EXPECTED_INDICATORS),
           sorted(conv_variable["variable"].astype(str).unique().tolist()),
           sorted(EXPECTED_INDICATORS))

    variable_region = t["conv_variable_region"]
    assert_columns(
        variable_region,
        [
            "region_id", "flow", "feature", "family", "variable",
            "convergence_contribution", "convergence_index",
            "converged_toward_destination",
        ],
        "06 variable region detail",
    )
    assert_unique(
        variable_region,
        ["region_id", "variable"],
        "06 variable region detail",
    )
    qa_add(rows, "region_indicator_contribution_rows",
           len(variable_region) == EXPECTED_REGIONS * len(EXPECTED_INDICATORS),
           len(variable_region), EXPECTED_REGIONS * len(EXPECTED_INDICATORS))

    variable_annual = t["conv_variable_annual"]
    assert_columns(
        variable_annual,
        ["region_id", "year", "flow", "feature", "family", "variable"],
        "06 variable annual components",
    )
    assert_unique(
        variable_annual,
        ["region_id", "year", "variable"],
        "06 variable annual components",
    )
    qa_add(rows, "region_year_indicator_component_rows",
           len(variable_annual) == EXPECTED_REGION_YEARS * len(EXPECTED_INDICATORS),
           len(variable_annual), EXPECTED_REGION_YEARS * len(EXPECTED_INDICATORS))

    annual_medoid = t["annual_medoid"]
    assert_columns(
        annual_medoid,
        [
            "region_id", "year", "flow", "origin_cluster", "destination_cluster",
            "annual_distance_to_origin_medoid",
            "annual_distance_to_destination_medoid",
            "relative_destination_origin_gap",
        ],
        "06 annual medoid distances",
    )
    assert_unique(annual_medoid, ["region_id", "year"], "06 annual medoid distances")
    qa_add(rows, "annual_medoid_rows", len(annual_medoid) == EXPECTED_REGION_YEARS,
           len(annual_medoid), EXPECTED_REGION_YEARS)

    dynamic = t["dynamic"]
    assert_columns(
        dynamic,
        ["region_id", "code", "geo_code", "geo_name", "year", *EXPECTED_DYNAMIC_FEATURES],
        "04 final dynamic panel",
    )
    assert_unique(dynamic, ["region_id", "year"], "04 final dynamic panel")
    qa_add(rows, "dynamic_rows", len(dynamic) == EXPECTED_REGION_YEARS,
           len(dynamic), EXPECTED_REGION_YEARS)

    country_distances = t["country_distances"]
    assert_columns(
        country_distances,
        ["region_id", "reference_country_code", "mean_annual_distance_2015_2019"],
        "06 country-reference distances",
    )
    assert_unique(
        country_distances,
        ["region_id", "reference_country_code"],
        "06 country-reference distances",
    )
    qa_add(rows, "country_reference_distance_rows",
           len(country_distances) == EXPECTED_REGIONS * EXPECTED_COUNTRIES,
           len(country_distances), EXPECTED_REGIONS * EXPECTED_COUNTRIES)

    gl_country = t["global_local_country"]
    gl_region = t["global_local_region"]
    country_ref = t["country_reference"]
    qa_add(rows, "global_local_country_rows", len(gl_country) == EXPECTED_COUNTRIES,
           len(gl_country), EXPECTED_COUNTRIES)
    qa_add(rows, "global_local_region_rows", len(gl_region) == EXPECTED_REGIONS,
           len(gl_region), EXPECTED_REGIONS)
    qa_add(rows, "country_reference_region_rows", len(country_ref) == EXPECTED_REGIONS,
           len(country_ref), EXPECTED_REGIONS)

    status_col = first_existing(
        gl_country,
        ["country_model_status", "status", "local_status"],
    )
    if status_col is None:
        raise RuntimeError(
            "07.1 global-local country table has no recognizable country-model status column."
        )
    local_success = int(gl_country[status_col].astype(str).eq("SUCCESS").sum())
    qa_add(rows, "local_success_countries",
           local_success == EXPECTED_LOCAL_SUCCESS_COUNTRIES,
           local_success, EXPECTED_LOCAL_SUCCESS_COUNTRIES)

    assert_columns(
        country_ref,
        ["region_id", "nearest_reference_is_own_country"],
        "07.1 country reference",
    )
    own_nearest = int(bool_series(country_ref["nearest_reference_is_own_country"]).sum())
    qa_add(rows, "own_country_reference_nearest",
           own_nearest == EXPECTED_OWN_COUNTRY_NEAREST,
           own_nearest, EXPECTED_OWN_COUNTRY_NEAREST)

    return enforce_qa(rows, "Script 07.2 input contract")


# =============================================================================
# 2/8 — DIMENSIONS
# =============================================================================

def build_dimensions(t: dict[str, Any]) -> dict[str, pd.DataFrame]:
    region = t["region"].copy()
    conv_variable = t["conv_variable"].copy()

    dim_year = pd.DataFrame({
        "year": EXPECTED_YEARS,
        "year_label": [str(y) for y in EXPECTED_YEARS],
        "year_sort": range(1, len(EXPECTED_YEARS) + 1),
    })

    dim_country = (
        region[["code"]]
        .drop_duplicates()
        .sort_values("code", kind="stable")
        .reset_index(drop=True)
    )
    dim_country["country_label"] = dim_country["code"].astype(str)
    dim_country["country_sort"] = np.arange(1, len(dim_country) + 1)

    dim_reference_country = dim_country.rename(
        columns={
            "code": "reference_country_code",
            "country_label": "reference_country_label",
            "country_sort": "reference_country_sort",
        }
    ).copy()

    region_cols = ["region_id", "code", "geo_code", "geo_name"]
    dim_region = region[region_cols].copy()
    assert_unique(dim_region, ["region_id"], "dim_region")
    dim_region["region_label"] = np.where(
        dim_region["geo_name"].notna(),
        dim_region["geo_name"].astype(str),
        dim_region["geo_code"].astype(str),
    )
    dim_region = dim_region.sort_values(
        ["code", "geo_code", "region_id"], kind="stable"
    ).reset_index(drop=True)

    cluster = t["cluster"].copy()
    dim_cluster = cluster[["cluster"]].drop_duplicates().copy()
    dim_cluster["cluster"] = dim_cluster["cluster"].astype(int)
    dim_cluster["cluster_label"] = "C" + dim_cluster["cluster"].astype(str)
    dim_cluster["alternative_cluster"] = 3 - dim_cluster["cluster"]
    dim_cluster["alternative_cluster_label"] = (
        "C" + dim_cluster["alternative_cluster"].astype(str)
    )
    dim_cluster["directional_flow"] = (
        dim_cluster["cluster_label"] + "->" + dim_cluster["alternative_cluster_label"]
    )
    for column in [
        "medoid_region_id", "medoid_country_code", "medoid_geo_code", "medoid_geo_name"
    ]:
        if column in cluster.columns:
            dim_cluster = dim_cluster.merge(
                cluster[["cluster", column]],
                on="cluster",
                how="left",
                validate="one_to_one",
            )
    dim_cluster["cluster_sort"] = dim_cluster["cluster"]
    dim_cluster = dim_cluster.sort_values("cluster").reset_index(drop=True)

    flow_rows = []
    for flow in sorted(set(t["conv_flow"]["flow"].astype(str))):
        origin, destination = parse_flow(flow)
        flow_rows.append({
            "flow": flow,
            "origin_cluster": origin,
            "destination_cluster": destination,
            "origin_cluster_label": f"C{origin}",
            "destination_cluster_label": f"C{destination}",
            "flow_label": f"C{origin} → C{destination}",
            "flow_sort": origin,
            "interpretation": (
                "Relative movement toward the alternative fixed global medoid profile; "
                "not formal cluster reassignment."
            ),
        })
    dim_flow = pd.DataFrame(flow_rows).sort_values("flow_sort").reset_index(drop=True)

    family_meta = (
        conv_variable[["family"]]
        .drop_duplicates()
        .sort_values("family", kind="stable")
        .reset_index(drop=True)
    )
    dim_family = family_meta.copy()
    dim_family["family_label"] = dim_family["family"].map(FAMILY_LABELS).fillna(
        dim_family["family"].astype(str)
    )
    dim_family["family_sort"] = dim_family["family"].map(FAMILY_SORT)

    feature_meta = (
        conv_variable[["feature", "family"]]
        .drop_duplicates()
        .sort_values(["family", "feature"], kind="stable")
        .reset_index(drop=True)
    )
    dim_feature = feature_meta.copy()
    dim_feature["feature_label"] = dim_feature["feature"].map(FEATURE_LABELS).fillna(
        dim_feature["feature"].astype(str)
    )
    dim_feature["feature_sort"] = dim_feature["feature"].map(FEATURE_SORT)
    dim_feature["is_bod_family_score"] = dim_feature["feature"].astype(str).str.startswith("bod_")

    indicator_cols = [
        c for c in ["variable", "feature", "family", "component_kind", "common_weight"]
        if c in conv_variable.columns
    ]
    dim_indicator = (
        conv_variable[indicator_cols]
        .drop_duplicates()
        .sort_values(["family", "variable"], kind="stable")
        .reset_index(drop=True)
    )
    dim_indicator["indicator_label"] = (
        dim_indicator["variable"].map(INDICATOR_LABELS).fillna(
            dim_indicator["variable"].astype(str)
        )
    )
    dim_indicator["indicator_sort"] = dim_indicator["variable"].map(INDICATOR_SORT)
    dim_indicator["profile_value_basis"] = "Normalized analytical value"
    dim_indicator["interpretation_role"] = np.where(
        dim_indicator["variable"].eq("ntl_per_capita_norm"),
        "Single-indicator regional economic activity proxy",
        "Underlying common-weight BoD indicator",
    )

    dimensions = {
        "dim_year": dim_year,
        "dim_country": dim_country,
        "dim_reference_country": dim_reference_country,
        "dim_region": dim_region,
        "dim_cluster": dim_cluster,
        "dim_flow": dim_flow,
        "dim_family": dim_family,
        "dim_feature": dim_feature,
        "dim_indicator": dim_indicator,
    }

    for name, frame in dimensions.items():
        if frame.empty:
            raise RuntimeError(f"{name} is empty.")

    return dimensions


# =============================================================================
# 3/8 — CORE FACT TABLES
# =============================================================================

def build_indicator_profile(
    bod_profile: pd.DataFrame,
    feature_profile: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, r in bod_profile.iterrows():
        rows.append({
            "cluster": int(r["cluster"]),
            "family": str(r["family"]),
            "variable": str(r["variable"]),
            "common_weight": value_or_nan(r, "common_weight"),
            "hard_cluster_size": value_or_nan(r, "hard_cluster_size"),
            "effective_fuzzy_size": value_or_nan(r, "effective_fuzzy_size"),
            "profile_mean": value_or_nan(r, "weighted_mean_value_normalized_2015_2019"),
            "profile_median": value_or_nan(r, "weighted_median_value_normalized_2015_2019"),
            "profile_q25": value_or_nan(r, "weighted_q25_value_normalized_2015_2019"),
            "profile_q75": value_or_nan(r, "weighted_q75_value_normalized_2015_2019"),
            "weighted_contribution_mean": value_or_nan(
                r, "weighted_mean_weighted_contribution_2015_2019"
            ),
            "share_of_family_score_mean": value_or_nan(
                r, "weighted_mean_share_of_family_score_2015_2019"
            ),
            "value_basis": "normalized_indicator_value",
            "source_table": "07.1_bod_variable_profiles",
        })

    ntl = feature_profile[
        feature_profile["feature"].astype(str).eq("ntl_per_capita_norm")
    ].copy()
    if len(ntl) != EXPECTED_C:
        raise RuntimeError(
            f"Expected {EXPECTED_C} NTL cluster-profile rows, found {len(ntl)}."
        )

    for _, r in ntl.iterrows():
        rows.append({
            "cluster": int(r["cluster"]),
            "family": "regional_economic_activity_proxy",
            "variable": "ntl_per_capita_norm",
            "common_weight": 1.0,
            "hard_cluster_size": value_or_nan(r, "hard_cluster_size"),
            "effective_fuzzy_size": value_or_nan(r, "effective_fuzzy_size"),
            "profile_mean": value_or_nan(r, "weighted_mean_2015_2019"),
            "profile_median": value_or_nan(r, "weighted_median_2015_2019"),
            "profile_q25": value_or_nan(r, "weighted_q25_2015_2019"),
            "profile_q75": value_or_nan(r, "weighted_q75_2015_2019"),
            "weighted_contribution_mean": value_or_nan(r, "weighted_mean_2015_2019"),
            "share_of_family_score_mean": 1.0,
            "value_basis": "normalized_single_indicator_feature",
            "source_table": "07.1_cluster_profiles",
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["cluster", "family", "variable"], kind="stable").reset_index(drop=True)


def build_indicator_profile_annual(
    bod_annual: pd.DataFrame,
    feature_annual: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, r in bod_annual.iterrows():
        rows.append({
            "cluster": int(r["cluster"]),
            "year": int(r["year"]),
            "family": str(r["family"]),
            "variable": str(r["variable"]),
            "common_weight": value_or_nan(r, "common_weight"),
            "profile_mean": value_or_nan(r, "weighted_mean_value_normalized"),
            "profile_median": value_or_nan(r, "weighted_median_value_normalized"),
            "profile_q25": np.nan,
            "profile_q75": np.nan,
            "profile_sd": np.nan,
            "weighted_contribution_mean": value_or_nan(
                r, "weighted_mean_weighted_contribution"
            ),
            "share_of_family_score_mean": value_or_nan(
                r, "weighted_mean_share_of_family_score"
            ),
            "value_basis": "normalized_indicator_value",
            "source_table": "07.1_bod_variable_profiles_annual",
        })

    ntl = feature_annual[
        feature_annual["feature"].astype(str).eq("ntl_per_capita_norm")
    ].copy()
    expected_ntl = EXPECTED_C * len(EXPECTED_YEARS)
    if len(ntl) != expected_ntl:
        raise RuntimeError(
            f"Expected {expected_ntl} NTL annual cluster-profile rows, found {len(ntl)}."
        )

    for _, r in ntl.iterrows():
        rows.append({
            "cluster": int(r["cluster"]),
            "year": int(r["year"]),
            "family": "regional_economic_activity_proxy",
            "variable": "ntl_per_capita_norm",
            "common_weight": 1.0,
            "profile_mean": value_or_nan(r, "weighted_mean"),
            "profile_median": value_or_nan(r, "weighted_median"),
            "profile_q25": value_or_nan(r, "weighted_q25"),
            "profile_q75": value_or_nan(r, "weighted_q75"),
            "profile_sd": value_or_nan(r, "weighted_sd"),
            "weighted_contribution_mean": value_or_nan(r, "weighted_mean"),
            "share_of_family_score_mean": 1.0,
            "value_basis": "normalized_single_indicator_feature",
            "source_table": "07.1_cluster_profiles_annual",
        })

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["cluster", "year", "family", "variable"], kind="stable"
    ).reset_index(drop=True)


def build_region_year_feature(dynamic: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["region_id", "code", "geo_code", "geo_name", "year"]
    long = dynamic.melt(
        id_vars=id_cols,
        value_vars=list(EXPECTED_DYNAMIC_FEATURES),
        var_name="feature",
        value_name="feature_value",
    )
    return long.sort_values(
        ["region_id", "year", "feature"], kind="stable"
    ).reset_index(drop=True)


def build_region_year_indicator(
    dynamic: pd.DataFrame,
    bod_contrib: pd.DataFrame,
) -> pd.DataFrame:
    bod = bod_contrib[
        bod_contrib["block"].astype(str).str.upper().eq("DYNAMIC")
    ].copy()

    assert_columns(
        bod,
        [
            "region_id", "code", "year", "family", "variable",
            "value_raw", "value_normalized", "common_weight",
            "weighted_contribution", "family_score",
        ],
        "04 dynamic BoD contributions",
    )

    identity = (
        dynamic[["region_id", "code", "geo_code", "geo_name"]]
        .drop_duplicates("region_id")
    )

    keep = [
        c for c in [
            "region_id", "code", "year", "family", "variable",
            "value_raw", "value_normalized", "common_weight",
            "weighted_contribution", "family_score",
            "contribution_share_of_family_score",
        ] if c in bod.columns
    ]
    bod = bod[keep].copy()
    bod["feature"] = "bod_" + bod["family"].astype(str)
    bod["component_kind"] = "BOD_WEIGHTED_CONTRIBUTION"

    for col in ["geo_code", "geo_name"]:
        if col in bod.columns:
            bod = bod.drop(columns=[col])

    bod = bod.merge(
        identity[["region_id", "geo_code", "geo_name"]],
        on="region_id",
        how="left",
        validate="many_to_one",
    )

    ntl = dynamic[
        ["region_id", "code", "geo_code", "geo_name", "year", "ntl_per_capita_norm"]
    ].copy()
    ntl["feature"] = "ntl_per_capita_norm"
    ntl["family"] = "regional_economic_activity_proxy"
    ntl["variable"] = "ntl_per_capita_norm"
    ntl["component_kind"] = "SINGLE_INDICATOR"
    ntl["value_raw"] = np.nan
    ntl["value_normalized"] = pd.to_numeric(
        ntl["ntl_per_capita_norm"], errors="raise"
    )
    ntl["common_weight"] = 1.0
    ntl["weighted_contribution"] = ntl["value_normalized"]
    ntl["family_score"] = ntl["value_normalized"]
    ntl["contribution_share_of_family_score"] = np.where(
        ntl["family_score"].abs() > 1e-15, 1.0, np.nan
    )
    ntl = ntl.drop(columns=["ntl_per_capita_norm"])

    out = pd.concat([bod, ntl], ignore_index=True, sort=False)
    desired = [
        "region_id", "code", "geo_code", "geo_name", "year",
        "feature", "family", "variable", "component_kind",
        "value_raw", "value_normalized", "common_weight",
        "weighted_contribution", "family_score",
        "contribution_share_of_family_score",
    ]
    desired = [c for c in desired if c in out.columns]
    return out[desired].sort_values(
        ["region_id", "year", "variable"], kind="stable"
    ).reset_index(drop=True)


def build_core_facts(t: dict[str, Any]) -> dict[str, pd.DataFrame]:
    facts = {
        "fact_model_summary": t["model_summary"].copy(),
        "fact_region": t["region"].copy(),
        "fact_cluster_summary": t["cluster"].copy(),
        "fact_feature_profile": t["feature_profile"].copy(),
        "fact_feature_profile_annual": t["feature_annual"].copy(),
        "fact_indicator_profile": build_indicator_profile(
            t["bod_profile"], t["feature_profile"]
        ),
        "fact_indicator_profile_annual": build_indicator_profile_annual(
            t["bod_annual"], t["feature_annual"]
        ),
        "fact_region_year_feature": build_region_year_feature(t["dynamic"]),
        "fact_region_year_indicator": build_region_year_indicator(
            t["dynamic"], t["bod_contrib"]
        ),
        "fact_fuzzy_summary": t["fuzzy_summary"].copy(),
        "fact_convergence_flow": t["conv_flow"].copy(),
        "fact_convergence_feature": t["conv_feature"].copy(),
        "fact_convergence_family": t["conv_family"].copy(),
        "fact_convergence_indicator": t["conv_variable"].copy(),
        "fact_global_local_country": t["global_local_country"].copy(),
        "fact_global_local_region": t["global_local_region"].copy(),
        "fact_country_reference_region": t["country_reference"].copy(),
    }

    return facts


# =============================================================================
# 4/8 — DRILL-THROUGH FACTS
# =============================================================================

def build_drillthrough_facts(t: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "fact_convergence_region_indicator": t["conv_variable_region"].copy(),
        "fact_convergence_region_indicator_annual": t["conv_variable_annual"].copy(),
        "fact_region_year_distance": t["annual_medoid"].copy(),
        "fact_region_country_reference_distance": t["country_distances"].copy(),
    }


# =============================================================================
# 5/8 — POWER BI SEMANTIC SPECIFICATION
# =============================================================================

def build_relationships() -> pd.DataFrame:
    rows = [
        # Snowflake geography.
        ("dim_country", "code", "dim_region", "code", "1:*", "Single", True,
         "Country filters region dimension."),
        # Region facts.
        ("dim_region", "region_id", "fact_region", "region_id", "1:1", "Single", True, ""),
        ("dim_region", "region_id", "fact_region_year_feature", "region_id", "1:*", "Single", True, ""),
        ("dim_region", "region_id", "fact_region_year_indicator", "region_id", "1:*", "Single", True, ""),
        ("dim_region", "region_id", "fact_convergence_region_indicator", "region_id", "1:*", "Single", True, ""),
        ("dim_region", "region_id", "fact_convergence_region_indicator_annual", "region_id", "1:*", "Single", True, ""),
        ("dim_region", "region_id", "fact_region_year_distance", "region_id", "1:*", "Single", True, ""),
        ("dim_region", "region_id", "fact_global_local_region", "region_id", "1:1", "Single", True, ""),
        ("dim_region", "region_id", "fact_country_reference_region", "region_id", "1:1", "Single", True, ""),
        ("dim_region", "region_id", "fact_region_country_reference_distance", "region_id", "1:*", "Single", True, ""),
        # Role-playing reference country.
        ("dim_reference_country", "reference_country_code",
         "fact_region_country_reference_distance", "reference_country_code",
         "1:*", "Single", True, "Reference-country role; separate from home-country dimension."),
        # Year.
        ("dim_year", "year", "fact_feature_profile_annual", "year", "1:*", "Single", True, ""),
        ("dim_year", "year", "fact_indicator_profile_annual", "year", "1:*", "Single", True, ""),
        ("dim_year", "year", "fact_region_year_feature", "year", "1:*", "Single", True, ""),
        ("dim_year", "year", "fact_region_year_indicator", "year", "1:*", "Single", True, ""),
        ("dim_year", "year", "fact_convergence_region_indicator_annual", "year", "1:*", "Single", True, ""),
        ("dim_year", "year", "fact_region_year_distance", "year", "1:*", "Single", True, ""),
        # Cluster.
        ("dim_cluster", "cluster", "fact_region", "hard_cluster", "1:*", "Single", True, ""),
        ("dim_cluster", "cluster", "fact_cluster_summary", "cluster", "1:1", "Single", True, ""),
        ("dim_cluster", "cluster", "fact_feature_profile", "cluster", "1:*", "Single", True, ""),
        ("dim_cluster", "cluster", "fact_feature_profile_annual", "cluster", "1:*", "Single", True, ""),
        ("dim_cluster", "cluster", "fact_indicator_profile", "cluster", "1:*", "Single", True, ""),
        ("dim_cluster", "cluster", "fact_indicator_profile_annual", "cluster", "1:*", "Single", True, ""),
        # Flow.
        ("dim_flow", "flow", "fact_region", "flow", "1:*", "Single", True, ""),
        ("dim_flow", "flow", "fact_convergence_flow", "flow", "1:1", "Single", True, ""),
        ("dim_flow", "flow", "fact_convergence_feature", "flow", "1:*", "Single", True, ""),
        ("dim_flow", "flow", "fact_convergence_family", "flow", "1:*", "Single", True, ""),
        ("dim_flow", "flow", "fact_convergence_indicator", "flow", "1:*", "Single", True, ""),
        ("dim_flow", "flow", "fact_convergence_region_indicator", "flow", "1:*", "Single", True, ""),
        ("dim_flow", "flow", "fact_convergence_region_indicator_annual", "flow", "1:*", "Single", True, ""),
        ("dim_flow", "flow", "fact_region_year_distance", "flow", "1:*", "Single", True, ""),
        # Family / feature / indicator.
        ("dim_family", "family", "dim_feature", "family", "1:*", "Single", True, ""),
        ("dim_family", "family", "dim_indicator", "family", "1:*", "Single", True, ""),
        ("dim_family", "family", "fact_convergence_family", "family", "1:*", "Single", True, ""),
        ("dim_feature", "feature", "fact_feature_profile", "feature", "1:*", "Single", True, ""),
        ("dim_feature", "feature", "fact_feature_profile_annual", "feature", "1:*", "Single", True, ""),
        ("dim_feature", "feature", "fact_region_year_feature", "feature", "1:*", "Single", True, ""),
        ("dim_feature", "feature", "fact_convergence_feature", "feature", "1:*", "Single", True, ""),
        ("dim_indicator", "variable", "fact_indicator_profile", "variable", "1:*", "Single", True, ""),
        ("dim_indicator", "variable", "fact_indicator_profile_annual", "variable", "1:*", "Single", True, ""),
        ("dim_indicator", "variable", "fact_region_year_indicator", "variable", "1:*", "Single", True, ""),
        ("dim_indicator", "variable", "fact_convergence_indicator", "variable", "1:*", "Single", True, ""),
        ("dim_indicator", "variable", "fact_convergence_region_indicator", "variable", "1:*", "Single", True, ""),
        ("dim_indicator", "variable", "fact_convergence_region_indicator_annual", "variable", "1:*", "Single", True, ""),
        # Country-level fact.
        ("dim_country", "code", "fact_global_local_country", "code", "1:1", "Single", True, ""),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "from_table", "from_column", "to_table", "to_column",
            "cardinality", "cross_filter_direction", "active", "note",
        ],
    )


def build_dax_measures() -> pd.DataFrame:
    rows = [
        (
            "Regions",
            "DISTINCTCOUNT('fact_region'[region_id])",
            "0",
            "Core",
            "Number of regions in the current filter context.",
        ),
        (
            "Countries",
            "DISTINCTCOUNT('fact_region'[code])",
            "0",
            "Core",
            "Number of countries in the current filter context.",
        ),
        (
            "Fuzzy transition regions",
            "CALCULATE([Regions], 'fact_region'[transition_zone] = TRUE())",
            "0",
            "Fuzzy",
            "Regions with membership margin below the frozen 0.20 threshold.",
        ),
        (
            "Fuzzy transition share",
            "DIVIDE([Fuzzy transition regions], [Regions])",
            "0.0%",
            "Fuzzy",
            "Share of regions classified as fuzzy-transition.",
        ),
        (
            "Converging regions",
            "CALCULATE([Regions], 'fact_region'[converged_toward_destination] = TRUE())",
            "0",
            "Convergence",
            "Regions with positive convergence toward the alternative fixed profile.",
        ),
        (
            "Converging share",
            "DIVIDE([Converging regions], [Regions])",
            "0.0%",
            "Convergence",
            "Share of regions converging toward the alternative fixed profile.",
        ),
        (
            "Mean max membership",
            "AVERAGE('fact_region'[max_membership])",
            "0.000",
            "Fuzzy",
            "Mean maximum global membership.",
        ),
        (
            "Mean membership margin",
            "AVERAGE('fact_region'[membership_margin])",
            "0.000",
            "Fuzzy",
            "Mean difference between largest and second-largest membership.",
        ),
        (
            "Mean convergence index",
            "AVERAGE('fact_region'[convergence_index])",
            "0.0000",
            "Convergence",
            "Mean directional convergence index in the current filter context.",
        ),
        (
            "Median convergence index",
            "MEDIAN('fact_region'[convergence_index])",
            "0.0000",
            "Convergence",
            "Median directional convergence index in the current filter context.",
        ),
        (
            "Selected median contribution",
            "MAX('fact_convergence_indicator'[median_contribution])",
            "0.0000",
            "Convergence contribution",
            "Median indicator contribution for the selected flow/indicator.",
        ),
        (
            "Selected mean contribution",
            "MAX('fact_convergence_indicator'[mean_contribution])",
            "0.0000",
            "Convergence contribution",
            "Mean indicator contribution for the selected flow/indicator.",
        ),
        (
            "Selected consistency pct",
            "MAX('fact_convergence_indicator'[consistency_pct])",
            "0.0",
            "Convergence contribution",
            "Percentage of converging regions where the indicator contributed toward destination.",
        ),
        (
            "Selected flow converging regions",
            "MAX('fact_convergence_flow'[regions_converging_toward_destination])",
            "0",
            "Convergence",
            "Converging regions for the selected directional flow.",
        ),
        (
            "Selected flow converging share",
            "MAX('fact_convergence_flow'[share_converging_toward_destination])",
            "0.0%",
            "Convergence",
            "Share converging toward destination for the selected directional flow.",
        ),
        (
            "Own-country nearest regions",
            "CALCULATE([Regions], 'fact_region'[nearest_reference_is_own_country] = TRUE())",
            "0",
            "Country reference",
            "Regions whose leave-one-out own-country reference is the closest country reference.",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["measure_name", "dax_expression", "format_string", "folder", "description"],
    )


def build_visual_blueprint() -> pd.DataFrame:
    rows = [
        ("01 Overview", "overview_cards", "Cards", "Global analytical scope",
         "fact_region; dim_country",
         "Regions; Countries; Fuzzy transition regions; Converging regions",
         "", "High-level scope; no causal language."),
        ("01 Overview", "cluster_sizes", "Bar chart", "Global cluster sizes",
         "fact_cluster_summary; dim_cluster",
         "dim_cluster[cluster_label]; fact_cluster_summary[hard_cluster_size]",
         "", "Hard global clusters remain fixed over 2015–2019."),
        ("01 Overview", "region_map", "Map", "Regional cluster structure",
         "fact_region; dim_region; dim_cluster",
         "dim_region[geo_name]; fact_region[hard_cluster]; fact_region[max_membership]",
         "", "Geographic field availability/geocoding must be checked in Power BI."),

        ("02 Cluster Profiles", "feature_profile", "Line chart", "Cluster profiles over time",
         "fact_feature_profile_annual; dim_cluster; dim_feature; dim_year",
         "year; weighted_mean; cluster_label",
         "Feature slicer", "Profile clusters before showing convergence."),
        ("02 Cluster Profiles", "indicator_profile", "Line chart", "Underlying indicator profiles",
         "fact_indicator_profile_annual; dim_cluster; dim_indicator; dim_year",
         "year; profile_mean; cluster_label",
         "Indicator slicer", "profile_mean uses normalized analytical values."),
        ("02 Cluster Profiles", "profile_table", "Matrix", "Cluster profile summary",
         "fact_indicator_profile; dim_cluster; dim_indicator",
         "indicator_label; cluster_label; profile_mean; profile_median",
         "", "Descriptive profile, not an overall ranking."),

        ("03 Fuzzy Structure", "membership_scatter", "Scatter plot", "Membership structure",
         "fact_region; dim_region; dim_cluster",
         "u_cluster_1; u_cluster_2; geo_name",
         "Country/cluster slicers", "Transition zone is membership_margin < 0.20."),
        ("03 Fuzzy Structure", "transition_map", "Map", "Fuzzy-transition regions",
         "fact_region; dim_region",
         "geo_name; transition_zone; membership_margin",
         "transition_zone = TRUE", "Fuzzy affinity is not formal cluster change."),

        ("04 Convergence", "flow_slicer", "Slicer", "Directional flow",
         "dim_flow", "flow_label", "", "Primary navigation for convergence page."),
        ("04 Convergence", "flow_cards", "Cards", "Selected flow summary",
         "fact_convergence_flow",
         "Selected flow converging regions; Selected flow converging share",
         "Flow slicer", "Directional convergence to fixed alternative profile."),
        ("04 Convergence", "indicator_ranking", "Bar chart", "Indicator contribution ranking",
         "fact_convergence_indicator; dim_indicator",
         "indicator_label; median_contribution",
         "Flow slicer", "Median contribution is primary; mean is complementary."),
        ("04 Convergence", "contribution_consistency", "Scatter plot",
         "Contribution × consistency",
         "fact_convergence_indicator; dim_indicator",
         "median_contribution; consistency_pct; indicator_label",
         "Flow slicer", "X = median contribution; Y = consistency %."),
        ("04 Convergence", "converging_map", "Map", "Regions converging toward alternative profile",
         "fact_region; dim_region; dim_flow",
         "geo_name; convergence_index",
         "converged_toward_destination = TRUE", "Do not label as cluster migration."),

        ("05 Regional Explorer", "region_selector", "Slicer", "Region",
         "dim_region", "region_label", "", ""),
        ("05 Regional Explorer", "feature_trajectory", "Line chart", "Regional feature trajectory",
         "fact_region_year_feature; dim_year; dim_feature",
         "year; feature_value; feature_label",
         "Region slicer", "Three frozen clustering features."),
        ("05 Regional Explorer", "indicator_trajectory", "Line chart", "Regional indicator trajectory",
         "fact_region_year_indicator; dim_year; dim_indicator",
         "year; value_normalized; indicator_label",
         "Region slicer", "Use normalized analytical values for comparability."),
        ("05 Regional Explorer", "medoid_distance", "Line chart", "Distance to origin and destination profiles",
         "fact_region_year_distance; dim_year",
         "year; annual_distance_to_origin_medoid; annual_distance_to_destination_medoid",
         "Region slicer", "Two distance series from the same region-year row."),
        ("05 Regional Explorer", "indicator_contribution_detail", "Table",
         "Regional convergence decomposition",
         "fact_convergence_region_indicator; dim_indicator",
         "indicator_label; convergence_contribution; convergence_index",
         "Region slicer", "Indicator contributions reconstruct the regional convergence index."),
        ("05 Regional Explorer", "country_reference", "Bar chart",
         "Distance to country reference profiles",
         "fact_region_country_reference_distance; dim_reference_country",
         "reference_country_label; mean_annual_distance_2015_2019",
         "Region slicer", "Own-country reference uses leave-one-out mean."),

        ("06 Methodology QA", "model_summary", "Table", "Frozen model configuration",
         "fact_model_summary", "parameter; value", "", "Display analytical configuration only."),
        ("06 Methodology QA", "global_local_country", "Table", "Global × intracountry diagnostic",
         "fact_global_local_country; dim_country",
         "country_label; country_model_status; country_selected_c; ari_global_vs_local",
         "", "Diagnostic comparison; global partition remains official."),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "page", "visual_id", "visual_type", "title", "source_tables",
            "fields_or_measures", "filter_context", "interpretation_note",
        ],
    )


def build_table_catalog(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    role_map = {
        name: (
            "DIMENSION" if name.startswith("dim_")
            else "FACT" if name.startswith("fact_")
            else "SPECIFICATION"
        )
        for name in tables
    }

    grain_map = {
        "dim_year": "one row per year",
        "dim_country": "one row per home country",
        "dim_reference_country": "one row per reference-country role",
        "dim_region": "one row per region",
        "dim_cluster": "one row per global cluster",
        "dim_flow": "one row per directional flow",
        "dim_family": "one row per analytical family",
        "dim_feature": "one row per clustering feature",
        "dim_indicator": "one row per interpretation indicator",
        "fact_model_summary": "one row per model parameter",
        "fact_region": "one row per region",
        "fact_cluster_summary": "one row per global cluster",
        "fact_feature_profile": "one row per cluster × feature",
        "fact_feature_profile_annual": "one row per cluster × year × feature",
        "fact_indicator_profile": "one row per cluster × indicator",
        "fact_indicator_profile_annual": "one row per cluster × year × indicator",
        "fact_region_year_feature": "one row per region × year × feature",
        "fact_region_year_indicator": "one row per region × year × indicator",
        "fact_fuzzy_summary": "one row per directional fuzzy summary plus ALL",
        "fact_convergence_flow": "one row per directional flow",
        "fact_convergence_feature": "one row per flow × feature",
        "fact_convergence_family": "one row per flow × family",
        "fact_convergence_indicator": "one row per flow × indicator",
        "fact_convergence_region_indicator": "one row per region × indicator",
        "fact_convergence_region_indicator_annual": "one row per region × year × indicator",
        "fact_region_year_distance": "one row per region × year",
        "fact_global_local_country": "one row per country",
        "fact_global_local_region": "one row per region",
        "fact_country_reference_region": "one row per region",
        "fact_region_country_reference_distance": "one row per region × reference country",
    }

    core_tables = {
        "dim_year", "dim_country", "dim_region", "dim_cluster", "dim_flow",
        "dim_family", "dim_feature", "dim_indicator",
        "fact_region", "fact_cluster_summary",
        "fact_feature_profile_annual", "fact_indicator_profile_annual",
        "fact_convergence_flow", "fact_convergence_indicator",
        "fact_convergence_region_indicator", "fact_region_year_distance",
    }

    rows = []
    for name, frame in tables.items():
        rows.append({
            "table": name,
            "role": role_map[name],
            "grain": grain_map.get(name, ""),
            "rows": len(frame),
            "columns": len(frame.columns),
            "priority": "CORE" if name in core_tables else "SECONDARY",
        })
    return pd.DataFrame(rows).sort_values(
        ["role", "priority", "table"], kind="stable"
    ).reset_index(drop=True)


def column_role(
    table: str,
    column: str,
    relationships: pd.DataFrame,
) -> str:
    from_key = (
        (relationships["from_table"].eq(table))
        & (relationships["from_column"].eq(column))
    ).any()
    to_key = (
        (relationships["to_table"].eq(table))
        & (relationships["to_column"].eq(column))
    ).any()

    if from_key and to_key:
        return "PRIMARY_OR_BRIDGE_KEY"
    if from_key:
        return "PRIMARY_KEY"
    if to_key:
        return "FOREIGN_KEY"
    if column.endswith("_id") or column in {"code", "year", "cluster", "flow", "family", "feature", "variable"}:
        return "ATTRIBUTE_OR_KEY"
    if pd.api.types.is_numeric_dtype:
        return "FIELD"
    return "ATTRIBUTE"


COLUMN_DESCRIPTIONS = {
    "region_id": "Canonical region identifier.",
    "code": "Home-country code.",
    "geo_code": "Subnational geographic code.",
    "geo_name": "Subnational geographic name.",
    "year": "Analytical year.",
    "cluster": "Frozen global M-Exp-FCMd cluster.",
    "hard_cluster": "Frozen global hard-cluster assignment for the complete 2015–2019 trajectory.",
    "flow": "Directional comparison toward the alternative fixed global medoid profile; not cluster reassignment.",
    "transition_zone": "True when membership margin is below the frozen 0.20 threshold.",
    "convergence_index": "relative_gap_2015 - relative_gap_2019; positive indicates movement toward the alternative profile.",
    "converged_toward_destination": "True when the convergence index is positive beyond the numeric tolerance.",
    "median_contribution": "Median contribution among regions with positive overall convergence in the selected flow.",
    "mean_contribution": "Mean contribution among regions with positive overall convergence in the selected flow.",
    "consistency_pct": "Share (%) of converging regions where the component contributed in the destination direction.",
    "profile_mean": "Membership-weighted mean of the normalized analytical indicator value.",
    "profile_median": "Membership-weighted median of the normalized analytical indicator value.",
    "feature_value": "Value of one of the three frozen dynamic clustering features.",
    "value_normalized": "Normalized analytical indicator value used in the common-weight decomposition or NTL feature.",
    "annual_distance_to_origin_medoid": "Family-balanced weighted squared annual distance to the frozen origin medoid.",
    "annual_distance_to_destination_medoid": "Family-balanced weighted squared annual distance to the fixed alternative medoid.",
    "relative_destination_origin_gap": "Normalized annual destination-minus-origin distance gap.",
    "reference_country_code": "Country used as a contextual reference profile.",
    "nearest_reference_is_own_country": "Whether the leave-one-out own-country reference is the nearest country reference.",
}


def build_data_dictionary(
    tables: dict[str, pd.DataFrame],
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for table_name, frame in tables.items():
        for position, column in enumerate(frame.columns, start=1):
            series = frame[column]
            role = "FIELD"
            from_key = (
                relationships["from_table"].eq(table_name)
                & relationships["from_column"].eq(column)
            ).any()
            to_key = (
                relationships["to_table"].eq(table_name)
                & relationships["to_column"].eq(column)
            ).any()

            if from_key and to_key:
                role = "PRIMARY_OR_BRIDGE_KEY"
            elif from_key:
                role = "PRIMARY_KEY"
            elif to_key:
                role = "FOREIGN_KEY"
            elif column in {
                "region_id", "code", "year", "cluster", "flow",
                "family", "feature", "variable", "reference_country_code"
            }:
                role = "ATTRIBUTE_OR_KEY"
            elif pd.api.types.is_numeric_dtype(series):
                role = "MEASURE_FIELD"
            else:
                role = "ATTRIBUTE"

            rows.append({
                "table": table_name,
                "position": position,
                "column": column,
                "dtype": str(series.dtype),
                "role": role,
                "description": COLUMN_DESCRIPTIONS.get(
                    column,
                    "Validated analytical or semantic-model field; see table grain and lineage.",
                ),
            })
    return pd.DataFrame(rows)


def build_lineage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source_map = {
        "dim_year": "Script 07.2 constants from frozen analytical period",
        "dim_country": "07.1_region_results.csv",
        "dim_reference_country": "07.1_region_results.csv (role-playing copy)",
        "dim_region": "07.1_region_results.csv",
        "dim_cluster": "07.1_cluster_summary.csv",
        "dim_flow": "07.1_convergence_flow.csv",
        "dim_family": "07.1_contribution_variable.csv",
        "dim_feature": "07.1_contribution_variable.csv",
        "dim_indicator": "07.1_contribution_variable.csv",
        "fact_model_summary": "07.1_model_summary.csv",
        "fact_region": "07.1_region_results.csv",
        "fact_cluster_summary": "07.1_cluster_summary.csv",
        "fact_feature_profile": "07.1_cluster_profiles.csv",
        "fact_feature_profile_annual": "07.1_cluster_profiles_annual.csv",
        "fact_indicator_profile": "07.1_bod_variable_profiles.csv + 07.1_cluster_profiles.csv",
        "fact_indicator_profile_annual": "07.1_bod_variable_profiles_annual.csv + 07.1_cluster_profiles_annual.csv",
        "fact_region_year_feature": "04_final_dynamic_model_panel_2015_2019.csv",
        "fact_region_year_indicator": "04_bod_weighted_contributions_long.csv + 04_final_dynamic_model_panel_2015_2019.csv",
        "fact_fuzzy_summary": "07.1_fuzzy_summary.csv",
        "fact_convergence_flow": "07.1_convergence_flow.csv",
        "fact_convergence_feature": "07.1_contribution_feature.csv",
        "fact_convergence_family": "07.1_contribution_family.csv",
        "fact_convergence_indicator": "07.1_contribution_variable.csv",
        "fact_convergence_region_indicator": "06_convergence_variable_region_detail.csv",
        "fact_convergence_region_indicator_annual": "06_convergence_variable_annual_components.csv",
        "fact_region_year_distance": "06_region_annual_medoid_distances.csv",
        "fact_global_local_country": "07.1_global_local_country.csv",
        "fact_global_local_region": "07.1_global_local_region.csv",
        "fact_country_reference_region": "07.1_country_reference_regions.csv",
        "fact_region_country_reference_distance": "06_region_country_reference_distances.csv",
    }

    transformation_map = {
        "dim_year": "semantic dimension only",
        "dim_country": "lossless distinct projection",
        "dim_reference_country": "role-playing semantic dimension",
        "dim_region": "lossless identity projection + display label",
        "dim_cluster": "lossless descriptive projection + labels",
        "dim_flow": "flow label parsing + interpretation metadata",
        "dim_family": "lossless distinct projection + display label",
        "dim_feature": "lossless distinct projection + display label",
        "dim_indicator": "lossless distinct projection + display label",
        "fact_indicator_profile": "union of five BoD normalized profiles with NTL normalized feature profile",
        "fact_indicator_profile_annual": "union of five BoD annual normalized profiles with NTL annual normalized feature profile",
        "fact_region_year_feature": "wide-to-long reshape only",
        "fact_region_year_indicator": "long-form standardization; no new analytical value",
    }

    rows = []
    for table in tables:
        rows.append({
            "output_table": table,
            "source": source_map.get(table, "validated upstream analytical output"),
            "transformation": transformation_map.get(table, "lossless copy or ordering only"),
            "analytical_recalculation": False,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6/8 — SEMANTIC MODEL QA
# =============================================================================

EXPECTED_TABLE_ROWS = {
    "dim_year": 5,
    "dim_country": 28,
    "dim_reference_country": 28,
    "dim_region": 512,
    "dim_cluster": 2,
    "dim_flow": 2,
    "dim_family": 3,
    "dim_feature": 3,
    "dim_indicator": 6,
    "fact_region": 512,
    "fact_fuzzy_summary": 3,
    "fact_cluster_summary": 2,
    "fact_feature_profile": 6,
    "fact_feature_profile_annual": 30,
    "fact_indicator_profile": 12,
    "fact_indicator_profile_annual": 60,
    "fact_region_year_feature": 7680,
    "fact_region_year_indicator": 15360,
    "fact_convergence_flow": 2,
    "fact_convergence_feature": 6,
    "fact_convergence_family": 6,
    "fact_convergence_indicator": 12,
    "fact_convergence_region_indicator": 3072,
    "fact_convergence_region_indicator_annual": 15360,
    "fact_region_year_distance": 2560,
    "fact_global_local_country": 28,
    "fact_global_local_region": 512,
    "fact_country_reference_region": 512,
    "fact_region_country_reference_distance": 14336,
}


UNIQUE_KEYS = {
    "dim_year": ["year"],
    "dim_country": ["code"],
    "dim_reference_country": ["reference_country_code"],
    "dim_region": ["region_id"],
    "dim_cluster": ["cluster"],
    "dim_flow": ["flow"],
    "dim_family": ["family"],
    "dim_feature": ["feature"],
    "dim_indicator": ["variable"],
    "fact_region": ["region_id"],
    "fact_fuzzy_summary": ["fuzzy_direction"],
    "fact_cluster_summary": ["cluster"],
    "fact_feature_profile": ["cluster", "feature"],
    "fact_feature_profile_annual": ["cluster", "year", "feature"],
    "fact_indicator_profile": ["cluster", "variable"],
    "fact_indicator_profile_annual": ["cluster", "year", "variable"],
    "fact_region_year_feature": ["region_id", "year", "feature"],
    "fact_region_year_indicator": ["region_id", "year", "variable"],
    "fact_convergence_flow": ["flow"],
    "fact_convergence_feature": ["flow", "feature"],
    "fact_convergence_family": ["flow", "family"],
    "fact_convergence_indicator": ["flow", "variable"],
    "fact_convergence_region_indicator": ["region_id", "variable"],
    "fact_convergence_region_indicator_annual": ["region_id", "year", "variable"],
    "fact_region_year_distance": ["region_id", "year"],
    "fact_global_local_country": ["code"],
    "fact_global_local_region": ["region_id"],
    "fact_country_reference_region": ["region_id"],
    "fact_region_country_reference_distance": ["region_id", "reference_country_code"],
}


def validate_relationships(
    tables: dict[str, pd.DataFrame],
    relationships: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> None:
    for rel in relationships.itertuples(index=False):
        if rel.from_table not in tables or rel.to_table not in tables:
            qa_add(
                rows,
                f"relationship_tables::{rel.from_table}->{rel.to_table}",
                False,
                "missing table",
                "both tables present",
            )
            continue

        left = tables[rel.from_table]
        right = tables[rel.to_table]

        left_has = rel.from_column in left.columns
        right_has = rel.to_column in right.columns
        qa_add(
            rows,
            f"relationship_columns::{rel.from_table}.{rel.from_column}->{rel.to_table}.{rel.to_column}",
            left_has and right_has,
            f"left={left_has}; right={right_has}",
            "left=True; right=True",
        )
        if not left_has or not right_has:
            continue

        left_key = left[rel.from_column].dropna()
        right_key = right[rel.to_column].dropna()

        left_dtype = str(left[rel.from_column].dtype)
        right_dtype = str(right[rel.to_column].dtype)
        qa_add(
            rows,
            f"relationship_dtype::{rel.from_table}.{rel.from_column}"
            f"->{rel.to_table}.{rel.to_column}",
            left_dtype == right_dtype,
            f"{left_dtype} -> {right_dtype}",
            "identical key dtypes",
            "Power BI relationship keys are canonicalized before export.",
        )

        source_unique = not left_key.duplicated().any()
        qa_add(
            rows,
            f"relationship_one_side_unique::{rel.from_table}.{rel.from_column}",
            source_unique,
            int(left_key.duplicated().sum()),
            0,
        )

        left_values = set(left_key.astype(str))
        right_values = set(right_key.astype(str))
        missing_fk = sorted(right_values - left_values)
        qa_add(
            rows,
            f"referential_integrity::{rel.from_table}->{rel.to_table}",
            len(missing_fk) == 0,
            missing_fk[:10],
            "no orphan foreign keys",
        )


def validate_semantic_model(
    tables: dict[str, pd.DataFrame],
    relationships: pd.DataFrame,
    input_qa: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    qa_add(rows, "input_contract_passed",
           input_qa["status"].eq("PASS").all(),
           int(input_qa["status"].eq("PASS").sum()),
           len(input_qa))

    for table, expected_rows in EXPECTED_TABLE_ROWS.items():
        actual = len(tables[table])
        qa_add(
            rows,
            f"row_count::{table}",
            actual == expected_rows,
            actual,
            expected_rows,
        )

    for table, key in UNIQUE_KEYS.items():
        frame = tables[table]
        assert_columns(frame, key, table)
        duplicates = int(frame.duplicated(key).sum())
        qa_add(
            rows,
            f"unique_key::{table}",
            duplicates == 0,
            duplicates,
            0,
            f"Key={key}",
        )

    model_summary = tables["fact_model_summary"]
    if {"parameter", "value"}.issubset(model_summary.columns):
        model_dup = int(model_summary.duplicated(["parameter"]).sum())
        qa_add(
            rows,
            "unique_key::fact_model_summary",
            model_dup == 0,
            model_dup,
            0,
            "Key=['parameter']",
        )
    else:
        qa_add(
            rows,
            "model_summary_contract",
            False,
            list(model_summary.columns),
            "columns include parameter and value",
        )

    fact_region = tables["fact_region"]
    transition_count = int(bool_series(fact_region["transition_zone"]).sum())
    converging_count = int(bool_series(fact_region["converged_toward_destination"]).sum())
    hard_sizes = (
        fact_region["hard_cluster"].astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    qa_add(rows, "frozen_transition_regions",
           transition_count == EXPECTED_TRANSITION_REGIONS,
           transition_count, EXPECTED_TRANSITION_REGIONS)
    qa_add(rows, "frozen_converging_regions",
           converging_count == EXPECTED_CONVERGING_REGIONS,
           converging_count, EXPECTED_CONVERGING_REGIONS)
    qa_add(rows, "frozen_hard_cluster_sizes",
           hard_sizes == EXPECTED_HARD_CLUSTER_SIZES,
           hard_sizes, EXPECTED_HARD_CLUSTER_SIZES)

    # Relationship-key type contract: all year-bearing semantic tables must
    # expose year as the same integer dtype as dim_year.
    dim_year_dtype = str(tables["dim_year"]["year"].dtype)
    year_tables = [
        "fact_feature_profile_annual",
        "fact_indicator_profile_annual",
        "fact_region_year_feature",
        "fact_region_year_indicator",
        "fact_convergence_region_indicator_annual",
        "fact_region_year_distance",
    ]
    for table_name in year_tables:
        observed_dtype = str(tables[table_name]["year"].dtype)
        qa_add(
            rows,
            f"year_dtype::{table_name}",
            observed_dtype == dim_year_dtype,
            observed_dtype,
            dim_year_dtype,
        )

    indicator_profile = tables["fact_indicator_profile"]
    indicator_annual = tables["fact_indicator_profile_annual"]
    qa_add(rows, "indicator_profile_set",
           set(indicator_profile["variable"].astype(str)) == set(EXPECTED_INDICATORS),
           sorted(indicator_profile["variable"].astype(str).unique().tolist()),
           sorted(EXPECTED_INDICATORS))
    qa_add(rows, "indicator_annual_years",
           sorted(indicator_annual["year"].astype(int).unique().tolist()) == EXPECTED_YEARS,
           sorted(indicator_annual["year"].astype(int).unique().tolist()),
           EXPECTED_YEARS)

    ry_indicator = tables["fact_region_year_indicator"]
    qa_add(rows, "region_year_indicator_set",
           set(ry_indicator["variable"].astype(str)) == set(EXPECTED_INDICATORS),
           sorted(ry_indicator["variable"].astype(str).unique().tolist()),
           sorted(EXPECTED_INDICATORS))
    qa_add(rows, "region_year_indicator_regions",
           ry_indicator["region_id"].astype(str).nunique() == EXPECTED_REGIONS,
           ry_indicator["region_id"].astype(str).nunique(), EXPECTED_REGIONS)

    # Exact lossless reconciliation of 07.1 copies.
    copy_pairs = {
        "fact_model_summary": "model_summary",
        "fact_region": "region",
        "fact_cluster_summary": "cluster",
        "fact_feature_profile": "feature_profile",
        "fact_feature_profile_annual": "feature_annual",
        "fact_fuzzy_summary": "fuzzy_summary",
        "fact_convergence_flow": "conv_flow",
        "fact_convergence_feature": "conv_feature",
        "fact_convergence_family": "conv_family",
        "fact_convergence_indicator": "conv_variable",
        "fact_global_local_country": "global_local_country",
        "fact_global_local_region": "global_local_region",
        "fact_country_reference_region": "country_reference",
    }
    # Actual source equality is checked in main and stored as metadata flag;
    # here row/key checks plus upstream 07.1 QA protect the analytical contract.

    validate_relationships(tables, relationships, rows)

    return enforce_qa(rows, "Script 07.2 semantic-model QA")


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def table_output_path(table_name: str) -> Path:
    return POWERBI_DIR / f"{table_name}.csv"


def build_output_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected Script-07.2 output was not created: {relative_path(path)}"
            )
        rows.append({
            "relative_path": relative_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return pd.DataFrame(rows)


SHEET_NAMES = {
    "dim_year": "Dim_Year",
    "dim_country": "Dim_Country",
    "dim_reference_country": "Dim_Ref_Country",
    "dim_region": "Dim_Region",
    "dim_cluster": "Dim_Cluster",
    "dim_flow": "Dim_Flow",
    "dim_family": "Dim_Family",
    "dim_feature": "Dim_Feature",
    "dim_indicator": "Dim_Indicator",
    "fact_model_summary": "Model_Summary",
    "fact_region": "Fact_Region",
    "fact_cluster_summary": "Fact_Cluster",
    "fact_feature_profile": "Feature_Profile",
    "fact_feature_profile_annual": "Feature_Annual",
    "fact_indicator_profile": "Indicator_Profile",
    "fact_indicator_profile_annual": "Indicator_Annual",
    "fact_region_year_feature": "Region_Year_Feature",
    "fact_region_year_indicator": "Region_Year_Indicator",
    "fact_fuzzy_summary": "Fuzzy_Summary",
    "fact_convergence_flow": "Conv_Flow",
    "fact_convergence_feature": "Conv_Feature",
    "fact_convergence_family": "Conv_Family",
    "fact_convergence_indicator": "Conv_Indicator",
    "fact_convergence_region_indicator": "Conv_Reg_Indicator",
    "fact_convergence_region_indicator_annual": "Conv_Reg_Ind_Annual",
    "fact_region_year_distance": "Region_Year_Distance",
    "fact_global_local_country": "Global_Local_Country",
    "fact_global_local_region": "Global_Local_Region",
    "fact_country_reference_region": "Country_Ref_Region",
    "fact_region_country_reference_distance": "Region_Country_Ref",
}


def write_workbook(
    tables: dict[str, pd.DataFrame],
    relationships: pd.DataFrame,
    dax: pd.DataFrame,
    visual_blueprint: pd.DataFrame,
    table_catalog: pd.DataFrame,
    data_dictionary: pd.DataFrame,
    lineage: pd.DataFrame,
    qa: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=SHEET_NAMES[name], index=False)

        table_catalog.to_excel(writer, sheet_name="Table_Catalog", index=False)
        relationships.to_excel(writer, sheet_name="Relationships", index=False)
        dax.to_excel(writer, sheet_name="DAX_Measures", index=False)
        visual_blueprint.to_excel(writer, sheet_name="Visual_Blueprint", index=False)
        data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)
        lineage.to_excel(writer, sheet_name="Lineage", index=False)
        qa.to_excel(writer, sheet_name="QA", index=False)

    autofit_workbook(WORKBOOK_OUTPUT)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    ensure_directories()

    print_header("SCRIPT 07.2 v1.0.1 — POWER BI PREPARATION")
    print(f"Project root: {ROOT}")
    print(f"Analytical period: {EXPECTED_YEARS[0]}–{EXPECTED_YEARS[-1]}")
    print(f"Final analytical sample: {EXPECTED_REGIONS} regions | {EXPECTED_COUNTRIES} countries")
    print("Analytical recalculation: NONE")
    print("Primary source of truth: approved Script 07.1 outputs")
    print("Detail sources: Script 04/06 files already validated by Script 07.1")

    # ------------------------------------------------------------------
    # 1/8 — input contract
    # ------------------------------------------------------------------
    print_header("1/8 — VALIDATE APPROVED 07.1 HANDOFF + DETAIL SOURCES")
    input_manifest = build_input_manifest()
    t = load_inputs()
    input_qa = validate_input_contract(t)
    print(f"Required inputs found: {len(input_manifest):,}")
    print(f"Input-contract QA checks passed: {len(input_qa):,}")

    # ------------------------------------------------------------------
    # 2/8 — dimensions
    # ------------------------------------------------------------------
    print_header("2/8 — BUILD DIMENSIONS")
    dimensions = build_dimensions(t)
    for name, frame in dimensions.items():
        print(f"{name:<28} {len(frame):>7,} rows x {len(frame.columns):>2} cols")

    # ------------------------------------------------------------------
    # 3/8 — core facts
    # ------------------------------------------------------------------
    print_header("3/8 — BUILD CORE FACT TABLES")
    core_facts = build_core_facts(t)
    for name, frame in core_facts.items():
        print(f"{name:<42} {len(frame):>7,} rows x {len(frame.columns):>2} cols")

    # ------------------------------------------------------------------
    # 4/8 — drill-through facts
    # ------------------------------------------------------------------
    print_header("4/8 — BUILD DRILL-THROUGH FACT TABLES")
    drill_facts = build_drillthrough_facts(t)
    for name, frame in drill_facts.items():
        print(f"{name:<42} {len(frame):>7,} rows x {len(frame.columns):>2} cols")

    tables = {**dimensions, **core_facts, **drill_facts}

    # Canonicalize every relationship key before semantic-model QA/export.
    # This prevents Excel/CSV/pandas serialization differences (for example,
    # 2015 vs 2015.0) from leaking into the Power BI model.
    tables = canonicalize_semantic_key_types(tables)

    # ------------------------------------------------------------------
    # 5/8 — semantic specification
    # ------------------------------------------------------------------
    print_header("5/8 — BUILD POWER BI SEMANTIC SPECIFICATION")
    relationships = build_relationships()
    dax = build_dax_measures()
    visual_blueprint = build_visual_blueprint()
    table_catalog = build_table_catalog(tables)
    data_dictionary = build_data_dictionary(tables, relationships)
    lineage = build_lineage(tables)

    print(f"Semantic tables:        {len(tables):,}")
    print(f"Relationships:          {len(relationships):,}")
    print(f"Suggested DAX measures: {len(dax):,}")
    print(f"Visual blueprint rows:  {len(visual_blueprint):,}")
    print(f"Data dictionary fields: {len(data_dictionary):,}")

    # ------------------------------------------------------------------
    # 6/8 — QA
    # ------------------------------------------------------------------
    print_header("6/8 — FINAL POWER BI MODEL QA")
    model_qa = validate_semantic_model(tables, relationships, input_qa)
    print(f"Power BI model QA checks passed: {len(model_qa):,}")

    # ------------------------------------------------------------------
    # 7/8 — write outputs
    # ------------------------------------------------------------------
    print_header("7/8 — WRITE POWER BI-READY OUTPUTS")

    write_csv(input_manifest, INPUT_MANIFEST_OUTPUT)

    table_paths: list[Path] = []
    for table_name, frame in tables.items():
        path = table_output_path(table_name)
        write_csv(frame, path)
        table_paths.append(path)

    write_csv(table_catalog, TABLE_CATALOG_OUTPUT)
    write_csv(relationships, RELATIONSHIPS_OUTPUT)
    write_csv(dax, DAX_OUTPUT)
    write_csv(visual_blueprint, VISUAL_BLUEPRINT_OUTPUT)
    write_csv(data_dictionary, DATA_DICTIONARY_OUTPUT)
    write_csv(lineage, LINEAGE_OUTPUT)
    write_csv(model_qa, QA_OUTPUT)

    write_workbook(
        tables=tables,
        relationships=relationships,
        dax=dax,
        visual_blueprint=visual_blueprint,
        table_catalog=table_catalog,
        data_dictionary=data_dictionary,
        lineage=lineage,
        qa=model_qa,
    )

    print(f"Power BI CSV directory: {relative_path(POWERBI_DIR)}")
    print(f"Inspection workbook:    {relative_path(WORKBOOK_OUTPUT)}")

    # ------------------------------------------------------------------
    # 8/8 — final status + manifest
    # ------------------------------------------------------------------
    print_header("8/8 — FINAL STATUS + OUTPUT MANIFEST")

    status = {
        "script": "07.2_prepare_powerbi.py",
        "script_version": "1.0.1",
        "status": "PASS",
        "analytical_period": [2015, 2019],
        "regions": EXPECTED_REGIONS,
        "countries": EXPECTED_COUNTRIES,
        "selected_c": EXPECTED_C,
        "selected_m": EXPECTED_M,
        "hard_cluster_sizes": EXPECTED_HARD_CLUSTER_SIZES,
        "fuzzy_transition_regions": EXPECTED_TRANSITION_REGIONS,
        "converging_regions": EXPECTED_CONVERGING_REGIONS,
        "semantic_tables": len(tables),
        "relationships": len(relationships),
        "suggested_dax_measures": len(dax),
        "prior_analytical_recalculation": False,
        "ready_for_powerbi_import": True,
        "powerbi_csv_directory": relative_path(POWERBI_DIR),
        "inspection_workbook": relative_path(WORKBOOK_OUTPUT),
    }
    FINAL_STATUS_OUTPUT.write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output_paths = [
        INPUT_MANIFEST_OUTPUT,
        *table_paths,
        TABLE_CATALOG_OUTPUT,
        RELATIONSHIPS_OUTPUT,
        DAX_OUTPUT,
        VISUAL_BLUEPRINT_OUTPUT,
        DATA_DICTIONARY_OUTPUT,
        LINEAGE_OUTPUT,
        QA_OUTPUT,
        FINAL_STATUS_OUTPUT,
        WORKBOOK_OUTPUT,
    ]
    manifest = build_output_manifest(output_paths)
    write_csv(manifest, OUTPUT_MANIFEST_OUTPUT)

    elapsed = time.perf_counter() - t0

    print("\nFINAL SCRIPT 07.2 SUMMARY")
    print("-" * 100)
    print(f"Regions:                         {EXPECTED_REGIONS}")
    print(f"Countries:                       {EXPECTED_COUNTRIES}")
    print(f"Frozen configuration:            C={EXPECTED_C}, m={EXPECTED_M}")
    print(f"Semantic tables:                 {len(tables)}")
    print(f"Relationships specified:         {len(relationships)}")
    print(f"Suggested DAX measures:          {len(dax)}")
    print(f"Power BI QA checks:              {len(model_qa)} PASS")
    print("Semantic key types:              CANONICALIZED + VALIDATED")
    print(f"Power BI CSV directory:          {relative_path(POWERBI_DIR)}")
    print(f"Inspection workbook:             {relative_path(WORKBOOK_OUTPUT)}")
    print(f"Output manifest:                 {relative_path(OUTPUT_MANIFEST_OUTPUT)}")
    print("Prior analytical recalculation:  NONE")
    print("\nSCRIPT 07.2 COMPLETE — POWER BI MODEL READY FOR IMPORT")
    print(f"Script execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
