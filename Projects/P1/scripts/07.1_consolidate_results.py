from __future__ import annotations

"""
SCRIPT 07.1 v1.0.4 — CONSOLIDATE APPROVED P1 RESULTS

Purpose
-------
Consolidate the validated outputs from Scripts 04, 05 and 06 into a single,
one-click results layer for P1. This script does NOT re-estimate BoD weights,
M-Exp-FCMd clusters, memberships, medoids, bootstrap models, temporal
convergence, or contribution decompositions.

Official P1 state consumed here
-------------------------------
- Analytical period: 2015–2019.
- Final analytical sample: 512 regions in 28 countries.
- The 513th candidate region is intentionally outside the final analytical
  sample and is not reintroduced at Stage 07.
- Dynamic-only M-Exp-FCMd clustering.
- Frozen global solution: C=2, m=2.5.
- Frozen hard-cluster sizes: C1=100, C2=412.
- Final dynamic clustering features:
    * bod_demographic_productive_potential
    * bod_socioeconomic_deprivation
    * ntl_per_capita_norm
- Common-weight BoD decomposition is preserved as an interpretation/audit
  layer and is never added as extra clustering dimensions.
- Fuzzy-transition threshold inherited from Script 06: membership margin < 0.20.
- Frozen Script-06 results:
    * 83 fuzzy-transition regions;
    * 182 regions converging toward the alternative fixed global profile;
    * 18 countries with viable local M-Exp-FCMd models;
    * 188 regions whose leave-one-out own-country reference is nearest.

Stage 1.7.1 implemented in this one-click script
-----------------------------------------------
1.7.1.1 Inputs + outputs + QA
1.7.1.2 Regional and cluster consolidation
1.7.1.3 Profiles and evolution
1.7.1.4 Fuzzy / transition
1.7.1.5 Convergence
1.7.1.6 Contributions and consistency
1.7.1.7 Global × local comparisons
1.7.1.8 Summary spreadsheet
1.7.1.9 Results report
1.7.1.10 Final QA and export

Interpretation guardrail
------------------------
A flow such as C1->C2 does NOT mean formal cluster reassignment. Every region
keeps one global cluster for the complete 2015–2019 trajectory. The flow means
that the region moved relatively closer to the alternative fixed medoid profile
between 2015 and 2019.
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
SCRIPT05_DIR = INTERIM_DIR / "script05"
SCRIPT06_DIR = INTERIM_DIR / "script06"
SCRIPT07_DIR = INTERIM_DIR / "script07"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

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
TRANSITION_MARGIN_THRESHOLD = 0.20
MEMBERSHIP_TOL = 1e-10
NUMERIC_TOL = 1e-10

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
EXPECTED_STRUCTURAL_FEATURES = (
    "bod_environmental_risk",
    "built_area_share_2015_norm",
    "ghs_urban_pop_share_norm",
    "ghs_urban_centre_pop_share_norm",
)
EXPECTED_DYNAMIC_BOD_VARIABLES = {
    "demographic_productive_potential": {"share_15_64", "share_65_plus"},
    "socioeconomic_deprivation": {"poor420", "gini", "prosgap2021"},
}
EXPECTED_CONVERGENCE_VARIABLES = {
    "share_15_64",
    "share_65_plus",
    "poor420",
    "gini",
    "prosgap2021",
    "ntl_per_capita_norm",
}

# -----------------------------------------------------------------------------
# Script 04 — final analytical handoff and common-weight decomposition
# -----------------------------------------------------------------------------

S04_WORKBOOK = OUTPUTS_DIR / "04_final_handoff_script05.xlsx"
S04_DYNAMIC = SCRIPT04_DIR / "04_final_dynamic_model_panel_2015_2019.csv"
S04_STRUCTURAL = SCRIPT04_DIR / "04_final_structural_model_panel.csv"
S04_FEATURE_WEIGHTS = SCRIPT04_DIR / "04_final_feature_weights.csv"
S04_PREPROCESSING_AUDIT = SCRIPT04_DIR / "04_preprocessing_audit.csv"
S04_BOD_WEIGHTS = SCRIPT04_DIR / "04_bod_common_weights.csv"
S04_BOD_SUMMARY = SCRIPT04_DIR / "04_bod_family_summary.csv"
S04_BOD_CONTRIBUTIONS = SCRIPT04_DIR / "04_bod_weighted_contributions_long.csv"
S04_BOD_DYNAMIC_DECOMP = SCRIPT04_DIR / "04_bod_dynamic_decomposition_2015_2019.csv"
S04_BOD_STRUCTURAL_DECOMP = SCRIPT04_DIR / "04_bod_structural_decomposition.csv"
S04_COUNTRY_DYNAMIC = SCRIPT04_DIR / "04_country_reference_dynamic_final.csv"
S04_COUNTRY_STRUCTURAL = SCRIPT04_DIR / "04_country_reference_structural_final.csv"
S04_OFFICIAL_REGIONS = SCRIPT04_DIR / "04_official_region_set_512.csv"

S04_WORKBOOK_MAP = {
    "dynamic_model": S04_DYNAMIC,
    "structural_model": S04_STRUCTURAL,
    "feature_weights": S04_FEATURE_WEIGHTS,
    "preprocessing_audit": S04_PREPROCESSING_AUDIT,
    "bod_weights": S04_BOD_WEIGHTS,
    "bod_summary": S04_BOD_SUMMARY,
    "bod_contributions": S04_BOD_CONTRIBUTIONS,
    "bod_dynamic_decomp": S04_BOD_DYNAMIC_DECOMP,
    "bod_struct_decomp": S04_BOD_STRUCTURAL_DECOMP,
    "country_dynamic": S04_COUNTRY_DYNAMIC,
    "country_structural": S04_COUNTRY_STRUCTURAL,
    "official_regions": S04_OFFICIAL_REGIONS,
}

# -----------------------------------------------------------------------------
# Script 05 — frozen M-Exp-FCMd results
# -----------------------------------------------------------------------------

S05_WORKBOOK = OUTPUTS_DIR / "05_mexp_fcmd_results.xlsx"
S05_MODEL_METADATA = SCRIPT05_DIR / "05_model_metadata.csv"
S05_DISTANCE_COMPONENTS = SCRIPT05_DIR / "05_distance_components_summary.csv"
S05_GLOBAL_GRID = SCRIPT05_DIR / "05_global_grid_search.csv"
S05_GLOBAL_C_SELECTION = SCRIPT05_DIR / "05_global_c_selection_summary.csv"
S05_GLOBAL_PARTITION = SCRIPT05_DIR / "05_global_partition.csv"
S05_GLOBAL_MEMBERSHIPS_LONG = SCRIPT05_DIR / "05_global_memberships_long.csv"
S05_GLOBAL_MEDOIDS = SCRIPT05_DIR / "05_global_medoids.csv"
S05_GLOBAL_BOOTSTRAP_DETAIL = SCRIPT05_DIR / "05_global_bootstrap_stability.csv"
S05_GLOBAL_BOOTSTRAP = SCRIPT05_DIR / "05_global_bootstrap_summary.csv"
S05_COUNTRY_SUMMARY = SCRIPT05_DIR / "05_country_cluster_summary.csv"
S05_COUNTRY_PARTITION = SCRIPT05_DIR / "05_country_partition.csv"
S05_COUNTRY_MEMBERSHIPS_LONG = SCRIPT05_DIR / "05_country_memberships_long.csv"
S05_COUNTRY_MEDOIDS = SCRIPT05_DIR / "05_country_medoids.csv"
S05_COUNTRY_BOOTSTRAP_DETAIL = SCRIPT05_DIR / "05_country_bootstrap_stability.csv"
S05_COUNTRY_BOOTSTRAP = SCRIPT05_DIR / "05_country_bootstrap_summary.csv"
S05_REGION_TO_MEDOIDS = SCRIPT05_DIR / "05_region_to_global_medoids_distances.csv"
S05_SELECTED_CONFIG = SCRIPT05_DIR / "05_selected_configuration.json"

S05_WORKBOOK_MAP = {
    "Model_Metadata": S05_MODEL_METADATA,
    "Distance_Components": S05_DISTANCE_COMPONENTS,
    "Global_Grid": S05_GLOBAL_GRID,
    "Global_C_Selection": S05_GLOBAL_C_SELECTION,
    "Global_Partition": S05_GLOBAL_PARTITION,
    "Global_Medoids": S05_GLOBAL_MEDOIDS,
    "Global_Bootstrap": S05_GLOBAL_BOOTSTRAP,
    "Country_Summary": S05_COUNTRY_SUMMARY,
    "Country_Partition": S05_COUNTRY_PARTITION,
    "Country_Medoids": S05_COUNTRY_MEDOIDS,
    "Country_Bootstrap": S05_COUNTRY_BOOTSTRAP,
}

S05_DETAIL_FILES = {
    "global_memberships_long": S05_GLOBAL_MEMBERSHIPS_LONG,
    "global_bootstrap_detail": S05_GLOBAL_BOOTSTRAP_DETAIL,
    "country_memberships_long": S05_COUNTRY_MEMBERSHIPS_LONG,
    "country_bootstrap_detail": S05_COUNTRY_BOOTSTRAP_DETAIL,
    "region_to_global_medoids": S05_REGION_TO_MEDOIDS,
}

# -----------------------------------------------------------------------------
# Script 06 — profiles, convergence and interpretation
# -----------------------------------------------------------------------------

S06_WORKBOOK = OUTPUTS_DIR / "06_evolution_interpretation.xlsx"
S06_METADATA = SCRIPT06_DIR / "06_analysis_metadata.csv"
S06_PROFILE_SUMMARY = SCRIPT06_DIR / "06_cluster_profile_summary.csv"
S06_PROFILE_ANNUAL = SCRIPT06_DIR / "06_cluster_profile_annual.csv"
S06_PROFILE_COMPARISON = SCRIPT06_DIR / "06_cluster_profile_comparison.csv"
S06_BOD_VAR_PROFILE = SCRIPT06_DIR / "06_bod_variable_profile_summary.csv"
S06_BOD_VAR_ANNUAL = SCRIPT06_DIR / "06_bod_variable_profile_annual.csv"
S06_FUZZY_REGIONS = SCRIPT06_DIR / "06_fuzzy_transition_regions.csv"
S06_FUZZY_SUMMARY = SCRIPT06_DIR / "06_fuzzy_transition_summary.csv"
S06_ANNUAL_MEDOID_DISTANCES = SCRIPT06_DIR / "06_region_annual_medoid_distances.csv"
S06_CONVERGENCE_REGION = SCRIPT06_DIR / "06_region_convergence_2015_2019.csv"
S06_CONVERGENCE_FLOW = SCRIPT06_DIR / "06_convergence_flow_summary.csv"
S06_CONV_FEATURE = SCRIPT06_DIR / "06_convergence_feature_contributions.csv"
S06_CONV_FAMILY = SCRIPT06_DIR / "06_convergence_family_contributions.csv"
S06_CONV_VARIABLE = SCRIPT06_DIR / "06_convergence_variable_contributions.csv"
S06_CONV_VARIABLE_REGION = SCRIPT06_DIR / "06_convergence_variable_region_detail.csv"
S06_CONV_VARIABLE_ANNUAL = SCRIPT06_DIR / "06_convergence_variable_annual_components.csv"
S06_COUNTRY_DISTANCES = SCRIPT06_DIR / "06_region_country_reference_distances.csv"
S06_COUNTRY_NEAREST = SCRIPT06_DIR / "06_region_country_reference_nearest.csv"
S06_GLOBAL_LOCAL_REGION = SCRIPT06_DIR / "06_global_local_region_comparison.csv"
S06_GLOBAL_LOCAL_COUNTRY = SCRIPT06_DIR / "06_global_local_country_summary.csv"

S06_WORKBOOK_MAP = {
    "Metadata": S06_METADATA,
    "Cluster_Profile": S06_PROFILE_SUMMARY,
    "Profile_Annual": S06_PROFILE_ANNUAL,
    "Profile_Compare": S06_PROFILE_COMPARISON,
    "BoD_Var_Profile": S06_BOD_VAR_PROFILE,
    "BoD_Var_Annual": S06_BOD_VAR_ANNUAL,
    "Fuzzy_Summary": S06_FUZZY_SUMMARY,
    "Fuzzy_Regions": S06_FUZZY_REGIONS,
    "Convergence_Flow": S06_CONVERGENCE_FLOW,
    "Convergence_Region": S06_CONVERGENCE_REGION,
    "Conv_Feature": S06_CONV_FEATURE,
    "Conv_Family": S06_CONV_FAMILY,
    "Conv_Variable": S06_CONV_VARIABLE,
    "Conv_Var_Region": S06_CONV_VARIABLE_REGION,
    "Country_Refs": S06_COUNTRY_NEAREST,
    "Global_Local_Country": S06_GLOBAL_LOCAL_COUNTRY,
    "Global_Local_Region": S06_GLOBAL_LOCAL_REGION,
}

S06_DETAIL_FILES = {
    "annual_region_medoid_distances": S06_ANNUAL_MEDOID_DISTANCES,
    "variable_annual_components": S06_CONV_VARIABLE_ANNUAL,
    "region_country_reference_distances": S06_COUNTRY_DISTANCES,
}

# -----------------------------------------------------------------------------
# Script 07.1 outputs
# -----------------------------------------------------------------------------

INPUT_MANIFEST_OUTPUT = SCRIPT07_DIR / "07.1_input_manifest.csv"
WORKBOOK_SYNC_OUTPUT = SCRIPT07_DIR / "07.1_workbook_sync_audit.csv"
MODEL_SUMMARY_OUTPUT = SCRIPT07_DIR / "07.1_model_summary.csv"
REGION_RESULTS_OUTPUT = SCRIPT07_DIR / "07.1_region_results.csv"
CLUSTER_SUMMARY_OUTPUT = SCRIPT07_DIR / "07.1_cluster_summary.csv"
PROFILE_SUMMARY_OUTPUT = SCRIPT07_DIR / "07.1_cluster_profiles.csv"
PROFILE_ANNUAL_OUTPUT = SCRIPT07_DIR / "07.1_cluster_profiles_annual.csv"
BOD_VAR_PROFILE_OUTPUT = SCRIPT07_DIR / "07.1_bod_variable_profiles.csv"
BOD_VAR_ANNUAL_OUTPUT = SCRIPT07_DIR / "07.1_bod_variable_profiles_annual.csv"
FUZZY_SUMMARY_OUTPUT = SCRIPT07_DIR / "07.1_fuzzy_summary.csv"
FUZZY_REGIONS_OUTPUT = SCRIPT07_DIR / "07.1_fuzzy_regions.csv"
CONVERGENCE_FLOW_OUTPUT = SCRIPT07_DIR / "07.1_convergence_flow.csv"
CONVERGENCE_REGIONS_OUTPUT = SCRIPT07_DIR / "07.1_convergence_regions.csv"
CONTRIB_FEATURE_OUTPUT = SCRIPT07_DIR / "07.1_contribution_feature.csv"
CONTRIB_FAMILY_OUTPUT = SCRIPT07_DIR / "07.1_contribution_family.csv"
CONTRIB_VARIABLE_OUTPUT = SCRIPT07_DIR / "07.1_contribution_variable.csv"
GLOBAL_LOCAL_COUNTRY_OUTPUT = SCRIPT07_DIR / "07.1_global_local_country.csv"
GLOBAL_LOCAL_REGION_OUTPUT = SCRIPT07_DIR / "07.1_global_local_region.csv"
COUNTRY_REFERENCE_OUTPUT = SCRIPT07_DIR / "07.1_country_reference_regions.csv"
FINAL_QA_OUTPUT = SCRIPT07_DIR / "07.1_final_qa.csv"
OUTPUT_MANIFEST_OUTPUT = SCRIPT07_DIR / "07.1_output_manifest.csv"
FINAL_STATUS_OUTPUT = SCRIPT07_DIR / "07.1_final_status.json"

CONSOLIDATED_WORKBOOK_OUTPUT = OUTPUTS_DIR / "07.1_consolidated_results.xlsx"
RESULTS_REPORT_OUTPUT = RESULTS_DIR / "07.1_results_report.md"


# =============================================================================
# BASIC HELPERS
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def ensure_directories() -> None:
    SCRIPT07_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def assert_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
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


def assert_region_set_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> None:
    left_ids = set(left["region_id"].astype(str))
    right_ids = set(right["region_id"].astype(str))
    if left_ids != right_ids:
        only_left = sorted(left_ids - right_ids)[:10]
        only_right = sorted(right_ids - left_ids)[:10]
        raise RuntimeError(
            f"Region-set mismatch: {left_label} vs {right_label}. "
            f"Only in {left_label}: {only_left}; only in {right_label}: {only_right}"
        )


def metadata_to_dict(frame: pd.DataFrame) -> dict[str, Any]:
    if {"parameter", "value"}.issubset(frame.columns):
        return dict(zip(frame["parameter"].astype(str), frame["value"]))
    return {}


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


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


def autofit_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            letter = column_cells[0].column_letter
            max_length = 0
            for cell in column_cells[:2500]:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))
            ws.column_dimensions[letter].width = min(max(max_length + 2, 10), 55)
    wb.save(path)


# =============================================================================
# 1.7.1.1 — INPUTS + OUTPUTS + QA
# =============================================================================


def required_inputs() -> dict[str, tuple[str, Path]]:
    files: dict[str, tuple[str, Path]] = {
        "script04::workbook": ("CORE_WORKBOOK", S04_WORKBOOK),
        "script05::workbook": ("CORE_WORKBOOK", S05_WORKBOOK),
        "script06::workbook": ("CORE_WORKBOOK", S06_WORKBOOK),
        "script05::selected_config": ("CONFIG_JSON", S05_SELECTED_CONFIG),
    }
    for sheet, path in S04_WORKBOOK_MAP.items():
        files[f"script04::{sheet}"] = ("MACHINE_READABLE", path)
    for sheet, path in S05_WORKBOOK_MAP.items():
        files[f"script05::{sheet}"] = ("MACHINE_READABLE", path)
    for name, path in S05_DETAIL_FILES.items():
        files[f"script05_detail::{name}"] = ("DETAIL_ONLY", path)
    for sheet, path in S06_WORKBOOK_MAP.items():
        files[f"script06::{sheet}"] = ("MACHINE_READABLE", path)
    for name, path in S06_DETAIL_FILES.items():
        files[f"script06_detail::{name}"] = ("DETAIL_ONLY", path)
    return files


def build_input_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, Path]] = []
    for logical_name, (role, path) in required_inputs().items():
        exists = path.exists()
        if not exists:
            missing.append((logical_name, path))
        rows.append({
            "logical_name": logical_name,
            "role": role,
            "relative_path": relative_path(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else np.nan,
            "sha256": file_sha256(path) if exists else None,
        })
    manifest = pd.DataFrame(rows)
    if missing:
        lines = "\n".join(f"- {name}: {relative_path(path)}" for name, path in missing)
        raise FileNotFoundError(
            f"Script 07.1 cannot start; {len(missing)} required input(s) are missing:\n{lines}"
        )
    return manifest


def validate_workbook_contract(
    workbook_path: Path,
    mapping: dict[str, Path],
    label: str,
) -> None:
    actual = pd.ExcelFile(workbook_path).sheet_names
    expected = list(mapping.keys())
    if actual != expected:
        raise RuntimeError(
            f"{label}: workbook sheet contract changed.\n"
            f"Expected: {expected}\nActual:   {actual}"
        )


def _is_missing_scalar(value: Any) -> bool:
    """Return True only for scalar missing values."""
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _normalize_text_scalar(value: Any) -> str:
    """Normalize harmless Excel/CSV whitespace differences without changing words."""
    return " ".join(str(value).strip().split())


def _try_float_scalar(value: Any) -> tuple[bool, float]:
    """Parse a scalar as a finite numeric value when that interpretation is valid."""
    if isinstance(value, (bool, np.bool_)):
        return False, float("nan")
    try:
        if isinstance(value, str) and not value.strip():
            return False, float("nan")
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False, float("nan")
    return (math.isfinite(number), number)


def _scalars_semantically_equal(left: Any, right: Any) -> bool:
    """
    Compare one Excel cell and one CSV cell semantically.

    Excel frequently reloads values from mixed-type metadata columns as numbers
    (for example 512 or 1.0), while CSV round-trips may reload the same values as
    strings ("512" or "1.0"). These are serialization differences, not analytical
    differences. Text content remains strict apart from harmless surrounding/
    repeated whitespace.
    """
    left_missing = _is_missing_scalar(left)
    right_missing = _is_missing_scalar(right)
    if left_missing or right_missing:
        return left_missing and right_missing

    left_is_num, left_num = _try_float_scalar(left)
    right_is_num, right_num = _try_float_scalar(right)
    if left_is_num and right_is_num:
        return bool(np.isclose(left_num, right_num, rtol=1e-10, atol=1e-12))

    return _normalize_text_scalar(left) == _normalize_text_scalar(right)


def _assert_frames_semantically_equal(
    excel_frame: pd.DataFrame,
    csv_frame: pd.DataFrame,
    label: str,
) -> None:
    """Strict structural comparison with serialization-aware scalar comparison."""
    excel_frame = excel_frame.reset_index(drop=True)
    csv_frame = csv_frame.reset_index(drop=True)

    if excel_frame.shape != csv_frame.shape:
        raise AssertionError(
            f"shape mismatch: Excel={excel_frame.shape}, CSV={csv_frame.shape}"
        )
    excel_columns = list(excel_frame.columns)
    csv_columns = list(csv_frame.columns)

    if len(excel_columns) != len(csv_columns):
        raise AssertionError(
            f"{label}: column-count mismatch: "
            f"Excel={len(excel_columns)}, CSV={len(csv_columns)}"
        )

    column_mismatches: list[str] = []
    for position, (excel_column, csv_column) in enumerate(
        zip(excel_columns, csv_columns)
    ):
        if not _scalars_semantically_equal(excel_column, csv_column):
            column_mismatches.append(
                f"position={position}, "
                f"Excel={excel_column!r}, CSV={csv_column!r}"
            )

    if column_mismatches:
        raise AssertionError(
            f"{label}: semantic column mismatch(es):\n- "
            + "\n- ".join(column_mismatches[:10])
        )

    mismatches: list[str] = []
    max_reported = 10
    for column_position, (excel_column, csv_column) in enumerate(
        zip(excel_columns, csv_columns)
    ):
        left_values = excel_frame.iloc[:, column_position].tolist()
        right_values = csv_frame.iloc[:, column_position].tolist()
        for row_index, (left, right) in enumerate(zip(left_values, right_values)):
            if not _scalars_semantically_equal(left, right):
                mismatches.append(
                    f"row={row_index}, "
                    f"column_position={column_position}, "
                    f"Excel_column={excel_column!r}, "
                    f"CSV_column={csv_column!r}, "
                    f"Excel={left!r}, CSV={right!r}"
                )
                if len(mismatches) >= max_reported:
                    break
        if len(mismatches) >= max_reported:
            break

    if mismatches:
        raise AssertionError(
            f"{label}: semantic cell mismatch(es). First {len(mismatches)}:\n- "
            + "\n- ".join(mismatches)
        )


def compare_workbook_sheet_to_csv(
    workbook_path: Path,
    sheet_name: str,
    csv_path: Path,
) -> tuple[int, int]:
    excel_frame = pd.read_excel(workbook_path, sheet_name=sheet_name)
    csv_frame = read_csv(csv_path)

    # Fast path: ordinary homogeneous tables should match directly.
    try:
        pd.testing.assert_frame_equal(
            excel_frame.reset_index(drop=True),
            csv_frame.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            check_names=True,
            rtol=1e-10,
            atol=1e-12,
        )
        return csv_frame.shape
    except AssertionError:
        pass

    # Fallback for heterogeneous metadata/value columns. This preserves strict
    # content checking while accepting Excel-vs-CSV numeric type inference such
    # as 512 vs "512" and 1 vs "1.0".
    try:
        _assert_frames_semantically_equal(
            excel_frame,
            csv_frame,
            label=f"{workbook_path.name}::{sheet_name} vs {csv_path.name}",
        )
    except AssertionError as exc:
        raise RuntimeError(
            f"Workbook/CSV mismatch: {workbook_path.name}::{sheet_name} "
            f"vs {csv_path.name}.\n{exc}"
        ) from exc

    return csv_frame.shape


def validate_workbook_csv_sync() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("Script04", S04_WORKBOOK, S04_WORKBOOK_MAP),
        ("Script05", S05_WORKBOOK, S05_WORKBOOK_MAP),
        ("Script06", S06_WORKBOOK, S06_WORKBOOK_MAP),
    ]
    for stage, workbook, mapping in groups:
        validate_workbook_contract(workbook, mapping, f"{stage} workbook")
        for sheet, csv_path in mapping.items():
            shape = compare_workbook_sheet_to_csv(workbook, sheet, csv_path)
            rows.append({
                "stage": stage,
                "workbook": workbook.name,
                "sheet": sheet,
                "machine_readable_file": csv_path.name,
                "rows": shape[0],
                "columns": shape[1],
                "status": "MATCH",
            })
            print(f"[MATCH] {stage:<8} {sheet:<24} {shape[0]:>7,} x {shape[1]:<3}")
    return pd.DataFrame(rows)


def load_inputs() -> dict[str, Any]:
    with S05_SELECTED_CONFIG.open("r", encoding="utf-8") as f:
        selected_config = json.load(f)

    return {
        "dynamic": read_csv(S04_DYNAMIC),
        "structural": read_csv(S04_STRUCTURAL),
        "feature_weights": read_csv(S04_FEATURE_WEIGHTS),
        "bod_weights": read_csv(S04_BOD_WEIGHTS),
        "bod_summary": read_csv(S04_BOD_SUMMARY),
        "bod_contrib": read_csv(S04_BOD_CONTRIBUTIONS),
        "official_regions": read_csv(S04_OFFICIAL_REGIONS),
        "model_metadata": read_csv(S05_MODEL_METADATA),
        "global_grid": read_csv(S05_GLOBAL_GRID),
        "c_selection": read_csv(S05_GLOBAL_C_SELECTION),
        "global_partition": read_csv(S05_GLOBAL_PARTITION),
        "global_medoids": read_csv(S05_GLOBAL_MEDOIDS),
        "global_bootstrap": read_csv(S05_GLOBAL_BOOTSTRAP),
        "country_summary": read_csv(S05_COUNTRY_SUMMARY),
        "country_partition": read_csv(S05_COUNTRY_PARTITION),
        "selected_config": selected_config,
        "analysis_metadata": read_csv(S06_METADATA),
        "profile_summary": read_csv(S06_PROFILE_SUMMARY),
        "profile_annual": read_csv(S06_PROFILE_ANNUAL),
        "profile_comparison": read_csv(S06_PROFILE_COMPARISON),
        "bod_var_profile": read_csv(S06_BOD_VAR_PROFILE),
        "bod_var_annual": read_csv(S06_BOD_VAR_ANNUAL),
        "fuzzy_regions": read_csv(S06_FUZZY_REGIONS),
        "fuzzy_summary": read_csv(S06_FUZZY_SUMMARY),
        "annual_medoid_distances": read_csv(S06_ANNUAL_MEDOID_DISTANCES),
        "convergence_region": read_csv(S06_CONVERGENCE_REGION),
        "convergence_flow": read_csv(S06_CONVERGENCE_FLOW),
        "conv_feature": read_csv(S06_CONV_FEATURE),
        "conv_family": read_csv(S06_CONV_FAMILY),
        "conv_variable": read_csv(S06_CONV_VARIABLE),
        "conv_variable_region": read_csv(S06_CONV_VARIABLE_REGION),
        "conv_variable_annual": read_csv(S06_CONV_VARIABLE_ANNUAL),
        "country_distances": read_csv(S06_COUNTRY_DISTANCES),
        "country_nearest": read_csv(S06_COUNTRY_NEAREST),
        "global_local_region": read_csv(S06_GLOBAL_LOCAL_REGION),
        "global_local_country": read_csv(S06_GLOBAL_LOCAL_COUNTRY),
    }


def validate_frozen_results(t: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dynamic = t["dynamic"]
    structural = t["structural"]
    feature_weights = t["feature_weights"]
    official = t["official_regions"]
    partition = t["global_partition"]
    medoids = t["global_medoids"]
    config = t["selected_config"]
    fuzzy_regions = t["fuzzy_regions"]
    convergence = t["convergence_region"]
    country_summary = t["country_summary"]
    country_nearest = t["country_nearest"]
    gl_region = t["global_local_region"]

    assert_columns(dynamic, ["region_id", "code", "geo_code", "geo_name", "year", *EXPECTED_DYNAMIC_FEATURES], "Script04 dynamic")
    assert_columns(structural, ["region_id", "code", "geo_code", "geo_name", *EXPECTED_STRUCTURAL_FEATURES], "Script04 structural")
    assert_columns(partition, ["region_id", "hard_cluster", "u_cluster_1", "u_cluster_2", "membership_margin"], "Script05 partition")
    assert_columns(convergence, ["region_id", "origin_cluster", "flow", "convergence_index", "converged_toward_destination"], "Script06 convergence")

    assert_unique(dynamic, ["region_id", "year"], "Script04 dynamic")
    assert_unique(structural, ["region_id"], "Script04 structural")
    assert_unique(official, ["region_id"], "Script04 official regions")
    assert_unique(partition, ["region_id"], "Script05 partition")
    assert_unique(convergence, ["region_id"], "Script06 convergence")
    assert_unique(country_nearest, ["region_id"], "Script06 country nearest")
    assert_unique(gl_region, ["region_id"], "Script06 global-local region")

    qa_add(rows, "official_region_count", len(official) == EXPECTED_REGIONS, len(official), EXPECTED_REGIONS)
    qa_add(rows, "official_country_count", official["code"].nunique() == EXPECTED_COUNTRIES, official["code"].nunique(), EXPECTED_COUNTRIES)
    qa_add(rows, "dynamic_row_count", len(dynamic) == EXPECTED_REGION_YEARS, len(dynamic), EXPECTED_REGION_YEARS)
    qa_add(rows, "dynamic_region_count", dynamic["region_id"].nunique() == EXPECTED_REGIONS, dynamic["region_id"].nunique(), EXPECTED_REGIONS)
    qa_add(rows, "dynamic_country_count", dynamic["code"].nunique() == EXPECTED_COUNTRIES, dynamic["code"].nunique(), EXPECTED_COUNTRIES)
    qa_add(rows, "dynamic_years", sorted(dynamic["year"].astype(int).unique().tolist()) == EXPECTED_YEARS, sorted(dynamic["year"].astype(int).unique().tolist()), EXPECTED_YEARS)
    qa_add(rows, "structural_region_count", len(structural) == EXPECTED_REGIONS, len(structural), EXPECTED_REGIONS)

    assert_region_set_equal(dynamic, official, "dynamic", "official_regions")
    assert_region_set_equal(structural, official, "structural", "official_regions")
    assert_region_set_equal(partition, official, "global_partition", "official_regions")
    assert_region_set_equal(convergence, official, "convergence", "official_regions")
    assert_region_set_equal(country_nearest, official, "country_nearest", "official_regions")
    assert_region_set_equal(gl_region, official, "global_local_region", "official_regions")

    fw_dynamic = feature_weights[feature_weights["block"].astype(str).eq("DYNAMIC")]
    fw_structural = feature_weights[feature_weights["block"].astype(str).eq("STRUCTURAL")]
    dynamic_features = tuple(fw_dynamic["feature"].astype(str).tolist())
    structural_features = tuple(fw_structural["feature"].astype(str).tolist())
    qa_add(rows, "dynamic_feature_set", set(dynamic_features) == set(EXPECTED_DYNAMIC_FEATURES), sorted(dynamic_features), sorted(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "dynamic_feature_count", len(dynamic_features) == 3, len(dynamic_features), 3)
    qa_add(rows, "structural_feature_set", set(structural_features) == set(EXPECTED_STRUCTURAL_FEATURES), sorted(structural_features), sorted(EXPECTED_STRUCTURAL_FEATURES))
    qa_add(rows, "removed_demographic_scale_features_absent", not ({"wp_age_population_norm", "population_density_per_km2_norm"} & set(dynamic_features)), sorted(set(dynamic_features) & {"wp_age_population_norm", "population_density_per_km2_norm"}), "none")

    membership_sum = partition[["u_cluster_1", "u_cluster_2"]].sum(axis=1).to_numpy(dtype=float)
    qa_add(rows, "memberships_sum_to_one", bool(np.allclose(membership_sum, 1.0, atol=MEMBERSHIP_TOL)), float(np.max(np.abs(membership_sum - 1.0))), f"<= {MEMBERSHIP_TOL}")

    observed_sizes = partition["hard_cluster"].astype(int).value_counts().sort_index().to_dict()
    qa_add(rows, "hard_cluster_sizes", observed_sizes == EXPECTED_HARD_CLUSTER_SIZES, observed_sizes, EXPECTED_HARD_CLUSTER_SIZES)
    qa_add(rows, "global_medoids", len(medoids) == EXPECTED_C, len(medoids), EXPECTED_C)

    qa_add(rows, "selected_config_regions", int(config["regions"]) == EXPECTED_REGIONS, int(config["regions"]), EXPECTED_REGIONS)
    qa_add(rows, "selected_config_countries", int(config["countries"]) == EXPECTED_COUNTRIES, int(config["countries"]), EXPECTED_COUNTRIES)
    qa_add(rows, "selected_config_period", list(map(int, config["analytical_period"])) == [2015, 2019], config["analytical_period"], [2015, 2019])
    qa_add(rows, "selected_c", int(config["selected_c"]) == EXPECTED_C, int(config["selected_c"]), EXPECTED_C)
    qa_add(rows, "selected_m", math.isclose(float(config["selected_m"]), EXPECTED_M, abs_tol=1e-12), float(config["selected_m"]), EXPECTED_M)
    qa_add(rows, "selected_dynamic_features", set(map(str, config.get("dynamic_features", []))) == set(EXPECTED_DYNAMIC_FEATURES), config.get("dynamic_features", []), list(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "selected_dynamic_families", set(map(str, config.get("dynamic_families", []))) == set(EXPECTED_DYNAMIC_FAMILIES), config.get("dynamic_families", []), list(EXPECTED_DYNAMIC_FAMILIES))
    qa_add(rows, "structural_excluded_from_clustering", not bool(config.get("structural_static_variables_included", True)), config.get("structural_static_variables_included"), False)
    qa_add(rows, "bod_components_not_extra_dimensions", not bool(config.get("script04_bod_decomposition_used_as_extra_dimensions", True)), config.get("script04_bod_decomposition_used_as_extra_dimensions"), False)

    qa_add(rows, "fuzzy_transition_regions", len(fuzzy_regions) == EXPECTED_TRANSITION_REGIONS, len(fuzzy_regions), EXPECTED_TRANSITION_REGIONS)
    transition_from_partition = int((partition["membership_margin"].astype(float) < TRANSITION_MARGIN_THRESHOLD).sum())
    qa_add(rows, "fuzzy_transition_reconstructed", transition_from_partition == EXPECTED_TRANSITION_REGIONS, transition_from_partition, EXPECTED_TRANSITION_REGIONS)

    convergence_flag = bool_series(convergence["converged_toward_destination"])
    convergers = int(convergence_flag.sum())
    qa_add(rows, "converging_regions", convergers == EXPECTED_CONVERGING_REGIONS, convergers, EXPECTED_CONVERGING_REGIONS)

    local_success = int(country_summary["status"].astype(str).eq("SUCCESS").sum())
    qa_add(rows, "viable_local_country_models", local_success == EXPECTED_LOCAL_SUCCESS_COUNTRIES, local_success, EXPECTED_LOCAL_SUCCESS_COUNTRIES)

    own_nearest = int(bool_series(country_nearest["nearest_reference_is_own_country"]).sum())
    qa_add(rows, "own_country_reference_nearest", own_nearest == EXPECTED_OWN_COUNTRY_NEAREST, own_nearest, EXPECTED_OWN_COUNTRY_NEAREST)

    qa_add(rows, "country_reference_comparison_rows", len(t["country_distances"]) == EXPECTED_REGIONS * EXPECTED_COUNTRIES, len(t["country_distances"]), EXPECTED_REGIONS * EXPECTED_COUNTRIES)
    # Script 06 stores ONE row per region-year. Each row already contains both
    # the distance to the origin medoid and the distance to the destination
    # medoid. Therefore the expected cardinality is 512 regions x 5 years =
    # 2,560 rows, not 2,560 x C.
    annual_medoid = t["annual_medoid_distances"].copy()

    assert_columns(
        annual_medoid,
        [
            "region_id",
            "year",
            "origin_cluster",
            "destination_cluster",
            "annual_distance_to_origin_medoid",
            "annual_distance_to_destination_medoid",
        ],
        "Script06 annual medoid distances",
    )

    annual_medoid["year"] = pd.to_numeric(
        annual_medoid["year"], errors="raise"
    ).astype(int)
    annual_medoid["origin_cluster"] = pd.to_numeric(
        annual_medoid["origin_cluster"], errors="raise"
    ).astype(int)
    annual_medoid["destination_cluster"] = pd.to_numeric(
        annual_medoid["destination_cluster"], errors="raise"
    ).astype(int)

    qa_add(
        rows,
        "annual_medoid_distance_rows",
        len(annual_medoid) == EXPECTED_REGION_YEARS,
        len(annual_medoid),
        EXPECTED_REGION_YEARS,
        "One row per region-year; origin and destination distances are stored as columns.",
    )

    annual_medoid_unique_region_year = (
        not annual_medoid.duplicated(["region_id", "year"]).any()
        and annual_medoid["region_id"].astype(str).nunique() == EXPECTED_REGIONS
        and sorted(annual_medoid["year"].unique().tolist()) == EXPECTED_YEARS
    )
    qa_add(
        rows,
        "annual_medoid_unique_region_year",
        annual_medoid_unique_region_year,
        (
            f"rows={len(annual_medoid)}; "
            f"regions={annual_medoid['region_id'].astype(str).nunique()}; "
            f"years={sorted(annual_medoid['year'].unique().tolist())}"
        ),
        (
            f"rows={EXPECTED_REGION_YEARS}; "
            f"regions={EXPECTED_REGIONS}; "
            f"years={EXPECTED_YEARS}"
        ),
    )

    expected_destination = EXPECTED_C + 1 - annual_medoid["origin_cluster"]
    medoid_pairing_valid = (
        set(annual_medoid["origin_cluster"].unique()).issubset(set(range(1, EXPECTED_C + 1)))
        and set(annual_medoid["destination_cluster"].unique()).issubset(set(range(1, EXPECTED_C + 1)))
        and (annual_medoid["destination_cluster"] == expected_destination).all()
        and (annual_medoid["origin_cluster"] != annual_medoid["destination_cluster"]).all()
    )
    qa_add(
        rows,
        "annual_medoid_origin_destination_pairing",
        bool(medoid_pairing_valid),
        (
            annual_medoid[
                ["origin_cluster", "destination_cluster"]
            ]
            .drop_duplicates()
            .sort_values(["origin_cluster", "destination_cluster"])
            .to_dict("records")
        ),
        [{"origin_cluster": 1, "destination_cluster": 2},
         {"origin_cluster": 2, "destination_cluster": 1}],
    )

    qa_add(rows, "profile_summary_rows", len(t["profile_summary"]) == EXPECTED_C * len(EXPECTED_DYNAMIC_FEATURES), len(t["profile_summary"]), EXPECTED_C * len(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "profile_annual_rows", len(t["profile_annual"]) == EXPECTED_C * len(EXPECTED_YEARS) * len(EXPECTED_DYNAMIC_FEATURES), len(t["profile_annual"]), EXPECTED_C * len(EXPECTED_YEARS) * len(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "bod_variable_profile_rows", len(t["bod_var_profile"]) == EXPECTED_C * 5, len(t["bod_var_profile"]), EXPECTED_C * 5)
    qa_add(rows, "bod_variable_annual_rows", len(t["bod_var_annual"]) == EXPECTED_C * len(EXPECTED_YEARS) * 5, len(t["bod_var_annual"]), EXPECTED_C * len(EXPECTED_YEARS) * 5)

    qa_add(rows, "feature_contribution_rows", len(t["conv_feature"]) == EXPECTED_C * len(EXPECTED_DYNAMIC_FEATURES), len(t["conv_feature"]), EXPECTED_C * len(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "feature_contribution_set", set(t["conv_feature"]["feature"].astype(str)) == set(EXPECTED_DYNAMIC_FEATURES), sorted(t["conv_feature"]["feature"].astype(str).unique()), sorted(EXPECTED_DYNAMIC_FEATURES))
    qa_add(rows, "variable_contribution_set", set(t["conv_variable"]["variable"].astype(str)) == EXPECTED_CONVERGENCE_VARIABLES, sorted(t["conv_variable"]["variable"].astype(str).unique()), sorted(EXPECTED_CONVERGENCE_VARIABLES))
    qa_add(rows, "variable_region_detail_rows", len(t["conv_variable_region"]) == EXPECTED_REGIONS * len(EXPECTED_CONVERGENCE_VARIABLES), len(t["conv_variable_region"]), EXPECTED_REGIONS * len(EXPECTED_CONVERGENCE_VARIABLES))
    qa_add(rows, "variable_annual_detail_rows", len(t["conv_variable_annual"]) == EXPECTED_REGION_YEARS * len(EXPECTED_CONVERGENCE_VARIABLES), len(t["conv_variable_annual"]), EXPECTED_REGION_YEARS * len(EXPECTED_CONVERGENCE_VARIABLES))

    # Exact Script-04 common-weight BoD score reconstruction.
    contrib = t["bod_contrib"].copy()
    assert_columns(contrib, ["block", "region_id", "family", "variable", "common_weight", "weighted_contribution", "family_score"], "Script04 BoD contributions")
    dynamic_contrib = contrib[contrib["block"].astype(str).eq("DYNAMIC")].copy()
    structural_contrib = contrib[contrib["block"].astype(str).eq("STRUCTURAL")].copy()
    dynamic_recon = (
        dynamic_contrib.groupby(["region_id", "year", "family"], as_index=False)
        .agg(reconstructed=("weighted_contribution", "sum"), score=("family_score", "first"))
    )
    structural_recon = (
        structural_contrib.groupby(["region_id", "family"], as_index=False)
        .agg(reconstructed=("weighted_contribution", "sum"), score=("family_score", "first"))
    )
    dyn_err = float(np.max(np.abs(dynamic_recon["reconstructed"] - dynamic_recon["score"])))
    struct_err = float(np.max(np.abs(structural_recon["reconstructed"] - structural_recon["score"])))
    qa_add(rows, "dynamic_bod_exact_reconstruction", dyn_err <= NUMERIC_TOL, dyn_err, f"<= {NUMERIC_TOL}")
    qa_add(rows, "structural_bod_exact_reconstruction", struct_err <= NUMERIC_TOL, struct_err, f"<= {NUMERIC_TOL}")

    observed_dynamic_bod = {
        family: set(group["variable"].astype(str))
        for family, group in dynamic_contrib.groupby("family")
    }
    qa_add(rows, "dynamic_bod_variable_contract", observed_dynamic_bod == EXPECTED_DYNAMIC_BOD_VARIABLES, observed_dynamic_bod, EXPECTED_DYNAMIC_BOD_VARIABLES)

    # Exact Script-06 variable-level contribution decomposition.
    variable_region = t["conv_variable_region"].copy()
    conv_check = (
        variable_region.groupby("region_id", as_index=False)
        .agg(contribution_sum=("convergence_contribution", "sum"))
        .merge(convergence[["region_id", "convergence_index"]], on="region_id", how="left", validate="one_to_one")
    )
    conv_err = float(np.max(np.abs(conv_check["contribution_sum"] - conv_check["convergence_index"])))
    qa_add(rows, "variable_contributions_reconstruct_convergence", conv_err <= NUMERIC_TOL, conv_err, f"<= {NUMERIC_TOL}")

    return enforce_qa(rows, "Frozen analytical QA")


# =============================================================================
# 1.7.1.2 — CONSOLIDATED REGIONAL AND CLUSTER RESULTS
# =============================================================================


def build_region_results(t: dict[str, Any]) -> pd.DataFrame:
    official = t["official_regions"].copy()
    structural = t["structural"].copy()
    partition = t["global_partition"].copy()
    convergence = t["convergence_region"].copy()
    country_nearest = t["country_nearest"].copy()
    global_local = t["global_local_region"].copy()

    base_cols = [c for c in ["region_id", "code", "geo_code", "geo_name"] if c in official.columns]
    region = official.copy()

    structural_keep = ["region_id", *EXPECTED_STRUCTURAL_FEATURES]
    region = region.merge(
        structural[structural_keep],
        on="region_id",
        how="left",
        validate="one_to_one",
    )

    partition_keep = [
        "region_id", "hard_cluster", "max_membership", "second_membership",
        "membership_margin", "membership_entropy_normalized", "u_cluster_1", "u_cluster_2",
    ]
    region = region.merge(
        partition[partition_keep], on="region_id", how="left", validate="one_to_one"
    )

    convergence_keep = [
        "region_id", "destination_cluster", "flow", "relative_gap_2015",
        "relative_gap_2019", "convergence_index",
        "no_direction_within_numeric_tolerance", "converged_toward_destination",
        "moved_away_from_destination",
    ]
    convergence_keep = [c for c in convergence_keep if c in convergence.columns]
    region = region.merge(
        convergence[convergence_keep], on="region_id", how="left", validate="one_to_one"
    )

    country_keep = ["region_id"] + [
        c for c in country_nearest.columns
        if c not in {"region_id", "code", "geo_code", "geo_name"}
    ]
    region = region.merge(
        country_nearest[country_keep], on="region_id", how="left", validate="one_to_one"
    )

    local_candidates = [
        "region_id", "country_model_status", "country_selected_c", "local_status",
        "local_c", "local_hard_cluster", "local_max_membership",
        "local_membership_margin", "local_membership_entropy", "local_transition_zone",
    ]
    local_keep = [c for c in local_candidates if c in global_local.columns]
    if len(local_keep) > 1:
        region = region.merge(
            global_local[local_keep], on="region_id", how="left", validate="one_to_one"
        )

    region["transition_zone"] = region["membership_margin"].astype(float) < TRANSITION_MARGIN_THRESHOLD
    region["structural_context_used_in_clustering"] = False
    region["formal_cluster_change_2015_2019"] = False
    region["flow_interpretation"] = "relative movement toward alternative fixed medoid profile"

    # Put identity columns first without discarding additional official-region metadata.
    front = [c for c in base_cols if c in region.columns]
    remaining = [c for c in region.columns if c not in front]
    return region[front + remaining].sort_values(["code", "geo_code", "region_id"], kind="stable").reset_index(drop=True)


def build_cluster_summary(t: dict[str, Any]) -> pd.DataFrame:
    partition = t["global_partition"].copy()
    medoids = t["global_medoids"].copy()
    fuzzy = t["fuzzy_summary"].copy()
    conv_flow = t["convergence_flow"].copy()

    rows: list[dict[str, Any]] = []
    for cluster in range(1, EXPECTED_C + 1):
        g = partition[partition["hard_cluster"].astype(int).eq(cluster)]
        membership_col = f"u_cluster_{cluster}"
        flow = f"C{cluster}->C{3 - cluster}"
        med = medoids[medoids["cluster"].astype(int).eq(cluster)]
        fuzzy_row = fuzzy[fuzzy["fuzzy_direction"].astype(str).eq(flow)]
        conv_row = conv_flow[conv_flow["flow"].astype(str).eq(flow)]

        row: dict[str, Any] = {
            "cluster": cluster,
            "hard_cluster_size": len(g),
            "effective_fuzzy_size": float(partition[membership_col].astype(float).sum()),
            "mean_max_membership": float(g["max_membership"].astype(float).mean()),
            "median_membership_margin": float(g["membership_margin"].astype(float).median()),
            "mean_membership_entropy_normalized": float(g["membership_entropy_normalized"].astype(float).mean()),
            "alternative_profile": 3 - cluster,
            "directional_flow_label": flow,
        }
        if not med.empty:
            for source, target in [
                ("region_id", "medoid_region_id"),
                ("code", "medoid_country_code"),
                ("geo_code", "medoid_geo_code"),
                ("geo_name", "medoid_geo_name"),
            ]:
                if source in med.columns:
                    row[target] = med.iloc[0][source]
        if not fuzzy_row.empty:
            row["fuzzy_transition_regions"] = int(fuzzy_row.iloc[0]["transition_regions"])
            row["fuzzy_transition_share_within_cluster"] = float(fuzzy_row.iloc[0]["transition_share_within_origin"])
        if not conv_row.empty:
            row["regions_converging_toward_alternative_profile"] = int(conv_row.iloc[0]["regions_converging_toward_destination"])
            row["share_converging_toward_alternative_profile"] = float(conv_row.iloc[0]["share_converging_toward_destination"])
            row["median_convergence_index_all"] = float(conv_row.iloc[0]["median_convergence_index_all"])
            row["mean_convergence_index_all"] = float(conv_row.iloc[0]["mean_convergence_index_all"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_model_summary(t: dict[str, Any]) -> pd.DataFrame:
    config = t["selected_config"]
    bootstrap = t["global_bootstrap"]
    country_summary = t["country_summary"]
    fuzzy_regions = t["fuzzy_regions"]
    convergence = t["convergence_region"]
    country_nearest = t["country_nearest"]

    bootstrap_row = bootstrap.iloc[0] if not bootstrap.empty else pd.Series(dtype=object)
    values = [
        ("analytical_period", "2015–2019"),
        ("regions", EXPECTED_REGIONS),
        ("countries", EXPECTED_COUNTRIES),
        ("clustering_method", config.get("model", "M-Exp-FCMd")),
        ("selected_c", int(config["selected_c"])),
        ("selected_m", float(config["selected_m"])),
        ("fuzzy_silhouette", float(config["fuzzy_silhouette"])),
        ("objective", float(config["objective"])),
        ("beta", float(config["beta"])),
        ("beta_central_region_id", config.get("beta_central_region_id")),
        ("hard_cluster_sizes", json.dumps(config.get("hard_cluster_sizes", []))),
        ("effective_cluster_sizes", json.dumps(config.get("effective_cluster_sizes", []))),
        ("dynamic_features", ",".join(EXPECTED_DYNAMIC_FEATURES)),
        ("dynamic_families", ",".join(EXPECTED_DYNAMIC_FAMILIES)),
        ("structural_context_used_in_clustering", False),
        ("bod_decomposition_used_as_extra_clustering_dimensions", False),
        ("fuzzy_transition_threshold_membership_margin", TRANSITION_MARGIN_THRESHOLD),
        ("fuzzy_transition_regions", len(fuzzy_regions)),
        ("regions_converging_toward_alternative_profile", int(bool_series(convergence["converged_toward_destination"]).sum())),
        ("countries_with_viable_local_models", int(country_summary["status"].astype(str).eq("SUCCESS").sum())),
        ("own_country_reference_nearest_regions", int(bool_series(country_nearest["nearest_reference_is_own_country"]).sum())),
    ]
    for column, parameter in [
        ("requested_reps", "global_bootstrap_requested_reps"),
        ("successful_reps", "global_bootstrap_successful_reps"),
        ("success_rate", "global_bootstrap_success_rate"),
        ("ari_mean", "global_bootstrap_ari_mean"),
        ("ari_median", "global_bootstrap_ari_median"),
        ("fuzzy_similarity_mean", "global_bootstrap_fuzzy_similarity_mean"),
        ("fuzzy_similarity_median", "global_bootstrap_fuzzy_similarity_median"),
    ]:
        if column in bootstrap_row.index:
            values.append((parameter, bootstrap_row[column]))
    return pd.DataFrame(values, columns=["parameter", "value"])


# =============================================================================
# 1.7.1.3–1.7.1.7 — APPROVED INTERPRETATION LAYERS
# =============================================================================


def prepare_interpretation_tables(t: dict[str, Any]) -> dict[str, pd.DataFrame]:
    # These are direct, lossless consolidations of approved Script-06 results.
    # No analytical value is recalculated here.
    return {
        "profile_summary": t["profile_summary"].copy(),
        "profile_annual": t["profile_annual"].copy(),
        "bod_var_profile": t["bod_var_profile"].copy(),
        "bod_var_annual": t["bod_var_annual"].copy(),
        "fuzzy_summary": t["fuzzy_summary"].copy(),
        "fuzzy_regions": t["fuzzy_regions"].copy(),
        "convergence_flow": t["convergence_flow"].copy(),
        "convergence_region": t["convergence_region"].copy(),
        "conv_feature": t["conv_feature"].copy(),
        "conv_family": t["conv_family"].copy(),
        "conv_variable": t["conv_variable"].copy(),
        "global_local_country": t["global_local_country"].copy(),
        "global_local_region": t["global_local_region"].copy(),
        "country_reference": t["country_nearest"].copy(),
    }


# =============================================================================
# 1.7.1.8 — SUMMARY SPREADSHEET
# =============================================================================


def write_consolidated_workbook(
    model_summary: pd.DataFrame,
    region_results: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    interpretation: dict[str, pd.DataFrame],
    t: dict[str, Any],
    input_manifest: pd.DataFrame,
    workbook_sync: pd.DataFrame,
    final_qa: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(CONSOLIDATED_WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        model_summary.to_excel(writer, sheet_name="Model_Summary", index=False)
        t["c_selection"].to_excel(writer, sheet_name="Model_Selection", index=False)
        t["global_bootstrap"].to_excel(writer, sheet_name="Bootstrap_Summary", index=False)
        t["global_medoids"].to_excel(writer, sheet_name="Global_Medoids", index=False)
        t["feature_weights"].to_excel(writer, sheet_name="Feature_Weights", index=False)
        t["bod_weights"].to_excel(writer, sheet_name="BoD_Common_Weights", index=False)
        region_results.to_excel(writer, sheet_name="Region_Results", index=False)
        cluster_summary.to_excel(writer, sheet_name="Cluster_Summary", index=False)
        interpretation["profile_summary"].to_excel(writer, sheet_name="Profile_Summary", index=False)
        interpretation["profile_annual"].to_excel(writer, sheet_name="Profile_Annual", index=False)
        interpretation["bod_var_profile"].to_excel(writer, sheet_name="BoD_Var_Profile", index=False)
        interpretation["bod_var_annual"].to_excel(writer, sheet_name="BoD_Var_Annual", index=False)
        interpretation["fuzzy_summary"].to_excel(writer, sheet_name="Fuzzy_Summary", index=False)
        interpretation["fuzzy_regions"].to_excel(writer, sheet_name="Fuzzy_Regions", index=False)
        interpretation["convergence_flow"].to_excel(writer, sheet_name="Convergence_Flow", index=False)
        interpretation["convergence_region"].to_excel(writer, sheet_name="Convergence_Region", index=False)
        interpretation["conv_feature"].to_excel(writer, sheet_name="Conv_Feature", index=False)
        interpretation["conv_family"].to_excel(writer, sheet_name="Conv_Family", index=False)
        interpretation["conv_variable"].to_excel(writer, sheet_name="Conv_Variable", index=False)
        t["conv_variable_region"].to_excel(writer, sheet_name="Conv_Var_Region", index=False)
        interpretation["country_reference"].to_excel(writer, sheet_name="Country_Reference", index=False)
        t["country_summary"].to_excel(writer, sheet_name="Local_Models", index=False)
        interpretation["global_local_country"].to_excel(writer, sheet_name="Global_Local_Country", index=False)
        interpretation["global_local_region"].to_excel(writer, sheet_name="Global_Local_Region", index=False)
        input_manifest.to_excel(writer, sheet_name="Input_Manifest", index=False)
        workbook_sync.to_excel(writer, sheet_name="Workbook_Sync", index=False)
        final_qa.to_excel(writer, sheet_name="Final_QA", index=False)
    autofit_workbook(CONSOLIDATED_WORKBOOK_OUTPUT)


# =============================================================================
# 1.7.1.9 — RESULTS REPORT
# =============================================================================


def fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def write_results_report(
    t: dict[str, Any],
    cluster_summary: pd.DataFrame,
    interpretation: dict[str, pd.DataFrame],
) -> None:
    config = t["selected_config"]
    bootstrap = t["global_bootstrap"].iloc[0]
    fuzzy_all = interpretation["fuzzy_summary"]
    fuzzy_all = fuzzy_all[fuzzy_all["fuzzy_direction"].astype(str).eq("ALL")]
    transition_share = (
        float(fuzzy_all.iloc[0]["transition_share_within_origin"])
        if not fuzzy_all.empty else EXPECTED_TRANSITION_REGIONS / EXPECTED_REGIONS
    )

    lines = [
        "# P1 — Consolidated Results",
        "",
        "## Scope",
        "",
        f"The final analytical sample contains **{EXPECTED_REGIONS} subnational regions in {EXPECTED_COUNTRIES} countries**, observed annually from **2015 to 2019**. The additional candidate region outside the complete five-year analytical sample is not part of Scripts 04–07 final results.",
        "",
        "## Global M-Exp-FCMd solution",
        "",
        f"- Selected number of clusters: **C = {int(config['selected_c'])}**.",
        f"- Selected fuzziness parameter: **m = {float(config['selected_m']):.1f}**.",
        f"- Fuzzy Silhouette: **{float(config['fuzzy_silhouette']):.6f}**.",
        f"- Hard cluster sizes: **C1 = {EXPECTED_HARD_CLUSTER_SIZES[1]}**, **C2 = {EXPECTED_HARD_CLUSTER_SIZES[2]}**.",
        f"- Robust exponential beta: **{float(config['beta']):.6f}**.",
        "- Clustering uses only the three final dynamic family-level features; structural/static variables remain contextual and are not part of the clustering distance.",
        "",
        "### Dynamic clustering features",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in EXPECTED_DYNAMIC_FEATURES)

    ari = bootstrap.get("ari_mean", np.nan)
    fuzzy_similarity = bootstrap.get("fuzzy_similarity_mean", np.nan)
    lines.extend([
        "",
        "## Stability",
        "",
        f"The global bootstrap completed **{int(bootstrap.get('successful_reps', 0))}/{int(bootstrap.get('requested_reps', 0))}** successful replications. The mean hard-partition Adjusted Rand Index was **{float(ari):.6f}**, while mean fuzzy-membership similarity was **{float(fuzzy_similarity):.6f}**. These statistics describe different aspects of stability and are not used to re-select C.",
        "",
        "## Fuzzy transition",
        "",
        f"**{EXPECTED_TRANSITION_REGIONS} regions ({fmt_pct(transition_share)})** have a membership margin below {TRANSITION_MARGIN_THRESHOLD:.2f} and are therefore flagged as fuzzy-transition regions.",
        "",
        "## Temporal convergence",
        "",
        f"**{EXPECTED_CONVERGING_REGIONS} regions** moved relatively closer to the alternative fixed global medoid profile between 2015 and 2019. This is a directional convergence measure and **does not represent formal cluster reassignment**.",
        "",
    ])

    for _, row in interpretation["convergence_flow"].sort_values("flow").iterrows():
        lines.append(
            f"- **{row['flow']}**: {int(row['regions_converging_toward_destination'])}/{int(row['origin_regions'])} regions ({100.0 * float(row['share_converging_toward_destination']):.2f}%)."
        )

    lines.extend([
        "",
        "## Common-weight BoD decomposition",
        "",
        "The annual BoD family scores are constructed with one common weight vector per family. Script 07.1 preserves the underlying DMU × year × variable × common-weight decomposition for interpretation and audit, but does not reintroduce those variables as independent clustering dimensions.",
        "",
        "### Variable contributions to convergence",
        "",
        "The table below reports the variable-level aggregation already produced by Script 06 for regions with positive overall convergence in each directional flow. Positive consistency means the variable contributed in the direction of the alternative profile for that share of converging regions.",
        "",
        "| Flow | Variable | Median contribution | Mean contribution | Consistency |",
        "|---|---|---:|---:|---:|",
    ])
    variable = interpretation["conv_variable"].copy()
    variable = variable.sort_values(["flow", "median_contribution"], ascending=[True, False], kind="stable")
    for _, row in variable.iterrows():
        lines.append(
            f"| {row['flow']} | `{row['variable']}` | {float(row['median_contribution']):.6f} | {float(row['mean_contribution']):.6f} | {float(row['consistency_pct']):.2f}% |"
        )

    own_count = int(bool_series(t["country_nearest"]["nearest_reference_is_own_country"]).sum())
    local_success = int(t["country_summary"]["status"].astype(str).eq("SUCCESS").sum())
    lines.extend([
        "",
        "## Country-reference and local-model context",
        "",
        f"The region × country-reference analysis contains **{EXPECTED_REGIONS * EXPECTED_COUNTRIES:,} comparisons**. The leave-one-out own-country reference is the nearest reference for **{own_count}/{EXPECTED_REGIONS} regions**.",
        "",
        f"Independent intracountry M-Exp-FCMd models are technically viable for **{local_success}/{EXPECTED_COUNTRIES} countries**. Global-vs-local ARI is descriptive only, especially where the locally selected number of clusters differs from the global C=2 solution.",
        "",
        "## Interpretation constraints",
        "",
        "- Clusters are descriptive trajectory profiles, not an overall ranking of regions.",
        "- `C1->C2` and `C2->C1` denote relative convergence toward a fixed alternative medoid profile, not cluster switching.",
        "- Common-weight BoD contributions explain family-score variation; they are not extra M-Exp-FCMd dimensions.",
        "- Structural variables are retained as contextual descriptors only and do not affect the frozen global clustering solution.",
        "",
        "## Consolidated artifacts",
        "",
        f"- Workbook: `{relative_path(CONSOLIDATED_WORKBOOK_OUTPUT)}`",
        f"- Machine-readable tables: `{relative_path(SCRIPT07_DIR)}/`",
        "",
    ])

    RESULTS_REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 1.7.1.10 — FINAL QA AND EXPORT
# =============================================================================


def build_final_qa(
    frozen_qa: pd.DataFrame,
    region_results: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    interpretation: dict[str, pd.DataFrame],
    t: dict[str, Any],
) -> pd.DataFrame:
    rows = frozen_qa.to_dict("records")

    qa_add(rows, "consolidated_region_rows", len(region_results) == EXPECTED_REGIONS, len(region_results), EXPECTED_REGIONS)
    qa_add(rows, "consolidated_region_unique_key", not region_results["region_id"].duplicated().any(), int(region_results["region_id"].duplicated().sum()), 0)
    qa_add(rows, "consolidated_cluster_rows", len(cluster_summary) == EXPECTED_C, len(cluster_summary), EXPECTED_C)
    qa_add(rows, "consolidated_cluster_size_sum", int(cluster_summary["hard_cluster_size"].sum()) == EXPECTED_REGIONS, int(cluster_summary["hard_cluster_size"].sum()), EXPECTED_REGIONS)
    qa_add(rows, "consolidated_transition_count", int(region_results["transition_zone"].sum()) == EXPECTED_TRANSITION_REGIONS, int(region_results["transition_zone"].sum()), EXPECTED_TRANSITION_REGIONS)
    qa_add(rows, "consolidated_convergence_count", int(bool_series(region_results["converged_toward_destination"]).sum()) == EXPECTED_CONVERGING_REGIONS, int(bool_series(region_results["converged_toward_destination"]).sum()), EXPECTED_CONVERGING_REGIONS)
    qa_add(rows, "no_formal_cluster_changes_encoded", not bool_series(region_results["formal_cluster_change_2015_2019"]).any(), int(bool_series(region_results["formal_cluster_change_2015_2019"]).sum()), 0)
    qa_add(rows, "structural_context_not_marked_as_clustering", not bool_series(region_results["structural_context_used_in_clustering"]).any(), int(bool_series(region_results["structural_context_used_in_clustering"]).sum()), 0)

    # Reconcile flow origin totals with hard-cluster sizes.
    flow = interpretation["convergence_flow"].copy()
    for cluster in [1, 2]:
        label = f"C{cluster}->C{3-cluster}"
        row = flow[flow["flow"].astype(str).eq(label)]
        observed = int(row.iloc[0]["origin_regions"]) if not row.empty else -1
        qa_add(rows, f"flow_origin_matches_cluster_{cluster}", observed == EXPECTED_HARD_CLUSTER_SIZES[cluster], observed, EXPECTED_HARD_CLUSTER_SIZES[cluster])

    # Reconcile current output tables with their Script-06 source tables.
    for name, source, consolidated in [
        ("profile_summary", t["profile_summary"], interpretation["profile_summary"]),
        ("profile_annual", t["profile_annual"], interpretation["profile_annual"]),
        ("bod_var_profile", t["bod_var_profile"], interpretation["bod_var_profile"]),
        ("bod_var_annual", t["bod_var_annual"], interpretation["bod_var_annual"]),
        ("fuzzy_summary", t["fuzzy_summary"], interpretation["fuzzy_summary"]),
        ("fuzzy_regions", t["fuzzy_regions"], interpretation["fuzzy_regions"]),
        ("convergence_flow", t["convergence_flow"], interpretation["convergence_flow"]),
        ("convergence_region", t["convergence_region"], interpretation["convergence_region"]),
        ("conv_feature", t["conv_feature"], interpretation["conv_feature"]),
        ("conv_family", t["conv_family"], interpretation["conv_family"]),
        ("conv_variable", t["conv_variable"], interpretation["conv_variable"]),
        ("global_local_country", t["global_local_country"], interpretation["global_local_country"]),
        ("global_local_region", t["global_local_region"], interpretation["global_local_region"]),
        ("country_reference", t["country_nearest"], interpretation["country_reference"]),
    ]:
        same = source.equals(consolidated)
        qa_add(rows, f"lossless_copy_{name}", same, source.shape, source.shape)

    return enforce_qa(rows, "Final Script-07.1 QA")


def write_machine_readable_outputs(
    model_summary: pd.DataFrame,
    region_results: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    interpretation: dict[str, pd.DataFrame],
    final_qa: pd.DataFrame,
) -> None:
    write_csv(model_summary, MODEL_SUMMARY_OUTPUT)
    write_csv(region_results, REGION_RESULTS_OUTPUT)
    write_csv(cluster_summary, CLUSTER_SUMMARY_OUTPUT)
    write_csv(interpretation["profile_summary"], PROFILE_SUMMARY_OUTPUT)
    write_csv(interpretation["profile_annual"], PROFILE_ANNUAL_OUTPUT)
    write_csv(interpretation["bod_var_profile"], BOD_VAR_PROFILE_OUTPUT)
    write_csv(interpretation["bod_var_annual"], BOD_VAR_ANNUAL_OUTPUT)
    write_csv(interpretation["fuzzy_summary"], FUZZY_SUMMARY_OUTPUT)
    write_csv(interpretation["fuzzy_regions"], FUZZY_REGIONS_OUTPUT)
    write_csv(interpretation["convergence_flow"], CONVERGENCE_FLOW_OUTPUT)
    write_csv(interpretation["convergence_region"], CONVERGENCE_REGIONS_OUTPUT)
    write_csv(interpretation["conv_feature"], CONTRIB_FEATURE_OUTPUT)
    write_csv(interpretation["conv_family"], CONTRIB_FAMILY_OUTPUT)
    write_csv(interpretation["conv_variable"], CONTRIB_VARIABLE_OUTPUT)
    write_csv(interpretation["global_local_country"], GLOBAL_LOCAL_COUNTRY_OUTPUT)
    write_csv(interpretation["global_local_region"], GLOBAL_LOCAL_REGION_OUTPUT)
    write_csv(interpretation["country_reference"], COUNTRY_REFERENCE_OUTPUT)
    write_csv(final_qa, FINAL_QA_OUTPUT)


def output_files() -> list[Path]:
    return [
        INPUT_MANIFEST_OUTPUT,
        WORKBOOK_SYNC_OUTPUT,
        MODEL_SUMMARY_OUTPUT,
        REGION_RESULTS_OUTPUT,
        CLUSTER_SUMMARY_OUTPUT,
        PROFILE_SUMMARY_OUTPUT,
        PROFILE_ANNUAL_OUTPUT,
        BOD_VAR_PROFILE_OUTPUT,
        BOD_VAR_ANNUAL_OUTPUT,
        FUZZY_SUMMARY_OUTPUT,
        FUZZY_REGIONS_OUTPUT,
        CONVERGENCE_FLOW_OUTPUT,
        CONVERGENCE_REGIONS_OUTPUT,
        CONTRIB_FEATURE_OUTPUT,
        CONTRIB_FAMILY_OUTPUT,
        CONTRIB_VARIABLE_OUTPUT,
        GLOBAL_LOCAL_COUNTRY_OUTPUT,
        GLOBAL_LOCAL_REGION_OUTPUT,
        COUNTRY_REFERENCE_OUTPUT,
        FINAL_QA_OUTPUT,
        CONSOLIDATED_WORKBOOK_OUTPUT,
        RESULTS_REPORT_OUTPUT,
    ]


def build_output_manifest() -> pd.DataFrame:
    rows = []
    for path in output_files():
        if not path.exists():
            raise FileNotFoundError(f"Expected Script-07.1 output was not created: {relative_path(path)}")
        rows.append({
            "relative_path": relative_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    t0 = time.perf_counter()
    ensure_directories()

    print_header("SCRIPT 07.1 v1.0.4 — CONSOLIDATE APPROVED P1 RESULTS")
    print(f"Project root: {ROOT}")
    print(f"Analytical period: {EXPECTED_YEARS[0]}–{EXPECTED_YEARS[-1]}")
    print(f"Final analytical sample: {EXPECTED_REGIONS} regions | {EXPECTED_COUNTRIES} countries")
    print("Prior analytical recalculation: NONE")
    print("Source of truth: validated Scripts 04, 05 and 06")

    # ------------------------------------------------------------------
    # 1.7.1.1
    # ------------------------------------------------------------------
    print_header("1/10 — 1.7.1.1 INPUTS + OUTPUTS + QA")
    input_manifest = build_input_manifest()
    write_csv(input_manifest, INPUT_MANIFEST_OUTPUT)
    print(f"Required inputs found: {len(input_manifest):,}")

    workbook_sync = validate_workbook_csv_sync()
    write_csv(workbook_sync, WORKBOOK_SYNC_OUTPUT)
    print(f"Workbook/CSV pairs synchronized: {len(workbook_sync):,}")

    t = load_inputs()
    frozen_qa = validate_frozen_results(t)
    print(f"Frozen analytical QA checks passed: {len(frozen_qa):,}")

    # ------------------------------------------------------------------
    # 1.7.1.2
    # ------------------------------------------------------------------
    print_header("2/10 — 1.7.1.2 REGIONAL + CLUSTER CONSOLIDATION")
    region_results = build_region_results(t)
    cluster_summary = build_cluster_summary(t)
    model_summary = build_model_summary(t)
    print(f"Consolidated regions: {len(region_results):,}")
    print(f"Cluster summary rows: {len(cluster_summary):,}")
    print(cluster_summary[["cluster", "hard_cluster_size", "effective_fuzzy_size", "fuzzy_transition_regions", "regions_converging_toward_alternative_profile"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 1.7.1.3
    # ------------------------------------------------------------------
    print_header("3/10 — 1.7.1.3 PROFILES + EVOLUTION")
    interpretation = prepare_interpretation_tables(t)
    print(f"Feature profile rows: {len(interpretation['profile_summary']):,}")
    print(f"Feature annual rows: {len(interpretation['profile_annual']):,}")
    print(f"BoD variable profile rows: {len(interpretation['bod_var_profile']):,}")
    print(f"BoD variable annual rows: {len(interpretation['bod_var_annual']):,}")

    # ------------------------------------------------------------------
    # 1.7.1.4
    # ------------------------------------------------------------------
    print_header("4/10 — 1.7.1.4 FUZZY / TRANSITION")
    print(interpretation["fuzzy_summary"].to_string(index=False))

    # ------------------------------------------------------------------
    # 1.7.1.5
    # ------------------------------------------------------------------
    print_header("5/10 — 1.7.1.5 TEMPORAL CONVERGENCE")
    print(interpretation["convergence_flow"].to_string(index=False))
    print("Interpretation: directional convergence toward a fixed alternative medoid; not formal cluster reassignment.")

    # ------------------------------------------------------------------
    # 1.7.1.6
    # ------------------------------------------------------------------
    print_header("6/10 — 1.7.1.6 CONTRIBUTIONS + CONSISTENCY")
    print("Variable-level contribution table:")
    print(
        interpretation["conv_variable"]
        .sort_values(["flow", "median_contribution"], ascending=[True, False], kind="stable")
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 1.7.1.7
    # ------------------------------------------------------------------
    print_header("7/10 — 1.7.1.7 GLOBAL × LOCAL + COUNTRY REFERENCES")
    local_success = int(t["country_summary"]["status"].astype(str).eq("SUCCESS").sum())
    own_nearest = int(bool_series(t["country_nearest"]["nearest_reference_is_own_country"]).sum())
    print(f"Countries with viable local model: {local_success}/{EXPECTED_COUNTRIES}")
    print(f"Own-country reference nearest: {own_nearest}/{EXPECTED_REGIONS}")

    # ------------------------------------------------------------------
    # 1.7.1.8 preparation + 1.7.1.10 analytical QA before file writing
    # ------------------------------------------------------------------
    print_header("8/10 — 1.7.1.8 SUMMARY TABLES + PRE-EXPORT QA")
    final_qa = build_final_qa(
        frozen_qa=frozen_qa,
        region_results=region_results,
        cluster_summary=cluster_summary,
        interpretation=interpretation,
        t=t,
    )
    print(f"Final QA checks passed: {len(final_qa):,}")

    write_machine_readable_outputs(
        model_summary=model_summary,
        region_results=region_results,
        cluster_summary=cluster_summary,
        interpretation=interpretation,
        final_qa=final_qa,
    )

    write_consolidated_workbook(
        model_summary=model_summary,
        region_results=region_results,
        cluster_summary=cluster_summary,
        interpretation=interpretation,
        t=t,
        input_manifest=input_manifest,
        workbook_sync=workbook_sync,
        final_qa=final_qa,
    )
    print(f"Consolidated workbook: {relative_path(CONSOLIDATED_WORKBOOK_OUTPUT)}")

    # ------------------------------------------------------------------
    # 1.7.1.9
    # ------------------------------------------------------------------
    print_header("9/10 — 1.7.1.9 RESULTS REPORT")
    write_results_report(t, cluster_summary, interpretation)
    print(f"Results report: {relative_path(RESULTS_REPORT_OUTPUT)}")

    # ------------------------------------------------------------------
    # 1.7.1.10
    # ------------------------------------------------------------------
    print_header("10/10 — 1.7.1.10 FINAL QA + EXPORT MANIFEST")
    output_manifest = build_output_manifest()
    write_csv(output_manifest, OUTPUT_MANIFEST_OUTPUT)

    status = {
        "script": "07.1_consolidate_results.py",
        "status": "PASS",
        "analytical_period": [2015, 2019],
        "final_regions": EXPECTED_REGIONS,
        "countries": EXPECTED_COUNTRIES,
        "selected_c": EXPECTED_C,
        "selected_m": EXPECTED_M,
        "hard_cluster_sizes": EXPECTED_HARD_CLUSTER_SIZES,
        "fuzzy_transition_regions": EXPECTED_TRANSITION_REGIONS,
        "converging_regions": EXPECTED_CONVERGING_REGIONS,
        "viable_local_country_models": EXPECTED_LOCAL_SUCCESS_COUNTRIES,
        "own_country_reference_nearest_regions": EXPECTED_OWN_COUNTRY_NEAREST,
        "prior_analytical_recalculation": False,
        "ready_for_script_07_2": True,
        "consolidated_workbook": relative_path(CONSOLIDATED_WORKBOOK_OUTPUT),
        "results_report": relative_path(RESULTS_REPORT_OUTPUT),
        "output_manifest": relative_path(OUTPUT_MANIFEST_OUTPUT),
    }
    FINAL_STATUS_OUTPUT.write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Append the final status file itself to a second/final manifest snapshot.
    final_manifest = output_manifest.copy()
    final_manifest = pd.concat([
        final_manifest,
        pd.DataFrame([{
            "relative_path": relative_path(FINAL_STATUS_OUTPUT),
            "size_bytes": FINAL_STATUS_OUTPUT.stat().st_size,
            "sha256": file_sha256(FINAL_STATUS_OUTPUT),
        }]),
    ], ignore_index=True)
    write_csv(final_manifest, OUTPUT_MANIFEST_OUTPUT)

    elapsed = time.perf_counter() - t0

    print("\nFINAL SCRIPT 07.1 SUMMARY")
    print("-" * 100)
    print(f"Final regions:                   {EXPECTED_REGIONS}")
    print(f"Countries:                       {EXPECTED_COUNTRIES}")
    print(f"Selected global configuration:   C={EXPECTED_C}, m={EXPECTED_M}")
    print(f"Hard cluster sizes:              {EXPECTED_HARD_CLUSTER_SIZES}")
    print(f"Fuzzy-transition regions:        {EXPECTED_TRANSITION_REGIONS}")
    print(f"Converging regions:              {EXPECTED_CONVERGING_REGIONS}")
    print(f"Viable local-country models:     {EXPECTED_LOCAL_SUCCESS_COUNTRIES}")
    print(f"Own-country reference nearest:   {EXPECTED_OWN_COUNTRY_NEAREST}")
    print(f"Consolidated workbook:           {relative_path(CONSOLIDATED_WORKBOOK_OUTPUT)}")
    print(f"Results report:                  {relative_path(RESULTS_REPORT_OUTPUT)}")
    print(f"Output manifest:                 {relative_path(OUTPUT_MANIFEST_OUTPUT)}")
    print("Prior analytical recalculation:  NONE")
    print("\nSCRIPT 07.1 COMPLETE — READY FOR SCRIPT 07.2 (POWER BI PREPARATION)")
    print(f"Script execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
