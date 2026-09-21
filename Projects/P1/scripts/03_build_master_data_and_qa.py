from __future__ import annotations

# =============================================================================
# SCRIPT 03 v3.0 — HARMONIZATION + FINAL QA
# P1 / Development Outcomes Analytics
#
# Official analytical window: 2015–2020
# Official scope: 28 countries | 513 regions | 3,078 region-year observations
#
# PURPOSE OF THIS VERSION
# -----------------------
# 1) Validate the final outputs produced by Script 02.
# 2) Build the canonical SPID region-year master key.
# 3) Inventory every retained source and persist the staging layer in DuckDB.
# 4) Build a conservative, auditable geographic crosswalk to the 513 SPID
#    target regions.
# 5) Use exact codes/names first, high-confidence fuzzy names only when unique,
#    spatial overlap for Niva GADM1 geometries and safe source-to-Niva bridges.
# 6) Diagnose geographic granularity instead of treating every finer/coarser
#    source unit as a matching failure.
# 7) Close temporal QA for the official 2015–2020 window.
# 8) Produce provisional source- and variable-level decisions:
#    KEEP / KEEP_WITH_FLAG / CONTEXT_ONLY / DROP.
# 9) Export a final QA package for Script 04 without forcing unsafe aggregation,
#    imputation or normalization.
#
# IMPORTANT
# ---------
# - This script DOES NOT impute missing years or values.
# - This script DOES NOT normalize analytical indicators.
# - This script DOES NOT aggregate variables whose aggregation rule depends on
#   the indicator semantics (rates, counts, monetary values, etc.).
# - Many-source-to-one-target relations are identified, not silently combined.
# - Script 04 remains responsible for analytical preparation after Script 03 is
#   fully validated.
# =============================================================================

from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
import re
import traceback
import unicodedata

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError as exc:
    raise ImportError(
        "DuckDB is required by Script 03. Install it inside the active venv with:\n"
        "    pip install duckdb\n"
    ) from exc

try:
    import geopandas as gpd
except Exception:
    gpd = None


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SCRIPT02_DIR = ROOT / "data" / "interim" / "script02"
SCRIPT03_DIR = ROOT / "data" / "interim" / "script03"
OUTPUT_DIR = ROOT / "outputs"

SCRIPT03_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUTPUT = SCRIPT03_DIR / "03_master_region_year_2015_2020.csv"
REGION_DIM_OUTPUT = SCRIPT03_DIR / "03_dim_region.csv"
SOURCE_INVENTORY_OUTPUT = SCRIPT03_DIR / "03_source_inventory.csv"
COLUMN_INVENTORY_OUTPUT = SCRIPT03_DIR / "03_column_inventory.csv"
YEAR_COVERAGE_OUTPUT = SCRIPT03_DIR / "03_year_coverage.csv"

GEO_CROSSWALK_OUTPUT = SCRIPT03_DIR / "03_geographic_crosswalk.csv"
GEO_QA_OUTPUT = SCRIPT03_DIR / "03_geographic_qa_summary.csv"
GEO_COUNTRY_QA_OUTPUT = SCRIPT03_DIR / "03_geographic_qa_by_country.csv"
GEO_REVIEW_OUTPUT = SCRIPT03_DIR / "03_geographic_review_queue.csv"
TEMPORAL_QA_OUTPUT = SCRIPT03_DIR / "03_temporal_qa_summary.csv"
SOURCE_DECISIONS_OUTPUT = SCRIPT03_DIR / "03_source_decisions.csv"
VARIABLE_QA_OUTPUT = SCRIPT03_DIR / "03_variable_qa.csv"

QA_WORKBOOK = OUTPUT_DIR / "03_harmonization_qa.xlsx"
DUCKDB_FILE = SCRIPT03_DIR / "03_harmonization.duckdb"

TARGET_GEOMETRIES = SCRIPT02_DIR / "02_target_regions.gpkg"


# =============================================================================
# OFFICIAL SCOPE
# =============================================================================

START_YEAR = 2015
END_YEAR = 2020
YEARS = list(range(START_YEAR, END_YEAR + 1))

EXPECTED_COUNTRIES = 28
EXPECTED_REGIONS = 513
EXPECTED_REGION_YEARS = EXPECTED_REGIONS * len(YEARS)

# Conservative automatic fuzzy-name rule.
FUZZY_AUTO_THRESHOLD = 0.970
FUZZY_MIN_MARGIN = 0.050

# Conservative spatial assignment rule.
SPATIAL_MIN_SOURCE_OVERLAP = 0.800
SPATIAL_MIN_MARGIN = 0.100
SPATIAL_EQUAL_AREA_CRS = "EPSG:6933"


# =============================================================================
# SCRIPT 02 SOURCES RETAINED FOR SCRIPT 03
# =============================================================================

SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "SPID": {
        "path": SCRIPT02_DIR / "02_spid_selected_panel_2015_2020.csv",
        "required": True,
        "role": "master_panel",
    },
    "TARGET_REGIONS": {
        "path": SCRIPT02_DIR / "02_target_regions.csv",
        "required": True,
        "role": "region_metadata",
    },
    "SPACE2STATS": {
        "path": SCRIPT02_DIR / "02_space2stats_candidates.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "WORLDPOP_AGESEX": {
        "path": SCRIPT02_DIR / "02_worldpop_agesex_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "SSGD": {
        "path": SCRIPT02_DIR / "02_ssgd_candidates_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "WDI": {
        "path": SCRIPT02_DIR / "02_wdi_candidates_2015_2020.csv",
        "required": False,
        "role": "national_context",
    },
    "KUMMU_GDP": {
        "path": SCRIPT02_DIR / "02_kummu_adm1_gdp_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "DOSE": {
        "path": SCRIPT02_DIR / "02_dose_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "NIVA_MIGRATION": {
        "path": SCRIPT02_DIR / "02_niva_adm1_2015_2019.gpkg",
        "required": False,
        "role": "candidate_indicators",
    },
    "OOKLA_WB": {
        "path": SCRIPT02_DIR / "02_ookla_adm1_annual_2019_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "GDL_SCD_BASELINE": {
        "path": SCRIPT02_DIR / "02_gdl_scd_baseline_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "GDL_SCD_COMPREHENSIVE": {
        "path": SCRIPT02_DIR / "02_gdl_scd_comprehensive_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "GDL_SHDI": {
        "path": SCRIPT02_DIR / "02_gdl_shdi_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
    "OECD_REGIONAL": {
        "path": SCRIPT02_DIR / "02_oecd_regional_unemployment_2015_2020.csv",
        "required": False,
        "role": "candidate_indicators",
    },
}


# =============================================================================
# SOURCE-SPECIFIC GEOGRAPHIC METADATA
# =============================================================================
#
# These overrides avoid false detection when a file contains both machine-code
# columns and human-readable labels with similar names. They are based on the
# schemas validated in Script 03 checkpoint 1.
# =============================================================================

SOURCE_COLUMN_OVERRIDES: dict[str, dict[str, str | None]] = {
    "SPID": {
        "country": "code",
        "region_code": "geo_code",
        "region_name": "sample",
        "year": "year",
    },
    "TARGET_REGIONS": {
        "country": "code",
        "region_code": "geo_code",
        "region_name": "geo_name",
        "year": None,
    },
    "SPACE2STATS": {
        "country": "code",
        "region_code": "geo_code",
        "region_name": "geo_name",
        "year": None,
    },
    "WORLDPOP_AGESEX": {
        "country": "code",
        "region_code": "geo_code",
        "region_name": "geo_name",
        "year": "year",
    },
    "SSGD": {
        "country": "countrycode",
        "region_code": "adm1_code",
        "region_name": "category",
        "year": "_year",
    },
    "WDI": {
        "country": "country_code",
        "region_code": None,
        "region_name": None,
        "year": "year",
    },
    "KUMMU_GDP": {
        "country": "iso3",
        "region_code": "GID_nmbr",
        "region_name": "Subnat",
        "year": None,
    },
    "DOSE": {
        "country": "GID_0",
        "region_code": "GID_1",
        "region_name": "region",
        "year": "year",
    },
    "NIVA_MIGRATION": {
        "country": "iso3",
        "region_code": "GID_1",
        "region_name": "NAME_1",
        "year": None,
    },
    "OOKLA_WB": {
        "country": "REF_AREA_0",
        "region_code": "REF_AREA_1",
        "region_name": "REF_AREA_1_NAME",
        "year": "TIME_PERIOD",
    },
    "GDL_SCD_BASELINE": {
        "country": "iso",
        "region_code": "GDLcode",
        "region_name": "region",
        "year": "year",
    },
    "GDL_SCD_COMPREHENSIVE": {
        "country": "iso",
        "region_code": "GDLcode",
        "region_name": "region",
        "year": "year",
    },
    "GDL_SHDI": {
        "country": "isocode3",
        "region_code": "gdlcode",
        "region_name": "region",
        "year": "year",
    },
    "OECD_REGIONAL": {
        "country": "COUNTRY",
        "region_code": "REF_AREA",
        "region_name": "Reference area",
        "year": "TIME_PERIOD",
    },
}

DIRECT_GEO_CODE_SOURCES = {
    "SPID",
    "TARGET_REGIONS",
    "SPACE2STATS",
    "WORLDPOP_AGESEX",
}

NATIONAL_CONTEXT_SOURCES = {
    "WDI",
}

# Sources whose raw subnational units are explicitly first-order administrative
# geographies. An exact normalized-name bridge through the already validated
# Niva GADM1 crosswalk is conservative for these sources and avoids unsafe
# fuzzy/manual matching.
SAFE_NIVA_NAME_BRIDGE_SOURCES = {
    "SSGD",
    "OOKLA_WB",
}

# Provisional engineering decisions for the handoff to Script 04. These are
# not substantive feature-selection verdicts: Script 04 still handles
# collinearity, transformations, missing-data treatment and final feature set.
SOURCE_DECISION_RULES: dict[str, dict[str, str]] = {
    "SPID": {
        "decision": "KEEP",
        "primary_use": "core_trajectory",
        "reason": "Master socioeconomic panel on the canonical 513-region geography.",
    },
    "TARGET_REGIONS": {
        "decision": "KEEP",
        "primary_use": "support",
        "reason": "Canonical region metadata and geographic identifiers.",
    },
    "SPACE2STATS": {
        "decision": "KEEP",
        "primary_use": "core_and_supporting_candidates",
        "reason": "Native 513-region coverage; dynamic and structural spatial indicators.",
    },
    "WORLDPOP_AGESEX": {
        "decision": "KEEP",
        "primary_use": "core_trajectory",
        "reason": "Complete 2015–2020 demographic trajectories on the canonical geography.",
    },
    "SSGD": {
        "decision": "CONTEXT_ONLY",
        "primary_use": "supplementary_context",
        "reason": "Only 8 project countries and sparse survey years; unsuitable as a global trajectory input.",
    },
    "WDI": {
        "decision": "CONTEXT_ONLY",
        "primary_use": "national_context",
        "reason": "National rather than subnational geography.",
    },
    "KUMMU_GDP": {
        "decision": "KEEP_WITH_FLAG",
        "primary_use": "candidate_or_sensitivity",
        "reason": "Full temporal window but heterogeneous subnational granularity requires controlled geographic use.",
    },
    "DOSE": {
        "decision": "KEEP_WITH_FLAG",
        "primary_use": "candidate_or_sensitivity",
        "reason": "High geographic coverage, but 2020 is absent and some targets require many-to-one aggregation.",
    },
    "NIVA_MIGRATION": {
        "decision": "KEEP_WITH_FLAG",
        "primary_use": "candidate_or_sensitivity",
        "reason": "High geographic coverage, but 2020 is absent and some boundary-vintage mismatches remain.",
    },
    "OOKLA_WB": {
        "decision": "CONTEXT_ONLY",
        "primary_use": "supplementary_context",
        "reason": "Only 2019–2020 are available, so it cannot represent the full 2015–2020 trajectory.",
    },
    "GDL_SCD_BASELINE": {
        "decision": "CONTEXT_ONLY",
        "primary_use": "supplementary_context",
        "reason": "Survey geography and coverage are too uneven for the common global trajectory matrix.",
    },
    "GDL_SCD_COMPREHENSIVE": {
        "decision": "CONTEXT_ONLY",
        "primary_use": "supplementary_context",
        "reason": "Complete calendar window but survey geography remains only partially compatible with the target panel.",
    },
    "GDL_SHDI": {
        "decision": "KEEP_WITH_FLAG",
        "primary_use": "candidate_or_sensitivity",
        "reason": "Complete 2015–2020 coverage, but subnational geography differs across countries and must be controlled.",
    },
    "OECD_REGIONAL": {
        "decision": "CONTEXT_ONLY",
        "primary_use": "supplementary_context",
        "reason": "Only 18 project countries and mixed TL2/TL3 levels; target-level unemployment aggregation is not automatically safe.",
    },
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def print_header(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_country(value: Any) -> str:
    text = normalize_text(value).upper()
    return re.sub(r"\s+", "", text)


def normalize_code(value: Any) -> str:
    text = normalize_text(value).upper()

    if not text:
        return ""

    # Pandas may represent integer identifiers as strings ending in .0.
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]

    return text.strip()


def normalize_region_name(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""

    text = text.replace("&", " and ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    # Keep the first physical column when normalized names collide. This is
    # important for OECD, where TIME_PERIOD and "Time period" coexist.
    lookup: dict[str, str] = {}
    for column in frame.columns:
        lookup.setdefault(normalize_column_name(column), column)

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in lookup:
            return lookup[key]

    return None


def resolve_source_column(
    source_id: str,
    frame: pd.DataFrame,
    kind: str,
) -> str | None:
    override = SOURCE_COLUMN_OVERRIDES.get(source_id, {}).get(kind)

    if override is not None:
        if override in frame.columns:
            return override

        # Case/spacing-insensitive fallback for the explicit override.
        found = find_column(frame, [override])
        if found is not None:
            return found

    if kind == "year":
        return find_column(
            frame,
            ["year", "time_period", "timeperiod", "period", "date", "time"],
        )

    if kind == "country":
        return find_column(
            frame,
            [
                "code",
                "countrycode",
                "country_code",
                "isocode3",
                "iso_code",
                "iso_code3",
                "iso3",
                "iso3c",
                "iso",
                "country_iso3",
                "countryiso3code",
                "gid_0",
                "adm0_a3",
                "wb_a3",
                "country",
            ],
        )

    if kind == "region_code":
        return find_column(
            frame,
            [
                "geo_code",
                "gdlcode",
                "region_code",
                "region_id",
                "geo_id",
                "gid_1",
                "adm1_code",
                "adm1_pcode",
                "ref_area_1",
                "ref_area",
                "gid_nmbr",
                "tl2",
                "tl3",
                "territorial_code",
            ],
        )

    if kind == "region_name":
        return find_column(
            frame,
            [
                "geo_name",
                "region",
                "region_name",
                "name",
                "adm1_name",
                "name_1",
                "category",
                "subnat",
                "ref_area_1_name",
                "reference area",
                "territorial_name",
            ],
        )

    raise ValueError(f"Unknown source-column kind: {kind}")


def safe_unique_count(series: pd.Series) -> int:
    try:
        return int(series.dropna().nunique())
    except Exception:
        return 0


def safe_sample(series: pd.Series, limit: int = 3) -> str:
    try:
        values = series.dropna().astype(str).str.strip()
        values = values[values.ne("")]
        return " | ".join(values.drop_duplicates().head(limit).tolist())
    except Exception:
        return ""


def parse_year_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    extracted = pd.to_numeric(
        series.astype(str).str.extract(
            r"(?<!\d)((?:19|20)\d{2})(?!\d)",
            expand=False,
        ),
        errors="coerce",
    )

    return numeric.fillna(extracted)


def detect_wide_year_columns(frame: pd.DataFrame) -> list[str]:
    result: list[str] = []

    for column in frame.columns:
        text = str(column)
        if any(
            re.search(rf"(?<!\d){year}(?!\d)", text)
            for year in YEARS
        ):
            result.append(column)

    return result


def sanitize_duckdb_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "table"


def load_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".gpkg":
        if gpd is None:
            raise ImportError(f"geopandas is required to read {path.name}.")

        gdf = gpd.read_file(path)
        frame = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        if "geometry" in gdf.columns:
            frame["geometry_wkt"] = gdf.geometry.astype(str)

        return frame

    raise ValueError(f"Unsupported source format: {path.suffix}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def first_existing_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return find_column(frame, names)


# =============================================================================
# MASTER SPID REGION-YEAR KEY
# =============================================================================

def build_master_panel(
    spid: pd.DataFrame,
    target_regions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    required_columns = ["code", "geo_code", "year"]
    missing_required = [
        column for column in required_columns if column not in spid.columns
    ]

    if missing_required:
        raise ValueError(
            "SPID master input is missing required columns: "
            f"{missing_required}"
        )

    master = spid.copy()
    master["code"] = master["code"].astype("string").str.strip().str.upper()
    master["geo_code"] = master["geo_code"].astype("string").str.strip()
    master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("Int64")
    master = master[master["year"].isin(YEARS)].copy()

    master["region_id"] = master["code"].astype(str) + "__" + master["geo_code"].astype(str)
    master["region_year_id"] = master["region_id"] + "__" + master["year"].astype(str)

    duplicated = int(
        master.duplicated(["code", "geo_code", "year"], keep=False).sum()
    )

    if duplicated:
        raise ValueError(
            f"SPID master has {duplicated} duplicated (code, geo_code, year) rows."
        )

    target = target_regions.copy()

    common_keys = [
        column
        for column in ["code", "geo_code"]
        if column in target.columns and column in master.columns
    ]

    if "geo_code" not in common_keys:
        raise ValueError("02_target_regions.csv must contain geo_code.")

    if "code" in target.columns:
        target["code"] = target["code"].astype("string").str.strip().str.upper()

    target["geo_code"] = target["geo_code"].astype("string").str.strip()
    target = target.drop_duplicates(common_keys).copy()

    metadata_columns = [column for column in target.columns if column not in common_keys]

    master = master.merge(
        target[common_keys + metadata_columns],
        how="left",
        on=common_keys,
        validate="many_to_one",
        suffixes=("", "_target"),
    )

    country_count = int(master["code"].nunique())
    region_count = int(master["region_id"].nunique())
    row_count = len(master)
    observed_years = sorted(master["year"].dropna().astype(int).unique().tolist())

    qa = pd.DataFrame(
        [
            {
                "check": "country_count",
                "observed": country_count,
                "expected": EXPECTED_COUNTRIES,
                "status": "PASS" if country_count == EXPECTED_COUNTRIES else "FAIL",
            },
            {
                "check": "region_count",
                "observed": region_count,
                "expected": EXPECTED_REGIONS,
                "status": "PASS" if region_count == EXPECTED_REGIONS else "FAIL",
            },
            {
                "check": "region_year_rows",
                "observed": row_count,
                "expected": EXPECTED_REGION_YEARS,
                "status": "PASS" if row_count == EXPECTED_REGION_YEARS else "FAIL",
            },
            {
                "check": "official_years",
                "observed": ",".join(map(str, observed_years)),
                "expected": ",".join(map(str, YEARS)),
                "status": "PASS" if observed_years == YEARS else "FAIL",
            },
            {
                "check": "duplicate_master_keys",
                "observed": duplicated,
                "expected": 0,
                "status": "PASS" if duplicated == 0 else "FAIL",
            },
        ]
    )

    if (qa["status"] == "FAIL").any():
        failed = qa.loc[
            qa["status"].eq("FAIL"),
            ["check", "observed", "expected"],
        ]
        raise ValueError(
            "Master-panel validation failed:\n" + failed.to_string(index=False)
        )

    region_dimension_columns = [
        column
        for column in [
            "region_id",
            "code",
            "geo_code",
            "geo_year",
            "geo_source",
            "geo_level",
            "geo_idvar",
            "geo_id",
            "geo_nvar",
            "geo_name",
        ]
        if column in master.columns
    ]

    dim_region = (
        master[region_dimension_columns]
        .drop_duplicates()
        .sort_values(["code", "geo_code"])
        .reset_index(drop=True)
    )

    return master, dim_region, qa


# =============================================================================
# SOURCE INVENTORY + INITIAL QA
# =============================================================================

def inventory_source(
    source_id: str,
    specification: dict[str, Any],
    frame: pd.DataFrame | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:

    path: Path = specification["path"]
    required = bool(specification["required"])
    role = str(specification["role"])

    if frame is None:
        return (
            {
                "source_id": source_id,
                "role": role,
                "required": required,
                "exists": False,
                "path": str(path),
                "file_type": path.suffix.lower(),
                "file_size_mb": np.nan,
                "rows": 0,
                "columns": 0,
                "country_column": "",
                "region_code_column": "",
                "region_name_column": "",
                "year_column": "",
                "wide_year_columns": "",
                "min_year": np.nan,
                "max_year": np.nan,
                "years_in_scope": "",
                "country_count": np.nan,
                "region_count": np.nan,
                "exact_duplicate_rows": np.nan,
                "status": "MISSING_REQUIRED" if required else "MISSING_OPTIONAL",
            },
            [],
            [],
        )

    country_col = resolve_source_column(source_id, frame, "country")
    region_code_col = resolve_source_column(source_id, frame, "region_code")
    region_name_col = resolve_source_column(source_id, frame, "region_name")
    year_col = resolve_source_column(source_id, frame, "year")
    wide_year_cols = detect_wide_year_columns(frame)

    exact_duplicates = int(frame.duplicated(keep=False).sum())

    country_count: int | float = np.nan
    region_count: int | float = np.nan

    if country_col is not None:
        country_count = safe_unique_count(frame[country_col])

    if region_code_col is not None:
        region_count = safe_unique_count(frame[region_code_col])
    elif region_name_col is not None:
        region_count = safe_unique_count(frame[region_name_col])

    years_in_scope: list[int] = []
    min_year: int | float = np.nan
    max_year: int | float = np.nan
    year_coverage_rows: list[dict[str, Any]] = []

    if year_col is not None:
        parsed_year = parse_year_series(frame[year_col])
        valid_years = parsed_year.dropna().astype(int)

        if not valid_years.empty:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())

        years_in_scope = sorted(
            valid_years[valid_years.isin(YEARS)].unique().tolist()
        )

        temp = frame.copy()
        temp["_qa_year"] = parsed_year

        for year in YEARS:
            subset = temp[temp["_qa_year"].eq(year)].copy()

            year_coverage_rows.append(
                {
                    "source_id": source_id,
                    "year": year,
                    "records": len(subset),
                    "countries": (
                        safe_unique_count(subset[country_col])
                        if country_col is not None and not subset.empty
                        else np.nan
                    ),
                    "regions": (
                        safe_unique_count(subset[region_code_col])
                        if region_code_col is not None and not subset.empty
                        else (
                            safe_unique_count(subset[region_name_col])
                            if region_name_col is not None and not subset.empty
                            else np.nan
                        )
                    ),
                }
            )

    elif wide_year_cols:
        detected_years: set[int] = set()

        for column in wide_year_cols:
            for year in YEARS:
                if re.search(rf"(?<!\d){year}(?!\d)", str(column)):
                    detected_years.add(year)

        years_in_scope = sorted(detected_years)

        if years_in_scope:
            min_year = min(years_in_scope)
            max_year = max(years_in_scope)

        for year in YEARS:
            columns_for_year = [
                column
                for column in wide_year_cols
                if re.search(rf"(?<!\d){year}(?!\d)", str(column))
            ]

            valid_rows = 0
            if columns_for_year:
                valid_rows = int(frame[columns_for_year].notna().any(axis=1).sum())

            year_coverage_rows.append(
                {
                    "source_id": source_id,
                    "year": year,
                    "records": valid_rows,
                    "countries": np.nan,
                    "regions": np.nan,
                }
            )

    column_rows: list[dict[str, Any]] = []

    for column in frame.columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        row_count = len(frame)

        column_rows.append(
            {
                "source_id": source_id,
                "column": str(column),
                "dtype": str(series.dtype),
                "rows": row_count,
                "null_count": null_count,
                "null_pct": null_count / row_count if row_count else np.nan,
                "unique_non_null": safe_unique_count(series),
                "sample": safe_sample(series),
            }
        )

    source_row = {
        "source_id": source_id,
        "role": role,
        "required": required,
        "exists": True,
        "path": str(path),
        "file_type": path.suffix.lower(),
        "file_size_mb": path.stat().st_size / (1024 ** 2) if path.exists() else np.nan,
        "rows": len(frame),
        "columns": len(frame.columns),
        "country_column": country_col or "",
        "region_code_column": region_code_col or "",
        "region_name_column": region_name_col or "",
        "year_column": year_col or "",
        "wide_year_columns": " | ".join(map(str, wide_year_cols)),
        "min_year": min_year,
        "max_year": max_year,
        "years_in_scope": ",".join(map(str, years_in_scope)),
        "country_count": country_count,
        "region_count": region_count,
        "exact_duplicate_rows": exact_duplicates,
        "status": "AVAILABLE",
    }

    return source_row, column_rows, year_coverage_rows


# =============================================================================
# GEOGRAPHIC CROSSWALK
# =============================================================================

def prepare_target_lookup(dim_region: pd.DataFrame) -> pd.DataFrame:
    target = dim_region.copy()

    required = ["region_id", "code", "geo_code"]
    missing = [column for column in required if column not in target.columns]
    if missing:
        raise ValueError(f"Region dimension is missing columns required for crosswalk: {missing}")

    target["target_country"] = target["code"].map(normalize_country)
    target["target_geo_code"] = target["geo_code"].map(normalize_code)

    if "geo_id" in target.columns:
        target["target_geo_id"] = target["geo_id"].map(normalize_code)
    else:
        target["target_geo_id"] = ""

    if "geo_name" in target.columns:
        target["target_geo_name"] = target["geo_name"].map(normalize_text)
    else:
        target["target_geo_name"] = ""

    target["target_name_norm"] = target["target_geo_name"].map(normalize_region_name)

    keep = [
        "region_id",
        "target_country",
        "target_geo_code",
        "target_geo_id",
        "target_geo_name",
        "target_name_norm",
    ]

    for optional in ["geo_source", "geo_level", "geo_idvar"]:
        if optional in target.columns:
            keep.append(optional)

    return target[keep].drop_duplicates("region_id").reset_index(drop=True)


def build_source_geographies(
    source_id: str,
    frame: pd.DataFrame,
    target_countries: set[str],
) -> pd.DataFrame:
    if source_id in NATIONAL_CONTEXT_SOURCES:
        return pd.DataFrame(
            columns=[
                "source_id",
                "source_geo_key",
                "source_country",
                "source_region_code",
                "source_region_name",
                "source_code_norm",
                "source_name_norm",
            ]
        )

    work = frame.copy()

    # GDL SHDI includes national observations. Only subnational rows belong in
    # the region crosswalk.
    if source_id == "GDL_SHDI" and "level" in work.columns:
        work = work[
            work["level"].astype(str).str.strip().str.casefold().eq("subnat")
        ].copy()

    # Geographic QA for the OECD source should reflect the retained
    # unemployment-rate measure, not the auxiliary sex-difference measure.
    if source_id == "OECD_REGIONAL":
        if "MEASURE" in work.columns:
            work = work[work["MEASURE"].astype(str).eq("UNE_RATE")].copy()
        if "UNIT_MEASURE" in work.columns:
            work = work[work["UNIT_MEASURE"].astype(str).eq("PT_LF_SUB")].copy()

    country_col = resolve_source_column(source_id, work, "country")
    code_col = resolve_source_column(source_id, work, "region_code")
    name_col = resolve_source_column(source_id, work, "region_name")

    if country_col is None:
        return pd.DataFrame(
            columns=[
                "source_id",
                "source_geo_key",
                "source_country",
                "source_region_code",
                "source_region_name",
                "source_code_norm",
                "source_name_norm",
            ]
        )

    geo = pd.DataFrame(index=work.index)
    geo["source_id"] = source_id
    geo["source_country"] = work[country_col].map(normalize_country)
    geo["source_region_code"] = (
        work[code_col].map(normalize_text) if code_col is not None else ""
    )
    geo["source_region_name"] = (
        work[name_col].map(normalize_text) if name_col is not None else ""
    )

    geo["source_code_norm"] = geo["source_region_code"].map(normalize_code)
    geo["source_name_norm"] = geo["source_region_name"].map(normalize_region_name)

    geo = geo[geo["source_country"].isin(target_countries)].copy()

    # Drop rows that contain no subnational identifier at all.
    geo = geo[
        geo["source_code_norm"].ne("") | geo["source_name_norm"].ne("")
    ].copy()

    geo["source_geo_key"] = np.where(
        geo["source_code_norm"].ne(""),
        geo["source_country"] + "__CODE__" + geo["source_code_norm"],
        geo["source_country"] + "__NAME__" + geo["source_name_norm"],
    )

    # One geographic record per source key. If labels vary over time, keep the
    # first non-empty representation while preserving the stable code key.
    geo = (
        geo.sort_values(
            ["source_country", "source_geo_key", "source_region_name"],
            na_position="last",
        )
        .drop_duplicates("source_geo_key", keep="first")
        .reset_index(drop=True)
    )

    return geo[
        [
            "source_id",
            "source_geo_key",
            "source_country",
            "source_region_code",
            "source_region_name",
            "source_code_norm",
            "source_name_norm",
        ]
    ]


def _candidate_row(target_row: pd.Series) -> dict[str, Any]:
    return {
        "target_region_id": target_row["region_id"],
        "target_geo_code": target_row["target_geo_code"],
        "target_geo_name": target_row["target_geo_name"],
    }


def match_source_geographies(
    source_id: str,
    source_geo: pd.DataFrame,
    target_lookup: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in source_geo.to_dict("records"):
        country = record["source_country"]
        code_norm = record["source_code_norm"]
        name_norm = record["source_name_norm"]

        candidates = target_lookup[target_lookup["target_country"].eq(country)].copy()

        output: dict[str, Any] = {
            **record,
            "target_region_id": "",
            "target_geo_code": "",
            "target_geo_name": "",
            "match_method": "",
            "match_score": np.nan,
            "match_status": "UNMATCHED",
            "candidate_target_region_id": "",
            "candidate_target_geo_name": "",
            "candidate_score": np.nan,
            "spatial_source_overlap": np.nan,
            "spatial_target_overlap": np.nan,
            "relation": "UNMATCHED",
            "notes": "",
        }

        if candidates.empty:
            output["match_status"] = "NO_TARGET_COUNTRY"
            output["notes"] = "No target region exists for source country."
            rows.append(output)
            continue

        # ------------------------------------------------------------------
        # 1. Exact canonical geo_code for sources already extracted on the
        #    project target geometry.
        # ------------------------------------------------------------------
        if source_id in DIRECT_GEO_CODE_SOURCES and code_norm:
            exact = candidates[candidates["target_geo_code"].eq(code_norm)]
            if len(exact) == 1:
                output.update(_candidate_row(exact.iloc[0]))
                output["match_method"] = "EXACT_GEO_CODE"
                output["match_score"] = 1.0
                output["match_status"] = "MATCHED"
                rows.append(output)
                continue

        # ------------------------------------------------------------------
        # 2. Exact code against canonical geo_code. This is safe when a source
        #    directly exposes the same project code even if it is not one of
        #    the native project sources.
        # ------------------------------------------------------------------
        if code_norm:
            exact = candidates[candidates["target_geo_code"].eq(code_norm)]
            if len(exact) == 1:
                output.update(_candidate_row(exact.iloc[0]))
                output["match_method"] = "EXACT_GEO_CODE"
                output["match_score"] = 1.0
                output["match_status"] = "MATCHED"
                rows.append(output)
                continue

        # ------------------------------------------------------------------
        # 3. Exact source region code against the target source-specific ID.
        #    This captures GAUL ADM1 codes, some NUTS/OECD codes, etc.
        # ------------------------------------------------------------------
        if code_norm:
            exact = candidates[candidates["target_geo_id"].eq(code_norm)]
            if len(exact) == 1:
                output.update(_candidate_row(exact.iloc[0]))
                output["match_method"] = "EXACT_TARGET_GEO_ID"
                output["match_score"] = 1.0
                output["match_status"] = "MATCHED"
                rows.append(output)
                continue

        # ------------------------------------------------------------------
        # 4. Unique normalized name inside the same country.
        # ------------------------------------------------------------------
        if name_norm:
            exact_name = candidates[candidates["target_name_norm"].eq(name_norm)]
            if len(exact_name) == 1:
                output.update(_candidate_row(exact_name.iloc[0]))
                output["match_method"] = "NORMALIZED_NAME"
                output["match_score"] = 1.0
                output["match_status"] = "MATCHED"
                rows.append(output)
                continue
            if len(exact_name) > 1:
                output["match_status"] = "AMBIGUOUS"
                output["notes"] = "Normalized source name matches multiple target regions."
                rows.append(output)
                continue

        # ------------------------------------------------------------------
        # 5. High-confidence fuzzy name. Automatic only when the best match is
        #    very strong and separated from the second-best candidate.
        # ------------------------------------------------------------------
        if name_norm:
            scored: list[tuple[float, int]] = []
            for idx, candidate in candidates.iterrows():
                target_name = candidate["target_name_norm"]
                if not target_name:
                    continue
                scored.append((similarity(name_norm, target_name), idx))

            scored.sort(reverse=True, key=lambda item: item[0])

            if scored:
                best_score, best_idx = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                best = candidates.loc[best_idx]

                output["candidate_target_region_id"] = best["region_id"]
                output["candidate_target_geo_name"] = best["target_geo_name"]
                output["candidate_score"] = best_score

                if (
                    best_score >= FUZZY_AUTO_THRESHOLD
                    and (best_score - second_score) >= FUZZY_MIN_MARGIN
                ):
                    output.update(_candidate_row(best))
                    output["match_method"] = "FUZZY_NAME_HIGH_CONF"
                    output["match_score"] = best_score
                    output["match_status"] = "MATCHED"
                    rows.append(output)
                    continue

                if best_score >= FUZZY_AUTO_THRESHOLD:
                    output["match_status"] = "AMBIGUOUS"
                    output["notes"] = (
                        "High fuzzy similarity but insufficient margin over second candidate."
                    )

        rows.append(output)

    return pd.DataFrame(rows)


def apply_niva_spatial_crosswalk(
    crosswalk: pd.DataFrame,
    target_lookup: pd.DataFrame,
) -> pd.DataFrame:
    if crosswalk.empty:
        return crosswalk

    mask_source = crosswalk["source_id"].eq("NIVA_MIGRATION")
    mask_unmatched = ~crosswalk["match_status"].eq("MATCHED")

    if not (mask_source & mask_unmatched).any():
        return crosswalk

    if gpd is None:
        crosswalk.loc[
            mask_source & mask_unmatched,
            "notes",
        ] = "geopandas unavailable; spatial crosswalk not attempted."
        return crosswalk

    if not TARGET_GEOMETRIES.exists():
        crosswalk.loc[
            mask_source & mask_unmatched,
            "notes",
        ] = f"Target geometry file not found: {TARGET_GEOMETRIES.name}"
        return crosswalk

    niva_path = SOURCE_REGISTRY["NIVA_MIGRATION"]["path"]
    if not niva_path.exists():
        return crosswalk

    try:
        source_gdf = gpd.read_file(niva_path)
        target_gdf = gpd.read_file(TARGET_GEOMETRIES)
    except Exception as exc:
        crosswalk.loc[
            mask_source & mask_unmatched,
            "notes",
        ] = f"Spatial files could not be read: {exc}"
        return crosswalk

    source_country_col = first_existing_column(source_gdf, ["iso3"])
    source_code_col = first_existing_column(source_gdf, ["GID_1"])
    target_country_col = first_existing_column(target_gdf, ["code"])
    target_geo_code_col = first_existing_column(target_gdf, ["geo_code"])

    if None in {
        source_country_col,
        source_code_col,
        target_country_col,
        target_geo_code_col,
    }:
        crosswalk.loc[
            mask_source & mask_unmatched,
            "notes",
        ] = "Spatial crosswalk skipped because required geometry identifiers are missing."
        return crosswalk

    if source_gdf.crs is None or target_gdf.crs is None:
        crosswalk.loc[
            mask_source & mask_unmatched,
            "notes",
        ] = "Spatial crosswalk skipped because source or target CRS is missing."
        return crosswalk

    source_gdf = source_gdf.copy()
    target_gdf = target_gdf.copy()

    source_gdf["_country"] = source_gdf[source_country_col].map(normalize_country)
    source_gdf["_code"] = source_gdf[source_code_col].map(normalize_code)
    target_gdf["_country"] = target_gdf[target_country_col].map(normalize_country)
    target_gdf["_geo_code"] = target_gdf[target_geo_code_col].map(normalize_code)

    countries = set(target_lookup["target_country"].unique())
    source_gdf = source_gdf[source_gdf["_country"].isin(countries)].copy()
    target_gdf = target_gdf[target_gdf["_country"].isin(countries)].copy()

    source_gdf = source_gdf[source_gdf.geometry.notna() & ~source_gdf.geometry.is_empty].copy()
    target_gdf = target_gdf[target_gdf.geometry.notna() & ~target_gdf.geometry.is_empty].copy()

    try:
        source_area = source_gdf.to_crs(SPATIAL_EQUAL_AREA_CRS)
        target_area = target_gdf.to_crs(SPATIAL_EQUAL_AREA_CRS)
    except Exception as exc:
        crosswalk.loc[
            mask_source & mask_unmatched,
            "notes",
        ] = f"Spatial reprojection failed: {exc}"
        return crosswalk

    target_map = target_lookup.set_index("target_geo_code", drop=False)

    source_geo_lookup = {
        (row["_country"], row["_code"]): idx
        for idx, row in source_area.iterrows()
        if row["_country"] and row["_code"]
    }

    for idx in crosswalk.index[mask_source & mask_unmatched]:
        country = crosswalk.at[idx, "source_country"]
        source_code = crosswalk.at[idx, "source_code_norm"]
        source_idx = source_geo_lookup.get((country, source_code))

        if source_idx is None:
            continue

        source_geometry = source_area.at[source_idx, "geometry"]
        source_polygon_area = float(source_geometry.area)

        if not np.isfinite(source_polygon_area) or source_polygon_area <= 0:
            continue

        candidate_targets = target_area[target_area["_country"].eq(country)].copy()
        if candidate_targets.empty:
            continue

        candidate_targets = candidate_targets[candidate_targets.geometry.intersects(source_geometry)]
        if candidate_targets.empty:
            continue

        scored: list[tuple[float, float, str]] = []

        for _, target_row in candidate_targets.iterrows():
            target_geometry = target_row.geometry
            target_polygon_area = float(target_geometry.area)

            try:
                intersection_area = float(source_geometry.intersection(target_geometry).area)
            except Exception:
                continue

            if intersection_area <= 0:
                continue

            source_share = intersection_area / source_polygon_area
            target_share = (
                intersection_area / target_polygon_area
                if target_polygon_area > 0
                else np.nan
            )

            scored.append(
                (
                    source_share,
                    target_share,
                    target_row["_geo_code"],
                )
            )

        scored.sort(reverse=True, key=lambda item: item[0])
        if not scored:
            continue

        best_source_share, best_target_share, best_geo_code = scored[0]
        second_source_share = scored[1][0] if len(scored) > 1 else 0.0

        if best_geo_code in target_map.index:
            best_target = target_map.loc[best_geo_code]
            if isinstance(best_target, pd.DataFrame):
                best_target = best_target.iloc[0]

            crosswalk.at[idx, "candidate_target_region_id"] = best_target["region_id"]
            crosswalk.at[idx, "candidate_target_geo_name"] = best_target["target_geo_name"]
            crosswalk.at[idx, "candidate_score"] = best_source_share
            crosswalk.at[idx, "spatial_source_overlap"] = best_source_share
            crosswalk.at[idx, "spatial_target_overlap"] = best_target_share

            if (
                best_source_share >= SPATIAL_MIN_SOURCE_OVERLAP
                and (best_source_share - second_source_share) >= SPATIAL_MIN_MARGIN
            ):
                crosswalk.at[idx, "target_region_id"] = best_target["region_id"]
                crosswalk.at[idx, "target_geo_code"] = best_target["target_geo_code"]
                crosswalk.at[idx, "target_geo_name"] = best_target["target_geo_name"]
                crosswalk.at[idx, "match_method"] = "SPATIAL_OVERLAP"
                crosswalk.at[idx, "match_score"] = best_source_share
                crosswalk.at[idx, "match_status"] = "MATCHED"
                crosswalk.at[idx, "notes"] = ""
            elif best_source_share > 0:
                crosswalk.at[idx, "match_status"] = "SPATIAL_SPLIT_SOURCE"
                crosswalk.at[idx, "notes"] = (
                    "Source geometry intersects multiple targets or overlap is below automatic threshold."
                )

    return crosswalk


def apply_dose_niva_bridge(crosswalk: pd.DataFrame) -> pd.DataFrame:
    if crosswalk.empty:
        return crosswalk

    niva = crosswalk[
        crosswalk["source_id"].eq("NIVA_MIGRATION")
        & crosswalk["match_status"].eq("MATCHED")
        & crosswalk["source_code_norm"].ne("")
    ].copy()

    if niva.empty:
        return crosswalk

    niva["_bridge_key"] = niva["source_country"] + "__" + niva["source_code_norm"]
    niva = niva.drop_duplicates("_bridge_key")

    bridge = niva.set_index("_bridge_key")

    dose_mask = (
        crosswalk["source_id"].eq("DOSE")
        & ~crosswalk["match_status"].eq("MATCHED")
        & crosswalk["source_code_norm"].ne("")
    )

    for idx in crosswalk.index[dose_mask]:
        key = (
            str(crosswalk.at[idx, "source_country"])
            + "__"
            + str(crosswalk.at[idx, "source_code_norm"])
        )

        if key not in bridge.index:
            continue

        matched = bridge.loc[key]
        if isinstance(matched, pd.DataFrame):
            matched = matched.iloc[0]

        crosswalk.at[idx, "target_region_id"] = matched["target_region_id"]
        crosswalk.at[idx, "target_geo_code"] = matched["target_geo_code"]
        crosswalk.at[idx, "target_geo_name"] = matched["target_geo_name"]
        crosswalk.at[idx, "match_method"] = "NIVA_GID1_BRIDGE"
        crosswalk.at[idx, "match_score"] = matched["match_score"]
        crosswalk.at[idx, "match_status"] = "MATCHED"
        crosswalk.at[idx, "spatial_source_overlap"] = matched["spatial_source_overlap"]
        crosswalk.at[idx, "spatial_target_overlap"] = matched["spatial_target_overlap"]
        crosswalk.at[idx, "notes"] = "DOSE GID_1 matched through Niva GID_1 crosswalk."

    return crosswalk


def annotate_crosswalk_relations(crosswalk: pd.DataFrame) -> pd.DataFrame:
    if crosswalk.empty:
        return crosswalk

    result = crosswalk.copy()
    matched = result[result["match_status"].eq("MATCHED")].copy()

    if matched.empty:
        return result

    counts = (
        matched.groupby(["source_id", "target_region_id"])["source_geo_key"]
        .nunique()
        .rename("source_units_per_target")
        .reset_index()
    )

    result = result.merge(
        counts,
        how="left",
        on=["source_id", "target_region_id"],
    )

    matched_mask = result["match_status"].eq("MATCHED")
    result.loc[
        matched_mask & result["source_units_per_target"].eq(1),
        "relation",
    ] = "ONE_SOURCE_TO_ONE_TARGET"
    result.loc[
        matched_mask & result["source_units_per_target"].gt(1),
        "relation",
    ] = "MANY_SOURCE_TO_ONE_TARGET"

    result["source_units_per_target"] = result["source_units_per_target"].astype("Int64")
    return result


def build_geographic_qa(
    crosswalk: pd.DataFrame,
    target_lookup: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []

    for source_id, specification in SOURCE_REGISTRY.items():
        role = specification["role"]

        if source_id in NATIONAL_CONTEXT_SOURCES:
            frame = frames.get(source_id)
            country_col = (
                resolve_source_column(source_id, frame, "country")
                if frame is not None
                else None
            )
            countries = (
                frame[country_col].map(normalize_country).nunique()
                if frame is not None and country_col is not None
                else 0
            )

            summary_rows.append(
                {
                    "source_id": source_id,
                    "role": role,
                    "source_geographies": 0,
                    "countries_in_scope": int(countries),
                    "matched_geographies": 0,
                    "unmatched_geographies": 0,
                    "ambiguous_geographies": 0,
                    "match_pct": np.nan,
                    "eligible_target_regions": 0,
                    "target_regions_covered": 0,
                    "target_coverage_pct": np.nan,
                    "many_source_to_one_rows": 0,
                    "exact_or_normalized_matches": 0,
                    "fuzzy_matches": 0,
                    "spatial_or_bridge_matches": 0,
                    "status": "NATIONAL_CONTEXT",
                }
            )
            continue

        subset = crosswalk[crosswalk["source_id"].eq(source_id)].copy()

        if subset.empty:
            status = "NO_GEOGRAPHY_ROWS" if source_id in frames else "SOURCE_NOT_AVAILABLE"
            summary_rows.append(
                {
                    "source_id": source_id,
                    "role": role,
                    "source_geographies": 0,
                    "countries_in_scope": 0,
                    "matched_geographies": 0,
                    "unmatched_geographies": 0,
                    "ambiguous_geographies": 0,
                    "match_pct": np.nan,
                    "eligible_target_regions": 0,
                    "target_regions_covered": 0,
                    "target_coverage_pct": np.nan,
                    "many_source_to_one_rows": 0,
                    "exact_or_normalized_matches": 0,
                    "fuzzy_matches": 0,
                    "spatial_or_bridge_matches": 0,
                    "status": status,
                }
            )
            continue

        countries = sorted(subset["source_country"].dropna().unique().tolist())
        eligible = target_lookup[target_lookup["target_country"].isin(countries)].copy()

        matched = subset[subset["match_status"].eq("MATCHED")].copy()
        ambiguous = subset[
            subset["match_status"].isin(["AMBIGUOUS", "SPATIAL_SPLIT_SOURCE"])
        ].copy()
        unmatched = subset[~subset["match_status"].eq("MATCHED")].copy()

        total = len(subset)
        match_pct = len(matched) / total if total else np.nan
        target_covered = matched["target_region_id"].replace("", np.nan).nunique()
        eligible_count = eligible["region_id"].nunique()

        methods = matched["match_method"].fillna("")

        if total == 0:
            status = "NO_GEOGRAPHY_ROWS"
        elif len(matched) == total:
            status = "COMPLETE_MATCH"
        elif match_pct >= 0.90:
            status = "HIGH_COVERAGE_REVIEW_REMAINDER"
        elif len(matched) > 0:
            status = "PARTIAL_MATCH_REVIEW_REQUIRED"
        else:
            status = "NO_MATCH_REVIEW_REQUIRED"

        summary_rows.append(
            {
                "source_id": source_id,
                "role": role,
                "source_geographies": total,
                "countries_in_scope": len(countries),
                "matched_geographies": len(matched),
                "unmatched_geographies": len(unmatched),
                "ambiguous_geographies": len(ambiguous),
                "match_pct": match_pct,
                "eligible_target_regions": eligible_count,
                "target_regions_covered": int(target_covered),
                "target_coverage_pct": (
                    target_covered / eligible_count if eligible_count else np.nan
                ),
                "many_source_to_one_rows": int(
                    matched["relation"].eq("MANY_SOURCE_TO_ONE_TARGET").sum()
                ),
                "exact_or_normalized_matches": int(
                    methods.isin(
                        ["EXACT_GEO_CODE", "EXACT_TARGET_GEO_ID", "NORMALIZED_NAME"]
                    ).sum()
                ),
                "fuzzy_matches": int(methods.eq("FUZZY_NAME_HIGH_CONF").sum()),
                "spatial_or_bridge_matches": int(
                    methods.isin(["SPATIAL_OVERLAP", "NIVA_GID1_BRIDGE", "NIVA_EXACT_NAME_BRIDGE"]).sum()
                ),
                "status": status,
            }
        )

        for country in countries:
            country_subset = subset[subset["source_country"].eq(country)].copy()
            country_matched = country_subset[country_subset["match_status"].eq("MATCHED")]
            country_targets = target_lookup[target_lookup["target_country"].eq(country)]
            eligible_country = country_targets["region_id"].nunique()
            covered_country = country_matched["target_region_id"].replace("", np.nan).nunique()

            country_rows.append(
                {
                    "source_id": source_id,
                    "country": country,
                    "source_geographies": len(country_subset),
                    "matched_geographies": len(country_matched),
                    "match_pct": (
                        len(country_matched) / len(country_subset)
                        if len(country_subset)
                        else np.nan
                    ),
                    "eligible_target_regions": eligible_country,
                    "target_regions_covered": int(covered_country),
                    "target_coverage_pct": (
                        covered_country / eligible_country
                        if eligible_country
                        else np.nan
                    ),
                    "many_source_to_one": int(
                        country_matched["relation"].eq("MANY_SOURCE_TO_ONE_TARGET").sum()
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    country_qa = pd.DataFrame(country_rows)

    review = crosswalk[~crosswalk["match_status"].eq("MATCHED")].copy()
    if not review.empty:
        review = review.sort_values(
            ["source_id", "source_country", "match_status", "source_region_name"]
        ).reset_index(drop=True)

    return summary, country_qa, review


def build_all_crosswalks(
    frames: dict[str, pd.DataFrame],
    dim_region: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_lookup = prepare_target_lookup(dim_region)
    target_countries = set(target_lookup["target_country"].unique())

    crosswalk_parts: list[pd.DataFrame] = []

    for source_id in SOURCE_REGISTRY:
        if source_id in NATIONAL_CONTEXT_SOURCES:
            continue

        frame = frames.get(source_id)
        if frame is None:
            continue

        source_geo = build_source_geographies(source_id, frame, target_countries)
        if source_geo.empty:
            continue

        matched = match_source_geographies(source_id, source_geo, target_lookup)
        crosswalk_parts.append(matched)

    if crosswalk_parts:
        crosswalk = pd.concat(crosswalk_parts, ignore_index=True)
    else:
        crosswalk = pd.DataFrame()

    if not crosswalk.empty:
        crosswalk = apply_niva_spatial_crosswalk(crosswalk, target_lookup)
        crosswalk = apply_safe_niva_name_bridges(crosswalk)
        crosswalk = apply_dose_niva_bridge(crosswalk)
        crosswalk = annotate_crosswalk_relations(crosswalk)

        sort_columns = [
            column
            for column in ["source_id", "source_country", "source_region_name", "source_region_code"]
            if column in crosswalk.columns
        ]
        crosswalk = crosswalk.sort_values(sort_columns).reset_index(drop=True)

    summary, country_qa, review = build_geographic_qa(
        crosswalk,
        target_lookup,
        frames,
    )
    summary, country_qa = add_granularity_diagnostics(summary, country_qa)
    review = classify_geographic_review(review, country_qa)

    return crosswalk, summary, country_qa, review



# =============================================================================
# SAFE CROSS-SOURCE BRIDGES + FINAL QA
# =============================================================================

def apply_safe_niva_name_bridges(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """
    Bridge exact ADM1 names from selected sources through the validated Niva
    source-to-target crosswalk. Only exact normalized names inside the same
    country are accepted, and only when that Niva name is unique.
    """
    if crosswalk.empty:
        return crosswalk

    result = crosswalk.copy()

    niva = result[
        result["source_id"].eq("NIVA_MIGRATION")
        & result["match_status"].eq("MATCHED")
        & result["source_name_norm"].ne("")
    ].copy()

    if niva.empty:
        return result

    key_counts = (
        niva.groupby(["source_country", "source_name_norm"])["source_geo_key"]
        .nunique()
        .rename("_name_count")
        .reset_index()
    )
    niva = niva.merge(
        key_counts,
        how="left",
        on=["source_country", "source_name_norm"],
    )
    niva = niva[niva["_name_count"].eq(1)].copy()
    niva["_bridge_key"] = niva["source_country"] + "__" + niva["source_name_norm"]
    niva = niva.drop_duplicates("_bridge_key").set_index("_bridge_key")

    mask = (
        result["source_id"].isin(SAFE_NIVA_NAME_BRIDGE_SOURCES)
        & ~result["match_status"].eq("MATCHED")
        & result["source_name_norm"].ne("")
    )

    for idx in result.index[mask]:
        key = (
            str(result.at[idx, "source_country"])
            + "__"
            + str(result.at[idx, "source_name_norm"])
        )

        if key not in niva.index:
            continue

        matched = niva.loc[key]
        if isinstance(matched, pd.DataFrame):
            matched = matched.iloc[0]

        result.at[idx, "target_region_id"] = matched["target_region_id"]
        result.at[idx, "target_geo_code"] = matched["target_geo_code"]
        result.at[idx, "target_geo_name"] = matched["target_geo_name"]
        result.at[idx, "match_method"] = "NIVA_EXACT_NAME_BRIDGE"
        result.at[idx, "match_score"] = 1.0
        result.at[idx, "match_status"] = "MATCHED"
        result.at[idx, "spatial_source_overlap"] = matched["spatial_source_overlap"]
        result.at[idx, "spatial_target_overlap"] = matched["spatial_target_overlap"]
        result.at[idx, "notes"] = (
            "Exact normalized ADM1 name bridged through the validated Niva crosswalk."
        )

    return result


def add_granularity_diagnostics(
    geo_summary: pd.DataFrame,
    geo_country_qa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def classify(source_n: Any, target_n: Any) -> str:
        if pd.isna(source_n) or pd.isna(target_n) or float(target_n) <= 0:
            return "NOT_APPLICABLE"

        ratio = float(source_n) / float(target_n)

        if ratio >= 1.50:
            return "SOURCE_FINER_OR_MULTI_LEVEL"
        if ratio <= 0.67:
            return "SOURCE_COARSER_OR_PARTIAL"
        return "COMPARABLE_SCALE"

    summary = geo_summary.copy()
    country = geo_country_qa.copy()

    if not summary.empty:
        summary["source_target_ratio"] = np.where(
            pd.to_numeric(summary["eligible_target_regions"], errors="coerce").gt(0),
            pd.to_numeric(summary["source_geographies"], errors="coerce")
            / pd.to_numeric(summary["eligible_target_regions"], errors="coerce"),
            np.nan,
        )
        summary["granularity_hint"] = [
            classify(source_n, target_n)
            for source_n, target_n in zip(
                summary["source_geographies"],
                summary["eligible_target_regions"],
            )
        ]

    if not country.empty:
        country["source_target_ratio"] = np.where(
            pd.to_numeric(country["eligible_target_regions"], errors="coerce").gt(0),
            pd.to_numeric(country["source_geographies"], errors="coerce")
            / pd.to_numeric(country["eligible_target_regions"], errors="coerce"),
            np.nan,
        )
        country["granularity_hint"] = [
            classify(source_n, target_n)
            for source_n, target_n in zip(
                country["source_geographies"],
                country["eligible_target_regions"],
            )
        ]

    return summary, country


def classify_geographic_review(
    review: pd.DataFrame,
    geo_country_qa: pd.DataFrame,
) -> pd.DataFrame:
    if review.empty:
        result = review.copy()
        result["review_reason"] = pd.Series(dtype="string")
        result["review_priority"] = pd.Series(dtype="string")
        return result

    result = review.copy()

    country_diag = geo_country_qa[
        [
            "source_id",
            "country",
            "source_target_ratio",
            "granularity_hint",
        ]
    ].copy()

    result = result.merge(
        country_diag,
        how="left",
        left_on=["source_id", "source_country"],
        right_on=["source_id", "country"],
    ).drop(columns=["country"], errors="ignore")

    def reason(row: pd.Series) -> tuple[str, str]:
        status = str(row.get("match_status", ""))
        granularity = str(row.get("granularity_hint", ""))
        candidate_score = pd.to_numeric(
            pd.Series([row.get("candidate_score")]),
            errors="coerce",
        ).iloc[0]

        if status == "SPATIAL_SPLIT_SOURCE":
            return (
                "BOUNDARY_VINTAGE_OR_TARGET_AGGREGATION",
                "MEDIUM",
            )

        if granularity == "SOURCE_FINER_OR_MULTI_LEVEL":
            return (
                "SOURCE_FINER_THAN_TARGET_OR_MULTI_LEVEL",
                "LOW",
            )

        if granularity == "SOURCE_COARSER_OR_PARTIAL":
            return (
                "SOURCE_COARSER_THAN_TARGET_OR_PARTIAL_COVERAGE",
                "MEDIUM",
            )

        if pd.notna(candidate_score) and float(candidate_score) >= 0.80:
            return (
                "LIKELY_NAME_OR_BOUNDARY_VARIANT",
                "MEDIUM",
            )

        return (
            "UNRESOLVED_GEOGRAPHIC_EQUIVALENCE",
            "HIGH",
        )

    classified = result.apply(reason, axis=1, result_type="expand")
    result["review_reason"] = classified[0]
    result["review_priority"] = classified[1]

    return result


def build_temporal_qa(
    source_inventory: pd.DataFrame,
    year_coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source_id, specification in SOURCE_REGISTRY.items():
        inv = source_inventory[source_inventory["source_id"].eq(source_id)]
        inv_row = inv.iloc[0] if not inv.empty else pd.Series(dtype=object)

        subset = year_coverage[year_coverage["source_id"].eq(source_id)].copy()
        subset["records"] = pd.to_numeric(subset.get("records"), errors="coerce").fillna(0)

        observed_years = sorted(
            subset.loc[subset["records"].gt(0), "year"]
            .dropna()
            .astype(int)
            .tolist()
        )
        missing_years = [year for year in YEARS if year not in observed_years]

        if source_id == "TARGET_REGIONS":
            status = "STATIC_METADATA"
            temporal_coverage_pct = np.nan
        elif source_id in NATIONAL_CONTEXT_SOURCES and len(observed_years) == len(YEARS):
            status = "COMPLETE_NATIONAL_CONTEXT"
            temporal_coverage_pct = 1.0
        else:
            temporal_coverage_pct = len(observed_years) / len(YEARS)
            if len(observed_years) == len(YEARS):
                status = "COMPLETE"
            elif len(observed_years) == len(YEARS) - 1:
                status = "NEAR_COMPLETE"
            elif len(observed_years) >= 3:
                status = "PARTIAL"
            elif len(observed_years) > 0:
                status = "SPARSE"
            else:
                status = "NO_TEMPORAL_DATA"

        records_by_year = "; ".join(
            f"{int(row['year'])}:{int(row['records'])}"
            for _, row in subset.sort_values("year").iterrows()
            if pd.notna(row["year"])
        )

        rows.append(
            {
                "source_id": source_id,
                "role": specification["role"],
                "min_year": inv_row.get("min_year", np.nan),
                "max_year": inv_row.get("max_year", np.nan),
                "years_observed": ",".join(map(str, observed_years)),
                "years_observed_count": len(observed_years),
                "missing_official_years": ",".join(map(str, missing_years)),
                "temporal_coverage_pct": temporal_coverage_pct,
                "records_by_year": records_by_year,
                "temporal_status": status,
            }
        )

    return pd.DataFrame(rows)


def build_source_decisions(
    geo_summary: pd.DataFrame,
    temporal_qa: pd.DataFrame,
) -> pd.DataFrame:
    base_rows: list[dict[str, Any]] = []

    for source_id, specification in SOURCE_REGISTRY.items():
        rule = SOURCE_DECISION_RULES[source_id]

        geo = geo_summary[geo_summary["source_id"].eq(source_id)]
        geo_row = geo.iloc[0] if not geo.empty else pd.Series(dtype=object)

        temporal = temporal_qa[temporal_qa["source_id"].eq(source_id)]
        temporal_row = temporal.iloc[0] if not temporal.empty else pd.Series(dtype=object)

        base_rows.append(
            {
                "source_id": source_id,
                "role": specification["role"],
                "geo_status": geo_row.get("status", ""),
                "target_coverage_pct": geo_row.get("target_coverage_pct", np.nan),
                "granularity_hint": geo_row.get("granularity_hint", ""),
                "temporal_status": temporal_row.get("temporal_status", ""),
                "temporal_coverage_pct": temporal_row.get("temporal_coverage_pct", np.nan),
                "provisional_decision": rule["decision"],
                "primary_use": rule["primary_use"],
                "decision_reason": rule["reason"],
            }
        )

    return pd.DataFrame(base_rows)


def _non_missing_stats_long(
    frame: pd.DataFrame,
    value_column: str,
    source_id: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    work = frame.copy()

    for column, expected in (filters or {}).items():
        if column not in work.columns:
            continue
        if isinstance(expected, (set, list, tuple)):
            work = work[work[column].isin(expected)].copy()
        else:
            work = work[work[column].eq(expected)].copy()

    if value_column not in work.columns:
        return {
            "observations": 0,
            "non_missing": 0,
            "missing_pct": np.nan,
            "countries_observed": 0,
            "years_observed": "",
            "years_observed_count": 0,
        }

    value = pd.to_numeric(work[value_column], errors="coerce")
    valid = value.notna()

    country_col = resolve_source_column(source_id, work, "country")
    year_col = resolve_source_column(source_id, work, "year")

    countries_observed = 0
    if country_col is not None:
        countries_observed = (
            work.loc[valid, country_col]
            .map(normalize_country)
            .replace("", np.nan)
            .dropna()
            .nunique()
        )

    years_observed: list[int] = []
    if year_col is not None:
        years = parse_year_series(work.loc[valid, year_col]).dropna().astype(int)
        years_observed = sorted(years[years.isin(YEARS)].unique().tolist())

    total = len(work)

    return {
        "observations": total,
        "non_missing": int(valid.sum()),
        "missing_pct": (1 - valid.mean()) if total else np.nan,
        "countries_observed": int(countries_observed),
        "years_observed": ",".join(map(str, years_observed)),
        "years_observed_count": len(years_observed),
    }


def _non_missing_stats_wide(
    frame: pd.DataFrame,
    columns: list[str],
    source_id: str,
) -> dict[str, Any]:
    available = [column for column in columns if column in frame.columns]

    if not available:
        return {
            "observations": 0,
            "non_missing": 0,
            "missing_pct": np.nan,
            "countries_observed": 0,
            "years_observed": "",
            "years_observed_count": 0,
        }

    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    valid_rows = numeric.notna().any(axis=1)

    country_col = resolve_source_column(source_id, frame, "country")
    countries_observed = 0
    if country_col is not None:
        countries_observed = (
            frame.loc[valid_rows, country_col]
            .map(normalize_country)
            .replace("", np.nan)
            .dropna()
            .nunique()
        )

    years_observed = []
    for column in available:
        match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(column))
        if match and int(match.group(1)) in YEARS:
            years_observed.append(int(match.group(1)))

    years_observed = sorted(set(years_observed))
    total_cells = int(numeric.size)
    non_missing = int(numeric.notna().sum().sum())

    return {
        "observations": total_cells,
        "non_missing": non_missing,
        "missing_pct": (
            1 - (non_missing / total_cells)
            if total_cells
            else np.nan
        ),
        "countries_observed": int(countries_observed),
        "years_observed": ",".join(map(str, years_observed)),
        "years_observed_count": len(years_observed),
    }


def build_variable_qa(
    frames: dict[str, pd.DataFrame],
    source_decisions: pd.DataFrame,
    geo_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    decision_lookup = (
        source_decisions
        .set_index("source_id")
        .to_dict("index")
        if not source_decisions.empty
        else {}
    )
    geo_lookup = (
        geo_summary
        .set_index("source_id")
        .to_dict("index")
        if not geo_summary.empty
        else {}
    )

    def append_long(
        source_id: str,
        variable_id: str,
        value_column: str,
        decision: str | None = None,
        filters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        frame = frames.get(source_id)
        if frame is None:
            return

        stats = _non_missing_stats_long(
            frame,
            value_column,
            source_id,
            filters=filters,
        )
        source_rule = decision_lookup.get(source_id, {})
        source_decision = source_rule.get("provisional_decision", "KEEP_WITH_FLAG")
        final_decision = decision or source_decision
        geo = geo_lookup.get(source_id, {})

        rows.append(
            {
                "source_id": source_id,
                "variable_id": variable_id,
                "representation": "long",
                **stats,
                "temporal_coverage_pct": (
                    stats["years_observed_count"] / len(YEARS)
                    if stats["years_observed_count"] > 0
                    else np.nan
                ),
                "target_coverage_pct": geo.get("target_coverage_pct", np.nan),
                "provisional_decision": final_decision,
                "notes": notes,
            }
        )

    def append_wide(
        source_id: str,
        variable_id: str,
        columns: list[str],
        decision: str | None = None,
        notes: str = "",
    ) -> None:
        frame = frames.get(source_id)
        if frame is None:
            return

        stats = _non_missing_stats_wide(frame, columns, source_id)
        source_rule = decision_lookup.get(source_id, {})
        source_decision = source_rule.get("provisional_decision", "KEEP_WITH_FLAG")
        final_decision = decision or source_decision
        geo = geo_lookup.get(source_id, {})

        rows.append(
            {
                "source_id": source_id,
                "variable_id": variable_id,
                "representation": "wide",
                **stats,
                "temporal_coverage_pct": (
                    stats["years_observed_count"] / len(YEARS)
                    if stats["years_observed_count"] > 0
                    else np.nan
                ),
                "target_coverage_pct": geo.get("target_coverage_pct", np.nan),
                "provisional_decision": final_decision,
                "notes": notes,
            }
        )

    # SPID socioeconomic trajectories.
    for variable in ["poor300", "poor420", "poor830", "prosgap2021", "gini", "theil"]:
        append_long("SPID", variable, variable, "KEEP")

    # Space2Stats dynamic families and structural/static candidates.
    append_wide(
        "SPACE2STATS",
        "viirs_ntl_sum",
        [f"sum_viirs_ntl_{year}" for year in YEARS],
        "KEEP",
        "Dynamic 2015–2020 nighttime-lights trajectory.",
    )
    append_wide(
        "SPACE2STATS",
        "population_space2stats",
        [f"sum_pop_{year}" for year in YEARS],
        "CONTEXT_ONLY",
        "Retained primarily as a cross-check because WorldPop is the demographic source.",
    )
    append_wide(
        "SPACE2STATS",
        "built_area",
        ["sum_built_area_m_2015", "sum_built_area_m_2020"],
        "KEEP_WITH_FLAG",
        "Only the 2015 and 2020 endpoints are available.",
    )

    space2stats = frames.get("SPACE2STATS")
    if space2stats is not None:
        static_columns = [
            column
            for column in space2stats.columns
            if (
                column.startswith("ghs_")
                or column in {
                    "drought_spei_1_5_rp100_mean",
                    "fires_density_mean",
                    "landslide_susceptibility_mean_2023",
                    "pop_flood",
                    "pop_flood_pct",
                }
            )
        ]
        for column in static_columns:
            append_long(
                "SPACE2STATS",
                column,
                column,
                "KEEP_WITH_FLAG",
                notes="Static/structural candidate; Script 04 decides whether it belongs in the trajectory representation.",
            )

    # WorldPop demographic trajectories.
    for variable, decision in [
        ("wp_age_population", "KEEP"),
        ("age_dependency_ratio", "KEEP"),
        ("share_0_14", "KEEP"),
        ("share_65_plus", "KEEP"),
        ("wp_age_0_14", "CONTEXT_ONLY"),
        ("wp_age_15_64", "CONTEXT_ONLY"),
        ("wp_age_65_plus", "CONTEXT_ONLY"),
    ]:
        append_long("WORLDPOP_AGESEX", variable, variable, decision)

    # SSGD survey indicators.
    ssgd = frames.get("SSGD")
    if ssgd is not None and "short" in ssgd.columns:
        for short in sorted(ssgd["short"].dropna().astype(str).unique().tolist()):
            append_long(
                "SSGD",
                short,
                "value",
                "CONTEXT_ONLY",
                filters={"short": short},
                notes="Survey indicator with incomplete country/year coverage.",
            )

    # WDI national context indicators.
    wdi = frames.get("WDI")
    if wdi is not None and "variable_id" in wdi.columns:
        for variable_id in sorted(
            wdi["variable_id"].dropna().astype(str).unique().tolist()
        ):
            append_long(
                "WDI",
                variable_id,
                "value",
                "CONTEXT_ONLY",
                filters={"variable_id": variable_id},
                notes="National context only.",
            )

    # Kummu regional GDP per capita.
    append_wide(
        "KUMMU_GDP",
        "regional_gdp_per_capita_ppp",
        [str(year) for year in YEARS],
        "KEEP_WITH_FLAG",
        "Full temporal window; geographic compatibility varies strongly by country.",
    )

    # DOSE constant-2015-USD regional production candidates.
    for variable in [
        "grp_pc_usd_2015",
        "ag_grp_pc_usd_2015",
        "man_grp_pc_usd_2015",
        "serv_grp_pc_usd_2015",
    ]:
        append_long(
            "DOSE",
            variable,
            variable,
            "KEEP_WITH_FLAG",
            notes="2015–2019 only; 2020 treatment must be explicit in Script 04.",
        )

    # Niva migration trajectory.
    append_wide(
        "NIVA_MIGRATION",
        "net_migration",
        [f"netMgr_{year}" for year in range(2015, 2020)],
        "KEEP_WITH_FLAG",
        "2015–2019 only; 2020 is absent.",
    )

    # Ookla context by service indicator.
    ookla = frames.get("OOKLA_WB")
    if ookla is not None and "INDICATOR" in ookla.columns:
        for indicator in sorted(
            ookla["INDICATOR"].dropna().astype(str).unique().tolist()
        ):
            append_long(
                "OOKLA_WB",
                f"{indicator}_mean",
                "MEAN_VALUE",
                "CONTEXT_ONLY",
                filters={"INDICATOR": indicator},
                notes="Only 2019–2020 are available.",
            )

    # GDL corruption indicators.
    append_long(
        "GDL_SCD_BASELINE",
        "SCI2",
        "SCI2",
        "CONTEXT_ONLY",
        notes="Baseline survey series retained for context/robustness only.",
    )
    append_long(
        "GDL_SCD_COMPREHENSIVE",
        "SCI",
        "SCI",
        "CONTEXT_ONLY",
        notes="Survey geography is not sufficiently uniform for the core global trajectory matrix.",
    )

    # GDL SHDI and components.
    for variable in [
        "shdi",
        "healthindex",
        "edindex",
        "incindex",
        "lifexp",
        "esch",
        "msch",
        "lgnic",
    ]:
        append_long(
            "GDL_SHDI",
            variable,
            variable,
            "KEEP_WITH_FLAG",
            filters={"level": "Subnat"},
            notes="Full temporal window; geographic compatibility must be respected.",
        )

    # OECD regional unemployment rate.
    append_long(
        "OECD_REGIONAL",
        "unemployment_rate_15_64",
        "OBS_VALUE",
        "CONTEXT_ONLY",
        filters={
            "MEASURE": "UNE_RATE",
            "UNIT_MEASURE": "PT_LF_SUB",
            "AGE": "Y15T64",
            "SEX": "_T",
        },
        notes="Mixed TL2/TL3 source and only 18 project countries; no unsafe rate aggregation.",
    )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["provisional_decision", "source_id", "variable_id"]
        ).reset_index(drop=True)

    return result

# =============================================================================
# DUCKDB STAGING
# =============================================================================

def persist_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    safe_name = sanitize_duckdb_name(table_name)
    temp_name = f"_tmp_{safe_name}"

    connection.register(temp_name, frame)
    connection.execute(f'DROP TABLE IF EXISTS "{safe_name}"')
    connection.execute(
        f'CREATE TABLE "{safe_name}" AS SELECT * FROM "{temp_name}"'
    )
    connection.unregister(temp_name)


# =============================================================================
# EXCEL QA OUTPUT
# =============================================================================

def write_qa_workbook(
    master_qa: pd.DataFrame,
    source_inventory: pd.DataFrame,
    column_inventory: pd.DataFrame,
    year_coverage: pd.DataFrame,
    dim_region: pd.DataFrame,
    geo_summary: pd.DataFrame,
    geo_country_qa: pd.DataFrame,
    geo_crosswalk: pd.DataFrame,
    geo_review: pd.DataFrame,
    temporal_qa: pd.DataFrame,
    source_decisions: pd.DataFrame,
    variable_qa: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(QA_WORKBOOK, engine="openpyxl") as writer:
        master_qa.to_excel(writer, sheet_name="master_key_qa", index=False)
        source_inventory.to_excel(writer, sheet_name="source_inventory", index=False)
        column_inventory.to_excel(writer, sheet_name="column_inventory", index=False)
        year_coverage.to_excel(writer, sheet_name="year_coverage", index=False)
        dim_region.to_excel(writer, sheet_name="region_dimension", index=False)

        geo_summary.to_excel(writer, sheet_name="geo_qa_summary", index=False)
        geo_country_qa.to_excel(writer, sheet_name="geo_qa_country", index=False)
        geo_crosswalk.to_excel(writer, sheet_name="geo_crosswalk", index=False)
        geo_review.to_excel(writer, sheet_name="geo_review_queue", index=False)
        temporal_qa.to_excel(writer, sheet_name="temporal_qa", index=False)
        source_decisions.to_excel(writer, sheet_name="source_decisions", index=False)
        variable_qa.to_excel(writer, sheet_name="variable_qa", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = 0

                for cell in column_cells[:200]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_length + 2, 10),
                    45,
                )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_header(
        "SCRIPT 03 v3.0 — HARMONIZATION + FINAL QA"
    )

    print(f"Project root: {ROOT}")
    print(f"Official period: {START_YEAR}–{END_YEAR}")
    print(
        f"Official scope: {EXPECTED_COUNTRIES} countries | "
        f"{EXPECTED_REGIONS} regions | "
        f"{EXPECTED_REGION_YEARS:,} region-year observations"
    )

    # -------------------------------------------------------------------------
    # 1. Validate Script 02 outputs
    # -------------------------------------------------------------------------
    print_header("1/6 — VALIDATE SCRIPT 02 OUTPUTS")

    missing_required: list[str] = []

    for source_id, specification in SOURCE_REGISTRY.items():
        path: Path = specification["path"]

        if path.exists():
            print(f"[FOUND]   {source_id:<24} {path.name}")
        else:
            print(f"[MISSING] {source_id:<24} {path.name}")
            if specification["required"]:
                missing_required.append(f"{source_id}: {path}")

    if TARGET_GEOMETRIES.exists():
        print(f"[FOUND]   {'TARGET_GEOMETRIES':<24} {TARGET_GEOMETRIES.name}")
    else:
        print(
            f"[OPTIONAL] {'TARGET_GEOMETRIES':<24} {TARGET_GEOMETRIES.name} "
            "not found — Niva spatial matching will be skipped"
        )

    if missing_required:
        raise FileNotFoundError(
            "Required Script 02 outputs are missing:\n" + "\n".join(missing_required)
        )

    # -------------------------------------------------------------------------
    # 2. Build strict SPID master region-year key
    # -------------------------------------------------------------------------
    print_header("2/6 — BUILD MASTER REGION-YEAR KEY")

    spid = load_source(SOURCE_REGISTRY["SPID"]["path"])
    target_regions = load_source(SOURCE_REGISTRY["TARGET_REGIONS"]["path"])

    master, dim_region, master_qa = build_master_panel(spid, target_regions)

    write_csv(master, MASTER_OUTPUT)
    write_csv(dim_region, REGION_DIM_OUTPUT)

    print(
        f"Master panel: {len(master):,} rows | "
        f"{master['code'].nunique()} countries | "
        f"{master['region_id'].nunique()} regions | "
        f"{master['year'].nunique()} years"
    )
    print(
        "Master key status: "
        f"{'PASS' if master_qa['status'].eq('PASS').all() else 'FAIL'}"
    )

    # -------------------------------------------------------------------------
    # 3. Inventory every retained source + DuckDB staging
    # -------------------------------------------------------------------------
    print_header("3/6 — SOURCE INVENTORY + DUCKDB STAGING")

    source_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}

    connection = duckdb.connect(str(DUCKDB_FILE))

    try:
        persist_duckdb_table(connection, "master_region_year", master)
        persist_duckdb_table(connection, "dim_region", dim_region)

        for source_id, specification in SOURCE_REGISTRY.items():
            path: Path = specification["path"]

            if not path.exists():
                source_row, columns, years = inventory_source(
                    source_id,
                    specification,
                    None,
                )
                source_rows.append(source_row)
                column_rows.extend(columns)
                year_rows.extend(years)
                print(f"[SKIP] {source_id:<24} file not found")
                continue

            try:
                frame = load_source(path)
                frames[source_id] = frame

                source_row, columns, years = inventory_source(
                    source_id,
                    specification,
                    frame,
                )

                source_rows.append(source_row)
                column_rows.extend(columns)
                year_rows.extend(years)

                persist_duckdb_table(connection, f"raw_{source_id}", frame)

                print(
                    f"[OK]   {source_id:<24} "
                    f"{len(frame):>8,} rows | "
                    f"{len(frame.columns):>3} columns"
                )

            except Exception as exc:
                source_rows.append(
                    {
                        "source_id": source_id,
                        "role": specification["role"],
                        "required": specification["required"],
                        "exists": True,
                        "path": str(path),
                        "file_type": path.suffix.lower(),
                        "file_size_mb": (
                            path.stat().st_size / (1024 ** 2)
                            if path.exists()
                            else np.nan
                        ),
                        "rows": np.nan,
                        "columns": np.nan,
                        "country_column": "",
                        "region_code_column": "",
                        "region_name_column": "",
                        "year_column": "",
                        "wide_year_columns": "",
                        "min_year": np.nan,
                        "max_year": np.nan,
                        "years_in_scope": "",
                        "country_count": np.nan,
                        "region_count": np.nan,
                        "exact_duplicate_rows": np.nan,
                        "status": f"READ_ERROR: {exc}",
                    }
                )

                print(f"[ERROR] {source_id}: {exc}")
                traceback.print_exc()

        source_inventory = pd.DataFrame(source_rows)
        column_inventory = pd.DataFrame(column_rows)
        year_coverage = pd.DataFrame(year_rows)

        persist_duckdb_table(connection, "qa_master_key", master_qa)
        persist_duckdb_table(connection, "qa_source_inventory", source_inventory)
        persist_duckdb_table(connection, "qa_column_inventory", column_inventory)
        persist_duckdb_table(connection, "qa_year_coverage", year_coverage)

        # ---------------------------------------------------------------------
        # 4. Geographic crosswalk + geographic QA
        # ---------------------------------------------------------------------
        print_header("4/6 — GEOGRAPHIC CROSSWALK + QA")

        geo_crosswalk, geo_summary, geo_country_qa, geo_review = build_all_crosswalks(
            frames,
            dim_region,
        )

        if not geo_crosswalk.empty:
            persist_duckdb_table(connection, "geo_crosswalk", geo_crosswalk)
        else:
            persist_duckdb_table(connection, "geo_crosswalk", pd.DataFrame({"empty": []}))

        persist_duckdb_table(connection, "qa_geographic_summary", geo_summary)
        persist_duckdb_table(connection, "qa_geographic_country", geo_country_qa)

        if not geo_review.empty:
            persist_duckdb_table(connection, "qa_geographic_review", geo_review)
        else:
            persist_duckdb_table(connection, "qa_geographic_review", pd.DataFrame({"empty": []}))

        # ---------------------------------------------------------------------
        # 5. Temporal QA + provisional source/variable decisions
        # ---------------------------------------------------------------------
        print_header("5/6 — TEMPORAL QA + SOURCE / VARIABLE DECISIONS")

        temporal_qa = build_temporal_qa(source_inventory, year_coverage)
        source_decisions = build_source_decisions(geo_summary, temporal_qa)
        variable_qa = build_variable_qa(frames, source_decisions, geo_summary)

        persist_duckdb_table(connection, "qa_temporal_summary", temporal_qa)
        persist_duckdb_table(connection, "qa_source_decisions", source_decisions)
        persist_duckdb_table(connection, "qa_variable", variable_qa)

    finally:
        connection.close()

    write_csv(source_inventory, SOURCE_INVENTORY_OUTPUT)
    write_csv(column_inventory, COLUMN_INVENTORY_OUTPUT)
    write_csv(year_coverage, YEAR_COVERAGE_OUTPUT)
    write_csv(geo_crosswalk, GEO_CROSSWALK_OUTPUT)
    write_csv(geo_summary, GEO_QA_OUTPUT)
    write_csv(geo_country_qa, GEO_COUNTRY_QA_OUTPUT)
    write_csv(geo_review, GEO_REVIEW_OUTPUT)
    write_csv(temporal_qa, TEMPORAL_QA_OUTPUT)
    write_csv(source_decisions, SOURCE_DECISIONS_OUTPUT)
    write_csv(variable_qa, VARIABLE_QA_OUTPUT)

    if not geo_summary.empty:
        display_columns = [
            "source_id",
            "source_geographies",
            "matched_geographies",
            "match_pct",
            "target_regions_covered",
            "status",
        ]
        print()
        print(
            geo_summary[display_columns]
            .assign(match_pct=lambda x: x["match_pct"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "—"
            ))
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # 5. Consolidated QA workbook
    # -------------------------------------------------------------------------
    print_header("6/6 — WRITE CONSOLIDATED QA")

    write_qa_workbook(
        master_qa,
        source_inventory,
        column_inventory,
        year_coverage,
        dim_region,
        geo_summary,
        geo_country_qa,
        geo_crosswalk,
        geo_review,
        temporal_qa,
        source_decisions,
        variable_qa,
    )

    available = int(source_inventory["status"].eq("AVAILABLE").sum())
    missing = int(
        source_inventory["status"].str.startswith("MISSING", na=False).sum()
    )
    read_errors = int(
        source_inventory["status"].str.startswith("READ_ERROR", na=False).sum()
    )

    matched_geo = int(
        geo_crosswalk["match_status"].eq("MATCHED").sum()
        if not geo_crosswalk.empty
        else 0
    )
    total_geo = len(geo_crosswalk)
    review_geo = len(geo_review)

    print()
    print("CHECKPOINT SUMMARY")
    print("-" * 88)
    print(
        f"Master panel:      {len(master):,} rows | "
        f"{master['region_id'].nunique()} regions"
    )
    print(
        f"Sources:           {available} available | "
        f"{missing} missing | "
        f"{read_errors} read errors"
    )
    print(
        f"Geographic units:  {matched_geo:,}/{total_geo:,} automatically matched | "
        f"{review_geo:,} in review queue"
    )
    print(f"DuckDB:            {DUCKDB_FILE}")
    print(f"QA workbook:       {QA_WORKBOOK}")
    print(f"Master CSV:        {MASTER_OUTPUT}")
    print(f"Crosswalk CSV:     {GEO_CROSSWALK_OUTPUT}")
    print(f"Geo review CSV:    {GEO_REVIEW_OUTPUT}")

    print(f"Temporal QA CSV:    {TEMPORAL_QA_OUTPUT}")
    print(f"Source decisions:   {SOURCE_DECISIONS_OUTPUT}")
    print(f"Variable QA CSV:    {VARIABLE_QA_OUTPUT}")

    print()
    print("SOURCE DECISIONS")
    print("-" * 88)
    print(
        source_decisions[
            [
                "source_id",
                "provisional_decision",
                "geo_status",
                "temporal_status",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "SCRIPT 03 COMPLETE — GEOGRAPHIC + TEMPORAL HARMONIZATION QA CLOSED"
    )
    print(
        "Handoff to Script 04: analytical preparation, controlled aggregation, "
        "missing-data treatment, transformations and trajectory matrix."
    )


if __name__ == "__main__":
    main()
