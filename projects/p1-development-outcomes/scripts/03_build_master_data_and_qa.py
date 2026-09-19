from __future__ import annotations

# =============================================================================
# SCRIPT 03 v1.0 — MASTER PANEL + SOURCE INVENTORY + INITIAL QA
# P1 / Development Outcomes Analytics
#
# Official analytical window: 2015–2020
# Official scope: 28 countries | 513 regions | 3,078 region-year observations
#
# PURPOSE OF THIS CHECKPOINT
# --------------------------
# 1) Validate the final outputs produced by Script 02.
# 2) Build the canonical SPID region-year master key.
# 3) Inventory schema, years, countries, regions, missingness and exact duplicates
#    for every retained source.
# 4) Persist the staging layer in DuckDB.
# 5) Produce a single QA workbook that will guide the geographic/temporal
#    harmonization in the next Script 03 checkpoint.
#
# IMPORTANT
# ---------
# - This version DOES NOT force geographic matches between heterogeneous sources.
# - No fuzzy name match, spatial match, aggregation or interpolation is performed.
# - No imputation, normalization or clustering preparation is performed.
# - Script 04 remains responsible for analytical preparation.
# - The same Script 03 file will be updated incrementally after this checkpoint
#   is validated.
# =============================================================================

from pathlib import Path
from typing import Any
import re
import time
import traceback

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
QA_WORKBOOK = OUTPUT_DIR / "03_harmonization_qa.xlsx"
DUCKDB_FILE = SCRIPT03_DIR / "03_harmonization.duckdb"


# =============================================================================
# OFFICIAL SCOPE
# =============================================================================

START_YEAR = 2015
END_YEAR = 2020
YEARS = list(range(START_YEAR, END_YEAR + 1))

EXPECTED_COUNTRIES = 28
EXPECTED_REGIONS = 513
EXPECTED_REGION_YEARS = EXPECTED_REGIONS * len(YEARS)


# =============================================================================
# SCRIPT 02 SOURCES RETAINED FOR SCRIPT 03
# =============================================================================
#
# required=True means Script 03 cannot construct its master key without it.
# Other sources may be absent in a partial rerun, but this will be recorded
# explicitly in the QA workbook rather than silently ignored.
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
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {
        normalize_column_name(column): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in lookup:
            return lookup[key]

    return None


def safe_unique_count(series: pd.Series) -> int:
    try:
        return int(series.dropna().nunique())
    except Exception:
        return 0


def safe_sample(series: pd.Series, limit: int = 3) -> str:
    try:
        values = (
            series.dropna()
            .astype(str)
            .str.strip()
        )
        values = values[values.ne("")]
        return " | ".join(values.drop_duplicates().head(limit).tolist())
    except Exception:
        return ""


def detect_year_column(frame: pd.DataFrame) -> str | None:
    return find_column(
        frame,
        [
            "year",
            "time_period",
            "timeperiod",
            "period",
            "date",
            "time",
        ],
    )


def detect_country_column(frame: pd.DataFrame) -> str | None:
    return find_column(
        frame,
        [
            "code",
            "iso_code",
            "isocode3",
            "iso_code3",
            "iso3",
            "iso3c",
            "country_code",
            "country_iso3",
            "countryiso3code",
            "ref_area",
            "reference_area",
            "gid_0",
            "adm0_a3",
            "wb_a3",
        ],
    )


def detect_region_code_column(frame: pd.DataFrame) -> str | None:
    return find_column(
        frame,
        [
            "geo_code",
            "gdlcode",
            "region_code",
            "region_id",
            "geo_id",
            "gid_1",
            "adm1_pcode",
            "tl2",
            "tl3",
            "territorial_code",
        ],
    )


def detect_region_name_column(frame: pd.DataFrame) -> str | None:
    return find_column(
        frame,
        [
            "geo_name",
            "region",
            "region_name",
            "name",
            "adm1_name",
            "territorial_name",
        ],
    )


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
            re.search(
                rf"(?<!\d){year}(?!\d)",
                text,
            )
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
        return pd.read_csv(
            path,
            low_memory=False,
            encoding="utf-8-sig",
        )

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".gpkg":
        if gpd is None:
            raise ImportError(
                f"geopandas is required to read {path.name}."
            )

        gdf = gpd.read_file(path)
        frame = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        if "geometry" in gdf.columns:
            frame["geometry_wkt"] = gdf.geometry.astype(str)

        return frame

    raise ValueError(
        f"Unsupported source format: {path.suffix}"
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# MASTER SPID REGION-YEAR KEY
# =============================================================================

def build_master_panel(
    spid: pd.DataFrame,
    target_regions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    required_columns = [
        "code",
        "geo_code",
        "year",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in spid.columns
    ]

    if missing_required:
        raise ValueError(
            "SPID master input is missing required columns: "
            f"{missing_required}"
        )

    master = spid.copy()

    master["code"] = (
        master["code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    master["geo_code"] = (
        master["geo_code"]
        .astype("string")
        .str.strip()
    )

    master["year"] = pd.to_numeric(
        master["year"],
        errors="coerce",
    ).astype("Int64")

    master = master[
        master["year"].isin(YEARS)
    ].copy()

    master["region_id"] = (
        master["code"].astype(str)
        + "__"
        + master["geo_code"].astype(str)
    )

    master["region_year_id"] = (
        master["region_id"]
        + "__"
        + master["year"].astype(str)
    )

    duplicated = int(
        master.duplicated(
            ["code", "geo_code", "year"],
            keep=False,
        ).sum()
    )

    if duplicated:
        raise ValueError(
            f"SPID master has {duplicated} duplicated "
            "(code, geo_code, year) rows."
        )

    # -------------------------------------------------------------------------
    # Attach stable geographic metadata exported by Script 02.
    # -------------------------------------------------------------------------

    target = target_regions.copy()

    common_keys = [
        column
        for column in ["code", "geo_code"]
        if column in target.columns
        and column in master.columns
    ]

    if "geo_code" not in common_keys:
        raise ValueError(
            "02_target_regions.csv must contain geo_code."
        )

    if "code" in target.columns:
        target["code"] = (
            target["code"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    target["geo_code"] = (
        target["geo_code"]
        .astype("string")
        .str.strip()
    )

    target = target.drop_duplicates(
        common_keys
    ).copy()

    metadata_columns = [
        column
        for column in target.columns
        if column not in common_keys
    ]

    master = master.merge(
        target[
            common_keys + metadata_columns
        ],
        how="left",
        on=common_keys,
        validate="many_to_one",
        suffixes=("", "_target"),
    )

    # -------------------------------------------------------------------------
    # Strict scope validation.
    # -------------------------------------------------------------------------

    country_count = int(master["code"].nunique())
    region_count = int(master["region_id"].nunique())
    row_count = len(master)
    observed_years = sorted(
        master["year"].dropna().astype(int).unique().tolist()
    )

    qa = pd.DataFrame(
        [
            {
                "check": "country_count",
                "observed": country_count,
                "expected": EXPECTED_COUNTRIES,
                "status": (
                    "PASS"
                    if country_count == EXPECTED_COUNTRIES
                    else "FAIL"
                ),
            },
            {
                "check": "region_count",
                "observed": region_count,
                "expected": EXPECTED_REGIONS,
                "status": (
                    "PASS"
                    if region_count == EXPECTED_REGIONS
                    else "FAIL"
                ),
            },
            {
                "check": "region_year_rows",
                "observed": row_count,
                "expected": EXPECTED_REGION_YEARS,
                "status": (
                    "PASS"
                    if row_count == EXPECTED_REGION_YEARS
                    else "FAIL"
                ),
            },
            {
                "check": "official_years",
                "observed": ",".join(map(str, observed_years)),
                "expected": ",".join(map(str, YEARS)),
                "status": (
                    "PASS"
                    if observed_years == YEARS
                    else "FAIL"
                ),
            },
            {
                "check": "duplicate_master_keys",
                "observed": duplicated,
                "expected": 0,
                "status": (
                    "PASS"
                    if duplicated == 0
                    else "FAIL"
                ),
            },
        ]
    )

    if (qa["status"] == "FAIL").any():
        failed = qa.loc[
            qa["status"].eq("FAIL"),
            ["check", "observed", "expected"],
        ]
        raise ValueError(
            "Master-panel validation failed:\n"
            + failed.to_string(index=False)
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
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

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
                "status": (
                    "MISSING_REQUIRED"
                    if required
                    else "MISSING_OPTIONAL"
                ),
            },
            [],
            [],
        )

    country_col = detect_country_column(frame)
    region_code_col = detect_region_code_column(frame)
    region_name_col = detect_region_name_column(frame)
    year_col = detect_year_column(frame)
    wide_year_cols = detect_wide_year_columns(frame)

    exact_duplicates = int(
        frame.duplicated(keep=False).sum()
    )

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

        valid_years = (
            parsed_year
            .dropna()
            .astype(int)
        )

        if not valid_years.empty:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())

        years_in_scope = sorted(
            valid_years[
                valid_years.isin(YEARS)
            ]
            .unique()
            .tolist()
        )

        temp = frame.copy()
        temp["_qa_year"] = parsed_year

        for year in YEARS:
            subset = temp[
                temp["_qa_year"].eq(year)
            ].copy()

            year_coverage_rows.append(
                {
                    "source_id": source_id,
                    "year": year,
                    "records": len(subset),
                    "countries": (
                        safe_unique_count(subset[country_col])
                        if country_col is not None
                        and not subset.empty
                        else np.nan
                    ),
                    "regions": (
                        safe_unique_count(subset[region_code_col])
                        if region_code_col is not None
                        and not subset.empty
                        else (
                            safe_unique_count(subset[region_name_col])
                            if region_name_col is not None
                            and not subset.empty
                            else np.nan
                        )
                    ),
                }
            )

    elif wide_year_cols:
        detected_years = set()

        for column in wide_year_cols:
            for year in YEARS:
                if re.search(
                    rf"(?<!\d){year}(?!\d)",
                    str(column),
                ):
                    detected_years.add(year)

        years_in_scope = sorted(detected_years)

        if years_in_scope:
            min_year = min(years_in_scope)
            max_year = max(years_in_scope)

        for year in YEARS:
            columns_for_year = [
                column
                for column in wide_year_cols
                if re.search(
                    rf"(?<!\d){year}(?!\d)",
                    str(column),
                )
            ]

            valid_rows = 0

            if columns_for_year:
                valid_rows = int(
                    frame[columns_for_year]
                    .notna()
                    .any(axis=1)
                    .sum()
                )

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
                "null_pct": (
                    null_count / row_count
                    if row_count
                    else np.nan
                ),
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
        "file_size_mb": (
            path.stat().st_size / (1024 ** 2)
            if path.exists()
            else np.nan
        ),
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

    connection.execute(
        f'DROP TABLE IF EXISTS "{safe_name}"'
    )

    connection.execute(
        f'CREATE TABLE "{safe_name}" AS '
        f'SELECT * FROM "{temp_name}"'
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
) -> None:

    with pd.ExcelWriter(
        QA_WORKBOOK,
        engine="openpyxl",
    ) as writer:

        master_qa.to_excel(
            writer,
            sheet_name="master_key_qa",
            index=False,
        )

        source_inventory.to_excel(
            writer,
            sheet_name="source_inventory",
            index=False,
        )

        column_inventory.to_excel(
            writer,
            sheet_name="column_inventory",
            index=False,
        )

        year_coverage.to_excel(
            writer,
            sheet_name="year_coverage",
            index=False,
        )

        dim_region.to_excel(
            writer,
            sheet_name="region_dimension",
            index=False,
        )

        # Basic readability without introducing presentation complexity.
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = 0

                for cell in column_cells[:200]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))

                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max(max_length + 2, 10), 45)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    global_start = time.perf_counter()

    print_header(
        "SCRIPT 03 v1.0 — MASTER PANEL + SOURCE INVENTORY + INITIAL QA"
    )

    print(f"Project root: {ROOT}")
    print(f"Official period: {START_YEAR}–{END_YEAR}")
    print(
        f"Official scope: {EXPECTED_COUNTRIES} countries | "
        f"{EXPECTED_REGIONS} regions | "
        f"{EXPECTED_REGION_YEARS:,} region-year observations"
    )

    # -------------------------------------------------------------------------
    # 1. Validate required Script 02 outputs
    # -------------------------------------------------------------------------

    print_header(
        "1/4 — VALIDATE SCRIPT 02 OUTPUTS"
    )

    missing_required = []

    for source_id, specification in SOURCE_REGISTRY.items():
        path: Path = specification["path"]

        if path.exists():
            print(
                f"[FOUND]   {source_id:<24} "
                f"{path.name}"
            )
        else:
            print(
                f"[MISSING] {source_id:<24} "
                f"{path.name}"
            )

            if specification["required"]:
                missing_required.append(
                    f"{source_id}: {path}"
                )

    if missing_required:
        raise FileNotFoundError(
            "Required Script 02 outputs are missing:\n"
            + "\n".join(missing_required)
        )

    # -------------------------------------------------------------------------
    # 2. Build strict SPID master region-year key
    # -------------------------------------------------------------------------

    print_header(
        "2/4 — BUILD MASTER REGION-YEAR KEY"
    )

    spid = load_source(
        SOURCE_REGISTRY["SPID"]["path"]
    )

    target_regions = load_source(
        SOURCE_REGISTRY["TARGET_REGIONS"]["path"]
    )

    master, dim_region, master_qa = build_master_panel(
        spid,
        target_regions,
    )

    write_csv(
        master,
        MASTER_OUTPUT,
    )

    write_csv(
        dim_region,
        REGION_DIM_OUTPUT,
    )

    print(
        f"Master panel: {len(master):,} rows | "
        f"{master['code'].nunique()} countries | "
        f"{master['region_id'].nunique()} regions | "
        f"{master['year'].nunique()} years"
    )

    print(
        f"Master key status: "
        f"{'PASS' if master_qa['status'].eq('PASS').all() else 'FAIL'}"
    )

    # -------------------------------------------------------------------------
    # 3. Inventory every retained source + persist in DuckDB
    # -------------------------------------------------------------------------

    print_header(
        "3/4 — SOURCE INVENTORY + DUCKDB STAGING"
    )

    source_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []

    connection = duckdb.connect(
        str(DUCKDB_FILE)
    )

    try:
        persist_duckdb_table(
            connection,
            "master_region_year",
            master,
        )

        persist_duckdb_table(
            connection,
            "dim_region",
            dim_region,
        )

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

                print(
                    f"[SKIP] {source_id:<24} "
                    f"file not found"
                )

                continue

            try:
                frame = load_source(path)

                source_row, columns, years = inventory_source(
                    source_id,
                    specification,
                    frame,
                )

                source_rows.append(source_row)
                column_rows.extend(columns)
                year_rows.extend(years)

                persist_duckdb_table(
                    connection,
                    f"raw_{source_id}",
                    frame,
                )

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

                print(
                    f"[ERROR] {source_id}: {exc}"
                )

                traceback.print_exc()

        source_inventory = pd.DataFrame(
            source_rows
        )

        column_inventory = pd.DataFrame(
            column_rows
        )

        year_coverage = pd.DataFrame(
            year_rows
        )

        # Persist QA tables inside the same DuckDB file.
        persist_duckdb_table(
            connection,
            "qa_master_key",
            master_qa,
        )

        persist_duckdb_table(
            connection,
            "qa_source_inventory",
            source_inventory,
        )

        persist_duckdb_table(
            connection,
            "qa_column_inventory",
            column_inventory,
        )

        persist_duckdb_table(
            connection,
            "qa_year_coverage",
            year_coverage,
        )

    finally:
        connection.close()

    write_csv(
        source_inventory,
        SOURCE_INVENTORY_OUTPUT,
    )

    write_csv(
        column_inventory,
        COLUMN_INVENTORY_OUTPUT,
    )

    write_csv(
        year_coverage,
        YEAR_COVERAGE_OUTPUT,
    )

    # -------------------------------------------------------------------------
    # 4. Consolidated QA workbook
    # -------------------------------------------------------------------------

    print_header(
        "4/4 — WRITE CONSOLIDATED QA"
    )

    write_qa_workbook(
        master_qa,
        source_inventory,
        column_inventory,
        year_coverage,
        dim_region,
    )

    available = int(
        source_inventory["status"]
        .eq("AVAILABLE")
        .sum()
    )

    missing = int(
        source_inventory["status"]
        .str.startswith("MISSING", na=False)
        .sum()
    )

    read_errors = int(
        source_inventory["status"]
        .str.startswith("READ_ERROR", na=False)
        .sum()
    )

    print()
    print("CHECKPOINT SUMMARY")
    print("-" * 88)
    print(
        f"Master panel:   {len(master):,} rows | "
        f"{master['region_id'].nunique()} regions"
    )
    print(
        f"Sources:        {available} available | "
        f"{missing} missing | "
        f"{read_errors} read errors"
    )
    print(f"DuckDB:         {DUCKDB_FILE}")
    print(f"QA workbook:    {QA_WORKBOOK}")
    print(f"Master CSV:     {MASTER_OUTPUT}")

    elapsed = time.perf_counter() - global_start

    print_header(
        "GLOBAL EXECUTION TIME"
    )

    print(
        "Script: 03_build_master_data_and_qa.py"
    )

    print(
        f"Total execution time: "
        f"{int(elapsed // 60):02d} min "
        f"{elapsed % 60:05.2f} sec"
    )

    print()
    print(
        "SCRIPT 03 CHECKPOINT 1 COMPLETE — "
        "MASTER KEY + INVENTORY + INITIAL QA"
    )
    print(
        "Next checkpoint after validation: "
        "geographic crosswalk and source-by-source harmonization."
    )


if __name__ == "__main__":
    main()
