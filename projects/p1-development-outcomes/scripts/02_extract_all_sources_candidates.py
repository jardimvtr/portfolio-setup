from __future__ import annotations

# =============================================================================
# SCRIPT 02.3 — GDL SHDI PARSER FIX
# P1 / General Portfolio Data Pipeline
#
# Official analytical window: 2015–2020
#
# Purpose
# -------
# 1) Consume the availability audit produced by Script 01.
# 2) Reconstruct the strict SPID 2015–2020 country/region scope.
# 3) Extract/download all candidate sources that can be automated safely.
# 4) Preserve partial/snapshot/manual sources instead of silently discarding them.
# 5) Use checkpoints and resume logic for slow APIs.
# 6) Produce one extraction manifest summarizing what succeeded, failed, or still
#    requires credentials/manual download.
#
# IMPORTANT
# ---------
# - Script 02 performs ACQUISITION and source-level extraction.
# - Script 03 will perform the final spatial/temporal harmonization, derivations,
#   matching, QA and master-panel construction.
# - No 2015–2022 benchmark logic is carried forward.
#
# Script 02.1 finalization: GDL SHDI + OECD regional labour
# -------------------------------------
# - preserves all validated v2.2 extraction logic;
# - adds authenticated/manual ingestion of the current GDL SHDI file;
# - adds automatic OECD regional unemployment extraction from the official
#   OECD Data Explorer SDMX API for 2015–2020;
# - removes sources explicitly discarded from the P1 scope;
# - keeps one consolidated Script 02 execution point.
# =============================================================================

from pathlib import Path
from typing import Any, Iterable
import json
import math
from io import StringIO
import os
import re
import shutil
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    from shapely.geometry import box, mapping
except Exception:
    box = None
    mapping = None

try:
    import rasterio
    from rasterio.features import geometry_mask, geometry_window
except Exception:
    rasterio = None
    geometry_mask = None
    geometry_window = None


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "downloaded"
MANUAL_DIR = RAW_DIR / "manual"

INTERIM_DIR = ROOT / "data" / "interim" / "script02"
OUTPUT_DIR = ROOT / "outputs"

for directory in [
    DOWNLOAD_DIR,
    MANUAL_DIR,
    INTERIM_DIR,
    OUTPUT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# OFFICIAL PERIOD
# =============================================================================

START_YEAR = 2015
END_YEAR = 2020
YEARS = list(
    range(
        START_YEAR,
        END_YEAR + 1,
    )
)


# =============================================================================
# INPUT FILES
# =============================================================================

AUDIT_CANDIDATES = [
    OUTPUT_DIR / "01_candidate_availability_audit_v1_1.xlsx",
    OUTPUT_DIR / "01_candidate_availability_audit.xlsx",
]

AUDIT_FILE = next(
    (
        path
        for path in AUDIT_CANDIDATES
        if path.exists()
    ),
    AUDIT_CANDIDATES[0],
)

SPID_FILE = RAW_DIR / "AM25_-_SPID_data.xlsx"
SSGD_FILE = RAW_DIR / "ssgd_v2_long.xlsx"


# =============================================================================
# OUTPUT FILES
# =============================================================================

MANIFEST_FILE = (
    OUTPUT_DIR
    / "02_3_extraction_manifest.xlsx"
)

SPID_OUTPUT = (
    INTERIM_DIR
    / "02_spid_selected_panel_2015_2020.csv"
)

TARGET_REGIONS_CSV = (
    INTERIM_DIR
    / "02_target_regions.csv"
)

TARGET_REGIONS_GPKG = (
    INTERIM_DIR
    / "02_target_regions.gpkg"
)

SPACE2STATS_OUTPUT = (
    INTERIM_DIR
    / "02_space2stats_candidates.csv"
)

SPACE2STATS_CHECKPOINT = (
    INTERIM_DIR
    / "02_space2stats_checkpoint.csv"
)

SPACE2STATS_ERRORS = (
    INTERIM_DIR
    / "02_space2stats_errors.csv"
)

WORLDPOP_AGESEX_OUTPUT = (
    INTERIM_DIR
    / "02_worldpop_agesex_2015_2020.csv"
)

WORLDPOP_AGESEX_CHECKPOINT = (
    INTERIM_DIR
    / "02_worldpop_agesex_checkpoint.csv"
)

WORLDPOP_AGESEX_ERRORS = (
    INTERIM_DIR
    / "02_worldpop_agesex_errors.csv"
)

WORLDPOP_BULK_DIR = (
    DOWNLOAD_DIR
    / "worldpop_agesex_1km_R2025A"
)

WORLDPOP_BULK_DOWNLOAD_LOG = (
    INTERIM_DIR
    / "02_worldpop_bulk_download_log.csv"
)

SSGD_OUTPUT = (
    INTERIM_DIR
    / "02_ssgd_candidates_2015_2020.csv"
)

WDI_OUTPUT = (
    INTERIM_DIR
    / "02_wdi_candidates_2015_2020.csv"
)

KUMMU_OUTPUT = (
    INTERIM_DIR
    / "02_kummu_adm1_gdp_2015_2020.csv"
)

DOSE_OUTPUT = (
    INTERIM_DIR
    / "02_dose_2015_2020.csv"
)

NIVA_OUTPUT = (
    INTERIM_DIR
    / "02_niva_adm1_2015_2019.gpkg"
)

GDL_SCD_BASELINE_OUTPUT = (
    INTERIM_DIR
    / "02_gdl_scd_baseline_2015_2020.csv"
)

GDL_SCD_COMPREHENSIVE_OUTPUT = (
    INTERIM_DIR
    / "02_gdl_scd_comprehensive_2015_2020.csv"
)


OOKLA_OUTPUT = (
    INTERIM_DIR
    / "02_ookla_adm1_annual_2019_2020.csv"
)

GDL_SHDI_OUTPUT = (
    INTERIM_DIR
    / "02_gdl_shdi_2015_2020.csv"
)

OECD_REGIONAL_OUTPUT = (
    INTERIM_DIR
    / "02_oecd_regional_unemployment_2015_2020.csv"
)

OECD_REGIONAL_RAW_OUTPUT = (
    DOWNLOAD_DIR
    / "OECD_REGIONAL"
    / "oecd_regional_labour_rates_2015_2020.csv"
)


# =============================================================================
# EXECUTION OPTIONS
# =============================================================================
# Script 02.1 is a focused finalization pass. SPID remains enabled because the
# country/region scope is required by downstream logic and rebuilds quickly.
# Previously completed heavy/public sources are disabled so they are not
# re-executed. Only GDL SHDI and OECD Regional are added in this pass.

RUN_SPID = True
RUN_SPACE2STATS = False
RUN_SSGD = False
RUN_WDI = False
RUN_ZENODO_PUBLIC = False
RUN_WORLDPOP_AGESEX = False
RUN_OOKLA_WB = False
RUN_GDL_SHDI = True
RUN_OECD_REGIONAL = False

# WorldPop Global2 R2025A bulk-raster extraction.
#
# We no longer use the asynchronous polygon API. The official WorldPop Global2
# release provides annual country-level 1km GeoTIFFs for 2015-2030. Script 02
# downloads the frozen R2025A v1 files and performs the subnational aggregation
# locally. This removes the 1,000-request/day API bottleneck.
WORLDPOP_RELEASE = "R2025A"
WORLDPOP_VERSION = "v1"
WORLDPOP_RESOLUTION = "1km"
WORLDPOP_UN_ADJUSTMENT_TAG = "UA"
WORLDPOP_KEEP_RASTERS = False
WORLDPOP_DOWNLOAD_WORKERS = 4
WORLDPOP_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
WORLDPOP_DOWNLOAD_CONNECT_TIMEOUT = 30
WORLDPOP_DOWNLOAD_READ_TIMEOUT = 600

# Only these age bands are needed. Working-age population (15-64) is recovered
# as total population minus 0-14 minus 65+, which cuts the age raster downloads
# from 20 to 10 per country-year.
WORLDPOP_AGE_0_14 = ("00", "01", "05", "10")
WORLDPOP_AGE_65_PLUS = ("65", "70", "75", "80", "85", "90")

# Space2Stats fallback controls. Only regions that fail the normal request use
# these fallbacks.
SPACE2STATS_SIMPLIFY_RELATIVE_TOLERANCES = (
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
)
SPACE2STATS_FIELD_CHUNK_SIZE = 8

# Avoid silently downloading several GB of optional global rasters.
# Script 02 records these sources and accepts manually downloaded files.
DOWNLOAD_LARGE_RASTERS = False

# Network behavior
REQUEST_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 300
MAX_RETRIES = 5
BACKOFF_SECONDS = 2

SPACE2STATS_URL = (
    "https://space2stats.ds.io"
)

WDI_API = (
    "https://api.worldbank.org/v2"
)

OECD_REGIONAL_LABOUR_API = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.CFE.EDS,DSD_REG_LAB@DF_RATES,2.4/all"
)


# =============================================================================
# SOURCE-SPECIFIC PUBLIC DOWNLOADS
# =============================================================================

PUBLIC_DOWNLOADS = {
    "KUMMU_GDP": {
        "files": [
            {
                "name": "tabulated_adm1_gdp_perCapita.csv",
                "url": (
                    "https://zenodo.org/records/18429133/files/"
                    "tabulated_adm1_gdp_perCapita.csv?download=1"
                ),
            },
            {
                "name": "adm1_metaData.csv",
                "url": (
                    "https://zenodo.org/records/18429133/files/"
                    "adm1_metaData.csv?download=1"
                ),
            },
        ],
    },
    "DOSE": {
        # Current public release as of 2026. It supersedes V2.11 while retaining
        # the 2015–2020 years needed by this project.
        "files": [
            {
                "name": "DOSE_V2.14.csv",
                "url": (
                    "https://zenodo.org/records/20035157/files/"
                    "DOSE_V2.14.csv?download=1"
                ),
            },
        ],
    },
    "NIVA_MIGRATION": {
        "files": [
            {
                "name": "polyg_adm1_dataNetMgr.gpkg",
                "url": (
                    "https://zenodo.org/records/7997134/files/"
                    "polyg_adm1_dataNetMgr.gpkg?download=1"
                ),
            },
        ],
    },
    "OOKLA_WB": {
        "files": [
            {
                "name": "OOKLA_ST_ADM1_A.csv",
                "url": (
                    "https://datacatalogfiles.worldbank.org/"
                    "ddh-published/0067161/DR0096333/"
                    "OOKLA_ST_ADM1_A.csv"
                ),
            },
        ],
    },
}


# =============================================================================
# SSGD CANDIDATES
# =============================================================================

SSGD_CODES = {
    "si_intuse": "internet_access",
    "re_enofoo": "food_security",
    "re_savmon": "savings",
    "re_govtra": "government_transfers",
    "re_rem": "remittances",
}


# =============================================================================
# WDI CANDIDATES
# =============================================================================

WDI_CODES = {
    "EG.ELC.ACCS.ZS": "access_to_electricity",
    "EG.CFT.ACCS.ZS": "clean_cooking_access",
    "SH.H2O.BASW.ZS": "basic_drinking_water",
    "SH.H2O.SMDW.ZS": "safely_managed_drinking_water",
    "SH.STA.BASS.ZS": "basic_sanitation",
    "SH.STA.SMSS.ZS": "safely_managed_sanitation",
    "SN.ITK.MSFI.ZS": "moderate_or_severe_food_insecurity",
    "SN.ITK.SVFI.ZS": "severe_food_insecurity",
    "SN.ITK.DEFC.ZS": "undernourishment",
    "EG.FEC.RNEW.ZS": "renewable_energy_consumption",
    "EG.ELC.RNEW.ZS": "renewable_electricity_output",
    "EG.EGY.PRIM.PP.KD": "energy_intensity",
    "EN.ATM.GHGT.KT.CE": "ghg_total_legacy",
    "EN.GHG.ALL.MT.CE.AR5": "ghg_total_ar5",
    "EN.GHG.ALL.PC.CE.AR5": "ghg_per_capita_ar5",
    "ER.H2O.FWST.ZS": "water_stress",
    "ER.H2O.FWTL.K3": "freshwater_withdrawals_total",
    "ER.H2O.FWTL.ZS": "freshwater_withdrawals_share",
    "SP.POP.TOTL": "population_national",
    "SP.URB.TOTL.IN.ZS": "urban_population_share_national",
    "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp_national",
}


# =============================================================================
# SPACE2STATS FAMILY RULES
# =============================================================================

SPACE2STATS_RULES: list[
    tuple[
        str,
        list[str],
    ]
] = [
    (
        "population",
        [
            r"^sum_pop_",
        ],
    ),
    (
        "nighttime_lights",
        [
            r"viirs",
            r"ntl",
            r"night.*light",
        ],
    ),
    (
        "urbanization_ghs",
        [
            r"^ghs_",
        ],
    ),
    (
        "built_up_area",
        [
            r"built.*area",
            r"built_area",
        ],
    ),
    (
        "flood_exposure",
        [
            r"flood",
        ],
    ),
    (
        "drought",
        [
            r"drought",
            r"spei",
        ],
    ),
    (
        "cyclone",
        [
            r"cyclone",
            r"tropical.*storm",
            r"hurricane",
        ],
    ),
    (
        "wildfire",
        [
            r"wildfire",
            r"fire",
            r"burned",
            r"burnt",
        ],
    ),
    (
        "landslide",
        [
            r"landslide",
        ],
    ),
]


# =============================================================================
# LOGGING / MANIFEST
# =============================================================================

MANIFEST_ROWS: list[
    dict[str, Any]
] = []

ERROR_ROWS: list[
    dict[str, Any]
] = []


def add_manifest(
    source_id: str,
    status: str,
    records: int | float | None = None,
    output: str | Path | None = None,
    message: str = "",
    seconds: float | None = None,
) -> None:

    MANIFEST_ROWS.append(
        {
            "source_id": source_id,
            "status": status,
            "records": records,
            "output": (
                str(output)
                if output is not None
                else ""
            ),
            "message": message,
            "elapsed_seconds": seconds,
        }
    )


def print_header(
    text: str,
) -> None:

    print()
    print(
        "=" * 88
    )
    print(
        text
    )
    print(
        "=" * 88
    )


def timed_stage(
    label: str,
):
    """
    Minimal context manager implemented as a class below so the script does not
    depend on contextlib-specific custom objects.
    """

    class StageTimer:

        def __enter__(
            self,
        ):

            self.start = (
                time.perf_counter()
            )

            print_header(
                label
            )

            return self

        def __exit__(
            self,
            exc_type,
            exc,
            tb,
        ):

            self.seconds = (
                time.perf_counter()
                - self.start
            )

            print(
                f"\nStage time: "
                f"{self.seconds / 60:.2f} min"
            )

            return False

    return StageTimer()


# =============================================================================
# GENERIC HELPERS
# =============================================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "WBG-Outcomes-Portfolio-"
            "Consolidated-Extraction/2.1"
        )
    }
)


def normalize_text(
    value: Any,
) -> str:

    if pd.isna(
        value
    ):

        return ""

    return str(
        value
    ).strip()


def normalize_code(
    value: Any,
) -> str:

    return (
        normalize_text(
            value
        )
        .upper()
    )


def normalize_spid_comparability(
    value: Any,
) -> str:

    if pd.isna(
        value
    ):

        return "NA"

    text = str(
        value
    ).strip()

    if not text:

        return "NA"

    numeric = pd.to_numeric(
        text,
        errors="coerce",
    )

    if pd.notna(
        numeric
    ):

        if float(
            numeric
        ).is_integer():

            return str(
                int(
                    numeric
                )
            )

        return (
            f"{float(numeric):.12g}"
        )

    return text.upper()


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:

    normalized = {
        re.sub(
            r"[^a-z0-9]+",
            "",
            str(column).lower(),
        ): column
        for column in df.columns
    }

    for candidate in candidates:

        key = re.sub(
            r"[^a-z0-9]+",
            "",
            str(candidate).lower(),
        )

        if key in normalized:

            return normalized[
                key
            ]

    return None


def safe_bool(
    value: Any,
) -> bool:

    if isinstance(
        value,
        bool,
    ):

        return value

    if pd.isna(
        value
    ):

        return False

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def read_excel_required(
    path: Path,
    required_columns: list[str],
    preferred_sheet: str | None = None,
) -> tuple[
    pd.DataFrame,
    str,
]:

    excel = pd.ExcelFile(
        path
    )

    sheets = (
        [
            preferred_sheet
        ]
        if (
            preferred_sheet
            and preferred_sheet
            in excel.sheet_names
        )
        else []
    )

    sheets += [
        sheet
        for sheet in excel.sheet_names
        if sheet not in sheets
    ]

    for sheet in sheets:

        preview = pd.read_excel(
            path,
            sheet_name=sheet,
            nrows=5,
        )

        normalized = {
            re.sub(
                r"[^a-z0-9]+",
                "",
                str(column).lower(),
            )
            for column in preview.columns
        }

        if all(
            re.sub(
                r"[^a-z0-9]+",
                "",
                required.lower(),
            )
            in normalized
            for required in required_columns
        ):

            return (
                pd.read_excel(
                    path,
                    sheet_name=sheet,
                ),
                sheet,
            )

    raise ValueError(
        f"No sheet in {path.name} contains "
        f"{required_columns}."
    )


def safe_json_request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Any:

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            if method.upper() == "GET":

                response = SESSION.get(
                    url,
                    params=params,
                    timeout=timeout,
                )

            else:

                response = SESSION.post(
                    url,
                    json=payload,
                    timeout=timeout,
                )

            response.raise_for_status()

            return response.json()

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                wait = (
                    BACKOFF_SECONDS
                    * (
                        2
                        ** (
                            attempt - 1
                        )
                    )
                )

                time.sleep(
                    wait
                )

    raise RuntimeError(
        f"Request failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def download_file(
    url: str,
    destination: Path,
) -> Path:

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        print(
            f"  cached: "
            f"{destination.name}"
        )

        return destination

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix
        + ".part"
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            with SESSION.get(
                url,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:

                response.raise_for_status()

                with open(
                    temporary,
                    "wb",
                ) as file:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024,
                    ):

                        if chunk:

                            file.write(
                                chunk
                            )

            temporary.replace(
                destination
            )

            return destination

        except Exception as exc:

            last_error = exc

            if temporary.exists():

                temporary.unlink(
                    missing_ok=True
                )

            if attempt < MAX_RETRIES:

                wait = (
                    BACKOFF_SECONDS
                    * (
                        2
                        ** (
                            attempt - 1
                        )
                    )
                )

                print(
                    f"  retry {attempt}/"
                    f"{MAX_RETRIES} after "
                    f"{wait}s: {exc}"
                )

                time.sleep(
                    wait
                )

    raise RuntimeError(
        f"Download failed: "
        f"{url}\n{last_error}"
    )


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def find_geometry_file() -> Path:

    candidates = list(
        RAW_DIR.rglob(
            "*.gpkg"
        )
    )

    preferred = [
        path
        for path in candidates
        if (
            "am25"
            in path.name.lower()
            and "subnat"
            in path.name.lower()
        )
    ]

    if preferred:

        return preferred[
            0
        ]

    if candidates:

        return candidates[
            0
        ]

    raise FileNotFoundError(
        "No GPKG geometry file found "
        "under data/raw."
    )


# =============================================================================
# AUDIT WORKBOOK
# =============================================================================

def load_audit_workbook() -> dict[
    str,
    pd.DataFrame,
]:

    if not AUDIT_FILE.exists():

        raise FileNotFoundError(
            f"Script 01 workbook not found: "
            f"{AUDIT_FILE}"
        )

    excel = pd.ExcelFile(
        AUDIT_FILE
    )

    required = [
        "04_Extraction_Queue",
        "06_SPID_Countries",
        "07_SPID_Selected_Series",
        "11_S2S_All_Fields",
        "12_SSGD_Indicators",
    ]

    missing = [
        sheet
        for sheet in required
        if sheet not in excel.sheet_names
    ]

    if missing:

        raise ValueError(
            "Audit workbook is missing required "
            f"sheets: {missing}"
        )

    return {
        sheet: pd.read_excel(
            AUDIT_FILE,
            sheet_name=sheet,
        )
        for sheet in excel.sheet_names
    }


def get_primary_country_scope(
    workbook: dict[
        str,
        pd.DataFrame,
    ],
) -> list[str]:

    countries = workbook[
        "06_SPID_Countries"
    ].copy()

    flag = (
        "strict_all_metrics_complete_2015_2020"
    )

    if flag not in countries.columns:

        raise ValueError(
            f"{flag} not found in "
            "06_SPID_Countries."
        )

    countries = countries[
        countries[
            flag
        ].map(
            safe_bool
        )
    ].copy()

    result = sorted(
        countries[
            "code"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
    )

    if not result:

        raise ValueError(
            "No strict SPID countries found "
            "for 2015–2020."
        )

    return result


def get_selected_spid_series(
    workbook: dict[
        str,
        pd.DataFrame,
    ],
    countries: list[str],
) -> pd.DataFrame:

    selected = workbook[
        "07_SPID_Selected_Series"
    ].copy()

    if "selected_for" in selected.columns:

        selected = selected[
            selected[
                "selected_for"
            ]
            .astype(str)
            .str.strip()
            .eq(
                "2015-2020"
            )
        ].copy()

    if (
        "strict_all_metrics_complete"
        in selected.columns
    ):

        selected = selected[
            selected[
                "strict_all_metrics_complete"
            ]
            .map(
                safe_bool
            )
        ].copy()

    selected[
        "code"
    ] = (
        selected[
            "code"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    selected = selected[
        selected[
            "code"
        ].isin(
            countries
        )
    ].copy()

    if selected[
        "code"
    ].duplicated().any():

        selected = (
            selected.sort_values(
                [
                    "code",
                    "rank_within_country",
                ]
                if (
                    "rank_within_country"
                    in selected.columns
                )
                else [
                    "code"
                ]
            )
            .drop_duplicates(
                "code",
                keep="first",
            )
        )

    missing = sorted(
        set(
            countries
        )
        - set(
            selected[
                "code"
            ]
        )
    )

    if missing:

        raise ValueError(
            "Strict countries without selected "
            f"SPID series: {missing}"
        )

    return selected


# =============================================================================
# SPID EXTRACTION
# =============================================================================

def extract_spid(
    selected_series: pd.DataFrame,
    countries: list[str],
) -> pd.DataFrame:

    if not SPID_FILE.exists():

        raise FileNotFoundError(
            SPID_FILE
        )

    df, sheet = (
        read_excel_required(
            SPID_FILE,
            required_columns=[
                "code",
                "year",
                "geo_code",
            ],
            preferred_sheet="Data",
        )
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    df["year"] = pd.to_numeric(
        df[
            "year"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    df[
        "code"
    ] = (
        df[
            "code"
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df[
        "geo_code"
    ] = (
        df[
            "geo_code"
        ]
        .astype("string")
        .str.strip()
    )

    mask = (
        df[
            "year"
        ].isin(
            YEARS
        )
        & df[
            "code"
        ].isin(
            countries
        )
    )

    if "data" in df.columns:

        mask &= (
            df[
                "data"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(
                "ALL"
            )
        )

    if "data_group" in df.columns:

        mask &= (
            df[
                "data_group"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(
                "ALL"
            )
        )

    df = df[
        mask
    ].copy()

    if "welfaretype" in df.columns:

        df[
            "_welfaretype"
        ] = (
            df[
                "welfaretype"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    else:

        df[
            "_welfaretype"
        ] = (
            "UNKNOWN"
        )

    if "comparability" in df.columns:

        df[
            "_comparability"
        ] = (
            df[
                "comparability"
            ]
            .map(
                normalize_spid_comparability
            )
        )

    else:

        df[
            "_comparability"
        ] = (
            "NA"
        )

    pieces = []

    for _, row in (
        selected_series.iterrows()
    ):

        code = normalize_code(
            row[
                "code"
            ]
        )

        welfare = normalize_code(
            row[
                "welfaretype"
            ]
        )

        comparability = (
            normalize_spid_comparability(
                row[
                    "comparability"
                ]
            )
        )

        piece = df[
            df[
                "code"
            ].eq(
                code
            )
            & df[
                "_welfaretype"
            ].eq(
                welfare
            )
            & df[
                "_comparability"
            ].eq(
                comparability
            )
        ].copy()

        pieces.append(
            piece
        )

    panel = pd.concat(
        pieces,
        ignore_index=True,
    )

    key = [
        "code",
        "geo_code",
        "year",
    ]

    duplicated = int(
        panel.duplicated(
            key,
            keep=False,
        ).sum()
    )

    if duplicated:

        raise ValueError(
            f"SPID selected panel still has "
            f"{duplicated} duplicate key rows."
        )

    expected = (
        panel[
            [
                "code",
                "geo_code",
            ]
        ]
        .drop_duplicates()
        .shape[
            0
        ]
        * len(
            YEARS
        )
    )

    if len(
        panel
    ) != expected:

        raise ValueError(
            "SPID selected panel is not a "
            "complete region-year rectangle: "
            f"observed={len(panel)}, "
            f"expected={expected}."
        )

    write_csv(
        panel,
        SPID_OUTPUT,
    )

    print(
        f"SPID selected panel: "
        f"{len(panel):,} rows | "
        f"{panel['code'].nunique()} countries | "
        f"{panel['geo_code'].nunique()} regions"
    )

    return panel


# =============================================================================
# TARGET GEOMETRIES
# =============================================================================

def build_target_geometries(
    spid_panel: pd.DataFrame,
) -> Any:

    if gpd is None:

        raise ImportError(
            "geopandas is required for "
            "regional extraction."
        )

    geometry_file = (
        find_geometry_file()
    )

    regions = gpd.read_file(
        geometry_file
    )

    if "geo_code" not in regions.columns:

        raise ValueError(
            f"geo_code not found in "
            f"{geometry_file}"
        )

    regions[
        "geo_code"
    ] = (
        regions[
            "geo_code"
        ]
        .astype(str)
        .str.strip()
    )

    target_codes = set(
        spid_panel[
            "geo_code"
        ]
        .astype(str)
        .str.strip()
        .unique()
    )

    regions = regions[
        regions[
            "geo_code"
        ].isin(
            target_codes
        )
    ].copy()

    found = set(
        regions[
            "geo_code"
        ]
    )

    missing = sorted(
        target_codes
        - found
    )

    if missing:

        raise ValueError(
            f"{len(missing)} target geo_codes "
            "are missing from the geometry file. "
            f"Examples: {missing[:10]}"
        )

    if regions[
        "geo_code"
    ].duplicated().any():

        duplicate_codes = (
            regions.loc[
                regions[
                    "geo_code"
                ].duplicated(
                    keep=False
                ),
                "geo_code",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate geometry geo_code values: "
            f"{duplicate_codes[:10]}"
        )

    if regions.geometry.isna().any():

        raise ValueError(
            "Missing target geometry."
        )

    if regions.crs is None:

        raise ValueError(
            "Target geometry has no CRS. "
            "Space2Stats and WorldPop require WGS84 coordinates."
        )

    # Both Space2Stats GeoJSON requests and WorldPop v2 polygon requests expect
    # WGS84 longitude/latitude.
    if (
        regions.crs.to_epsg()
        != 4326
    ):

        regions = regions.to_crs(
            epsg=4326
        )

    if (
        ~regions.geometry.is_valid
    ).any():

        invalid_n = int(
            (
                ~regions.geometry.is_valid
            ).sum()
        )

        raise ValueError(
            f"{invalid_n} invalid target "
            "geometries found."
        )

    regions = regions.sort_values(
        [
            column
            for column in [
                "code",
                "geo_code",
            ]
            if column in regions.columns
        ]
    ).reset_index(
        drop=True
    )

    metadata_columns = [
        column
        for column in [
            "code",
            "geo_year",
            "geo_source",
            "geo_level",
            "geo_idvar",
            "geo_id",
            "geo_nvar",
            "geo_name",
            "geo_code",
        ]
        if column in regions.columns
    ]

    write_csv(
        pd.DataFrame(
            regions[
                metadata_columns
            ]
        ),
        TARGET_REGIONS_CSV,
    )

    if TARGET_REGIONS_GPKG.exists():

        TARGET_REGIONS_GPKG.unlink()

    regions.to_file(
        TARGET_REGIONS_GPKG,
        driver="GPKG",
    )

    print(
        f"Target geometries: "
        f"{len(regions):,} regions"
    )

    return regions


# =============================================================================
# SPACE2STATS
# =============================================================================

def extract_years(
    field: str,
) -> set[int]:

    return {
        int(
            value
        )
        for value in re.findall(
            r"(?<!\d)((?:19|20)\d{2})(?!\d)",
            str(
                field
            ),
        )
    }


def classify_s2s_field(
    field: str,
) -> str:

    text = field.lower()

    for family, patterns in (
        SPACE2STATS_RULES
    ):

        if any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        ):

            return family

    return "other"


def get_space2stats_fields() -> pd.DataFrame:

    payload = safe_json_request(
        "GET",
        f"{SPACE2STATS_URL}/fields",
    )

    fields = []

    if isinstance(
        payload,
        list,
    ):

        for item in payload:

            if isinstance(
                item,
                str,
            ):

                fields.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                for key in [
                    "field",
                    "name",
                    "id",
                    "variable",
                    "property",
                ]:

                    if isinstance(
                        item.get(
                            key
                        ),
                        str,
                    ):

                        fields.append(
                            item[
                                key
                            ]
                        )

                        break

    elif isinstance(
        payload,
        dict,
    ):

        for key in [
            "fields",
            "data",
            "results",
        ]:

            if key in payload:

                nested = payload[
                    key
                ]

                if isinstance(
                    nested,
                    list,
                ):

                    for item in nested:

                        if isinstance(
                            item,
                            str,
                        ):

                            fields.append(
                                item
                            )

                        elif isinstance(
                            item,
                            dict,
                        ):

                            for candidate in [
                                "field",
                                "name",
                                "id",
                                "variable",
                                "property",
                            ]:

                                if isinstance(
                                    item.get(
                                        candidate
                                    ),
                                    str,
                                ):

                                    fields.append(
                                        item[
                                            candidate
                                        ]
                                    )

                                    break

    rows = []

    for field in sorted(
        set(
            fields
        )
    ):

        family = (
            classify_s2s_field(
                field
            )
        )

        years = (
            extract_years(
                field
            )
        )

        include = False

        if family in {
            "population",
            "nighttime_lights",
        }:

            include = bool(
                years.intersection(
                    YEARS
                )
            )

        elif family == "built_up_area":

            include = bool(
                years.intersection(
                    {
                        2015,
                        2020,
                    }
                )
            )

        elif family in {
            "urbanization_ghs",
            "flood_exposure",
            "drought",
            "cyclone",
            "wildfire",
            "landslide",
        }:

            include = True

        rows.append(
            {
                "field": field,
                "family": family,
                "years": (
                    ", ".join(
                        str(year)
                        for year in sorted(
                            years
                        )
                    )
                ),
                "include_2015_2020": (
                    include
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def unpack_s2s_result(
    payload: Any,
) -> dict[
    str,
    Any,
]:
    if isinstance(
        payload,
        dict,
    ):

        if (
            "data"
            in payload
            and isinstance(
                payload[
                    "data"
                ],
                dict,
            )
        ):

            return payload[
                "data"
            ]

        return payload

    return {}



def make_s2s_payload(
    geometry: Any,
    fields: list[str],
) -> dict[str, Any]:

    return {
        "aoi": {
            "type": "Feature",
            "geometry": (
                geometry
                .__geo_interface__
            ),
            "properties": {},
        },
        "spatial_join_method": (
            "centroid"
        ),
        "fields": fields,
        "aggregation_type": (
            "sum"
        ),
    }


def request_space2stats(
    geometry: Any,
    fields: list[str],
) -> dict[str, Any]:

    response = safe_json_request(
        "POST",
        (
            f"{SPACE2STATS_URL}"
            "/aggregate"
        ),
        payload=make_s2s_payload(
            geometry,
            fields,
        ),
    )

    return unpack_s2s_result(
        response
    )


def space2stats_geometry_candidates(
    geometry: Any,
) -> list[
    tuple[
        str,
        float,
        Any,
    ]
]:

    candidates = [
        (
            "original",
            0.0,
            geometry,
        )
    ]

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    diagonal = math.hypot(
        maxx - minx,
        maxy - miny,
    )

    if diagonal <= 0:

        return candidates

    seen = {
        geometry.wkb
    }

    for relative in (
        SPACE2STATS_SIMPLIFY_RELATIVE_TOLERANCES
    ):

        tolerance = (
            diagonal
            * relative
        )

        simplified = geometry.simplify(
            tolerance,
            preserve_topology=True,
        )

        if (
            simplified.is_empty
            or not simplified.is_valid
        ):

            continue

        signature = simplified.wkb

        if signature in seen:

            continue

        seen.add(
            signature
        )

        candidates.append(
            (
                "simplified",
                tolerance,
                simplified,
            )
        )

    return candidates


def extract_space2stats_region(
    geometry: Any,
    fields: list[str],
) -> tuple[
    dict[str, Any],
    str,
    float,
    list[str],
]:

    last_error: Exception | None = None
    geometry_candidates = (
        space2stats_geometry_candidates(
            geometry
        )
    )

    # First try the complete field set. This preserves the original request
    # whenever possible. The simplification fallback is intentionally small and
    # topology-preserving; it is used only for regions that return HTTP 500.
    for (
        mode,
        tolerance,
        candidate_geometry,
    ) in geometry_candidates:

        try:

            values = request_space2stats(
                candidate_geometry,
                fields,
            )

            return (
                values,
                mode,
                tolerance,
                [],
            )

        except Exception as exc:

            last_error = exc

    # If a complex region still fails, retry the most simplified valid geometry
    # in field chunks. This prevents one problematic field/backend operation from
    # discarding all other candidate variables for that region.
    (
        fallback_mode,
        fallback_tolerance,
        fallback_geometry,
    ) = geometry_candidates[
        -1
    ]

    combined: dict[
        str,
        Any,
    ] = {}

    failed_fields: list[
        str
    ] = []

    for start in range(
        0,
        len(fields),
        SPACE2STATS_FIELD_CHUNK_SIZE,
    ):

        chunk = fields[
            start:
            start
            + SPACE2STATS_FIELD_CHUNK_SIZE
        ]

        try:

            combined.update(
                request_space2stats(
                    fallback_geometry,
                    chunk,
                )
            )

            continue

        except Exception:

            pass

        # Last-resort isolation: retry each field individually. This is only
        # reached for regions that failed the normal and simplified requests.
        for field in chunk:

            try:

                value = request_space2stats(
                    fallback_geometry,
                    [
                        field
                    ],
                )

                combined.update(
                    value
                )

            except Exception:

                failed_fields.append(
                    field
                )

    if combined:

        return (
            combined,
            (
                "field_chunk_fallback"
            ),
            fallback_tolerance,
            failed_fields,
        )

    raise RuntimeError(
        "Space2Stats region failed under original, "
        "topology-preserving simplification, and "
        "field-isolation fallbacks. "
        f"Last error: {last_error}"
    )


def extract_space2stats(
    regions: Any,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    fields_df = (
        get_space2stats_fields()
    )

    fields = (
        fields_df.loc[
            fields_df[
                "include_2015_2020"
            ],
            "field",
        ]
        .tolist()
    )

    if not fields:

        raise ValueError(
            "No Space2Stats candidate fields "
            "selected for 2015–2020."
        )

    print(
        f"Space2Stats candidate fields: "
        f"{len(fields)}"
    )

    print(
        fields_df[
            fields_df[
                "include_2015_2020"
            ]
        ]
        .groupby(
            "family"
        )
        .size()
        .to_string()
    )

    cached: dict[
        str,
        dict[str, Any],
    ] = {}

    if SPACE2STATS_CHECKPOINT.exists():

        previous = pd.read_csv(
            SPACE2STATS_CHECKPOINT
        )

        previous_columns = set(
            previous.columns
        )

        checkpoint_is_current = (
            set(
                fields
            ).issubset(
                previous_columns
            )
        )

        if checkpoint_is_current:

            for _, row in (
                previous.iterrows()
            ):

                # Cache only fully successful rows. Partial rows are retried on
                # the next run so that a transient backend problem can heal.
                if (
                    row.get(
                        "status"
                    )
                    == "success"
                ):

                    cached[
                        str(
                            row[
                                "geo_code"
                            ]
                        )
                    ] = row.to_dict()

        else:

            print(
                "Existing Space2Stats checkpoint "
                "does not contain the current "
                "field set; it will be rebuilt."
            )

    records = []
    errors = []

    total = len(
        regions
    )

    for position, (
        _,
        row,
    ) in enumerate(
        regions.iterrows(),
        start=1,
    ):

        geo_code = str(
            row[
                "geo_code"
            ]
        )

        if geo_code in cached:

            records.append(
                cached[
                    geo_code
                ]
            )

            print(
                f"[S2S {position:03d}/"
                f"{total:03d}] "
                f"{geo_code} -> cached"
            )

            continue

        try:

            (
                values,
                request_mode,
                tolerance,
                failed_fields,
            ) = (
                extract_space2stats_region(
                    row.geometry,
                    fields,
                )
            )

            status = (
                "success"
                if not failed_fields
                else "partial_fields"
            )

            record = {
                "code": (
                    row.get(
                        "code",
                        "",
                    )
                ),
                "geo_code": geo_code,
                "geo_name": (
                    row.get(
                        "geo_name",
                        "",
                    )
                ),
                "status": status,
                "error": "",
                "request_mode": (
                    request_mode
                ),
                "simplification_tolerance": (
                    tolerance
                ),
                "failed_fields": (
                    " | ".join(
                        failed_fields
                    )
                ),
            }

            for field in fields:

                record[
                    field
                ] = values.get(
                    field
                )

            records.append(
                record
            )

            if failed_fields:

                errors.append(
                    {
                        "geo_code": (
                            geo_code
                        ),
                        "error": (
                            "Partial field recovery. "
                            "Failed fields: "
                            + " | ".join(
                                failed_fields
                            )
                        ),
                    }
                )

                print(
                    f"[S2S {position:03d}/"
                    f"{total:03d}] "
                    f"{geo_code} -> PARTIAL "
                    f"({len(failed_fields)} fields)"
                )

            else:

                print(
                    f"[S2S {position:03d}/"
                    f"{total:03d}] "
                    f"{geo_code} -> success "
                    f"[{request_mode}]"
                )

        except Exception as exc:

            record = {
                "code": (
                    row.get(
                        "code",
                        "",
                    )
                ),
                "geo_code": geo_code,
                "geo_name": (
                    row.get(
                        "geo_name",
                        "",
                    )
                ),
                "status": (
                    "failed"
                ),
                "error": str(
                    exc
                ),
                "request_mode": (
                    "failed_all_fallbacks"
                ),
                "simplification_tolerance": (
                    np.nan
                ),
                "failed_fields": (
                    " | ".join(
                        fields
                    )
                ),
            }

            for field in fields:

                record[
                    field
                ] = np.nan

            records.append(
                record
            )

            errors.append(
                {
                    "geo_code": (
                        geo_code
                    ),
                    "error": str(
                        exc
                    ),
                }
            )

            print(
                f"[S2S {position:03d}/"
                f"{total:03d}] "
                f"{geo_code} -> FAILED"
            )

        if (
            position
            % 20
            == 0
            or position
            == total
        ):

            write_csv(
                pd.DataFrame(
                    records
                ),
                SPACE2STATS_CHECKPOINT,
            )

    result = pd.DataFrame(
        records
    )

    errors_df = pd.DataFrame(
        errors
    )

    write_csv(
        result,
        SPACE2STATS_OUTPUT,
    )

    write_csv(
        errors_df,
        SPACE2STATS_ERRORS,
    )

    return (
        result,
        fields_df,
    )


# =============================================================================
# WORLDPOP AGE-SEX — BULK RASTERS + LOCAL ZONAL AGGREGATION
# =============================================================================


WORLDPOP_LAST_RUN_META: dict[str, Any] = {
    "expected_region_years": 0,
    "completed_region_years": 0,
    "failed_region_years": 0,
    "country_years_processed": 0,
    "country_years_cached": 0,
    "files_downloaded": 0,
    "files_cached": 0,
    "bytes_downloaded": 0,
    "method": "bulk_country_rasters_local_zonal",
}


def worldpop_population_url(
    iso3: str,
    year: int,
) -> str:

    iso_upper = str(iso3).upper()
    iso_lower = iso_upper.lower()

    filename = (
        f"{iso_lower}_pop_{year}_CN_1km_"
        f"{WORLDPOP_RELEASE}_{WORLDPOP_UN_ADJUSTMENT_TAG}_"
        f"{WORLDPOP_VERSION}.tif"
    )

    return (
        "https://data.worldpop.org/GIS/Population/"
        "Global_2015_2030/"
        f"{WORLDPOP_RELEASE}/{year}/{iso_upper}/"
        f"{WORLDPOP_VERSION}/1km_ua/constrained/"
        f"{filename}"
    )


def worldpop_age_url(
    iso3: str,
    year: int,
    age_code: str,
) -> str:

    iso_upper = str(iso3).upper()
    iso_lower = iso_upper.lower()

    filename = (
        f"{iso_lower}_t_{age_code}_{year}_CN_1km_"
        f"{WORLDPOP_RELEASE}_{WORLDPOP_UN_ADJUSTMENT_TAG}_"
        f"{WORLDPOP_VERSION}.tif"
    )

    return (
        "https://data.worldpop.org/GIS/AgeSex_structures/"
        "Global_2015_2030/"
        f"{WORLDPOP_RELEASE}/{year}/{iso_upper}/"
        f"{WORLDPOP_VERSION}/1km_ua/constrained/"
        f"{filename}"
    )


def worldpop_local_path(
    iso3: str,
    year: int,
    kind: str,
    age_code: str | None = None,
) -> Path:

    folder = (
        WORLDPOP_BULK_DIR
        / str(iso3).upper()
        / str(year)
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    iso_lower = str(iso3).lower()

    if kind == "population":

        filename = (
            f"{iso_lower}_pop_{year}_CN_1km_"
            f"{WORLDPOP_RELEASE}_{WORLDPOP_UN_ADJUSTMENT_TAG}_"
            f"{WORLDPOP_VERSION}.tif"
        )

    elif kind == "age":

        if age_code is None:
            raise ValueError(
                "age_code is required for a WorldPop age raster."
            )

        filename = (
            f"{iso_lower}_t_{age_code}_{year}_CN_1km_"
            f"{WORLDPOP_RELEASE}_{WORLDPOP_UN_ADJUSTMENT_TAG}_"
            f"{WORLDPOP_VERSION}.tif"
        )

    else:
        raise ValueError(
            f"Unsupported WorldPop raster kind: {kind}"
        )

    return folder / filename


def download_worldpop_raster(
    url: str,
    destination: Path,
) -> dict[str, Any]:

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        return {
            "url": url,
            "path": str(destination),
            "status": "cached",
            "bytes": int(destination.stat().st_size),
        }

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial = destination.with_suffix(
        destination.suffix + ".part"
    )

    if partial.exists():
        partial.unlink()

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            with SESSION.get(
                url,
                stream=True,
                timeout=(
                    WORLDPOP_DOWNLOAD_CONNECT_TIMEOUT,
                    WORLDPOP_DOWNLOAD_READ_TIMEOUT,
                ),
            ) as response:

                response.raise_for_status()

                with partial.open("wb") as handle:

                    for chunk in response.iter_content(
                        chunk_size=(
                            WORLDPOP_DOWNLOAD_CHUNK_BYTES
                        )
                    ):

                        if chunk:
                            handle.write(chunk)

            if (
                not partial.exists()
                or partial.stat().st_size <= 0
            ):
                raise RuntimeError(
                    f"Downloaded empty WorldPop raster: {url}"
                )

            partial.replace(destination)

            return {
                "url": url,
                "path": str(destination),
                "status": "downloaded",
                "bytes": int(destination.stat().st_size),
            }

        except Exception as exc:

            last_error = exc

            if partial.exists():
                partial.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(
                    BACKOFF_SECONDS
                    * attempt
                )

    raise RuntimeError(
        "WorldPop raster download failed after "
        f"{MAX_RETRIES} attempts: {url} | {last_error}"
    )


def build_worldpop_region_masks(
    population_raster: Path,
    country_regions: Any,
) -> tuple[
    Any,
    dict[str, dict[str, Any]],
]:

    if (
        rasterio is None
        or geometry_window is None
        or geometry_mask is None
        or mapping is None
    ):
        raise ImportError(
            "WorldPop bulk extraction requires rasterio. "
            "Install it in the project venv with: pip install rasterio"
        )

    with rasterio.open(
        population_raster
    ) as src:

        target_crs = src.crs

        if target_crs is None:
            raise RuntimeError(
                f"WorldPop raster has no CRS: {population_raster}"
            )

        local_regions = (
            country_regions
            if country_regions.crs == target_crs
            else country_regions.to_crs(target_crs)
        )

        masks: dict[
            str,
            dict[str, Any],
        ] = {}

        for _, row in local_regions.iterrows():

            geo_code = str(
                row["geo_code"]
            )

            geom = row.geometry

            if (
                geom is None
                or geom.is_empty
            ):
                masks[geo_code] = {
                    "error": "EMPTY_GEOMETRY",
                }
                continue

            try:

                window = geometry_window(
                    src,
                    [mapping(geom)],
                    pad_x=0,
                    pad_y=0,
                )

                window = window.round_offsets().round_lengths()

                out_shape = (
                    int(window.height),
                    int(window.width),
                )

                if (
                    out_shape[0] <= 0
                    or out_shape[1] <= 0
                ):
                    raise ValueError(
                        "Zero-sized raster window."
                    )

                transform = src.window_transform(
                    window
                )

                inside = geometry_mask(
                    [mapping(geom)],
                    out_shape=out_shape,
                    transform=transform,
                    invert=True,
                    all_touched=False,
                )

                masks[geo_code] = {
                    "window": window,
                    "inside": inside,
                    "error": "",
                }

            except Exception as exc:

                masks[geo_code] = {
                    "error": str(exc),
                }

    return local_regions, masks


def zonal_sums_from_raster(
    raster_path: Path,
    region_masks: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, float]:

    if rasterio is None:
        raise ImportError(
            "rasterio is required for WorldPop local zonal sums."
        )

    sums: dict[str, float] = {}

    with rasterio.open(
        raster_path
    ) as src:

        for (
            geo_code,
            info,
        ) in region_masks.items():

            if info.get("error"):
                sums[geo_code] = np.nan
                continue

            try:

                data = src.read(
                    1,
                    window=info["window"],
                    masked=True,
                )

                inside = info["inside"]

                # Raster and mask should share the same Global2 grid for a
                # country-year. Defensive clipping handles rare one-pixel edge
                # inconsistencies without silently changing the geometry.
                rows = min(
                    data.shape[0],
                    inside.shape[0],
                )
                cols = min(
                    data.shape[1],
                    inside.shape[1],
                )

                data = data[
                    :rows,
                    :cols,
                ]
                inside = inside[
                    :rows,
                    :cols,
                ]

                values = np.asarray(
                    data.filled(np.nan),
                    dtype="float64",
                )

                valid = (
                    inside
                    & np.isfinite(values)
                )

                if not valid.any():
                    sums[geo_code] = 0.0
                else:
                    sums[geo_code] = float(
                        values[valid].sum()
                    )

            except Exception:
                sums[geo_code] = np.nan

    return sums


def archive_legacy_worldpop_checkpoint() -> None:

    if not WORLDPOP_AGESEX_CHECKPOINT.exists():
        return

    try:
        preview = pd.read_csv(
            WORLDPOP_AGESEX_CHECKPOINT,
            nrows=10,
        )
    except Exception:
        return

    method_col = (
        "worldpop_method"
        in preview.columns
    )

    is_bulk = (
        method_col
        and preview["worldpop_method"]
        .astype(str)
        .str.contains(
            "bulk_country_rasters",
            na=False,
        )
        .any()
    )

    if is_bulk:
        return

    stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    legacy = (
        WORLDPOP_AGESEX_CHECKPOINT
        .with_name(
            "02_worldpop_agesex_checkpoint_API_legacy_"
            f"{stamp}.csv"
        )
    )

    shutil.copy2(
        WORLDPOP_AGESEX_CHECKPOINT,
        legacy,
    )

    WORLDPOP_AGESEX_CHECKPOINT.unlink()

    print(
        "WorldPop: previous API checkpoint archived as "
        f"{legacy.name}. Bulk extraction will rebuild a "
        "consistent R2025A dataset."
    )


def extract_worldpop_agesex(
    regions: Any,
) -> pd.DataFrame:

    global WORLDPOP_LAST_RUN_META

    if gpd is None:
        raise ImportError(
            "geopandas is required for WorldPop extraction."
        )

    if rasterio is None:
        raise ImportError(
            "WorldPop bulk extraction requires rasterio. "
            "Run once in the project venv: pip install rasterio"
        )

    archive_legacy_worldpop_checkpoint()

    WORLDPOP_BULK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_region_years = (
        len(regions)
        * len(YEARS)
    )

    cached_records: list[
        dict[str, Any]
    ] = []

    if WORLDPOP_AGESEX_CHECKPOINT.exists():

        previous = pd.read_csv(
            WORLDPOP_AGESEX_CHECKPOINT
        )

        if not previous.empty:
            cached_records = (
                previous
                .to_dict(
                    orient="records"
                )
            )

    records = cached_records.copy()

    success_keys = {
        (
            str(row.get("geo_code")),
            int(row.get("year")),
        )
        for row in cached_records
        if (
            str(row.get("status"))
            == "success"
            and str(
                row.get(
                    "worldpop_method",
                    "",
                )
            ).startswith(
                "bulk_country_rasters"
            )
        )
    }

    errors: list[
        dict[str, Any]
    ] = []

    download_log: list[
        dict[str, Any]
    ] = []

    files_downloaded = 0
    files_cached = 0
    bytes_downloaded = 0
    country_years_processed = 0
    country_years_cached = 0

    countries = sorted(
        regions["code"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    total_country_years = (
        len(countries)
        * len(YEARS)
    )

    cy_counter = 0

    for iso3 in countries:

        country_regions = regions[
            regions["code"]
            .astype(str)
            .str.upper()
            .eq(iso3)
        ].copy()

        if country_regions.empty:
            continue

        geo_codes = (
            country_regions["geo_code"]
            .astype(str)
            .tolist()
        )

        for year in YEARS:

            cy_counter += 1

            expected_keys = {
                (geo_code, year)
                for geo_code in geo_codes
            }

            if expected_keys.issubset(
                success_keys
            ):

                country_years_cached += 1

                print(
                    f"[WP BULK {cy_counter:03d}/"
                    f"{total_country_years:03d}] "
                    f"{iso3} {year} -> cached"
                )

                continue

            print(
                f"[WP BULK {cy_counter:03d}/"
                f"{total_country_years:03d}] "
                f"{iso3} {year} -> downloading 11 rasters"
            )

            population_path = worldpop_local_path(
                iso3,
                year,
                "population",
            )

            population_url = worldpop_population_url(
                iso3,
                year,
            )

            age_codes = (
                list(WORLDPOP_AGE_0_14)
                + list(WORLDPOP_AGE_65_PLUS)
            )

            age_specs = [
                (
                    age_code,
                    worldpop_age_url(
                        iso3,
                        year,
                        age_code,
                    ),
                    worldpop_local_path(
                        iso3,
                        year,
                        "age",
                        age_code,
                    ),
                )
                for age_code in age_codes
            ]

            try:

                pop_log = download_worldpop_raster(
                    population_url,
                    population_path,
                )

                download_log.append(
                    {
                        "iso3": iso3,
                        "year": year,
                        "kind": "population",
                        "age_code": "",
                        **pop_log,
                    }
                )

                if pop_log["status"] == "downloaded":
                    files_downloaded += 1
                    bytes_downloaded += int(
                        pop_log["bytes"]
                    )
                else:
                    files_cached += 1

                with ThreadPoolExecutor(
                    max_workers=(
                        WORLDPOP_DOWNLOAD_WORKERS
                    )
                ) as executor:

                    futures = {
                        executor.submit(
                            download_worldpop_raster,
                            url,
                            path,
                        ): (
                            age_code,
                            url,
                            path,
                        )
                        for (
                            age_code,
                            url,
                            path,
                        ) in age_specs
                    }

                    for future in as_completed(
                        futures
                    ):

                        age_code, _, _ = (
                            futures[future]
                        )

                        file_log = future.result()

                        download_log.append(
                            {
                                "iso3": iso3,
                                "year": year,
                                "kind": "age",
                                "age_code": age_code,
                                **file_log,
                            }
                        )

                        if (
                            file_log["status"]
                            == "downloaded"
                        ):
                            files_downloaded += 1
                            bytes_downloaded += int(
                                file_log["bytes"]
                            )
                        else:
                            files_cached += 1

                _, masks = build_worldpop_region_masks(
                    population_path,
                    country_regions,
                )

                population_sums = zonal_sums_from_raster(
                    population_path,
                    masks,
                )

                young = {
                    geo_code: 0.0
                    for geo_code in geo_codes
                }

                older = {
                    geo_code: 0.0
                    for geo_code in geo_codes
                }

                for (
                    age_code,
                    _,
                    raster_path,
                ) in age_specs:

                    age_sums = zonal_sums_from_raster(
                        raster_path,
                        masks,
                    )

                    target = (
                        young
                        if age_code
                        in WORLDPOP_AGE_0_14
                        else older
                    )

                    for geo_code in geo_codes:

                        value = age_sums.get(
                            geo_code,
                            np.nan,
                        )

                        if np.isfinite(value):
                            target[geo_code] += float(
                                value
                            )
                        else:
                            target[geo_code] = np.nan

                metadata = (
                    country_regions
                    .set_index("geo_code")
                )

                # Replace any stale record for this country-year before adding
                # the new bulk-raster result.
                records = [
                    item
                    for item in records
                    if not (
                        str(
                            item.get("geo_code")
                        ) in set(geo_codes)
                        and int(
                            item.get(
                                "year",
                                -1,
                            )
                        ) == year
                    )
                ]

                for geo_code in geo_codes:

                    pop = population_sums.get(
                        geo_code,
                        np.nan,
                    )
                    age_0_14 = young.get(
                        geo_code,
                        np.nan,
                    )
                    age_65_plus = older.get(
                        geo_code,
                        np.nan,
                    )

                    mask_error = masks.get(
                        geo_code,
                        {},
                    ).get(
                        "error",
                        "",
                    )

                    values_ok = all(
                        np.isfinite(value)
                        for value in [
                            pop,
                            age_0_14,
                            age_65_plus,
                        ]
                    )

                    if (
                        not values_ok
                        or mask_error
                    ):

                        error_text = (
                            mask_error
                            or "NONFINITE_ZONAL_SUM"
                        )

                        records.append(
                            {
                                "code": iso3,
                                "geo_code": geo_code,
                                "geo_name": (
                                    metadata.loc[
                                        geo_code
                                    ].get(
                                        "geo_name",
                                        "",
                                    )
                                    if geo_code
                                    in metadata.index
                                    else ""
                                ),
                                "year": year,
                                "status": "failed",
                                "error": error_text,
                                "worldpop_method": (
                                    "bulk_country_rasters_"
                                    "local_zonal"
                                ),
                                "worldpop_release": (
                                    f"{WORLDPOP_RELEASE}_"
                                    f"{WORLDPOP_VERSION}"
                                ),
                                "worldpop_resolution": (
                                    WORLDPOP_RESOLUTION
                                ),
                                "worldpop_files_used": 11,
                                "worldpop_api_calls": 0,
                            }
                        )

                        errors.append(
                            {
                                "iso3": iso3,
                                "geo_code": geo_code,
                                "year": year,
                                "error": error_text,
                            }
                        )

                        continue

                    working = (
                        float(pop)
                        - float(age_0_14)
                        - float(age_65_plus)
                    )

                    # Tiny negatives may appear from independent raster
                    # compression/rounding; material negatives are flagged.
                    if working < 0:

                        tolerance = max(
                            1.0,
                            abs(float(pop))
                            * 1e-5,
                        )

                        if abs(working) <= tolerance:
                            working = 0.0
                        else:
                            errors.append(
                                {
                                    "iso3": iso3,
                                    "geo_code": geo_code,
                                    "year": year,
                                    "error": (
                                        "NEGATIVE_WORKING_AGE_"
                                        f"{working}"
                                    ),
                                }
                            )

                    dependency = (
                        (
                            float(age_0_14)
                            + float(age_65_plus)
                        )
                        / working
                        * 100.0
                        if working > 0
                        else np.nan
                    )

                    share_0_14 = (
                        float(age_0_14)
                        / float(pop)
                        if float(pop) > 0
                        else np.nan
                    )

                    share_65_plus = (
                        float(age_65_plus)
                        / float(pop)
                        if float(pop) > 0
                        else np.nan
                    )

                    records.append(
                        {
                            "code": iso3,
                            "geo_code": geo_code,
                            "geo_name": (
                                metadata.loc[
                                    geo_code
                                ].get(
                                    "geo_name",
                                    "",
                                )
                                if geo_code
                                in metadata.index
                                else ""
                            ),
                            "year": year,
                            "status": "success",
                            "error": "",
                            "worldpop_method": (
                                "bulk_country_rasters_"
                                "local_zonal"
                            ),
                            "worldpop_release": (
                                f"{WORLDPOP_RELEASE}_"
                                f"{WORLDPOP_VERSION}"
                            ),
                            "worldpop_resolution": (
                                WORLDPOP_RESOLUTION
                            ),
                            "worldpop_files_used": 11,
                            "worldpop_api_calls": 0,
                            "wp_age_0_14": float(
                                age_0_14
                            ),
                            "wp_age_15_64": float(
                                working
                            ),
                            "wp_age_65_plus": float(
                                age_65_plus
                            ),
                            "wp_age_population": float(
                                pop
                            ),
                            "age_dependency_ratio": (
                                dependency
                            ),
                            "share_0_14": share_0_14,
                            "share_65_plus": (
                                share_65_plus
                            ),
                        }
                    )

                    success_keys.add(
                        (
                            geo_code,
                            year,
                        )
                    )

                country_years_processed += 1

                checkpoint_df = (
                    pd.DataFrame(records)
                    .sort_values(
                        [
                            "code",
                            "geo_code",
                            "year",
                        ],
                        kind="stable",
                    )
                    .drop_duplicates(
                        subset=[
                            "geo_code",
                            "year",
                        ],
                        keep="last",
                    )
                    .reset_index(drop=True)
                )

                write_csv(
                    checkpoint_df,
                    WORLDPOP_AGESEX_CHECKPOINT,
                )

                write_csv(
                    pd.DataFrame(download_log),
                    WORLDPOP_BULK_DOWNLOAD_LOG,
                )

                success_count = sum(
                    (
                        geo_code,
                        year,
                    ) in success_keys
                    for geo_code in geo_codes
                )

                print(
                    f"    -> {success_count}/"
                    f"{len(geo_codes)} regions aggregated locally"
                )

            except Exception as exc:

                error_text = str(exc)

                errors.append(
                    {
                        "iso3": iso3,
                        "geo_code": "",
                        "year": year,
                        "error": error_text,
                    }
                )

                print(
                    f"    -> FAILED: {error_text}"
                )

            finally:

                if not WORLDPOP_KEEP_RASTERS:

                    folder = (
                        WORLDPOP_BULK_DIR
                        / iso3
                        / str(year)
                    )

                    if folder.exists():

                        for tif in folder.glob(
                            "*.tif"
                        ):
                            try:
                                tif.unlink()
                            except Exception:
                                pass

                        for partial in folder.glob(
                            "*.part"
                        ):
                            try:
                                partial.unlink()
                            except Exception:
                                pass

    result = pd.DataFrame(
        records
    )

    if not result.empty:
        result = (
            result
            .sort_values(
                [
                    "code",
                    "geo_code",
                    "year",
                ],
                kind="stable",
            )
            .drop_duplicates(
                subset=[
                    "geo_code",
                    "year",
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )

    write_csv(
        result,
        WORLDPOP_AGESEX_CHECKPOINT,
    )

    write_csv(
        result,
        WORLDPOP_AGESEX_OUTPUT,
    )

    write_csv(
        pd.DataFrame(errors),
        WORLDPOP_AGESEX_ERRORS,
    )

    write_csv(
        pd.DataFrame(download_log),
        WORLDPOP_BULK_DOWNLOAD_LOG,
    )

    completed = (
        int(
            (
                result.get(
                    "status",
                    pd.Series(dtype=str),
                )
                == "success"
            ).sum()
        )
        if not result.empty
        else 0
    )

    failed = (
        int(
            (
                result.get(
                    "status",
                    pd.Series(dtype=str),
                )
                == "failed"
            ).sum()
        )
        if not result.empty
        else 0
    )

    WORLDPOP_LAST_RUN_META = {
        "expected_region_years": expected_region_years,
        "completed_region_years": completed,
        "failed_region_years": failed,
        "country_years_processed": country_years_processed,
        "country_years_cached": country_years_cached,
        "files_downloaded": files_downloaded,
        "files_cached": files_cached,
        "bytes_downloaded": bytes_downloaded,
        "method": "bulk_country_rasters_local_zonal",
    }

    return result


# =============================================================================
# SSGD
# =============================================================================

def extract_ssgd(
    countries: list[str],
) -> pd.DataFrame:

    if not SSGD_FILE.exists():

        raise FileNotFoundError(
            SSGD_FILE
        )

    excel = pd.ExcelFile(
        SSGD_FILE
    )

    df = None

    for sheet in (
        excel.sheet_names
    ):

        preview = pd.read_excel(
            SSGD_FILE,
            sheet_name=sheet,
            nrows=5,
        )

        if (
            find_column(
                preview,
                [
                    "period",
                    "year",
                ],
            )
            and find_column(
                preview,
                [
                    "value",
                    "val",
                ],
            )
        ):

            df = pd.read_excel(
                SSGD_FILE,
                sheet_name=sheet,
            )

            break

    if df is None:

        raise ValueError(
            "No SSGD long-format sheet found."
        )

    area_col = find_column(
        df,
        [
            "area",
        ],
    )

    year_col = find_column(
        df,
        [
            "period",
            "year",
        ],
    )

    indicator_col = find_column(
        df,
        [
            "short",
            "indicator_code",
            "indicatorcode",
            "indicator",
            "variable",
        ],
    )

    country_col = find_column(
        df,
        [
            "countrycode",
            "country_code",
            "iso3",
            "code",
        ],
    )

    if not (
        year_col
        and indicator_col
    ):

        raise ValueError(
            "SSGD year/indicator columns "
            "could not be identified."
        )

    if area_col:

        df = df[
            df[
                area_col
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(
                "subnational"
            )
        ].copy()

    # SSGD stores the period as strings such as "Year 2018". A direct
    # pd.to_numeric() therefore returns NaN and previously produced a false
    # SUCCESS with zero rows. Parse a four-digit year when needed.
    raw_period = (
        df[
            year_col
        ]
    )

    numeric_year = pd.to_numeric(
        raw_period,
        errors="coerce",
    )

    extracted_year = pd.to_numeric(
        raw_period
        .astype(str)
        .str.extract(
            r"(?<!\d)((?:19|20)\d{2})(?!\d)",
            expand=False,
        ),
        errors="coerce",
    )

    df[
        "_year"
    ] = (
        numeric_year
        .fillna(
            extracted_year
        )
        .astype(
            "Int64"
        )
    )

    df = df[
        df[
            "_year"
        ].isin(
            YEARS
        )
    ].copy()

    df = df[
        df[
            indicator_col
        ]
        .astype(str)
        .str.strip()
        .isin(
            SSGD_CODES
        )
    ].copy()

    if country_col:

        df[
            "_country"
        ] = (
            df[
                country_col
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Only apply if SSGD country identifiers actually look like ISO3.
        overlap = set(
            df[
                "_country"
            ].dropna()
        ).intersection(
            countries
        )

        if overlap:

            df = df[
                df[
                    "_country"
                ].isin(
                    countries
                )
            ].copy()

    df[
        "_candidate_family"
    ] = (
        df[
            indicator_col
        ]
        .astype(str)
        .str.strip()
        .map(
            SSGD_CODES
        )
    )

    if df.empty:

        raise ValueError(
            "SSGD extraction returned zero rows after filtering. "
            "Check period parsing, candidate codes, and country scope."
        )

    write_csv(
        df,
        SSGD_OUTPUT,
    )

    return df


# =============================================================================
# WDI
# =============================================================================

def extract_wdi(
    countries: list[str],
) -> pd.DataFrame:

    rows = []

    country_path = ";".join(
        countries
    )

    total = len(
        WDI_CODES
    )

    for index, (
        code,
        variable_id,
    ) in enumerate(
        WDI_CODES.items(),
        start=1,
    ):

        print(
            f"[WDI {index:02d}/"
            f"{total:02d}] "
            f"{code}"
        )

        payload = (
            safe_json_request(
                "GET",
                (
                    f"{WDI_API}/country/"
                    f"{country_path}"
                    f"/indicator/{code}"
                ),
                params={
                    "format": (
                        "json"
                    ),
                    "date": (
                        f"{START_YEAR}:"
                        f"{END_YEAR}"
                    ),
                    "per_page": (
                        20000
                    ),
                },
            )
        )

        if (
            not isinstance(
                payload,
                list,
            )
            or len(
                payload
            )
            < 2
            or payload[
                1
            ]
            is None
        ):

            continue

        for item in payload[
            1
        ]:

            rows.append(
                {
                    "country_code": (
                        str(
                            item.get(
                                "countryiso3code",
                                "",
                            )
                        )
                        .strip()
                        .upper()
                    ),
                    "year": (
                        pd.to_numeric(
                            item.get(
                                "date"
                            ),
                            errors="coerce",
                        )
                    ),
                    "indicator_code": (
                        code
                    ),
                    "variable_id": (
                        variable_id
                    ),
                    "value": (
                        pd.to_numeric(
                            item.get(
                                "value"
                            ),
                            errors="coerce",
                        )
                    ),
                    "unit": (
                        item.get(
                            "unit",
                            "",
                        )
                    ),
                    "obs_status": (
                        item.get(
                            "obs_status",
                            "",
                        )
                    ),
                    "decimal": (
                        item.get(
                            "decimal",
                            "",
                        )
                    ),
                }
            )

    df = pd.DataFrame(
        rows
    )

    if not df.empty:

        df[
            "year"
        ] = pd.to_numeric(
            df[
                "year"
            ],
            errors="coerce",
        ).astype(
            "Int64"
        )

        df = df[
            df[
                "country_code"
            ].isin(
                countries
            )
            & df[
                "year"
            ].isin(
                YEARS
            )
        ].copy()

    write_csv(
        df,
        WDI_OUTPUT,
    )

    return df


# =============================================================================
# PUBLIC ZENODO SOURCES
# =============================================================================

def candidate_country_columns(
    df: pd.DataFrame,
) -> list[str]:

    preferred = [
        "iso3",
        "ISO3",
        "iso3c",
        "ISO3C",
        "country_code",
        "CountryCode",
        "countrycode",
        "code",
        "GID_0",
        "adm0_pcode",
        "country_iso3",
        "country_iso",
        "WB_A3",
        "wb_a3",
        "ADM0_A3",
        "adm0_a3",
    ]

    result = [
        column
        for column in preferred
        if column in df.columns
    ]

    if result:

        return result

    return [
        column
        for column in df.columns
        if (
            "iso3"
            in str(
                column
            ).lower()
            or "country"
            in str(
                column
            ).lower()
            and "code"
            in str(
                column
            ).lower()
        )
    ]


def subset_tabular_scope(
    source_file: Path,
    countries: list[str],
    output_file: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        source_file,
        low_memory=False,
    )

    country_columns = (
        candidate_country_columns(
            df
        )
    )

    filtered = df.copy()

    for column in (
        country_columns
    ):

        values = (
            filtered[
                column
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        overlap = set(
            values
        ).intersection(
            countries
        )

        if overlap:

            filtered = filtered[
                values.isin(
                    countries
                )
            ].copy()

            break

    year_col = find_column(
        filtered,
        [
            "year",
            "Year",
            "YEAR",
            "time_period",
            "TIME_PERIOD",
            "period",
            "Period",
        ],
    )

    if year_col:

        raw_year = (
            filtered[
                year_col
            ]
        )

        numeric_year = pd.to_numeric(
            raw_year,
            errors="coerce",
        )

        extracted_year = pd.to_numeric(
            raw_year
            .astype(str)
            .str.extract(
                r"(?<!\d)((?:19|20)\d{2})(?!\d)",
                expand=False,
            ),
            errors="coerce",
        )

        numeric_year = (
            numeric_year
            .fillna(
                extracted_year
            )
        )

        if numeric_year.notna().any():

            filtered = filtered[
                numeric_year.isin(
                    YEARS
                )
            ].copy()

    else:

        all_year_columns = [
            column
            for column in filtered.columns
            if re.search(
                r"(?<!\d)(?:19|20)\d{2}(?!\d)",
                str(column),
            )
        ]

        year_columns = [
            column
            for column in all_year_columns
            if any(
                str(year) in str(column)
                for year in YEARS
            )
        ]

        if year_columns:

            identifier_columns = [
                column
                for column in filtered.columns
                if column not in all_year_columns
            ]

            # Preserve non-year metadata columns plus only the six official
            # target-year columns. Script 03 will reshape if necessary.
            filtered = filtered[
                identifier_columns
                + year_columns
            ].copy()

    write_csv(
        filtered,
        output_file,
    )

    return filtered


def download_public_source(
    source_id: str,
) -> list[Path]:

    specification = (
        PUBLIC_DOWNLOADS[
            source_id
        ]
    )

    directory = (
        DOWNLOAD_DIR
        / source_id.lower()
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    downloaded = []

    for file_info in (
        specification[
            "files"
        ]
    ):

        destination = (
            directory
            / file_info[
                "name"
            ]
        )

        print(
            f"Downloading "
            f"{source_id}: "
            f"{destination.name}"
        )

        downloaded.append(
            download_file(
                file_info[
                    "url"
                ],
                destination,
            )
        )

    return downloaded


def extract_kummu(
    countries: list[str],
) -> pd.DataFrame:

    files = (
        download_public_source(
            "KUMMU_GDP"
        )
    )

    tabular = next(
        path
        for path in files
        if path.name.lower().startswith(
            "tabulated"
        )
    )

    return subset_tabular_scope(
        tabular,
        countries,
        KUMMU_OUTPUT,
    )


def extract_dose(
    countries: list[str],
) -> pd.DataFrame:

    files = (
        download_public_source(
            "DOSE"
        )
    )

    source_file = files[
        0
    ]

    return subset_tabular_scope(
        source_file,
        countries,
        DOSE_OUTPUT,
    )


def extract_niva(
    countries: list[str],
) -> Any:

    if gpd is None:

        raise ImportError(
            "geopandas is required for "
            "Niva migration GPKG."
        )

    files = (
        download_public_source(
            "NIVA_MIGRATION"
        )
    )

    source_file = files[
        0
    ]

    gdf = gpd.read_file(
        source_file
    )

    country_columns = (
        candidate_country_columns(
            pd.DataFrame(
                gdf.drop(
                    columns="geometry"
                )
            )
        )
    )

    filtered = gdf.copy()

    for column in (
        country_columns
    ):

        values = (
            filtered[
                column
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        overlap = set(
            values
        ).intersection(
            countries
        )

        if overlap:

            filtered = filtered[
                values.isin(
                    countries
                )
            ].copy()

            break

    # Keep only target-year fields where the dataset is wide, but preserve all
    # identification/metadata columns.
    year_columns = [
        column
        for column in filtered.columns
        if (
            column
            != "geometry"
            and any(
                str(
                    year
                )
                in str(
                    column
                )
                for year in range(
                    2015,
                    2020,
                )
            )
        )
    ]

    if year_columns:

        keep = [
            column
            for column in filtered.columns
            if (
                column
                == "geometry"
                or not re.search(
                    r"(?:19|20)\d{2}",
                    str(
                        column
                    ),
                )
                or column
                in year_columns
            )
        ]

        filtered = filtered[
            keep
        ].copy()

    if NIVA_OUTPUT.exists():

        NIVA_OUTPUT.unlink()

    filtered.to_file(
        NIVA_OUTPUT,
        driver="GPKG",
    )

    return filtered


def extract_ookla_wb(
    countries: list[str],
) -> pd.DataFrame:

    files = (
        download_public_source(
            "OOKLA_WB"
        )
    )

    source_file = files[
        0
    ]

    return subset_tabular_scope(
        source_file,
        countries,
        OOKLA_OUTPUT,
    )



# =============================================================================
# SUBNATIONAL CORRUPTION DATABASE — PUBLIC FIGSHARE ADAPTER
# =============================================================================


def filter_generic_dataframe_scope(
    df: pd.DataFrame,
    countries: list[str],
) -> pd.DataFrame:

    filtered = df.copy()

    country_columns = candidate_country_columns(
        filtered
    )

    for column in country_columns:

        values = (
            filtered[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        overlap = set(values).intersection(
            countries
        )

        if overlap:

            filtered = filtered[
                values.isin(countries)
            ].copy()

            break

    year_col = find_column(
        filtered,
        [
            "year",
            "Year",
            "YEAR",
        ],
    )

    if year_col:

        years_numeric = pd.to_numeric(
            filtered[year_col],
            errors="coerce",
        )

        if years_numeric.notna().any():

            filtered = filtered[
                years_numeric.isin(YEARS)
            ].copy()

    return filtered


def extract_gdl_scd(
    countries: list[str],
) -> dict[str, pd.DataFrame]:

    # Public Figshare deposit cited by the peer-reviewed data descriptor.
    article_api = (
        "https://api.figshare.com/v2/articles/25893019"
    )

    metadata = safe_json_request(
        "GET",
        article_api,
    )

    files = metadata.get(
        "files",
        [],
    ) if isinstance(metadata, dict) else []

    if not files:

        raise RuntimeError(
            "Figshare returned no files for the "
            "Subnational Corruption Database."
        )

    # The article is distributed as one archive in the public deposit.
    file_info = sorted(
        files,
        key=lambda item: item.get("size", 0),
        reverse=True,
    )[0]

    download_url = (
        file_info.get("download_url")
        or file_info.get("url_private_api")
    )

    if not download_url:

        raise RuntimeError(
            "No public Figshare download URL was returned."
        )

    archive_name = (
        file_info.get("name")
        or "subnational_corruption_database.zip"
    )

    archive_path = (
        DOWNLOAD_DIR
        / "gdl_scd"
        / archive_name
    )

    download_file(
        download_url,
        archive_path,
    )

    extract_dir = (
        DOWNLOAD_DIR
        / "gdl_scd"
        / "extracted"
    )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if zipfile.is_zipfile(
        archive_path
    ):

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:

            archive.extractall(
                extract_dir
            )

    else:

        raise RuntimeError(
            f"Expected a ZIP archive from Figshare, got: "
            f"{archive_path.name}"
        )

    spreadsheet_files = sorted(
        list(extract_dir.rglob("*.xlsx"))
        + list(extract_dir.rglob("*.csv"))
    )

    baseline_file = next(
        (
            path
            for path in spreadsheet_files
            if "baseline" in path.name.lower()
            and "sci" in path.name.lower()
        ),
        None,
    )

    comprehensive_file = next(
        (
            path
            for path in spreadsheet_files
            if "comprehensive" in path.name.lower()
            and "sci" in path.name.lower()
        ),
        None,
    )

    results: dict[str, pd.DataFrame] = {}

    for label, source_file, output_file in [
        (
            "baseline",
            baseline_file,
            GDL_SCD_BASELINE_OUTPUT,
        ),
        (
            "comprehensive",
            comprehensive_file,
            GDL_SCD_COMPREHENSIVE_OUTPUT,
        ),
    ]:

        if source_file is None:

            continue

        if source_file.suffix.lower() == ".csv":

            frame = pd.read_csv(
                source_file,
                low_memory=False,
            )

        else:

            frame = pd.read_excel(
                source_file
            )

        frame = filter_generic_dataframe_scope(
            frame,
            countries,
        )

        write_csv(
            frame,
            output_file,
        )

        results[label] = frame

    if not results:

        raise RuntimeError(
            "The corruption archive was downloaded, but "
            "Baseline/Comprehensive Excel or CSV files "
            "could not be identified automatically."
        )

    return results


# =============================================================================
# GDL SHDI + OECD REGIONAL LABOUR
# =============================================================================

GDL_SHDI_CORE_COLUMNS = [
    "iso_code",
    "country",
    "year",
    "gdlcode",
    "level",
    "region",
    "shdi",
    "healthindex",
    "incindex",
    "edindex",
    "lifexp",
    "lgnic",
    "esch",
    "msch",
]


def _normalized_column_name(value: Any) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def _gdl_header_score(columns: Iterable[Any]) -> int:

    normalized = {
        _normalized_column_name(column)
        for column in columns
    }

    expected_groups = [
        {"isocode", "iso3", "iso3c", "countrycode", "countryiso3"},
        {"country", "countryname"},
        {"year", "time", "timeperiod"},
        {"gdlcode", "regioncode"},
        {"level", "aggregationlevel"},
        {"region", "regionname"},
        {"shdi"},
        {"healthindex"},
        {"incindex", "incomeindex"},
        {"edindex", "educationindex", "educationalindex"},
        {"lifexp", "lifeexpectancy"},
        {"lgnic"},
        {"esch"},
        {"msch"},
    ]

    return sum(
        bool(normalized.intersection(group))
        for group in expected_groups
    )


def _read_manual_table(path: Path) -> pd.DataFrame:

    suffix = path.suffix.lower()

    if suffix == ".csv":

        # GDL archive files have changed CSV serialization between releases.
        # Detect encoding, separator and a possible metadata row instead of
        # assuming a comma-delimited file with the header on row 1.
        candidates = []
        last_error = None

        for encoding in ["utf-8-sig", "utf-8", "latin-1"]:

            for separator in [None, ",", ";", "\t", "|"]:

                for header_row in range(5):

                    try:

                        preview_kwargs = {
                            "encoding": encoding,
                            "header": header_row,
                            "nrows": 5,
                        }

                        if separator is None:
                            preview_kwargs.update(
                                {
                                    "sep": None,
                                    "engine": "python",
                                }
                            )
                        else:
                            preview_kwargs["sep"] = separator

                        preview = pd.read_csv(
                            path,
                            **preview_kwargs,
                        )

                        score = _gdl_header_score(
                            preview.columns
                        )

                        candidates.append(
                            (
                                score,
                                encoding,
                                separator,
                                header_row,
                                len(preview.columns),
                            )
                        )

                    except Exception as exc:

                        last_error = exc

        if not candidates:

            raise RuntimeError(
                f"Could not inspect manual CSV {path.name}: {last_error}"
            )

        best_score, encoding, separator, header_row, column_count = max(
            candidates,
            key=lambda item: (item[0], item[4], -item[3]),
        )

        read_kwargs = {
            "encoding": encoding,
            "header": header_row,
        }

        if separator is None:
            read_kwargs.update(
                {
                    "sep": None,
                    "engine": "python",
                }
            )
        else:
            read_kwargs.update(
                {
                    "sep": separator,
                    "low_memory": False,
                }
            )

        frame = pd.read_csv(
            path,
            **read_kwargs,
        )

        separator_label = (
            "auto"
            if separator is None
            else repr(separator)
        )

        print(
            "GDL SHDI CSV detection: "
            f"encoding={encoding} | separator={separator_label} | "
            f"header_row={header_row + 1} | columns={len(frame.columns)} | "
            f"header_score={best_score}"
        )

        return frame

    if suffix in {".xlsx", ".xls"}:

        excel = pd.ExcelFile(path)

        best = None
        best_score = -1

        for sheet in excel.sheet_names:

            frame = pd.read_excel(path, sheet_name=sheet)
            score = _gdl_header_score(frame.columns)

            if score > best_score:

                best = frame
                best_score = score

        if best is None:

            raise RuntimeError(
                f"No readable worksheet found in {path.name}."
            )

        return best

    raise ValueError(
        f"Unsupported manual GDL file type: {path.suffix}"
    )


def extract_gdl_shdi(
    countries: list[str],
) -> pd.DataFrame:

    matches: list[Path] = []

    for pattern in MANUAL_SOURCE_PATTERNS["GDL_SHDI"]:

        matches.extend(MANUAL_DIR.rglob(pattern))

    matches = sorted(
        set(matches),
        key=lambda path: (
            path.stat().st_mtime,
            path.stat().st_size,
        ),
        reverse=True,
    )

    if not matches:

        raise FileNotFoundError(
            "GDL SHDI file not found. Download the current "
            "'Subnational HDI Data' CSV from Global Data Lab and place it in "
            "data/raw/manual/."
        )

    source_file = matches[0]

    print(
        f"GDL SHDI source: {source_file.name}"
    )

    frame = _read_manual_table(source_file)

    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    year_col = find_column(
        frame,
        ["year", "time", "time_period"],
    )

    iso_col = find_column(
        frame,
        ["iso_code", "isocode3", "iso_code3", "iso3", "iso3c", "country_code", "country_iso3", "iso"],
    )

    level_col = find_column(
        frame,
        ["level", "aggregation_level"],
    )

    if year_col is None or iso_col is None:

        detected_columns = ", ".join(
            str(column)
            for column in frame.columns[:40]
        )

        raise ValueError(
            "The GDL SHDI file must contain a recognizable ISO country code "
            "and year field. Detected columns: "
            f"{detected_columns}"
        )

    frame[year_col] = pd.to_numeric(
        frame[year_col],
        errors="coerce",
    )

    frame[iso_col] = (
        frame[iso_col]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    frame = frame[
        frame[year_col].isin(YEARS)
        & frame[iso_col].isin(countries)
    ].copy()

    if level_col is not None:

        level_text = (
            frame[level_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        subnational_mask = level_text.str.contains(
            "subnational|sub-national|region",
            regex=True,
            na=False,
        )

        if subnational_mask.any():

            frame = frame[subnational_mask].copy()

    canonical_columns = []

    for expected in GDL_SHDI_CORE_COLUMNS:

        found = find_column(
            frame,
            [expected],
        )

        if found is not None and found not in canonical_columns:

            canonical_columns.append(found)

    remaining = [
        column
        for column in frame.columns
        if column not in canonical_columns
    ]

    frame = frame[
        canonical_columns + remaining
    ].copy()

    write_csv(
        frame,
        GDL_SHDI_OUTPUT,
    )

    print(
        f"GDL SHDI: {len(frame):,} rows | "
        f"{frame[iso_col].nunique()} project countries | "
        f"2015–2020"
    )

    return frame


def _columns_containing(
    frame: pd.DataFrame,
    token: str,
) -> list[str]:

    token = token.lower()

    return [
        column
        for column in frame.columns
        if token in str(column).lower()
    ]


def _match_any_column(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    exact_codes: set[str] | None = None,
    contains_text: list[str] | None = None,
) -> pd.Series:

    mask = pd.Series(
        False,
        index=frame.index,
    )

    exact_codes = {
        value.upper()
        for value in (exact_codes or set())
    }

    contains_text = [
        value.lower()
        for value in (contains_text or [])
    ]

    for column in columns:

        values = (
            frame[column]
            .astype(str)
            .str.strip()
        )

        if exact_codes:

            mask |= values.str.upper().isin(exact_codes)

        for text_value in contains_text:

            mask |= values.str.lower().str.contains(
                re.escape(text_value),
                regex=True,
                na=False,
            )

    return mask


def extract_oecd_regional_unemployment() -> pd.DataFrame:

    params = {
        "startPeriod": START_YEAR,
        "endPeriod": END_YEAR,
        "dimensionAtObservation": "AllDimensions",
        "format": "csvfilewithlabels",
    }

    OECD_REGIONAL_RAW_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        OECD_REGIONAL_RAW_OUTPUT.exists()
        and OECD_REGIONAL_RAW_OUTPUT.stat().st_size > 0
    ):

        print(
            f"  cached: {OECD_REGIONAL_RAW_OUTPUT.name}"
        )

        raw = pd.read_csv(
            OECD_REGIONAL_RAW_OUTPUT,
            low_memory=False,
        )

    else:

        response = SESSION.get(
            OECD_REGIONAL_LABOUR_API,
            params=params,
            timeout=DOWNLOAD_TIMEOUT,
        )

        response.raise_for_status()

        OECD_REGIONAL_RAW_OUTPUT.write_bytes(
            response.content
        )

        raw = pd.read_csv(
            StringIO(response.text),
            low_memory=False,
        )

    if raw.empty:

        raise RuntimeError(
            "OECD Regional Labour API returned an empty dataset."
        )

    time_columns = _columns_containing(
        raw,
        "time",
    )

    if time_columns:

        time_mask = pd.Series(
            False,
            index=raw.index,
        )

        for column in time_columns:

            numeric = pd.to_numeric(
                raw[column],
                errors="coerce",
            )

            time_mask |= numeric.isin(YEARS)

        if time_mask.any():

            raw = raw[time_mask].copy()

    measure_columns = _columns_containing(
        raw,
        "measure",
    )

    unemployment_mask = _match_any_column(
        raw,
        measure_columns,
        exact_codes={
            "UNEMP_RATE",
            "UNE_RATE",
            "UNEMPLOYMENT_RATE",
        },
        contains_text=[
            "unemployment rate",
        ],
    )

    if not unemployment_mask.any():

        raise ValueError(
            "OECD regional unemployment-rate series could not be identified "
            "from the current DF_RATES response. Inspect the cached raw CSV "
            "before changing indicator codes."
        )

    frame = raw[unemployment_mask].copy()

    age_columns = _columns_containing(
        frame,
        "age",
    )

    age_mask = _match_any_column(
        frame,
        age_columns,
        exact_codes={
            "Y15T64",
        },
        contains_text=[
            "15 to 64",
            "15-64",
        ],
    )

    if age_mask.any():

        frame = frame[age_mask].copy()

    sex_columns = _columns_containing(
        frame,
        "sex",
    )

    sex_mask = _match_any_column(
        frame,
        sex_columns,
        exact_codes={
            "_T",
            "T",
            "TOTAL",
        },
        contains_text=[
            "total",
        ],
    )

    if sex_mask.any():

        frame = frame[sex_mask].copy()

    # Explicitly retain annual observations when the frequency dimension is
    # present in the labelled OECD CSV.
    frequency_columns = (
        _columns_containing(frame, "frequency")
        + _columns_containing(frame, "freq")
    )
    frequency_columns = list(dict.fromkeys(frequency_columns))

    frequency_mask = _match_any_column(
        frame,
        frequency_columns,
        exact_codes={"A", "ANNUAL"},
        contains_text=["annual"],
    )

    if frequency_mask.any():

        frame = frame[frequency_mask].copy()

    # DF_RATES contains national and regional reference areas. Where the
    # territorial-level dimension is available, keep OECD TL2/TL3 only.
    territorial_columns = _columns_containing(
        frame,
        "territorial",
    )

    territorial_mask = _match_any_column(
        frame,
        territorial_columns,
        exact_codes={"TL2", "TL3"},
        contains_text=[
            "large region",
            "small region",
            "tl2",
            "tl3",
        ],
    )

    if territorial_mask.any():

        frame = frame[territorial_mask].copy()

    write_csv(
        frame,
        OECD_REGIONAL_OUTPUT,
    )

    print(
        f"OECD Regional unemployment: {len(frame):,} rows | "
        f"2015–2020 | age 15–64 | total sex"
    )

    return frame


# =============================================================================
# MANUAL / AUTHENTICATED SOURCES
# =============================================================================

MANUAL_SOURCE_PATTERNS = {
    "GDL_SHDI": [
        "*Subnational*HDI*Data*v10.2*.csv",
        "*Subnational*HDI*Data*.csv",
        "*shdi*.csv",
        "*human*development*.csv",
        "*gdl*hdi*.csv",
        "*shdi*.xlsx",
        "*human*development*.xlsx",
    ],
}

MANUAL_SOURCE_DEFAULTS = {
    "GDL_SHDI": {
        "status": "MANUAL_OR_AUTHENTICATED_DOWNLOAD_REQUIRED",
        "note": (
            "Global Data Lab requires a free login for the current SHDI CSV/Excel download. "
            "Download the current Subnational HDI Data file and place it in data/raw/manual/."
        ),
    },
}


def discover_manual_sources() -> pd.DataFrame:

    rows = []

    for (
        source_id,
        patterns,
    ) in MANUAL_SOURCE_PATTERNS.items():

        matches = []

        for pattern in patterns:

            matches.extend(
                MANUAL_DIR.rglob(
                    pattern
                )
            )

        matches = sorted(
            set(
                matches
            )
        )

        default = MANUAL_SOURCE_DEFAULTS.get(
            source_id,
            {
                "status": (
                    "MANUAL_OR_AUTHENTICATED_"
                    "DOWNLOAD_REQUIRED"
                ),
                "note": (
                    "Source requires manual review "
                    "before automated extraction."
                ),
            },
        )

        if matches:

            status = (
                "LOCAL_MANUAL_SOURCE_FOUND"
            )

            note = (
                "Local source file(s) detected. "
                "Script 03 can harmonize them."
            )

        else:

            status = str(
                default[
                    "status"
                ]
            )

            note = str(
                default[
                    "note"
                ]
            )

        rows.append(
            {
                "source_id": (
                    source_id
                ),
                "status": status,
                "files_found": (
                    len(
                        matches
                    )
                ),
                "files": (
                    " | ".join(
                        str(
                            path
                        )
                        for path in matches
                    )
                ),
                "note": note,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MANIFEST WORKBOOK
# =============================================================================

def build_extraction_summary(
    countries: list[str],
    spid_panel: pd.DataFrame,
    regions: Any,
    workbook: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "metric": (
                    "Official period"
                ),
                "value": (
                    "2015–2020"
                ),
            },
            {
                "metric": (
                    "Strict SPID countries"
                ),
                "value": (
                    len(
                        countries
                    )
                ),
            },
            {
                "metric": (
                    "Target subnational regions"
                ),
                "value": (
                    len(
                        regions
                    )
                ),
            },
            {
                "metric": (
                    "SPID region-year observations"
                ),
                "value": (
                    len(
                        spid_panel
                    )
                ),
            },
            {
                "metric": (
                    "Script 01 audit workbook"
                ),
                "value": (
                    str(
                        AUDIT_FILE
                    )
                ),
            },
            {
                "metric": (
                    "Script 02 interim directory"
                ),
                "value": (
                    str(
                        INTERIM_DIR
                    )
                ),
            },
            {
                "metric": (
                    "WorldPop extraction mode"
                ),
                "value": (
                    "Bulk R2025A 1km rasters + local zonal aggregation"
                ),
            },
        ]
    )


def write_manifest_workbook(
    summary: pd.DataFrame,
    source_manifest: pd.DataFrame,
    manual_sources: pd.DataFrame,
    workbook: dict[
        str,
        pd.DataFrame,
    ],
    s2s_fields: pd.DataFrame | None,
) -> None:

    with pd.ExcelWriter(
        MANIFEST_FILE,
        engine="openpyxl",
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="00_Summary",
            index=False,
        )

        source_manifest.to_excel(
            writer,
            sheet_name="01_Source_Status",
            index=False,
        )

        manual_sources.to_excel(
            writer,
            sheet_name="02_Manual_Sources",
            index=False,
        )

        if s2s_fields is not None:

            s2s_fields.to_excel(
                writer,
                sheet_name="03_S2S_Fields",
                index=False,
            )

        workbook[
            "04_Extraction_Queue"
        ].to_excel(
            writer,
            sheet_name="04_Original_Queue",
            index=False,
        )

        workbook[
            "06_SPID_Countries"
        ].to_excel(
            writer,
            sheet_name="05_SPID_Countries",
            index=False,
        )

        workbook[
            "07_SPID_Selected_Series"
        ].to_excel(
            writer,
            sheet_name="06_SPID_Series",
            index=False,
        )

        if ERROR_ROWS:

            pd.DataFrame(
                ERROR_ROWS
            ).to_excel(
                writer,
                sheet_name="07_Errors",
                index=False,
            )

        else:

            pd.DataFrame(
                [
                    {
                        "source": (
                            "GLOBAL"
                        ),
                        "error": (
                            "No unhandled source "
                            "errors recorded."
                        ),
                    }
                ]
            ).to_excel(
                writer,
                sheet_name="07_Errors",
                index=False,
            )

    from openpyxl import load_workbook
    from openpyxl.utils import (
        get_column_letter,
    )

    book = load_workbook(
        MANIFEST_FILE
    )

    for sheet in book.worksheets:

        if sheet.max_row >= 2:

            sheet.freeze_panes = (
                "A2"
            )

        if (
            sheet.max_row >= 1
            and sheet.max_column >= 1
        ):

            sheet.auto_filter.ref = (
                sheet.dimensions
            )

        for index in range(
            1,
            sheet.max_column + 1,
        ):

            letter = (
                get_column_letter(
                    index
                )
            )

            maximum = 0

            for cell in (
                sheet[
                    letter
                ]
            ):

                if cell.value is not None:

                    maximum = max(
                        maximum,
                        len(
                            str(
                                cell.value
                            )
                        ),
                    )

            sheet.column_dimensions[
                letter
            ].width = min(
                max(
                    maximum + 2,
                    10,
                ),
                55,
            )

    book.save(
        MANIFEST_FILE
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print_header(
        "SCRIPT 02.2 — GDL SHDI PARSER FIX"
    )

    print(
        f"Project root: "
        f"{ROOT}"
    )

    print(
        f"Official period: "
        f"{START_YEAR}–{END_YEAR}"
    )

    print(
        f"Audit workbook: "
        f"{AUDIT_FILE}"
    )

    # -------------------------------------------------------------------------
    # 0. Audit workbook / scope
    # -------------------------------------------------------------------------

    workbook = (
        load_audit_workbook()
    )

    countries = (
        get_primary_country_scope(
            workbook
        )
    )

    selected_series = (
        get_selected_spid_series(
            workbook,
            countries,
        )
    )

    print(
        f"Strict SPID country scope: "
        f"{len(countries)}"
    )

    print(
        ", ".join(
            countries
        )
    )

    # -------------------------------------------------------------------------
    # 1. SPID + target geography
    # -------------------------------------------------------------------------

    if not RUN_SPID:

        raise RuntimeError(
            "RUN_SPID must remain True because "
            "all regional sources depend on the "
            "target geography."
        )

    with timed_stage(
        "1/7 — SPID + TARGET GEOGRAPHY"
    ) as timer:

        try:

            spid_panel = (
                extract_spid(
                    selected_series,
                    countries,
                )
            )

            regions = (
                build_target_geometries(
                    spid_panel
                )
            )

            add_manifest(
                "SPID",
                "SUCCESS",
                records=len(
                    spid_panel
                ),
                output=SPID_OUTPUT,
                message=(
                    f"{len(regions)} target "
                    "regions reconstructed."
                ),
            )

        except Exception as exc:

            ERROR_ROWS.append(
                {
                    "source": "SPID",
                    "error": str(
                        exc
                    ),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )

            raise

    MANIFEST_ROWS[
        -1
    ][
        "elapsed_seconds"
    ] = timer.seconds

    # -------------------------------------------------------------------------
    # 2. Space2Stats
    # -------------------------------------------------------------------------

    s2s_fields = None

    with timed_stage(
        "2/7 — SPACE2STATS"
    ) as timer:

        if RUN_SPACE2STATS:

            try:

                (
                    s2s_data,
                    s2s_fields,
                ) = (
                    extract_space2stats(
                        regions
                    )
                )

                success = int(
                    (
                        s2s_data[
                            "status"
                        ]
                        == "success"
                    ).sum()
                )

                partial = int(
                    (
                        s2s_data[
                            "status"
                        ]
                        == "partial_fields"
                    ).sum()
                )

                failed = int(
                    (
                        s2s_data[
                            "status"
                        ]
                        == "failed"
                    ).sum()
                )

                add_manifest(
                    "SPACE2STATS",
                    (
                        "SUCCESS"
                        if (
                            partial == 0
                            and failed == 0
                        )
                        else "PARTIAL"
                    ),
                    records=(
                        success
                        + partial
                    ),
                    output=(
                        SPACE2STATS_OUTPUT
                    ),
                    message=(
                        f"{success} complete regions; "
                        f"{partial} partial-field regions; "
                        f"{failed} failed regions."
                    ),
                    seconds=(time.perf_counter() - timer.start),
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": (
                            "SPACE2STATS"
                        ),
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

                add_manifest(
                    "SPACE2STATS",
                    "FAILED",
                    records=0,
                    output=(
                        SPACE2STATS_OUTPUT
                    ),
                    message=str(
                        exc
                    ),
                )

                print(
                    f"Space2Stats failed but "
                    f"pipeline continues: {exc}"
                )

    # -------------------------------------------------------------------------
    # 3. WorldPop age-sex
    # -------------------------------------------------------------------------

    with timed_stage(
        "3/7 — WORLDPOP AGE-SEX (BULK RASTERS)"
    ) as timer:

        if RUN_WORLDPOP_AGESEX:

            try:

                wp = (
                    extract_worldpop_agesex(
                        regions
                    )
                )

                success = int(
                    (
                        wp[
                            "status"
                        ]
                        == "success"
                    ).sum()
                ) if not wp.empty else 0

                failed = int(
                    (
                        wp[
                            "status"
                        ]
                        == "failed"
                    ).sum()
                ) if not wp.empty else 0

                expected = int(
                    WORLDPOP_LAST_RUN_META.get(
                        "expected_region_years",
                        len(regions)
                        * len(YEARS),
                    )
                )

                if (
                    success == expected
                    and failed == 0
                ):

                    wp_status = (
                        "SUCCESS"
                    )

                else:

                    wp_status = (
                        "PARTIAL"
                    )

                add_manifest(
                    "WORLDPOP_AGESEX",
                    wp_status,
                    records=success,
                    output=(
                        WORLDPOP_AGESEX_OUTPUT
                    ),
                    message=(
                        f"{success}/{expected} "
                        "region-years complete; "
                        f"{failed} failed; "
                        f"{WORLDPOP_LAST_RUN_META.get('country_years_processed', 0)} "
                        "country-years processed locally; "
                        f"{WORLDPOP_LAST_RUN_META.get('files_downloaded', 0)} "
                        "rasters downloaded; "
                        f"{WORLDPOP_LAST_RUN_META.get('bytes_downloaded', 0) / (1024**3):.2f} GiB "
                        "downloaded in this run. "
                        "Method: official WorldPop Global2 R2025A 1km bulk rasters "
                        "+ local zonal aggregation."
                    ),
                    seconds=(time.perf_counter() - timer.start),
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": (
                            "WORLDPOP_AGESEX"
                        ),
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

                add_manifest(
                    "WORLDPOP_AGESEX",
                    "FAILED",
                    records=0,
                    output=(
                        WORLDPOP_AGESEX_OUTPUT
                    ),
                    message=str(
                        exc
                    ),
                )

                print(
                    f"WorldPop failed but "
                    f"pipeline continues: {exc}"
                )

    # -------------------------------------------------------------------------
    # 4. Local supplementary SSGD
    # -------------------------------------------------------------------------

    with timed_stage(
        "4/7 — SSGD"
    ) as timer:

        if RUN_SSGD:

            try:

                ssgd = (
                    extract_ssgd(
                        countries
                    )
                )

                add_manifest(
                    "SSGD",
                    "SUCCESS",
                    records=len(
                        ssgd
                    ),
                    output=(
                        SSGD_OUTPUT
                    ),
                    message=(
                        "Five pre-selected "
                        "candidate indicators "
                        "retained when present."
                    ),
                    seconds=(time.perf_counter() - timer.start),
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": (
                            "SSGD"
                        ),
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

                add_manifest(
                    "SSGD",
                    "FAILED",
                    records=0,
                    output=(
                        SSGD_OUTPUT
                    ),
                    message=str(
                        exc
                    ),
                )

                print(
                    f"SSGD failed but "
                    f"pipeline continues: {exc}"
                )

    # -------------------------------------------------------------------------
    # 5. WDI
    # -------------------------------------------------------------------------

    with timed_stage(
        "5/7 — WDI"
    ) as timer:

        if RUN_WDI:

            try:

                wdi = (
                    extract_wdi(
                        countries
                    )
                )

                add_manifest(
                    "WDI",
                    "SUCCESS",
                    records=len(
                        wdi
                    ),
                    output=WDI_OUTPUT,
                    message=(
                        f"{len(WDI_CODES)} "
                        "candidate indicators "
                        "queried."
                    ),
                    seconds=(time.perf_counter() - timer.start),
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": "WDI",
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

                add_manifest(
                    "WDI",
                    "FAILED",
                    records=0,
                    output=WDI_OUTPUT,
                    message=str(
                        exc
                    ),
                )

                print(
                    f"WDI failed but "
                    f"pipeline continues: {exc}"
                )

    # -------------------------------------------------------------------------
    # 6. Public downloadable sources
    # -------------------------------------------------------------------------

    with timed_stage(
        "6/7 — KUMMU + DOSE + NIVA + OOKLA + GDL SCD"
    ) as timer:

        if RUN_ZENODO_PUBLIC:

            public_extractors = [
                (
                    "KUMMU_GDP",
                    extract_kummu,
                    KUMMU_OUTPUT,
                ),
                (
                    "DOSE",
                    extract_dose,
                    DOSE_OUTPUT,
                ),
                (
                    "NIVA_MIGRATION",
                    extract_niva,
                    NIVA_OUTPUT,
                ),
            ]

            if RUN_OOKLA_WB:

                public_extractors.append(
                    (
                        "OOKLA_WB",
                        extract_ookla_wb,
                        OOKLA_OUTPUT,
                    )
                )

            for (
                source_id,
                function,
                output,
            ) in public_extractors:

                source_start = (
                    time.perf_counter()
                )

                try:

                    data = function(
                        countries
                    )

                    add_manifest(
                        source_id,
                        "SUCCESS",
                        records=len(
                            data
                        ),
                        output=output,
                        message=(
                            "Public source downloaded "
                            "and reduced to the "
                            "project scope where "
                            "identifiers allowed."
                        ),
                        seconds=(
                            time.perf_counter()
                            - source_start
                        ),
                    )

                except Exception as exc:

                    ERROR_ROWS.append(
                        {
                            "source": (
                                source_id
                            ),
                            "error": str(
                                exc
                            ),
                            "traceback": (
                                traceback.format_exc()
                            ),
                        }
                    )

                    add_manifest(
                        source_id,
                        "FAILED",
                        records=0,
                        output=output,
                        message=str(
                            exc
                        ),
                        seconds=(
                            time.perf_counter()
                            - source_start
                        ),
                    )

                    print(
                        f"{source_id} failed but "
                        f"pipeline continues: {exc}"
                    )

            # Public corruption database (Figshare)
            source_start = time.perf_counter()

            try:

                scd = extract_gdl_scd(
                    countries
                )

                scd_records = sum(
                    len(frame)
                    for frame in scd.values()
                )

                add_manifest(
                    "GDL_SCD",
                    "SUCCESS",
                    records=scd_records,
                    output=(
                        f"{GDL_SCD_BASELINE_OUTPUT} | "
                        f"{GDL_SCD_COMPREHENSIVE_OUTPUT}"
                    ),
                    message=(
                        "Public Figshare archive downloaded; "
                        "Baseline and/or Comprehensive SCI "
                        "tables retained for 2015–2020."
                    ),
                    seconds=(
                        time.perf_counter()
                        - source_start
                    ),
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": "GDL_SCD",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

                add_manifest(
                    "GDL_SCD",
                    "FAILED",
                    records=0,
                    output="",
                    message=str(exc),
                    seconds=(
                        time.perf_counter()
                        - source_start
                    ),
                )

                print(
                    "GDL_SCD failed but pipeline continues: "
                    f"{exc}"
                )

    # -------------------------------------------------------------------------
    # 7. GDL SHDI + OECD Regional + manifest
    # -------------------------------------------------------------------------

    with timed_stage(
        "7/7 — GDL SHDI + OECD REGIONAL + MANIFEST"
    ) as timer:

        if RUN_GDL_SHDI:

            source_start = time.perf_counter()

            try:

                shdi = extract_gdl_shdi(
                    countries
                )

                add_manifest(
                    "GDL_SHDI",
                    "SUCCESS",
                    records=len(shdi),
                    output=GDL_SHDI_OUTPUT,
                    message=(
                        "Current GDL Subnational HDI source retained for the "
                        "28-country 2015–2020 scope; final regional matching "
                        "and indicator selection are deferred to Script 03."
                    ),
                    seconds=(time.perf_counter() - source_start),
                )

            except FileNotFoundError as exc:

                add_manifest(
                    "GDL_SHDI",
                    "MANUAL_OR_AUTHENTICATED_DOWNLOAD_REQUIRED",
                    records=0,
                    output="",
                    message=str(exc),
                    seconds=(time.perf_counter() - source_start),
                )

                print(
                    f"GDL SHDI pending manual download: {exc}"
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": "GDL_SHDI",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

                add_manifest(
                    "GDL_SHDI",
                    "FAILED",
                    records=0,
                    output=GDL_SHDI_OUTPUT,
                    message=str(exc),
                    seconds=(time.perf_counter() - source_start),
                )

                print(
                    f"GDL SHDI failed but pipeline continues: {exc}"
                )

        if RUN_OECD_REGIONAL:

            source_start = time.perf_counter()

            try:

                oecd = extract_oecd_regional_unemployment()

                add_manifest(
                    "OECD_REGIONAL",
                    "SUCCESS",
                    records=len(oecd),
                    output=OECD_REGIONAL_OUTPUT,
                    message=(
                        "Official OECD regional unemployment-rate slice "
                        "downloaded for 2015–2020, age 15–64 and total sex. "
                        "TL2/TL3-to-target-region harmonization is deferred "
                        "to Script 03."
                    ),
                    seconds=(time.perf_counter() - source_start),
                )

            except Exception as exc:

                ERROR_ROWS.append(
                    {
                        "source": "OECD_REGIONAL",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

                add_manifest(
                    "OECD_REGIONAL",
                    "FAILED",
                    records=0,
                    output=OECD_REGIONAL_OUTPUT,
                    message=str(exc),
                    seconds=(time.perf_counter() - source_start),
                )

                print(
                    f"OECD Regional failed but pipeline continues: {exc}"
                )

        manual_sources = discover_manual_sources()

        source_manifest = pd.DataFrame(
            MANIFEST_ROWS
        )

        summary = build_extraction_summary(
            countries,
            spid_panel,
            regions,
            workbook,
        )

        write_manifest_workbook(
            summary,
            source_manifest,
            manual_sources,
            workbook,
            s2s_fields,
        )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print_header(
        "FINAL SUMMARY"
    )

    source_manifest = (
        pd.DataFrame(
            MANIFEST_ROWS
        )
    )

    if not source_manifest.empty:

        print(
            source_manifest[
                [
                    "source_id",
                    "status",
                    "records",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Countries: "
        f"{len(countries)}"
    )

    print(
        f"Regions: "
        f"{len(regions)}"
    )

    print(
        f"Official years: "
        f"{START_YEAR}–{END_YEAR}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_FILE}"
    )

    print(
        f"Interim data: "
        f"{INTERIM_DIR}"
    )

    print()
    print(
        "Script 03 will perform the final "
        "harmonization, derived indicators, "
        "spatial matching and consolidated QA."
    )


if __name__ == "__main__":

    main()
