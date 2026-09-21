from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import time

import numpy as np
import pandas as pd
import requests


# =============================================================================
# P1 / GENERAL PORTFOLIO DATA PIPELINE
# SCRIPT 01 v1.1 — CONSOLIDATED AVAILABILITY AUDIT
# =============================================================================
# Purpose
# -------
# 1) Register ALL candidate variables pre-selected for the portfolio/P1.
# 2) Audit local sources (SPID, SSGD).
# 3) Audit machine-readable public sources (Space2Stats, WDI, WorldPop, Zenodo).
# 4) Check reachability of complementary sources whose extraction adapters will be
#    implemented in Script 02.
# 5) Compare the original 2015–2022 window with the alternative 2015–2020 window.
# 6) Identify variables/countries that are recovered by dropping 2021–2022.
# 7) Produce ONE Excel workbook. No bulk geospatial extraction is performed here.
#
# IMPORTANT
# ---------
# - Do not hard-code a final number of countries or subnational units.
# - The SPID geography is mixed (GAUL1/GAULx/NUTS1/NUTS2/etc.); therefore this
#   script uses the generic term "subnational_region" rather than assuming ADM1.
# - Snapshot/static/survey variables are NOT judged by the same completeness rule
#   as annual panel variables.
#
# v1.1 corrections
# ----------------
# - 2015–2020 is the primary analytical window; 2015–2022 is retained as benchmark.
# - SPID eligibility is evaluated within code + welfaretype + comparability series.
# - SSGD pre-selected indicators use an explicit code-to-family map.
# - Missing Space2Stats cyclone fields are classified as unavailable in current /fields.
# - Temporary source outages are distinguished from data/source absence.
# - SPID selected/all comparable series are exported for transparent QA.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPID_FILE = RAW_DIR / "AM25_-_SPID_data.xlsx"
SSGD_FILE = RAW_DIR / "ssgd_v2_long.xlsx"

OUTPUT_FILE = OUTPUT_DIR / "01_candidate_availability_audit_v1_1.xlsx"


# =============================================================================
# PERIODS
# =============================================================================

PERIOD_LONG = list(range(2015, 2023))   # benchmark window: 2015–2022
PERIOD_SHORT = list(range(2015, 2021))  # PRIMARY window: 2015–2020
PRIMARY_PERIOD = PERIOD_SHORT
BENCHMARK_PERIOD = PERIOD_LONG


# =============================================================================
# NETWORK SETTINGS
# =============================================================================

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_WAIT = 2
RETRY_BACKOFF = 2

SPACE2STATS_FIELDS_URL = "https://space2stats.ds.io/fields"
WDI_API = "https://api.worldbank.org/v2"
WORLDPOP_SERVICES_URL = "https://api.worldpop.org/v1/services"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "WBG-Outcomes-Portfolio-Availability-Audit/1.0 "
            "(academic reproducibility workflow)"
        )
    }
)


# =============================================================================
# RUN LOG
# =============================================================================

RUN_LOG: list[dict[str, Any]] = []


def log(level: str, source: str, message: str) -> None:
    row = {"level": level, "source": source, "message": message}
    RUN_LOG.append(row)
    print(f"[{level}] {source}: {message}")


# =============================================================================
# SOURCE REGISTRY
# =============================================================================
# "documented_period" is descriptive metadata used for planning. Script 02 will
# perform actual extraction and spatial coverage measurement where possible.
# =============================================================================

SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "SPID": {
        "label": "World Bank Subnational Poverty and Inequality Database (SPID)",
        "kind": "local_file",
        "path": SPID_FILE,
        "machine_access": "LOCAL",
        "documented_period": "1999–2022",
    },
    "SSGD": {
        "label": "World Bank Subnational Social/Gender Database (SSGD)",
        "kind": "local_file",
        "path": SSGD_FILE,
        "machine_access": "LOCAL",
        "documented_period": "survey years; irregular",
    },
    "SPACE2STATS": {
        "label": "World Bank Space2Stats",
        "kind": "json_api",
        "url": SPACE2STATS_FIELDS_URL,
        "machine_access": "API",
        "documented_period": "field-specific",
    },
    "WORLDPOP": {
        "label": "WorldPop Global Population 2000–2020",
        "kind": "worldpop_api",
        "url": WORLDPOP_SERVICES_URL,
        "machine_access": "API",
        "documented_period": "2000–2020",
        "expected_dataset": "wpgppop",
    },
    "WORLDPOP_AGESEX": {
        "label": "WorldPop Age-Sex Structures 2000–2020",
        "kind": "worldpop_api",
        "url": WORLDPOP_SERVICES_URL,
        "machine_access": "API",
        "documented_period": "2000–2020",
        "expected_dataset": "wpgpas",
    },
    "KUMMU_GDP": {
        "label": "Kummu et al. downscaled GDP per capita PPP",
        "kind": "zenodo_api",
        "url": "https://zenodo.org/api/records/18429133",
        "public_url": "https://zenodo.org/records/18429133",
        "machine_access": "DIRECT_DOWNLOAD",
        "documented_period": "1990–2024",
        "expected_file_patterns": ["adm1", "gdp", "percapita"],
    },
    "DOSE": {
        "label": "DOSE — Database of Sub-national Economic Output V2.11",
        "kind": "zenodo_api",
        "url": "https://zenodo.org/api/records/16313760",
        "public_url": "https://zenodo.org/records/16313760",
        "machine_access": "DIRECT_DOWNLOAD",
        "documented_period": "1953–2020; irregular by region/year",
        "expected_file_patterns": ["dose"],
    },
    "NIVA_MIGRATION": {
        "label": "Niva et al. global net migration",
        "kind": "zenodo_api",
        "url": "https://zenodo.org/api/records/7997134",
        "public_url": "https://zenodo.org/records/7997134",
        "machine_access": "DIRECT_DOWNLOAD",
        "documented_period": "2000–2019",
        "expected_file_patterns": ["adm1", "netmgr"],
    },
    "GDL_SHDI": {
        "label": "Global Data Lab — Subnational Human Development",
        "kind": "web_source",
        "url": "https://globaldatalab.org/shdi/",
        "machine_access": "WEB_TABLE_OR_DOWNLOAD",
        "documented_period": "1990–2023",
    },
    "GDL_EDU_WORK": {
        "label": "Global Data Lab — Education & Work",
        "kind": "web_source",
        "url": "https://globaldatalab.org/",
        "machine_access": "WEB_TABLE_OR_DOWNLOAD",
        "documented_period": "survey years; country-dependent",
    },
    "GDL_SCD": {
        "label": "Global Data Lab — Subnational Corruption Database",
        "kind": "web_source",
        "url": "https://globaldatalab.org/governance/",
        "machine_access": "WEB_TABLE; DOWNLOAD MAY REQUIRE REGISTRATION",
        "documented_period": "1995–2022",
    },
    "OECD_REGIONAL": {
        "label": "OECD Regional / Territorial Statistics",
        "kind": "web_source",
        "url": "https://data-explorer.oecd.org/",
        "machine_access": "SDMX/API + DOWNLOAD",
        "documented_period": "indicator-specific; strong 2015–2020 coverage in subsets",
    },
    "EUROSTAT": {
        "label": "Eurostat Regional Statistics",
        "kind": "web_source",
        "url": "https://ec.europa.eu/eurostat/web/main/data/database",
        "machine_access": "API + DOWNLOAD",
        "documented_period": "indicator-specific",
    },
    "OOKLA_WB": {
        "label": "Ookla Speedtest Data — World Bank catalog",
        "kind": "web_source",
        "url": "https://datacatalog.worldbank.org/search/dataset/0067161/ookla-speedtest-data",
        "machine_access": "DOWNLOAD",
        "documented_period": "2019 onward",
    },
    "WEISS_ACCESS": {
        "label": "Weiss et al. travel time to cities",
        "kind": "web_source",
        "url": "https://doi.org/10.1038/nature25181",
        "machine_access": "RASTER DOWNLOAD",
        "documented_period": "2015 snapshot",
    },
    "WEISS_SETTLEMENT_ACCESS": {
        "label": "Weiss et al. suite of global accessibility indicators",
        "kind": "web_source",
        "url": "https://doi.org/10.1038/s41597-019-0265-5",
        "machine_access": "FIGSHARE RASTER DOWNLOAD",
        "documented_period": "2015 snapshot; nine settlement-size layers",
    },
    "WDI": {
        "label": "World Development Indicators API",
        "kind": "json_api",
        "url": "https://api.worldbank.org/v2/country/BRA?format=json",
        "machine_access": "API",
        "documented_period": "indicator-specific",
    },
    "DERIVED": {
        "label": "Derived variable",
        "kind": "derived",
        "machine_access": "DERIVED",
        "documented_period": "depends on inputs",
    },
}


# =============================================================================
# VARIABLE REGISTRY HELPERS
# =============================================================================

VARIABLES: list[dict[str, Any]] = []


def add_variable(
    variable_id: str,
    block: str,
    name: str,
    source_id: str,
    level: str,
    temporal_type: str,
    coverage_mode: str,
    documented_start: int | None = None,
    documented_end: int | None = None,
    documented_years: list[int] | None = None,
    source_variable: str = "",
    backup_sources: str = "",
    role_candidate: str = "candidate_to_classify",
    dependencies: str = "",
    measurement_type: str = "",
    expected_flags: str = "",
    spatial_notes: str = "",
    temporal_notes: str = "",
    extraction_notes: str = "",
) -> None:
    VARIABLES.append(
        {
            "variable_id": variable_id,
            "block": block,
            "variable_name": name,
            "source_id": source_id,
            "source_variable": source_variable,
            "backup_sources": backup_sources,
            "target_level": level,
            "temporal_type": temporal_type,
            "coverage_mode": coverage_mode,
            "documented_start": documented_start,
            "documented_end": documented_end,
            "documented_years": (
                ", ".join(str(x) for x in documented_years)
                if documented_years
                else ""
            ),
            "role_candidate": role_candidate,
            "dependencies": dependencies,
            "measurement_type": measurement_type,
            "expected_flags": expected_flags,
            "spatial_notes": spatial_notes,
            "temporal_notes": temporal_notes,
            "extraction_notes": extraction_notes,
        }
    )


# =============================================================================
# REGIONAL / SUBNATIONAL CANDIDATES FROM THE SUPPLEMENTARY AUDIT
# =============================================================================

add_variable(
    "population",
    "Territorial",
    "Population",
    "SPACE2STATS",
    "subnational_region",
    "ANNUAL",
    "actual_space2stats_family",
    source_variable="population",
    backup_sources="WORLDPOP",
    role_candidate="context / denominator / clustering_candidate",
    measurement_type="modelled raster aggregation",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
    spatial_notes="Aggregate to portfolio region geometry; do not assume all units are ADM1.",
)
add_variable(
    "population_density",
    "Territorial",
    "Population density",
    "DERIVED",
    "subnational_region",
    "ANNUAL",
    "derived",
    documented_start=2015,
    documented_end=2020,
    dependencies="population; region_area_km2",
    role_candidate="context / clustering_candidate",
    measurement_type="derived",
)
add_variable(
    "population_growth",
    "Territorial",
    "Population growth",
    "DERIVED",
    "subnational_region",
    "ANNUAL",
    "derived",
    documented_start=2015,
    documented_end=2020,
    dependencies="population",
    role_candidate="context / clustering_candidate",
    measurement_type="derived",
)
add_variable(
    "built_up_area",
    "Territorial",
    "Built-up area",
    "SPACE2STATS",
    "subnational_region",
    "SNAPSHOT_SERIES",
    "actual_space2stats_family",
    documented_years=[2015, 2020],
    source_variable="built_up_area",
    backup_sources="GHSL GHS-BUILT-S direct",
    role_candidate="structural_context / clustering_candidate",
    measurement_type="modelled raster aggregation",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
)
add_variable(
    "built_up_share",
    "Territorial",
    "Built-up share",
    "DERIVED",
    "subnational_region",
    "SNAPSHOT_SERIES",
    "derived",
    documented_years=[2015, 2020],
    dependencies="built_up_area; region_area_km2",
    role_candidate="structural_context / clustering_candidate",
    measurement_type="derived",
)
add_variable(
    "urbanization_degree",
    "Territorial",
    "Urbanization / Degree of Urbanisation",
    "SPACE2STATS",
    "subnational_region",
    "STATIC_OR_SNAPSHOT",
    "actual_space2stats_family",
    source_variable="urbanization_ghs",
    backup_sources="GHSL GHS-SMOD direct",
    role_candidate="structural_context / clustering_candidate",
    measurement_type="GHS-SMOD raster aggregation",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
    temporal_notes="Space2Stats ghs_* fields currently do not expose an epoch in the field name; direct GHSL has 2015/2020 epochs.",
)
add_variable(
    "nighttime_lights_total",
    "Geospatial",
    "Nighttime lights total",
    "SPACE2STATS",
    "subnational_region",
    "ANNUAL",
    "actual_space2stats_family",
    source_variable="nighttime_lights",
    backup_sources="NASA Black Marble",
    role_candidate="economic_proxy / context / clustering_candidate",
    measurement_type="satellite-derived raster aggregation",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
)
add_variable(
    "ntl_per_capita",
    "Geospatial",
    "Nighttime lights per capita",
    "DERIVED",
    "subnational_region",
    "ANNUAL",
    "derived",
    documented_start=2015,
    documented_end=2020,
    dependencies="nighttime_lights_total; population",
    role_candidate="economic_proxy / context / clustering_candidate",
    measurement_type="derived",
)

# Economic
add_variable(
    "gdp_pc_ppp_kummu",
    "Economic",
    "GDP per capita PPP — Kummu et al.",
    "KUMMU_GDP",
    "subnational_region",
    "ANNUAL",
    "full_series_source",
    documented_start=1990,
    documented_end=2024,
    source_variable="ADM1 GDP per capita PPP product",
    role_candidate="economic_context / clustering_candidate",
    measurement_type="mixed reported / downscaled / national fallback",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED; ADM0_FALLBACK; SPATIAL_GAP; BOUNDARY_MISMATCH",
    spatial_notes="Native ADM1 boundaries may not coincide with SPID regional systems.",
    extraction_notes="Preserve provenance/metadata distinguishing reported ADM1, downscaled and ADM0 fallback where available.",
)
add_variable(
    "grp_per_capita_dose",
    "Economic",
    "GRP / regional production per capita",
    "DOSE",
    "subnational_region",
    "ANNUAL_IRREGULAR",
    "irregular_source",
    documented_start=1953,
    documented_end=2020,
    role_candidate="economic_context / clustering_candidate",
    measurement_type="reported subnational economic output",
    expected_flags="SOURCE_TYPE; TEMPORAL_GAP; SPATIAL_GAP; BOUNDARY_MISMATCH",
    temporal_notes="Annual when observed; DOSE does not imply a complete annual panel for every region.",
)
for var_id, name in [
    ("agriculture_share_dose", "Agriculture share"),
    ("industry_share_dose", "Manufacturing / industry share"),
    ("services_share_dose", "Services share"),
]:
    add_variable(
        var_id,
        "Economic",
        name,
        "DOSE",
        "subnational_region",
        "ANNUAL_IRREGULAR",
        "irregular_source",
        documented_end=2020,
        role_candidate="economic_structure / clustering_candidate",
        dependencies="sector GRP; total GRP",
        measurement_type="reported and/or derived from reported sectoral output",
        expected_flags="SOURCE_TYPE; TEMPORAL_GAP; SPATIAL_GAP; BOUNDARY_MISMATCH",
    )

# Labour
for var_id, name in [
    ("employment_rate", "Employment rate"),
    ("unemployment_rate", "Unemployment rate"),
    ("labor_force_participation", "Labor-force participation"),
]:
    add_variable(
        var_id,
        "Labour",
        name,
        "OECD_REGIONAL",
        "subnational_region",
        "ANNUAL_OR_SURVEY",
        "subset_source",
        documented_start=2015,
        documented_end=2020,
        backup_sources="EUROSTAT; GDL_EDU_WORK; national statistical sources",
        role_candidate="labour_context / clustering_candidate",
        measurement_type="official regional statistics / survey",
        expected_flags="SOURCE_TYPE; TEMPORAL_GAP; SPATIAL_GAP; BOUNDARY_MISMATCH; DEFINITION_MISMATCH",
    )
add_variable(
    "agricultural_employment_share",
    "Labour",
    "Agricultural employment / occupational structure",
    "GDL_EDU_WORK",
    "subnational_region",
    "SURVEY",
    "survey_source",
    documented_start=2015,
    documented_end=2020,
    role_candidate="labour_structure / clustering_candidate",
    measurement_type="survey",
    expected_flags="SURVEY_DEPENDENT; TEMPORAL_GAP; SPATIAL_GAP; BOUNDARY_MISMATCH; DEFINITION_MISMATCH",
)

# Human capital
for var_id, name, source_var in [
    ("mean_years_schooling", "Mean years of schooling", "msch"),
    ("expected_years_schooling", "Expected years of schooling", "esch"),
    ("education_index", "Educational Index", "edindex"),
]:
    add_variable(
        var_id,
        "Human capital",
        name,
        "GDL_SHDI",
        "subnational_region",
        "ANNUAL_WITH_INTERPOLATION_OPTIONS",
        "full_series_spatial_audit",
        documented_start=1990,
        documented_end=2023,
        source_variable=source_var,
        role_candidate="human_capital / clustering_candidate",
        measurement_type="mixed observed / interpolated / extrapolated",
        expected_flags="SOURCE_TYPE; INTERPOLATED; EXTRAPOLATED; SPATIAL_GAP; BOUNDARY_MISMATCH",
    )

# SPID social variables — actual column names
SPID_METRICS = {
    "mean2021": ("Welfare mean (2021 PPP framework)", "welfare"),
    "poor300": ("Poverty rate — $3.00 threshold", "poverty"),
    "poor420": ("Poverty rate — $4.20 threshold", "poverty"),
    "poor830": ("Poverty rate — $8.30 threshold", "poverty"),
    "prosgap2021": ("Prosperity gap", "poverty_gap"),
    "gini": ("Gini", "inequality"),
    "theil": ("Theil", "inequality"),
}
for code, (name, family) in SPID_METRICS.items():
    add_variable(
        f"spid_{code}",
        "Social",
        name,
        "SPID",
        "subnational_region",
        "SURVEY_DERIVED_ANNUAL_RECORDS",
        "actual_spid",
        documented_start=1999,
        documented_end=2022,
        source_variable=code,
        role_candidate="outcome / characterization / clustering_candidate",
        measurement_type="survey-derived",
        expected_flags="SURVEY_DEPENDENT; TEMPORAL_GAP; SOURCE_TYPE",
    )

# Demographic
for var_id, name in [
    ("age_dependency_ratio", "Age dependency ratio"),
    ("share_65_plus", "Share age 65+ / aging"),
    ("share_0_14", "Share age 0–14"),
]:
    add_variable(
        var_id,
        "Demographic",
        name,
        "WORLDPOP_AGESEX",
        "subnational_region",
        "ANNUAL",
        "full_series_source",
        documented_start=2000,
        documented_end=2020,
        source_variable="wpgpas",
        role_candidate="demographic_context / clustering_candidate",
        dependencies="age-sex raster aggregation" if var_id != "age_dependency_ratio" else "population age 0–14; population age 15–64; population age 65+",
        measurement_type="modelled raster; derived indicator",
        expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
    )
add_variable(
    "net_migration",
    "Demographic",
    "Annual net migration",
    "NIVA_MIGRATION",
    "subnational_region",
    "ANNUAL",
    "partial_series_source",
    documented_start=2000,
    documented_end=2019,
    role_candidate="demographic_dynamics / clustering_candidate",
    measurement_type="modelled annual net migration",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED; TEMPORAL_GAP; SPATIAL_GAP; ADM0_FALLBACK",
)
add_variable(
    "net_migration_rate",
    "Demographic",
    "Net migration rate",
    "DERIVED",
    "subnational_region",
    "ANNUAL",
    "derived",
    documented_start=2015,
    documented_end=2019,
    dependencies="net_migration; population",
    role_candidate="demographic_dynamics / clustering_candidate",
    measurement_type="derived",
    expected_flags="TEMPORAL_GAP",
)

# Accessibility
add_variable(
    "travel_time_to_cities_2015",
    "Accessibility",
    "Travel time to cities",
    "WEISS_ACCESS",
    "subnational_region",
    "SNAPSHOT",
    "snapshot_source",
    documented_years=[2015],
    role_candidate="structural_context / remoteness",
    measurement_type="modelled global raster",
    expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
)
for i in range(1, 10):
    add_variable(
        f"accessibility_settlement_class_{i}",
        "Accessibility",
        f"Travel time to settlement-size class {i}",
        "WEISS_SETTLEMENT_ACCESS",
        "subnational_region",
        "SNAPSHOT",
        "snapshot_source",
        documented_years=[2015],
        source_variable=f"travel_time_to_cities_{i}.tif",
        role_candidate="structural_context / remoteness",
        measurement_type="modelled global raster",
        expected_flags="SOURCE_TYPE; OBSERVED_MODELED",
    )

# Connectivity
for var_id, name in [
    ("internet_download_speed", "Internet download speed"),
    ("internet_upload_speed", "Internet upload speed"),
    ("internet_latency", "Internet latency"),
    ("internet_test_count", "Internet speed-test count"),
    ("internet_device_count", "Internet speed-test device count"),
]:
    add_variable(
        var_id,
        "Connectivity",
        name,
        "OOKLA_WB",
        "subnational_region",
        "ANNUAL_OR_QUARTERLY",
        "partial_series_source",
        documented_start=2019,
        documented_end=2020,
        role_candidate="connectivity_context / quality_control" if "count" in var_id else "connectivity_context / clustering_candidate",
        measurement_type="user-initiated Speedtest observations aggregated spatially",
        expected_flags="SOURCE_TYPE; TEMPORAL_GAP; SPATIAL_GAP; SELECTION_BIAS",
    )

# Governance
for var_id, name, note in [
    (
        "subnational_corruption_index_baseline",
        "Subnational Corruption Index — baseline",
        "Survey years only; recommended for academic analysis of observed survey information.",
    ),
    (
        "subnational_corruption_index_comprehensive",
        "Subnational Corruption Index — comprehensive",
        "Series constructed with estimation/interpolation/extrapolation; useful for descriptive comparisons.",
    ),
]:
    add_variable(
        var_id,
        "Governance",
        name,
        "GDL_SCD",
        "subnational_region",
        "SURVEY_OR_MODELLED_SERIES",
        "survey_source" if var_id.endswith("baseline") else "full_series_spatial_audit",
        documented_start=1995,
        documented_end=2022,
        role_candidate="governance_context / clustering_candidate",
        measurement_type="survey" if var_id.endswith("baseline") else "survey + estimated/interpolated/extrapolated",
        expected_flags="SURVEY_DEPENDENT; TEMPORAL_GAP; SPATIAL_GAP; BOUNDARY_MISMATCH" + ("; INTERPOLATED; EXTRAPOLATED" if var_id.endswith("comprehensive") else ""),
        temporal_notes=note,
    )

# Innovation
for var_id, name in [
    ("rd_expenditure_gdp", "R&D expenditure / GDP"),
    ("rd_personnel", "R&D personnel"),
    ("patents_per_capita", "Patents per capita"),
    ("high_tech_employment", "High-tech employment"),
]:
    add_variable(
        var_id,
        "Innovation",
        name,
        "OECD_REGIONAL",
        "subnational_region",
        "ANNUAL",
        "subset_source",
        documented_start=2015,
        documented_end=2020,
        backup_sources="EUROSTAT",
        role_candidate="innovation_context / clustering_candidate",
        measurement_type="official regional statistics",
        expected_flags="SOURCE_TYPE; SPATIAL_GAP; BOUNDARY_MISMATCH; TEMPORAL_GAP; DEFINITION_MISMATCH",
    )

# Environmental Space2Stats
for var_id, name, family, temporal in [
    ("flood_exposure", "Flood exposure", "flood_exposure", "SNAPSHOT_OR_STATIC"),
    ("drought_exposure", "Drought exposure / SPEI", "drought", "CONTEXTUAL"),
    ("cyclone_frequency", "Cyclone frequency", "cyclone", "HISTORICAL_CONTEXT"),
    ("fire_density", "Fire density", "wildfire", "CONTEXTUAL"),
    ("landslide_susceptibility", "Landslide susceptibility", "landslide", "REFERENCE_SNAPSHOT"),
]:
    add_variable(
        var_id,
        "Environmental",
        name,
        "SPACE2STATS",
        "subnational_region",
        temporal,
        "actual_space2stats_family",
        source_variable=family,
        role_candidate="environmental_risk / context / clustering_candidate",
        measurement_type="Space2Stats raster-derived/contextual indicator",
        expected_flags="SOURCE_TYPE; OBSERVED_MODELED; TEMPORAL_GAP",
    )

# SSGD candidates already identified in earlier exploratory work
for var_id, name, family in [
    ("ssgd_food_deprivation", "Food deprivation / food security — SSGD", "food_security"),
    ("ssgd_internet_access", "Internet access — SSGD", "internet_access"),
    ("ssgd_savings", "Savings — SSGD", "savings"),
    ("ssgd_government_transfers", "Government transfers — SSGD", "government_transfers"),
    ("ssgd_remittances", "Remittances — SSGD", "remittances"),
]:
    add_variable(
        var_id,
        "Supplementary social",
        name,
        "SSGD",
        "subnational_region",
        "SURVEY",
        "actual_ssgd_family",
        source_variable=family,
        role_candidate="supplementary / outcome / characterization",
        measurement_type="survey",
        expected_flags="SURVEY_DEPENDENT; TEMPORAL_GAP; SPATIAL_GAP; BOUNDARY_MISMATCH",
    )


# =============================================================================
# NATIONAL OUTCOME / CONTEXT VARIABLES — WDI
# These preserve the original substantive P1 scope in addition to the regional
# candidates listed in the supplementary PDF.
# =============================================================================

WDI_CANDIDATES = [
    # Essential services
    ("wdi_access_electricity", "Essential services", "Access to electricity (% population)", "EG.ELC.ACCS.ZS", "outcome"),
    ("wdi_clean_cooking", "Essential services", "Access to clean fuels and technologies for cooking (% population)", "EG.CFT.ACCS.ZS", "outcome"),
    ("wdi_basic_water", "Essential services", "At least basic drinking water services (% population)", "SH.H2O.BASW.ZS", "outcome"),
    ("wdi_safely_managed_water", "Essential services", "Safely managed drinking water services (% population)", "SH.H2O.SMDW.ZS", "outcome"),
    ("wdi_basic_sanitation", "Essential services", "At least basic sanitation services (% population)", "SH.STA.BASS.ZS", "outcome"),
    ("wdi_safely_managed_sanitation", "Essential services", "Safely managed sanitation services (% population)", "SH.STA.SMSS.ZS", "outcome"),
    # Food security
    ("wdi_moderate_severe_food_insecurity", "Food security", "Moderate or severe food insecurity (% population)", "SN.ITK.MSFI.ZS", "outcome"),
    ("wdi_severe_food_insecurity", "Food security", "Severe food insecurity (% population)", "SN.ITK.SVFI.ZS", "outcome"),
    ("wdi_undernourishment", "Food security", "Prevalence of undernourishment (% population)", "SN.ITK.DEFC.ZS", "outcome"),
    # Energy
    ("wdi_renewable_energy", "Energy", "Renewable energy consumption (% total final energy consumption)", "EG.FEC.RNEW.ZS", "outcome"),
    ("wdi_renewable_electricity", "Energy", "Renewable electricity output (% total electricity output)", "EG.ELC.RNEW.ZS", "outcome"),
    ("wdi_energy_intensity", "Energy", "Energy intensity level of primary energy", "EG.EGY.PRIM.PP.KD", "outcome"),
    # GHG — keep legacy and newer AR5 series to test 2015–2020 recovery
    ("wdi_ghg_legacy", "GHG emissions", "Total greenhouse gas emissions — legacy WDI series", "EN.ATM.GHGT.KT.CE", "outcome"),
    ("wdi_ghg_ar5", "GHG emissions", "Total greenhouse gas emissions excluding LULUCF — AR5", "EN.GHG.ALL.MT.CE.AR5", "outcome"),
    ("wdi_ghg_ar5_pc", "GHG emissions", "GHG emissions excluding LULUCF per capita — AR5", "EN.GHG.ALL.PC.CE.AR5", "characterization"),
    # Water
    ("wdi_water_stress", "Water", "Level of water stress", "ER.H2O.FWST.ZS", "outcome"),
    ("wdi_freshwater_withdrawal", "Water", "Annual freshwater withdrawals, total", "ER.H2O.FWTL.K3", "context"),
    ("wdi_freshwater_withdrawal_pct", "Water", "Annual freshwater withdrawals (% internal resources)", "ER.H2O.FWTL.ZS", "context"),
    # National context
    ("wdi_population", "National context", "Population, total", "SP.POP.TOTL", "context"),
    ("wdi_urban_population", "National context", "Urban population (% total population)", "SP.URB.TOTL.IN.ZS", "context"),
    ("wdi_gdp_pc_ppp", "National context", "GDP per capita, PPP", "NY.GDP.PCAP.PP.KD", "context"),
]

for var_id, block, name, code, role in WDI_CANDIDATES:
    add_variable(
        var_id,
        block,
        name,
        "WDI",
        "national",
        "ANNUAL_OR_IRREGULAR",
        "actual_wdi",
        source_variable=code,
        role_candidate=role,
        measurement_type="official/international indicator series",
        expected_flags="SOURCE_TYPE; TEMPORAL_GAP; DEFINITION_MISMATCH; METHODOLOGY_CHANGE; SOURCE_CHANGE",
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {normalize_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in mapping:
            return mapping[key]
    return None


def coerce_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        extracted = (
            series.loc[missing]
            .astype(str)
            .str.extract(r"((?:19|20)\d{2})", expand=False)
        )
        numeric.loc[missing] = pd.to_numeric(extracted, errors="coerce")
    return numeric.astype("Int64")


def years_string(years: list[int] | set[int] | np.ndarray) -> str:
    clean = sorted({int(y) for y in years if pd.notna(y)})
    return ", ".join(str(y) for y in clean)


def year_set_from_registry(row: pd.Series) -> set[int]:
    explicit = str(row.get("documented_years", "") or "").strip()
    if explicit:
        return {
            int(x.strip())
            for x in explicit.split(",")
            if x.strip().isdigit()
        }
    start = row.get("documented_start")
    end = row.get("documented_end")
    if pd.notna(start) and pd.notna(end):
        return set(range(int(start), int(end) + 1))
    return set()


def year_coverage(years: set[int], period: list[int]) -> float:
    if not years:
        return np.nan
    return len(years.intersection(period)) / len(period)


def unit_year_coverage(
    df: pd.DataFrame,
    value_col: str,
    unit_cols: list[str],
    year_col: str,
    period: list[int],
) -> dict[str, Any]:
    """Compute unit-year coverage and complete-panel share for a target period.

    Coverage = unique non-missing unit-year observations / (eligible units * years).
    Complete-unit share = units with non-missing values in every target year / eligible units.

    The function intentionally uses the units present in ``df`` as the denominator.
    Therefore, callers should pass the already selected/eligible analytical panel when
    they want coverage for a strict sample (e.g., selected SPID country-series).
    """
    required = [*unit_cols, year_col, value_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return {
            "n_units": np.nan,
            "expected_cells": np.nan,
            "observed_cells": np.nan,
            "coverage": np.nan,
            "complete_units": np.nan,
            "complete_unit_share": np.nan,
        }

    if not period:
        return {
            "n_units": 0,
            "expected_cells": 0,
            "observed_cells": 0,
            "coverage": np.nan,
            "complete_units": 0,
            "complete_unit_share": np.nan,
        }

    work = df.loc[df[year_col].isin(period), required].copy()
    work = work.dropna(subset=unit_cols + [year_col])

    units = work[unit_cols].drop_duplicates()
    n_units = len(units)
    expected_cells = n_units * len(period)

    observed = (
        work.loc[work[value_col].notna(), unit_cols + [year_col]]
        .drop_duplicates()
    )
    observed_cells = len(observed)

    coverage = (
        observed_cells / expected_cells
        if expected_cells > 0
        else np.nan
    )

    if n_units > 0:
        counts = observed.groupby(unit_cols, dropna=False)[year_col].nunique()
        complete_units = int((counts == len(period)).sum())
        complete_unit_share = complete_units / n_units
    else:
        complete_units = 0
        complete_unit_share = np.nan

    return {
        "n_units": n_units,
        "expected_cells": expected_cells,
        "observed_cells": observed_cells,
        "coverage": coverage,
        "complete_units": complete_units,
        "complete_unit_share": complete_unit_share,
    }


def period_fit_from_years(
    years: set[int],
    period: list[int],
    temporal_type: str,
    coverage_mode: str,
) -> str:
    t = temporal_type.upper()
    mode = coverage_mode.lower()

    if "SNAPSHOT" in t or "STATIC" in t or "CONTEXTUAL" in t or "HISTORICAL" in t or "REFERENCE" in t:
        return "SNAPSHOT_OR_STATIC"
    if mode in {"survey_source", "irregular_source", "subset_source"}:
        if years.intersection(period):
            return "CONDITIONAL_OR_PARTIAL"
        return "NO_TARGET_YEARS"
    if not years:
        return "UNKNOWN"
    if set(period).issubset(years):
        return "FULL_YEAR_RANGE"
    if years.intersection(period):
        return "PARTIAL_YEAR_RANGE"
    return "NO_TARGET_YEARS"


def classify_network_failure(http_status: int | None, error: str) -> str:
    """Classify reachability failures without confusing them with missing data."""
    if http_status in {401, 403}:
        return "ACCESS_RESTRICTED"
    if http_status == 404:
        return "URL_OR_RECORD_NOT_FOUND"
    if http_status is not None and http_status >= 500:
        return "TEMPORARY_SERVER_ERROR"

    text = str(error).lower()
    temporary_terms = [
        "timeout",
        "timed out",
        "connection aborted",
        "connection reset",
        "remote end closed",
        "remotedisconnected",
        "temporarily unavailable",
        "max retries exceeded",
        "name resolution",
        "dns",
    ]
    if any(term in text for term in temporary_terms):
        return "TEMPORARY_CONNECTION_ERROR"
    return "UNREACHABLE_UNKNOWN"


def safe_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    source: str = "API",
) -> tuple[Any | None, int | None, str, str]:
    last_error = ""
    last_status: int | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            last_status = response.status_code
            response.raise_for_status()
            return response.json(), last_status, "", "NONE"
        except Exception as exc:
            last_error = str(exc)
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                last_status = getattr(response_obj, "status_code", last_status)

            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * (RETRY_BACKOFF ** (attempt - 1))
                time.sleep(wait)

    failure_class = classify_network_failure(last_status, last_error)
    log(
        "WARNING",
        source,
        f"Request failed after {MAX_RETRIES} attempts "
        f"[{failure_class}]: {last_error}",
    )
    return None, last_status, last_error, failure_class


def safe_get_status(url: str, source: str) -> tuple[bool, int | None, str, str]:
    last_error = ""
    last_status: int | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            last_status = response.status_code

            if response.status_code in {401, 403}:
                return True, response.status_code, "", "ACCESS_RESTRICTED"
            if response.status_code == 404:
                return False, response.status_code, "HTTP 404", "URL_OR_RECORD_NOT_FOUND"
            if response.status_code < 500:
                return True, response.status_code, "", "NONE"

            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                last_status = getattr(response_obj, "status_code", last_status)

        if attempt < MAX_RETRIES:
            wait = RETRY_WAIT * (RETRY_BACKOFF ** (attempt - 1))
            time.sleep(wait)

    failure_class = classify_network_failure(last_status, last_error)
    log(
        "WARNING",
        source,
        f"Source reachability check failed after {MAX_RETRIES} attempts "
        f"[{failure_class}]: {last_error}",
    )
    return False, last_status, last_error, failure_class


def extract_years_from_text(text: str) -> set[int]:
    return {
        int(y)
        for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(text))
    }


# =============================================================================
# SOURCE CHECKS
# =============================================================================


def audit_sources() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    zenodo_files_rows: list[dict[str, Any]] = []

    for source_id, meta in SOURCE_REGISTRY.items():
        kind = meta["kind"]
        row = {
            "source_id": source_id,
            "source_label": meta["label"],
            "kind": kind,
            "machine_access": meta.get("machine_access", ""),
            "documented_period": meta.get("documented_period", ""),
            "documented_source": True,
            "reachable": np.nan,
            "http_status": np.nan,
            "status": "",
            "failure_class": "",
            "details": "",
            "url_or_path": "",
        }
        detail: dict[str, Any] = {}

        if kind == "local_file":
            path = Path(meta["path"])
            exists = path.exists()
            row["reachable"] = exists
            row["status"] = "LOCAL_FILE_FOUND" if exists else "LOCAL_FILE_MISSING"
            row["failure_class"] = "" if exists else "LOCAL_FILE_MISSING"
            row["url_or_path"] = str(path)
            if exists:
                row["details"] = f"size_bytes={path.stat().st_size}"
            details[source_id] = {
                "reachable": exists,
                "path": str(path),
                "failure_class": row["failure_class"],
            }

        elif kind == "derived":
            row["reachable"] = True
            row["status"] = "DERIVED"
            row["details"] = "No external source required."
            details[source_id] = {"reachable": True, "failure_class": ""}

        elif kind == "worldpop_api":
            payload, http_status, error, failure_class = safe_get_json(
                meta["url"], source=source_id
            )
            row["url_or_path"] = meta["url"]
            row["http_status"] = http_status
            row["failure_class"] = failure_class if failure_class != "NONE" else ""

            if isinstance(payload, dict):
                datasets = payload.get("datasets", {})
                expected = meta.get("expected_dataset", "")
                found = expected in datasets if isinstance(datasets, dict) else False
                row["reachable"] = True
                row["status"] = (
                    "API_DATASET_FOUND"
                    if found
                    else "API_REACHABLE_DATASET_NOT_CONFIRMED"
                )
                row["details"] = (
                    f"expected_dataset={expected}; "
                    f"available_datasets={list(datasets) if isinstance(datasets, dict) else datasets}"
                )
                detail = {
                    "reachable": True,
                    "dataset_found": found,
                    "payload": payload,
                    "failure_class": "",
                }
            else:
                row["reachable"] = False
                if failure_class.startswith("TEMPORARY_"):
                    row["status"] = "API_TEMPORARILY_UNREACHABLE"
                elif failure_class == "URL_OR_RECORD_NOT_FOUND":
                    row["status"] = "API_ENDPOINT_NOT_FOUND"
                else:
                    row["status"] = "API_UNREACHABLE"
                row["details"] = error
                detail = {
                    "reachable": False,
                    "error": error,
                    "failure_class": failure_class,
                }
            details[source_id] = detail

        elif kind == "json_api" and source_id in {"SPACE2STATS", "WDI"}:
            payload, http_status, error, failure_class = safe_get_json(
                meta["url"], source=source_id
            )
            row["url_or_path"] = meta["url"]
            row["http_status"] = http_status
            row["failure_class"] = failure_class if failure_class != "NONE" else ""
            ok = payload is not None
            row["reachable"] = ok

            if ok:
                row["status"] = "API_REACHABLE"
                row["details"] = (
                    "Space2Stats /fields"
                    if source_id == "SPACE2STATS"
                    else "World Bank API v2"
                )
            else:
                if failure_class.startswith("TEMPORARY_"):
                    row["status"] = "API_TEMPORARILY_UNREACHABLE"
                elif failure_class == "URL_OR_RECORD_NOT_FOUND":
                    row["status"] = "API_ENDPOINT_NOT_FOUND"
                else:
                    row["status"] = "API_UNREACHABLE"
                row["details"] = error

            details[source_id] = {
                "reachable": ok,
                "payload": payload,
                "error": error,
                "failure_class": failure_class,
            }

        elif kind == "zenodo_api":
            payload, http_status, error, failure_class = safe_get_json(
                meta["url"], source=source_id
            )
            row["url_or_path"] = meta.get("public_url", meta["url"])
            row["http_status"] = http_status
            row["failure_class"] = failure_class if failure_class != "NONE" else ""

            if isinstance(payload, dict):
                files = payload.get("files", []) or []
                file_names = [str(f.get("key", "")) for f in files if isinstance(f, dict)]
                patterns = [p.lower() for p in meta.get("expected_file_patterns", [])]
                hits = (
                    [
                        f
                        for f in file_names
                        if all(p in f.lower() for p in patterns)
                    ]
                    if patterns
                    else file_names
                )
                row["reachable"] = True
                row["status"] = "ZENODO_RECORD_FOUND"
                row["details"] = (
                    f"files={len(file_names)}; expected_pattern_hits={len(hits)}"
                )
                detail = {
                    "reachable": True,
                    "file_names": file_names,
                    "pattern_hits": hits,
                    "title": payload.get("metadata", {}).get("title", ""),
                    "failure_class": "",
                }
                for f in files:
                    if not isinstance(f, dict):
                        continue
                    zenodo_files_rows.append(
                        {
                            "source_id": source_id,
                            "file": f.get("key", ""),
                            "size_bytes": f.get("size", np.nan),
                            "checksum": f.get("checksum", ""),
                            "download_url": (f.get("links", {}) or {}).get("self", ""),
                        }
                    )
            else:
                row["reachable"] = False
                if failure_class.startswith("TEMPORARY_"):
                    row["status"] = "ZENODO_TEMPORARILY_UNREACHABLE"
                elif failure_class == "URL_OR_RECORD_NOT_FOUND":
                    row["status"] = "ZENODO_RECORD_NOT_FOUND"
                else:
                    row["status"] = "ZENODO_RECORD_UNREACHABLE"
                row["details"] = error
                detail = {
                    "reachable": False,
                    "error": error,
                    "failure_class": failure_class,
                }
            details[source_id] = detail

        elif kind == "web_source":
            reachable, http_status, error, failure_class = safe_get_status(
                meta["url"], source_id
            )
            row["url_or_path"] = meta["url"]
            row["reachable"] = reachable
            row["http_status"] = http_status
            row["failure_class"] = failure_class if failure_class != "NONE" else ""

            if reachable and http_status in {401, 403}:
                row["status"] = "SOURCE_EXISTS_ACCESS_RESTRICTED"
            elif reachable:
                row["status"] = "SOURCE_REACHABLE"
            elif failure_class.startswith("TEMPORARY_"):
                row["status"] = "SOURCE_TEMPORARILY_UNREACHABLE"
            elif failure_class == "URL_OR_RECORD_NOT_FOUND":
                row["status"] = "SOURCE_URL_NOT_FOUND"
            else:
                row["status"] = "SOURCE_UNREACHABLE"

            row["details"] = error
            details[source_id] = {
                "reachable": reachable,
                "http_status": http_status,
                "failure_class": failure_class,
            }

        rows.append(row)

    return pd.DataFrame(rows), details, pd.DataFrame(zenodo_files_rows)


# =============================================================================
# SPID AUDIT — v1.1
# =============================================================================
# SPID may contain more than one comparable series for a country. The valid
# analytical unit is therefore not simply code + geo_code + year. First select
# a country-series identified by code + welfaretype + comparability, then test
# temporal and spatial completeness inside that series.
# =============================================================================


def read_excel_sheet_with_required_columns(
    path: Path,
    required_columns: list[str],
    preferred_sheet: str | None = None,
) -> tuple[pd.DataFrame, str]:
    excel = pd.ExcelFile(path)
    sheet_order = []
    if preferred_sheet and preferred_sheet in excel.sheet_names:
        sheet_order.append(preferred_sheet)
    sheet_order.extend([s for s in excel.sheet_names if s not in sheet_order])

    required_norm = {normalize_name(c) for c in required_columns}
    for sheet in sheet_order:
        preview = pd.read_excel(path, sheet_name=sheet, nrows=5)
        columns_norm = {normalize_name(c) for c in preview.columns}
        if required_norm.issubset(columns_norm):
            return pd.read_excel(path, sheet_name=sheet), sheet

    raise ValueError(f"No sheet in {path.name} contains {required_columns}")


def normalize_spid_comparability(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return str(value).strip()


def evaluate_spid_series(
    series_df: pd.DataFrame,
    years: list[int],
) -> dict[str, Any]:
    subset = series_df[series_df["year"].isin(years)].copy()

    if subset.empty:
        return {
            "years_present": 0,
            "years_available": "",
            "n_regions": 0,
            "min_regions_per_year": 0,
            "max_regions_per_year": 0,
            "observed_cells": 0,
            "expected_cells": 0,
            "coverage_pct": 0.0,
            "stable_geography": False,
            "geography_complete": False,
            "duplicate_key_rows": 0,
            "metric_complete_cells": 0,
            "metric_complete_share": 0.0,
            "strict_all_metrics_complete": False,
            "surveys": 0,
            "survey_coverage_codes": "",
        }

    target_year_sets: dict[int, set[str]] = {}
    region_counts: list[int] = []

    for year in years:
        codes = set(
            subset.loc[subset["year"].eq(year), "geo_code"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        target_year_sets[year] = codes
        region_counts.append(len(codes))

    available_years = [year for year in years if target_year_sets[year]]
    years_present = len(available_years)

    stable_geography = False
    common_regions: set[str] = set()
    if years_present == len(years):
        common_regions = target_year_sets[years[0]]
        stable_geography = (
            len(common_regions) > 0
            and all(target_year_sets[year] == common_regions for year in years)
        )

    all_regions = set().union(*target_year_sets.values()) if target_year_sets else set()
    n_regions = len(common_regions) if stable_geography else len(all_regions)

    duplicate_key_rows = int(
        subset.duplicated(["geo_code", "year"], keep=False).sum()
    )

    observed_cells = len(subset[["geo_code", "year"]].drop_duplicates())
    expected_cells = n_regions * len(years) if n_regions else 0
    coverage_pct = (
        observed_cells / expected_cells * 100.0
        if expected_cells
        else 0.0
    )

    geography_complete = bool(
        years_present == len(years)
        and stable_geography
        and duplicate_key_rows == 0
        and observed_cells == expected_cells
    )

    metrics_present = [metric for metric in SPID_METRICS if metric in subset.columns]
    if len(metrics_present) == len(SPID_METRICS):
        metric_complete_mask = subset[metrics_present].notna().all(axis=1)
        metric_complete_cells = len(
            subset.loc[metric_complete_mask, ["geo_code", "year"]].drop_duplicates()
        )
    else:
        metric_complete_cells = 0

    metric_complete_share = (
        metric_complete_cells / expected_cells
        if expected_cells
        else 0.0
    )

    strict_all_metrics_complete = bool(
        geography_complete
        and len(metrics_present) == len(SPID_METRICS)
        and metric_complete_cells == expected_cells
    )

    survey_col = "survname" if "survname" in subset.columns else None
    surveys = (
        int(subset[survey_col].dropna().astype(str).nunique())
        if survey_col
        else np.nan
    )

    coverage_codes = ""
    if "survey_coverage" in subset.columns:
        coverage_codes = " | ".join(
            sorted(
                subset["survey_coverage"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
        )

    return {
        "years_present": years_present,
        "years_available": years_string(available_years),
        "n_regions": n_regions,
        "min_regions_per_year": min(region_counts) if region_counts else 0,
        "max_regions_per_year": max(region_counts) if region_counts else 0,
        "observed_cells": observed_cells,
        "expected_cells": expected_cells,
        "coverage_pct": coverage_pct,
        "stable_geography": stable_geography,
        "geography_complete": geography_complete,
        "duplicate_key_rows": duplicate_key_rows,
        "metric_complete_cells": metric_complete_cells,
        "metric_complete_share": metric_complete_share,
        "strict_all_metrics_complete": strict_all_metrics_complete,
        "surveys": surveys,
        "survey_coverage_codes": coverage_codes,
    }


def evaluate_all_spid_series(
    df: pd.DataFrame,
    years: list[int],
    period_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["code", "_welfaretype", "_comparability"]

    for (code, welfaretype, comparability), group in df.groupby(
        group_cols,
        dropna=False,
        sort=True,
    ):
        result = evaluate_spid_series(group, years)
        rows.append(
            {
                "period": period_label,
                "code": code,
                "welfaretype": welfaretype,
                "comparability": comparability,
                **result,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Deterministic quality ordering. Comparability is an identifier, not a
    # quality rank, so it is used only as the final tie-breaker.
    out["selection_score_strict"] = out["strict_all_metrics_complete"].astype(int)
    out["selection_score_geo"] = out["geography_complete"].astype(int)
    out = out.sort_values(
        [
            "code",
            "selection_score_strict",
            "selection_score_geo",
            "coverage_pct",
            "metric_complete_share",
            "years_present",
            "n_regions",
            "welfaretype",
            "comparability",
        ],
        ascending=[True, False, False, False, False, False, False, True, True],
    ).reset_index(drop=True)

    out["rank_within_country"] = out.groupby("code").cumcount() + 1
    return out


def select_best_spid_series(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df.empty:
        return series_df.copy()

    selected = series_df[series_df["rank_within_country"].eq(1)].copy()

    strict_counts = (
        series_df.groupby("code")["strict_all_metrics_complete"]
        .sum()
        .astype(int)
        .rename("n_strict_complete_series")
    )
    geo_counts = (
        series_df.groupby("code")["geography_complete"]
        .sum()
        .astype(int)
        .rename("n_geography_complete_series")
    )
    total_counts = series_df.groupby("code").size().rename("n_series_evaluated")

    selected = selected.merge(total_counts, on="code", how="left")
    selected = selected.merge(geo_counts, on="code", how="left")
    selected = selected.merge(strict_counts, on="code", how="left")
    return selected


def build_spid_selected_panel(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    years: list[int],
    strict_only: bool = True,
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=df.columns)

    pieces: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        if strict_only and not bool(row["strict_all_metrics_complete"]):
            continue

        piece = df[
            df["code"].eq(row["code"])
            & df["_welfaretype"].eq(row["welfaretype"])
            & df["_comparability"].eq(row["comparability"])
            & df["year"].isin(years)
        ].copy()
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(pieces, ignore_index=True)


def audit_spid() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not SPID_FILE.exists():
        log("ERROR", "SPID", f"Missing local file: {SPID_FILE}")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    df, sheet = read_excel_sheet_with_required_columns(
        SPID_FILE,
        ["code", "year", "geo_code"],
        preferred_sheet="Data",
    )
    df.columns = [str(c).strip() for c in df.columns]
    df["year"] = coerce_year(df["year"])
    df["code"] = (
        df["code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    df["geo_code"] = (
        df["geo_code"]
        .astype("string")
        .str.strip()
    )

    mask = pd.Series(True, index=df.index)
    if "data" in df.columns:
        mask &= df["data"].astype(str).str.strip().str.upper().eq("ALL")
    if "data_group" in df.columns:
        mask &= df["data_group"].astype(str).str.strip().str.upper().eq("ALL")

    df = df[
        mask
        & df["year"].between(2015, 2022)
        & df["code"].notna()
        & df["geo_code"].notna()
        & df["geo_code"].ne("")
    ].copy()

    if "welfaretype" in df.columns:
        df["_welfaretype"] = (
            df["welfaretype"].astype(str).str.strip().str.upper()
        )
    else:
        df["_welfaretype"] = "UNKNOWN"
        log("WARNING", "SPID", "'welfaretype' not found; using UNKNOWN series key.")

    if "comparability" in df.columns:
        df["_comparability"] = df["comparability"].map(normalize_spid_comparability)
    else:
        df["_comparability"] = "NA"
        log("WARNING", "SPID", "'comparability' not found; using NA series key.")

    log(
        "INFO",
        "SPID",
        f"Loaded sheet '{sheet}'; total-population rows 2015–2022 = {len(df):,}",
    )

    series_long = evaluate_all_spid_series(df, PERIOD_LONG, "2015-2022")
    series_short = evaluate_all_spid_series(df, PERIOD_SHORT, "2015-2020")
    selected_long = select_best_spid_series(series_long)
    selected_short = select_best_spid_series(series_short)

    all_series = pd.concat([series_long, series_short], ignore_index=True)

    country_codes = sorted(df["code"].dropna().unique())
    country_rows: list[dict[str, Any]] = []

    long_map = selected_long.set_index("code").to_dict("index") if not selected_long.empty else {}
    short_map = selected_short.set_index("code").to_dict("index") if not selected_short.empty else {}

    for code in country_codes:
        long_row = long_map.get(code, {})
        short_row = short_map.get(code, {})

        country_rows.append(
            {
                "code": code,
                "selected_welfaretype_2015_2022": long_row.get("welfaretype", ""),
                "selected_comparability_2015_2022": long_row.get("comparability", ""),
                "geography_complete_2015_2022": bool(long_row.get("geography_complete", False)),
                "strict_all_metrics_complete_2015_2022": bool(long_row.get("strict_all_metrics_complete", False)),
                "regions_2015_2022": int(long_row.get("n_regions", 0) or 0),
                "duplicate_key_rows_2015_2022": int(long_row.get("duplicate_key_rows", 0) or 0),
                "coverage_pct_2015_2022": float(long_row.get("coverage_pct", 0.0) or 0.0),
                "metric_complete_share_2015_2022": float(long_row.get("metric_complete_share", 0.0) or 0.0),
                "n_series_evaluated_2015_2022": int(long_row.get("n_series_evaluated", 0) or 0),
                "n_strict_complete_series_2015_2022": int(long_row.get("n_strict_complete_series", 0) or 0),
                "selected_welfaretype_2015_2020": short_row.get("welfaretype", ""),
                "selected_comparability_2015_2020": short_row.get("comparability", ""),
                "geography_complete_2015_2020": bool(short_row.get("geography_complete", False)),
                "strict_all_metrics_complete_2015_2020": bool(short_row.get("strict_all_metrics_complete", False)),
                "regions_2015_2020": int(short_row.get("n_regions", 0) or 0),
                "duplicate_key_rows_2015_2020": int(short_row.get("duplicate_key_rows", 0) or 0),
                "coverage_pct_2015_2020": float(short_row.get("coverage_pct", 0.0) or 0.0),
                "metric_complete_share_2015_2020": float(short_row.get("metric_complete_share", 0.0) or 0.0),
                "n_series_evaluated_2015_2020": int(short_row.get("n_series_evaluated", 0) or 0),
                "n_strict_complete_series_2015_2020": int(short_row.get("n_strict_complete_series", 0) or 0),
                "country_recovered_geographically_2015_2020": (
                    bool(short_row.get("geography_complete", False))
                    and not bool(long_row.get("geography_complete", False))
                ),
                "country_recovered_strict_2015_2020": (
                    bool(short_row.get("strict_all_metrics_complete", False))
                    and not bool(long_row.get("strict_all_metrics_complete", False))
                ),
            }
        )

    countries_df = pd.DataFrame(country_rows)

    primary_panel = build_spid_selected_panel(
        df,
        selected_short,
        PERIOD_SHORT,
        strict_only=True,
    )
    benchmark_panel = build_spid_selected_panel(
        df,
        selected_long,
        PERIOD_LONG,
        strict_only=True,
    )

    metric_rows: list[dict[str, Any]] = []
    for metric in SPID_METRICS:
        if metric not in df.columns:
            metric_rows.append(
                {
                    "source_variable": metric,
                    "years_available": "",
                    "period_fit_2015_2022": "VARIABLE_NOT_FOUND",
                    "period_fit_2015_2020": "VARIABLE_NOT_FOUND",
                    "coverage_2015_2022": 0.0,
                    "coverage_2015_2020": 0.0,
                    "complete_unit_share_2015_2022": 0.0,
                    "complete_unit_share_2015_2020": 0.0,
                    "recovered_2015_2020": False,
                }
            )
            continue

        years_all = set(df.loc[df[metric].notna(), "year"].dropna().astype(int).unique())

        if benchmark_panel.empty:
            cov_long = {"coverage": np.nan, "complete_unit_share": np.nan}
        else:
            cov_long = unit_year_coverage(
                benchmark_panel,
                metric,
                ["code", "geo_code"],
                "year",
                PERIOD_LONG,
            )

        if primary_panel.empty:
            cov_short = {"coverage": np.nan, "complete_unit_share": np.nan}
        else:
            cov_short = unit_year_coverage(
                primary_panel,
                metric,
                ["code", "geo_code"],
                "year",
                PERIOD_SHORT,
            )

        fit_long = period_fit_from_years(
            years_all,
            PERIOD_LONG,
            "ANNUAL",
            "full_series_source",
        )
        fit_short = period_fit_from_years(
            years_all,
            PERIOD_SHORT,
            "ANNUAL",
            "full_series_source",
        )

        metric_rows.append(
            {
                "source_variable": metric,
                "years_available": years_string(years_all),
                "period_fit_2015_2022": fit_long,
                "period_fit_2015_2020": fit_short,
                "coverage_2015_2022": cov_long["coverage"],
                "coverage_2015_2020": cov_short["coverage"],
                "complete_unit_share_2015_2022": cov_long["complete_unit_share"],
                "complete_unit_share_2015_2020": cov_short["complete_unit_share"],
                "recovered_2015_2020": (
                    fit_long != "FULL_YEAR_RANGE"
                    and fit_short == "FULL_YEAR_RANGE"
                ),
            }
        )

    selected_series = pd.concat(
        [
            selected_long.assign(selected_for="2015-2022"),
            selected_short.assign(selected_for="2015-2020"),
        ],
        ignore_index=True,
    )

    strict_long = int(countries_df["strict_all_metrics_complete_2015_2022"].sum())
    strict_short = int(countries_df["strict_all_metrics_complete_2015_2020"].sum())
    recovered_strict = int(countries_df["country_recovered_strict_2015_2020"].sum())

    log(
        "INFO",
        "SPID",
        (
            f"Comparable series evaluated: long={len(series_long)}, short={len(series_short)}; "
            f"strict countries 2015–2022={strict_long}; "
            f"strict countries 2015–2020={strict_short}; "
            f"recovered={recovered_strict}."
        ),
    )

    return (
        df,
        countries_df,
        pd.DataFrame(metric_rows),
        selected_series,
        all_series,
    )


# =============================================================================
# SPACE2STATS AUDIT
# =============================================================================

SPACE2STATS_RULES: list[tuple[str, list[str]]] = [
    ("population", [r"^sum_pop_"]),
    ("nighttime_lights", [r"viirs", r"ntl", r"night.*light"]),
    ("urbanization_ghs", [r"^ghs_"]),
    ("built_up_area", [r"built.*area", r"built_area"]),
    ("flood_exposure", [r"flood"]),
    ("drought", [r"drought", r"spei"]),
    ("cyclone", [r"cyclone", r"tropical.*storm", r"hurricane"]),
    ("wildfire", [r"wildfire", r"fire", r"burned", r"burnt"]),
    ("landslide", [r"landslide"]),
]

SPACE2STATS_EXPECTED_FAMILIES = [family for family, _ in SPACE2STATS_RULES]


def extract_space2stats_fields(payload: Any) -> list[str]:
    fields: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                fields.append(item)
            elif isinstance(item, dict):
                for key in ["field", "name", "id", "variable", "property"]:
                    if isinstance(item.get(key), str):
                        fields.append(item[key])
                        break
    elif isinstance(payload, dict):
        for key in ["fields", "data", "results"]:
            if key in payload:
                return extract_space2stats_fields(payload[key])
        for key in payload:
            if re.match(r"^[A-Za-z][A-Za-z0-9_]+$", str(key)):
                fields.append(str(key))
    return sorted(set(f.strip() for f in fields if str(f).strip()))


def classify_s2s_field(field: str) -> str:
    text = field.lower()
    for family, patterns in SPACE2STATS_RULES:
        if any(re.search(p, text) for p in patterns):
            return family
    return "other_space2stats"


def audit_space2stats(
    source_details: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = source_details.get("SPACE2STATS", {}).get("payload")
    fields = extract_space2stats_fields(payload) if payload is not None else []
    if not fields:
        log("WARNING", "Space2Stats", "No fields were recovered from /fields.")
        return pd.DataFrame(), pd.DataFrame()

    all_rows = []
    for field in fields:
        family = classify_s2s_field(field)
        years = extract_years_from_text(field)
        all_rows.append(
            {
                "field": field,
                "family": family,
                "candidate_family": family != "other_space2stats",
                "years_in_field": years_string(years),
            }
        )
    all_df = pd.DataFrame(all_rows)

    family_rows = []
    for family in SPACE2STATS_EXPECTED_FAMILIES:
        group = all_df[
            all_df["candidate_family"]
            & all_df["family"].eq(family)
        ].copy()

        years: set[int] = set()
        for field in group["field"] if not group.empty else []:
            years |= extract_years_from_text(field)

        if family == "urbanization_ghs":
            temporal = "STATIC_OR_UNDATED"
            mode = "snapshot_source"
        elif family == "built_up_area":
            temporal = "SNAPSHOT_SERIES"
            mode = "snapshot_source"
        elif family in {"population", "nighttime_lights"}:
            temporal = "ANNUAL_FIELD_SERIES"
            mode = "full_series_source"
        elif years:
            temporal = "MULTIYEAR_OR_SNAPSHOT_FIELDS"
            mode = "irregular_source"
        else:
            temporal = "STATIC_OR_UNDATED"
            mode = "snapshot_source"

        if group.empty:
            field_status = "NOT_AVAILABLE_IN_CURRENT_SPACE2STATS_FIELDS"
            fit_long = "NOT_AVAILABLE_IN_CURRENT_FIELDS"
            fit_short = "NOT_AVAILABLE_IN_CURRENT_FIELDS"
        else:
            field_status = "FIELDS_CONFIRMED"
            fit_long = period_fit_from_years(years, PERIOD_LONG, temporal, mode)
            fit_short = period_fit_from_years(years, PERIOD_SHORT, temporal, mode)

        family_rows.append(
            {
                "family": family,
                "n_fields": len(group),
                "fields": " | ".join(group["field"].tolist()) if not group.empty else "",
                "years_available": years_string(years),
                "temporal_type": temporal,
                "field_status": field_status,
                "period_fit_2015_2022": fit_long,
                "period_fit_2015_2020": fit_short,
                "year_coverage_2015_2022": year_coverage(years, PERIOD_LONG) if years else np.nan,
                "year_coverage_2015_2020": year_coverage(years, PERIOD_SHORT) if years else np.nan,
            }
        )

    log(
        "INFO",
        "Space2Stats",
        (
            f"Fields detected: {len(all_df)}; "
            f"candidate-family fields: {int(all_df['candidate_family'].sum())}; "
            f"missing expected families: "
            f"{', '.join(pd.DataFrame(family_rows).loc[pd.DataFrame(family_rows)['n_fields'].eq(0), 'family'].tolist()) or 'none'}"
        ),
    )
    return all_df, pd.DataFrame(family_rows)


# =============================================================================
# SSGD AUDIT
# =============================================================================


SSGD_EXPLICIT_FAMILY_MAP = {
    "si_intuse": "internet_access",
    "re_enofoo": "food_security",
    "re_savmon": "savings",
    "re_govtra": "government_transfers",
    "re_rem": "remittances",
}


def classify_ssgd_family(code: str, label: str) -> str:
    code_clean = str(code).strip().lower()
    if code_clean in SSGD_EXPLICIT_FAMILY_MAP:
        return SSGD_EXPLICIT_FAMILY_MAP[code_clean]

    text = f"{code} {label}".lower()
    if any(x in text for x in ["food", "hunger", "hungry", "depriv"]):
        return "food_security"
    if any(x in text for x in ["internet", "intuse"]):
        return "internet_access"
    if any(x in text for x in ["saving", "savings", "savmon", "save "]):
        return "savings"
    if any(x in text for x in ["government transfer", "govtra", "transfer"]):
        return "government_transfers"
    if any(x in text for x in ["remit", "remittance", "re_rem"]):
        return "remittances"
    return "other_ssgd_subnational"


def audit_ssgd() -> pd.DataFrame:
    if not SSGD_FILE.exists():
        log("ERROR", "SSGD", f"Missing local file: {SSGD_FILE}")
        return pd.DataFrame()

    excel = pd.ExcelFile(SSGD_FILE)
    df = None
    sheet_used = None
    for sheet in excel.sheet_names:
        preview = pd.read_excel(SSGD_FILE, sheet_name=sheet, nrows=5)
        if find_column(preview, ["period", "year"]) and find_column(preview, ["value", "val"]):
            df = pd.read_excel(SSGD_FILE, sheet_name=sheet)
            sheet_used = sheet
            break
    if df is None:
        log("ERROR", "SSGD", "No long-format sheet with year/period and value was found.")
        return pd.DataFrame()

    area_col = find_column(df, ["area"])
    year_col = find_column(df, ["period", "year"])
    value_col = find_column(df, ["value", "val"])
    indicator_col = find_column(df, ["short", "indicator_code", "indicatorcode", "indicator", "variable"])
    label_col = find_column(df, ["indicator_name", "indicatorname", "label", "description", "long"])
    country_col = find_column(df, ["countrycode", "country_code", "iso3", "code"])
    unit_col = find_column(df, ["category", "adm1", "region", "subnational_unit", "subnationalunit"])

    if not all([year_col, value_col, indicator_col]):
        log("ERROR", "SSGD", "Could not detect required indicator/year/value columns.")
        return pd.DataFrame()

    if area_col:
        sub = df[df[area_col].astype(str).str.strip().str.lower().eq("subnational")].copy()
    else:
        sub = df.copy()
        log("WARNING", "SSGD", "'area' column not detected; true-subnational filter could not be applied.")

    sub["_year"] = coerce_year(sub[year_col])
    sub["_value"] = pd.to_numeric(sub[value_col], errors="coerce")
    if country_col:
        sub["_country"] = sub[country_col].astype(str).str.strip().str.upper()
    if unit_col:
        sub["_unit"] = sub[unit_col].astype(str).str.strip()

    rows = []
    for code in sorted(sub[indicator_col].dropna().astype(str).str.strip().unique()):
        item = sub[sub[indicator_col].astype(str).str.strip().eq(code)].copy()
        label = ""
        if label_col:
            labels = item[label_col].dropna().astype(str).str.strip().unique()
            if len(labels):
                label = labels[0]
        family = classify_ssgd_family(code, label)
        years = set(item.loc[item["_value"].notna(), "_year"].dropna().astype(int).unique())

        cov_long = np.nan
        cov_short = np.nan
        complete_long = np.nan
        complete_short = np.nan
        if country_col and unit_col:
            c1 = unit_year_coverage(item, "_value", ["_country", "_unit"], "_year", PERIOD_LONG)
            c2 = unit_year_coverage(item, "_value", ["_country", "_unit"], "_year", PERIOD_SHORT)
            cov_long = c1["coverage"]
            cov_short = c2["coverage"]
            complete_long = c1["complete_unit_share"]
            complete_short = c2["complete_unit_share"]

        rows.append(
            {
                "indicator_code": code,
                "indicator_name": label or code,
                "family": family,
                "records": len(item),
                "nonmissing_values": int(item["_value"].notna().sum()),
                "years_available": years_string(years),
                "period_fit_2015_2022": period_fit_from_years(years, PERIOD_LONG, "SURVEY", "survey_source"),
                "period_fit_2015_2020": period_fit_from_years(years, PERIOD_SHORT, "SURVEY", "survey_source"),
                "unit_year_coverage_2015_2022": cov_long,
                "unit_year_coverage_2015_2020": cov_short,
                "complete_unit_share_2015_2022": complete_long,
                "complete_unit_share_2015_2020": complete_short,
            }
        )

    log("INFO", "SSGD", f"Loaded sheet '{sheet_used}'; true-subnational rows={len(sub):,}; indicators={len(rows)}")
    return pd.DataFrame(rows)


# =============================================================================
# WDI AUDIT
# =============================================================================


def wdi_metadata(code: str) -> dict[str, Any] | None:
    payload, _, _, _ = safe_get_json(
        f"{WDI_API}/indicator/{code}",
        params={"format": "json", "per_page": 100},
        source=f"WDI {code}",
    )
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return None
    item = payload[1][0]
    return {
        "name": item.get("name", ""),
        "source_note": item.get("sourceNote", ""),
        "source_organization": item.get("sourceOrganization", ""),
    }


def wdi_data(code: str, countries: list[str]) -> pd.DataFrame:
    if not countries:
        return pd.DataFrame(columns=["country_code", "year", "value"])

    rows: list[dict[str, Any]] = []
    # Keep URL lengths reasonable.
    for start in range(0, len(countries), 25):
        chunk = countries[start : start + 25]
        path = ";".join(chunk)
        payload, _, _, _ = safe_get_json(
            f"{WDI_API}/country/{path}/indicator/{code}",
            params={
                "format": "json",
                "date": "2015:2022",
                "per_page": 20000,
            },
            source=f"WDI {code}",
        )
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            continue
        for item in payload[1]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "country_code": str(item.get("countryiso3code", "")).upper(),
                    "year": pd.to_numeric(item.get("date"), errors="coerce"),
                    "value": pd.to_numeric(item.get("value"), errors="coerce"),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["country_code", "year", "value"])
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    return result


def wdi_coverage(data: pd.DataFrame, countries: list[str], years: list[int]) -> dict[str, Any]:
    expected = len(countries) * len(years)
    subset = (
        data[
            data["country_code"].isin(countries)
            & data["year"].isin(years)
            & data["value"].notna()
        ][["country_code", "year"]]
        .drop_duplicates()
    )
    observed = len(subset)
    coverage = observed / expected if expected else np.nan
    if countries:
        counts = subset.groupby("country_code")["year"].nunique()
        complete = int((counts == len(years)).sum())
        complete_share = complete / len(countries)
    else:
        complete = 0
        complete_share = np.nan
    return {
        "expected_cells": expected,
        "observed_cells": observed,
        "coverage": coverage,
        "complete_countries": complete,
        "complete_country_share": complete_share,
    }


def audit_wdi(countries: list[str]) -> pd.DataFrame:
    rows = []
    unique_codes = []
    seen = set()
    for _, _, _, code, _ in WDI_CANDIDATES:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    log("INFO", "WDI", f"Auditing {len(unique_codes)} indicators for {len(countries)} countries.")

    for i, code in enumerate(unique_codes, start=1):
        print(f"  WDI [{i:02d}/{len(unique_codes):02d}] {code}")
        meta = wdi_metadata(code)
        if meta is None:
            rows.append(
                {
                    "indicator_code": code,
                    "indicator_name": "",
                    "api_valid": False,
                    "years_available": "",
                    "period_fit_2015_2022": "INDICATOR_NOT_FOUND",
                    "period_fit_2015_2020": "INDICATOR_NOT_FOUND",
                    "country_year_coverage_2015_2022": 0.0,
                    "country_year_coverage_2015_2020": 0.0,
                    "complete_country_share_2015_2022": 0.0,
                    "complete_country_share_2015_2020": 0.0,
                    "recovered_2015_2020": False,
                    "source_organization": "",
                }
            )
            continue

        data = wdi_data(code, countries)
        years = set(data.loc[data["value"].notna(), "year"].dropna().astype(int).unique())
        c_long = wdi_coverage(data, countries, PERIOD_LONG)
        c_short = wdi_coverage(data, countries, PERIOD_SHORT)
        fit_long = period_fit_from_years(years, PERIOD_LONG, "ANNUAL", "full_series_source")
        fit_short = period_fit_from_years(years, PERIOD_SHORT, "ANNUAL", "full_series_source")
        rows.append(
            {
                "indicator_code": code,
                "indicator_name": meta["name"],
                "api_valid": True,
                "years_available": years_string(years),
                "period_fit_2015_2022": fit_long,
                "period_fit_2015_2020": fit_short,
                "country_year_coverage_2015_2022": c_long["coverage"],
                "country_year_coverage_2015_2020": c_short["coverage"],
                "complete_country_share_2015_2022": c_long["complete_country_share"],
                "complete_country_share_2015_2020": c_short["complete_country_share"],
                "recovered_2015_2020": fit_long != "FULL_YEAR_RANGE" and fit_short == "FULL_YEAR_RANGE",
                "source_organization": meta["source_organization"],
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# CONCEPTUAL REGISTRY -> AVAILABILITY MATRIX
# =============================================================================


def registry_period_assessment(row: pd.Series) -> tuple[str, str, bool, float, float]:
    years = year_set_from_registry(row)
    fit_long = period_fit_from_years(years, PERIOD_LONG, row["temporal_type"], row["coverage_mode"])
    fit_short = period_fit_from_years(years, PERIOD_SHORT, row["temporal_type"], row["coverage_mode"])
    recovered = fit_long != "FULL_YEAR_RANGE" and fit_short == "FULL_YEAR_RANGE"
    return (
        fit_long,
        fit_short,
        recovered,
        year_coverage(years, PERIOD_LONG),
        year_coverage(years, PERIOD_SHORT),
    )


def build_availability_matrix(
    registry: pd.DataFrame,
    source_checks: pd.DataFrame,
    spid_metrics: pd.DataFrame,
    s2s_families: pd.DataFrame,
    ssgd_indicators: pd.DataFrame,
    wdi_df: pd.DataFrame,
) -> pd.DataFrame:
    source_status = (
        source_checks.set_index("source_id")["status"].to_dict()
        if not source_checks.empty
        else {}
    )
    source_reachable = (
        source_checks.set_index("source_id")["reachable"].to_dict()
        if not source_checks.empty
        else {}
    )

    spid_map = spid_metrics.set_index("source_variable").to_dict("index") if not spid_metrics.empty else {}
    s2s_map = s2s_families.set_index("family").to_dict("index") if not s2s_families.empty else {}
    wdi_map = wdi_df.set_index("indicator_code").to_dict("index") if not wdi_df.empty else {}

    ssgd_family_summary: dict[str, dict[str, Any]] = {}
    if not ssgd_indicators.empty:
        for family, group in ssgd_indicators.groupby("family"):
            years: set[int] = set()
            for text in group["years_available"].fillna(""):
                years |= {int(x.strip()) for x in str(text).split(",") if x.strip().isdigit()}
            ssgd_family_summary[family] = {
                "n_indicators": len(group),
                "years": years,
                "fit_long": period_fit_from_years(years, PERIOD_LONG, "SURVEY", "survey_source"),
                "fit_short": period_fit_from_years(years, PERIOD_SHORT, "SURVEY", "survey_source"),
                "coverage_long": group["unit_year_coverage_2015_2022"].max(skipna=True),
                "coverage_short": group["unit_year_coverage_2015_2020"].max(skipna=True),
                "complete_long": group["complete_unit_share_2015_2022"].max(skipna=True),
                "complete_short": group["complete_unit_share_2015_2020"].max(skipna=True),
            }

    rows = []
    for _, r in registry.iterrows():
        out = r.to_dict()
        source_id = r["source_id"]
        mode = r["coverage_mode"]
        fit_long, fit_short, recovered, yc_long, yc_short = registry_period_assessment(r)

        out.update(
            {
                "source_status": source_status.get(source_id, "NOT_CHECKED"),
                "source_reachable": source_reachable.get(source_id, np.nan),
                "years_available_actual_or_documented": r.get("documented_years", ""),
                "period_fit_2015_2022": fit_long,
                "period_fit_2015_2020": fit_short,
                "year_coverage_2015_2022": yc_long,
                "year_coverage_2015_2020": yc_short,
                "unit_year_coverage_2015_2022": np.nan,
                "unit_year_coverage_2015_2020": np.nan,
                "complete_unit_share_2015_2022": np.nan,
                "complete_unit_share_2015_2020": np.nan,
                "recovered_2015_2020": recovered,
                "availability_status": "DOCUMENTED_SOURCE_AUDIT_PENDING",
            }
        )

        if mode == "actual_spid":
            actual = spid_map.get(r["source_variable"])
            if actual:
                out["years_available_actual_or_documented"] = actual["years_available"]
                out["period_fit_2015_2022"] = actual["period_fit_2015_2022"]
                out["period_fit_2015_2020"] = actual["period_fit_2015_2020"]
                out["unit_year_coverage_2015_2022"] = actual["coverage_2015_2022"]
                out["unit_year_coverage_2015_2020"] = actual["coverage_2015_2020"]
                out["complete_unit_share_2015_2022"] = actual["complete_unit_share_2015_2022"]
                out["complete_unit_share_2015_2020"] = actual["complete_unit_share_2015_2020"]
                out["recovered_2015_2020"] = actual["recovered_2015_2020"]
                out["availability_status"] = "ACTUAL_LOCAL_DATA_AUDITED"
            else:
                out["availability_status"] = "SPID_VARIABLE_NOT_FOUND"

        elif mode == "actual_space2stats_family":
            actual = s2s_map.get(r["source_variable"])
            if actual:
                out["years_available_actual_or_documented"] = actual["years_available"]
                out["period_fit_2015_2022"] = actual["period_fit_2015_2022"]
                out["period_fit_2015_2020"] = actual["period_fit_2015_2020"]
                out["year_coverage_2015_2022"] = actual["year_coverage_2015_2022"]
                out["year_coverage_2015_2020"] = actual["year_coverage_2015_2020"]
                out["recovered_2015_2020"] = (
                    actual["period_fit_2015_2022"] != "FULL_YEAR_RANGE"
                    and actual["period_fit_2015_2020"] == "FULL_YEAR_RANGE"
                )

                if actual.get("field_status") == "NOT_AVAILABLE_IN_CURRENT_SPACE2STATS_FIELDS":
                    out["availability_status"] = "NOT_AVAILABLE_IN_CURRENT_SPACE2STATS_FIELDS"
                else:
                    out["availability_status"] = "SPACE2STATS_FIELDS_CONFIRMED"
            else:
                out["availability_status"] = "SPACE2STATS_FAMILY_NOT_FOUND"

        elif mode == "actual_ssgd_family":
            actual = ssgd_family_summary.get(r["source_variable"])
            if actual:
                out["years_available_actual_or_documented"] = years_string(actual["years"])
                out["period_fit_2015_2022"] = actual["fit_long"]
                out["period_fit_2015_2020"] = actual["fit_short"]
                out["unit_year_coverage_2015_2022"] = actual["coverage_long"]
                out["unit_year_coverage_2015_2020"] = actual["coverage_short"]
                out["complete_unit_share_2015_2022"] = actual["complete_long"]
                out["complete_unit_share_2015_2020"] = actual["complete_short"]
                out["recovered_2015_2020"] = False
                out["availability_status"] = f"SSGD_FAMILY_FOUND_{actual['n_indicators']}_INDICATORS"
            else:
                out["availability_status"] = "SSGD_FAMILY_NOT_FOUND"

        elif mode == "actual_wdi":
            actual = wdi_map.get(r["source_variable"])
            if actual:
                out["years_available_actual_or_documented"] = actual["years_available"]
                out["period_fit_2015_2022"] = actual["period_fit_2015_2022"]
                out["period_fit_2015_2020"] = actual["period_fit_2015_2020"]
                out["unit_year_coverage_2015_2022"] = actual["country_year_coverage_2015_2022"]
                out["unit_year_coverage_2015_2020"] = actual["country_year_coverage_2015_2020"]
                out["complete_unit_share_2015_2022"] = actual["complete_country_share_2015_2022"]
                out["complete_unit_share_2015_2020"] = actual["complete_country_share_2015_2020"]
                out["recovered_2015_2020"] = actual["recovered_2015_2020"]
                out["availability_status"] = "WDI_API_AUDITED" if actual["api_valid"] else "WDI_INDICATOR_NOT_FOUND"
            else:
                out["availability_status"] = "WDI_NOT_AUDITED"

        elif mode == "derived":
            out["availability_status"] = "DERIVABLE_AFTER_INPUT_EXTRACTION"

        elif mode == "full_series_source":
            current_status = str(source_status.get(source_id, "NOT_CHECKED"))
            if bool(source_reachable.get(source_id, False)):
                out["availability_status"] = "SOURCE_CONFIRMED_PERIOD_COMPATIBLE"
            elif "TEMPORARILY_UNREACHABLE" in current_status:
                out["availability_status"] = "SOURCE_DOCUMENTED_TEMPORARILY_UNREACHABLE_RETRY_SCRIPT_02"
            elif "NOT_FOUND" in current_status:
                out["availability_status"] = "SOURCE_URL_OR_RECORD_NOT_FOUND_REVIEW_REQUIRED"
            else:
                out["availability_status"] = "SOURCE_DOCUMENTED_NOT_REACHED_RETRY_SCRIPT_02"

        elif mode in {"irregular_source", "survey_source", "subset_source", "partial_series_source", "full_series_spatial_audit", "snapshot_source"}:
            current_status = str(source_status.get(source_id, "NOT_CHECKED"))
            if bool(source_reachable.get(source_id, False)):
                out["availability_status"] = "SOURCE_CONFIRMED_COVERAGE_TO_AUDIT_IN_SCRIPT_02"
            elif "TEMPORARILY_UNREACHABLE" in current_status:
                out["availability_status"] = "SOURCE_DOCUMENTED_TEMPORARILY_UNREACHABLE_RETRY_SCRIPT_02"
            elif "NOT_FOUND" in current_status:
                out["availability_status"] = "SOURCE_URL_OR_RECORD_NOT_FOUND_REVIEW_REQUIRED"
            else:
                out["availability_status"] = "SOURCE_DOCUMENTED_NOT_REACHED_RETRY_SCRIPT_02"

        out["primary_period"] = "2015-2020"
        out["primary_period_fit"] = out.get("period_fit_2015_2020", "")
        out["benchmark_period"] = "2015-2022"
        out["benchmark_period_fit"] = out.get("period_fit_2015_2022", "")
        rows.append(out)

    return pd.DataFrame(rows)


# =============================================================================
# PERIOD COMPARISON / QA FLAGS / SUMMARY
# =============================================================================

QA_FLAG_DEFINITIONS = pd.DataFrame(
    [
        ("SOURCE_TYPE", "Record source type and whether the value comes from survey, administrative data, raster/model, satellite or derived calculation."),
        ("OBSERVED_MODELED", "Distinguish directly observed/reported values from modelled/downscaled values."),
        ("TEMPORAL_GAP", "One or more target years are unavailable or not observed."),
        ("SPATIAL_GAP", "Native geography does not cover all portfolio regions/countries."),
        ("BOUNDARY_MISMATCH", "Native regional boundaries do not coincide with portfolio subnational units."),
        ("DEFINITION_MISMATCH", "Indicator definition differs across source/country/time."),
        ("SOURCE_CHANGE", "Underlying source changes across time."),
        ("METHODOLOGY_CHANGE", "Methodology changes across time."),
        ("DENOMINATOR_MISMATCH", "Indicator denominator is not consistent across observations."),
        ("SURVEY_DEPENDENT", "Availability depends on survey waves rather than an annual reporting system."),
        ("INTERPOLATED", "Value is interpolated between observed years."),
        ("EXTRAPOLATED", "Value is extrapolated beyond observed years."),
        ("ADM0_FALLBACK", "National information is used where subnational information is absent."),
        ("SELECTION_BIAS", "Observed sample may not represent the underlying population, e.g. voluntary Speedtest users."),
    ],
    columns=["flag", "definition"],
)


def build_period_comparison(availability: pd.DataFrame, spid_countries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "metric": "Registered candidate variables",
            "2015_2022": len(availability),
            "2015_2020": len(availability),
            "difference": 0,
        }
    )
    long_full = int(availability["period_fit_2015_2022"].eq("FULL_YEAR_RANGE").sum())
    short_full = int(availability["period_fit_2015_2020"].eq("FULL_YEAR_RANGE").sum())
    recovered = int(availability["recovered_2015_2020"].fillna(False).astype(bool).sum())
    rows.extend(
        [
            {
                "metric": "Variables with full target-year range",
                "2015_2022": long_full,
                "2015_2020": short_full,
                "difference": short_full - long_full,
            },
            {
                "metric": "Variables recovered by dropping 2021–2022",
                "2015_2022": 0,
                "2015_2020": recovered,
                "difference": recovered,
            },
            {
                "metric": "Variables classified as snapshot/static",
                "2015_2022": int(availability["period_fit_2015_2022"].eq("SNAPSHOT_OR_STATIC").sum()),
                "2015_2020": int(availability["period_fit_2015_2020"].eq("SNAPSHOT_OR_STATIC").sum()),
                "difference": 0,
            },
        ]
    )

    if not spid_countries.empty:
        for prefix, label in [
            ("geography_complete", "SPID countries with stable complete geography"),
            ("strict_all_metrics_complete", "SPID countries with stable geography + all SPID metrics complete"),
        ]:
            long_n = int(spid_countries[f"{prefix}_2015_2022"].sum())
            short_n = int(spid_countries[f"{prefix}_2015_2020"].sum())
            rows.append(
                {
                    "metric": label,
                    "2015_2022": long_n,
                    "2015_2020": short_n,
                    "difference": short_n - long_n,
                }
            )

    return pd.DataFrame(rows)


def build_summary(
    registry: pd.DataFrame,
    availability: pd.DataFrame,
    source_checks: pd.DataFrame,
    spid_countries: pd.DataFrame,
    s2s_all: pd.DataFrame,
    ssgd_indicators: pd.DataFrame,
    wdi_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = [
        ("Registered candidate variables", len(registry)),
        ("Registered source adapters/checks", len(source_checks)),
        ("Reachable/local sources", int(source_checks["reachable"].fillna(False).astype(bool).sum()) if not source_checks.empty else 0),
        ("Variables recovered in 2015–2020", int(availability["recovered_2015_2020"].fillna(False).astype(bool).sum())),
        ("Space2Stats fields discovered", len(s2s_all)),
        ("SSGD true-subnational indicators discovered", len(ssgd_indicators)),
        ("WDI indicators audited", len(wdi_df)),
        ("SPID countries audited", len(spid_countries)),
    ]
    if not spid_countries.empty:
        summary.extend(
            [
                ("SPID stable-geography countries 2015–2022", int(spid_countries["geography_complete_2015_2022"].sum())),
                ("SPID stable-geography countries 2015–2020", int(spid_countries["geography_complete_2015_2020"].sum())),
                ("SPID strict all-metric countries 2015–2022", int(spid_countries["strict_all_metrics_complete_2015_2022"].sum())),
                ("SPID strict all-metric countries 2015–2020", int(spid_countries["strict_all_metrics_complete_2015_2020"].sum())),
            ]
        )
    summary.extend(
        [
            ("Primary target period", "2015–2020"),
            ("Benchmark comparison period", "2015–2022"),
            ("Output workbook", str(OUTPUT_FILE)),
        ]
    )
    return pd.DataFrame(summary, columns=["metric", "value"])


# =============================================================================
# EXCEL FORMATTING
# =============================================================================


def format_excel(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    for ws in wb.worksheets:
        if ws.max_row >= 2:
            ws.freeze_panes = "A2"
        if ws.max_row >= 1 and ws.max_column >= 1:
            ws.auto_filter.ref = ws.dimensions
        for idx in range(1, ws.max_column + 1):
            letter = get_column_letter(idx)
            max_len = 0
            for cell in ws[letter]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 55)
    wb.save(path)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    script_start = time.perf_counter()

    print("=" * 88)
    print("SCRIPT 01 v1.1 — CONSOLIDATED AVAILABILITY AUDIT")
    print("=" * 88)
    print(f"Project root: {ROOT}")
    print(f"Primary period: {PERIOD_SHORT[0]}–{PERIOD_SHORT[-1]}")
    print(f"Benchmark period: {PERIOD_LONG[0]}–{PERIOD_LONG[-1]}")
    print()

    registry = pd.DataFrame(VARIABLES)

    # -------------------------------------------------------------------------
    # 1. Source reachability / metadata
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("1/5 — SOURCE CHECKS")
    print("=" * 88)
    source_checks, source_details, zenodo_files = audit_sources()
    print()

    # -------------------------------------------------------------------------
    # 2. SPID
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("2/5 — SPID")
    print("=" * 88)
    (
        _,
        spid_countries,
        spid_metrics,
        spid_selected_series,
        spid_all_series,
    ) = audit_spid()

    # PRIMARY WDI scope: countries with a strict SPID panel in 2015–2020 after
    # country-series selection by welfaretype + comparability. This prevents
    # unrelated comparable series from being mistaken for duplicate region-years.
    if not spid_countries.empty:
        country_scope = sorted(
            spid_countries.loc[
                spid_countries["strict_all_metrics_complete_2015_2020"],
                "code",
            ]
            .astype(str)
            .unique()
        )
    else:
        country_scope = []

    # Conservative fallback to the previously validated country set only if the
    # local SPID structure could not reconstruct any geography-complete country.
    if not country_scope:
        country_scope = [
            "AUT", "BEL", "BOL", "BRA", "CRI", "CZE", "ESP", "FIN", "GEO", "GRC",
            "IDN", "IRN", "ITA", "MNE", "PER", "ROU", "RUS", "SWE", "THA", "USA",
        ]
        log("WARNING", "SPID", "No geography-complete country scope reconstructed; using fallback 20-country scope for WDI audit only.")

    print(f"WDI audit country scope (strict SPID 2015–2020): {len(country_scope)} countries")
    print()

    # -------------------------------------------------------------------------
    # 3. Space2Stats + SSGD
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("3/5 — SPACE2STATS + SSGD")
    print("=" * 88)
    s2s_all, s2s_families = audit_space2stats(source_details)
    ssgd_indicators = audit_ssgd()
    print()

    # -------------------------------------------------------------------------
    # 4. WDI
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("4/5 — WDI")
    print("=" * 88)
    wdi_df = audit_wdi(country_scope)
    print()

    # -------------------------------------------------------------------------
    # 5. Consolidation
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("5/5 — CONSOLIDATION")
    print("=" * 88)

    availability = build_availability_matrix(
        registry,
        source_checks,
        spid_metrics,
        s2s_families,
        ssgd_indicators,
        wdi_df,
    )

    # Mark the actual WDI audit scope in the country table.
    if not spid_countries.empty:
        spid_countries["used_for_wdi_audit"] = spid_countries["code"].isin(country_scope)

    period_comparison = build_period_comparison(availability, spid_countries)
    summary = build_summary(
        registry,
        availability,
        source_checks,
        spid_countries,
        s2s_all,
        ssgd_indicators,
        wdi_df,
    )
    recovered = availability[availability["recovered_2015_2020"].fillna(False).astype(bool)].copy()

    # Practical Script-02 queue. Nothing is discarded here.
    extraction_queue = availability.copy()
    extraction_queue["script02_priority"] = np.select(
        [
            extraction_queue["availability_status"].eq("SPACE2STATS_FIELDS_CONFIRMED"),
            extraction_queue["availability_status"].eq("ACTUAL_LOCAL_DATA_AUDITED"),
            extraction_queue["availability_status"].eq("WDI_API_AUDITED"),
            extraction_queue["availability_status"].eq("SOURCE_CONFIRMED_PERIOD_COMPATIBLE"),
            extraction_queue["availability_status"].str.contains("TEMPORARILY_UNREACHABLE", na=False),
            extraction_queue["availability_status"].eq("DERIVABLE_AFTER_INPUT_EXTRACTION"),
            extraction_queue["availability_status"].eq("NOT_AVAILABLE_IN_CURRENT_SPACE2STATS_FIELDS"),
        ],
        [1, 1, 1, 2, 2, 3, 9],
        default=4,
    )
    extraction_queue = extraction_queue.sort_values(
        ["script02_priority", "block", "variable_id"]
    ).reset_index(drop=True)

    run_log_df = pd.DataFrame(RUN_LOG)
    if run_log_df.empty:
        run_log_df = pd.DataFrame(
            [{"level": "INFO", "source": "GLOBAL", "message": "No warnings/errors logged."}]
        )

    # -------------------------------------------------------------------------
    # ONE OUTPUT WORKBOOK
    # -------------------------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="00_Summary", index=False)
        period_comparison.to_excel(writer, sheet_name="01_Period_Comparison", index=False)
        availability.to_excel(writer, sheet_name="02_Availability", index=False)
        recovered.to_excel(writer, sheet_name="03_Recovered_2015_2020", index=False)
        extraction_queue.to_excel(writer, sheet_name="04_Extraction_Queue", index=False)
        source_checks.to_excel(writer, sheet_name="05_Source_Checks", index=False)
        spid_countries.to_excel(writer, sheet_name="06_SPID_Countries", index=False)
        spid_selected_series.to_excel(writer, sheet_name="07_SPID_Selected_Series", index=False)
        spid_all_series.to_excel(writer, sheet_name="08_SPID_All_Series", index=False)
        spid_metrics.to_excel(writer, sheet_name="09_SPID_Metrics", index=False)
        s2s_families.to_excel(writer, sheet_name="10_S2S_Families", index=False)
        s2s_all.to_excel(writer, sheet_name="11_S2S_All_Fields", index=False)
        ssgd_indicators.to_excel(writer, sheet_name="12_SSGD_Indicators", index=False)
        wdi_df.to_excel(writer, sheet_name="13_WDI_Coverage", index=False)
        zenodo_files.to_excel(writer, sheet_name="14_Zenodo_Files", index=False)
        registry.to_excel(writer, sheet_name="15_Variable_Registry", index=False)
        QA_FLAG_DEFINITIONS.to_excel(writer, sheet_name="16_QA_Flags", index=False)
        run_log_df.to_excel(writer, sheet_name="17_Run_Log", index=False)

    format_excel(OUTPUT_FILE)

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    for _, row in summary.iterrows():
        print(f"{row['metric']}: {row['value']}")

    print()
    print("Recovered variables (2015–2020):")
    if recovered.empty:
        print("  None classified as fully recovered at this audit stage.")
    else:
        cols = [
            "variable_id",
            "variable_name",
            "source_id",
            "period_fit_2015_2022",
            "period_fit_2015_2020",
        ]
        print(recovered[cols].to_string(index=False))

    print()
    print("=" * 88)
    print("OUTPUT")
    print("=" * 88)
    print(OUTPUT_FILE)
    print()
    print("No bulk regional raster/API extraction was performed in Script 01.")
    print("Script 02 will consume the registry/audit logic and extract all viable candidate families in one run.")

    elapsed = time.perf_counter() - script_start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print()
    print("=" * 88)
    print("EXECUTION TIME")
    print("=" * 88)
    print(f"Total execution time: {minutes:02d} min {seconds:05.2f} sec")


if __name__ == "__main__":
    main()
