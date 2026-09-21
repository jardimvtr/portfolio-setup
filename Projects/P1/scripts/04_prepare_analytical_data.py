from __future__ import annotations

"""
SCRIPT 04 v4.1.1 — FINAL ANALYTICAL PREPARATION + COMMON-WEIGHT BoD + ANNUAL DECOMPOSITION

Purpose
-------
Prepare the P1 analytical inputs after the methodological decisions adopted
through the 512-region coverage decision adopted after Script 04 v2.1.

Current official analytical window
----------------------------------
2015–2019.

Important: Scripts 01–03 are NOT retroactively truncated. Script 04 reads the
2015–2020 harmonized handoff and applies the analytical window locally. This
preserves 2020 in the acquisition/harmonization layer while using 2015–2019 for
trajectory analysis.

Eligibility rules implemented here
----------------------------------
1) Active dynamic variables must have a complete 2015–2019 trajectory for
   at least 512 regions.
2) Active structural variables may be observed at one reference point because they
   describe comparatively persistent regional configuration rather than annual
   trajectory. They must have at least 512 non-missing regions.
3) Structural variables are NOT duplicated across years.
4) Previously identified redundant variables are removed from the active
   registry rather than silently carried into the clustering matrix.
5) Sources/variables below the 512-region threshold remain auditable upstream
   but are excluded from the active P1 analytical feature registry.
6) The official common-region set is the intersection of all active dynamic and
   structural variables and must contain at least 512 regions; otherwise the script
   stops rather than silently dropping variables or regions.
7) WDI remains a country-level context layer. National values are NOT copied to
   every subnational region and are NOT treated as regional clustering features.

DEA / Benefit-of-the-Doubt decision
-----------------------------------
The project will use common-weight DEA/BoD for normative families where a
meaningful performance direction exists. In DEA terminology, indicators are
classified as DESEJAVEL (desirable), INDESEJAVEL (undesirable), or
NAO_NORMATIVA when no universal performance direction is defensible. This script completes the final preprocessing, estimates the common-weight BoD families,
builds balanced non-normative blocks, and exports the final analytical handoff to Script 05.

Reason: the user explicitly requested a decision checkpoint before statistical
normalization/standardization. BoD estimation requires a defensible choice of:
- subtraction treatment for INDESEJAVEL indicators before BoD;
- treatment of legitimate zero/negative values using the thesis epsilon protocol;
- normalization by the maximum (not Min-Max, not Z-score);
- common-weight restrictions if the unrestricted solution degenerates.

This revision applies the selected preprocessing rules, estimates one common BoD weight vector
per normative family over the pooled analytical window, and preserves the annual decomposition of
each family score into indicator-level weighted contributions.

Current family principle
------------------------
- Families with >=2 indicators and a defensible performance direction -> common-weight BoD candidate.
- Dynamic common weights are estimated jointly over all region-year observations in 2015-2019 and
  remain fixed across regions and years.
- For each region-year and variable, the script stores: raw value, preprocessed/normalized value,
  common BoD weight, weighted contribution, contribution share, and the reconstructed family score.
- The annual family score is the sum of weighted contributions; no second normalization is applied
  after the BoD aggregation.
- Single-indicator families -> no BoD; retain the indicator directly.
- The former demographic_scale_distribution family (population + density) is removed from the
  clustering representation; WorldPop population remains available only where technically needed
  as an upstream denominator/cross-check.
- Multi-indicator non-normative structural families -> retain as a balanced structural block, not a normative BoD.
- BoD aggregates within a family; it does not collapse all P1 dimensions into one global score.

Main use of SQL
---------------
DuckDB SQL integrates the canonical master panel with native SPID, WorldPop and
Space2Stats sources. Python/pandas/geopandas perform source-specific safe
aggregation, structural feature engineering, eligibility QA and BoD staging.
"""

from pathlib import Path
import re
from typing import Any, Iterable

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linprog


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
SCRIPT02_DIR = INTERIM_DIR / "script02"
SCRIPT03_DIR = INTERIM_DIR / "script03"
SCRIPT04_DIR = INTERIM_DIR / "script04"
OUTPUTS_DIR = ROOT / "outputs"

SCRIPT04_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Official analytical window from this revision onward.
START_YEAR = 2015
END_YEAR = 2019
YEARS = list(range(START_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)

EXPECTED_COUNTRIES = 28
EXPECTED_REGIONS = 513
EXPECTED_REGION_YEARS = EXPECTED_REGIONS * N_YEARS

MIN_COMPLETE_REGIONS = 512
MIN_STRUCTURAL_REGIONS = 512
HIGH_REDUNDANCY_THRESHOLD = 0.90
MIN_CORRELATION_PAIRS = 200
AREA_EQUAL_CRS = "EPSG:6933"

# BoD is intentionally staged, not estimated, until preprocessing alternatives
# and weight restrictions are explicitly selected.
ESTIMATE_BOD_WEIGHTS = True
BOD_METHOD = "COMMON_WEIGHT_BOD_MAXIMIZE_MEAN_SCORE"
BOD_MIN_CONTRIBUTION = 0.05
THESIS_EPSILON_SHARE = 0.01


# =============================================================================
# INPUTS
# =============================================================================

SCRIPT03_DB = SCRIPT03_DIR / "03_harmonization.duckdb"
MASTER_INPUT = SCRIPT03_DIR / "03_master_region_year_2015_2020.csv"
CROSSWALK_INPUT = SCRIPT03_DIR / "03_geographic_crosswalk.csv"
SOURCE_DECISIONS_INPUT = SCRIPT03_DIR / "03_source_decisions.csv"
VARIABLE_QA_INPUT = SCRIPT03_DIR / "03_variable_qa.csv"
TARGET_GEOMETRIES = SCRIPT02_DIR / "02_target_regions.gpkg"

REQUIRED_INPUTS = {
    "SCRIPT03_DUCKDB": SCRIPT03_DB,
    "MASTER_PANEL": MASTER_INPUT,
    "GEOGRAPHIC_CROSSWALK": CROSSWALK_INPUT,
    "SOURCE_DECISIONS": SOURCE_DECISIONS_INPUT,
    "VARIABLE_QA": VARIABLE_QA_INPUT,
    "TARGET_GEOMETRIES": TARGET_GEOMETRIES,
}


# =============================================================================
# OUTPUTS
# =============================================================================

DUCKDB_OUTPUT = SCRIPT04_DIR / "04_analytical_preparation.duckdb"
CORE_PANEL_OUTPUT = SCRIPT04_DIR / "04_core_analytical_panel_2015_2019.csv"
EXPANDED_PANEL_OUTPUT = SCRIPT04_DIR / "04_expanded_candidate_panel_2015_2019.csv"
DYNAMIC_COMPLETE_LONG_OUTPUT = SCRIPT04_DIR / "04_dynamic_complete_trajectories_long.csv"
STRUCTURAL_CONTEXT_OUTPUT = SCRIPT04_DIR / "04_structural_candidate_panel.csv"
FLAGGED_LONG_OUTPUT = SCRIPT04_DIR / "04_flagged_candidates_long_2015_2019.csv"
FLAGGED_QA_OUTPUT = SCRIPT04_DIR / "04_flagged_source_aggregation_qa.csv"
DYNAMIC_DIAGNOSTICS_OUTPUT = SCRIPT04_DIR / "04_dynamic_variable_diagnostics.csv"
STRUCTURAL_DIAGNOSTICS_OUTPUT = SCRIPT04_DIR / "04_structural_variable_diagnostics.csv"
FEATURE_ELIGIBILITY_OUTPUT = SCRIPT04_DIR / "04_feature_eligibility.csv"
SAMPLE_TRADEOFF_OUTPUT = SCRIPT04_DIR / "04_sample_tradeoff.csv"
CORRELATIONS_OUTPUT = SCRIPT04_DIR / "04_pairwise_correlations.csv"
REDUNDANCY_OUTPUT = SCRIPT04_DIR / "04_high_redundancy_pairs.csv"
BOD_FAMILY_DESIGN_OUTPUT = SCRIPT04_DIR / "04_bod_family_design.csv"
BOD_INPUT_RAW_OUTPUT = SCRIPT04_DIR / "04_bod_input_raw_long.csv"
MANUAL_EXCLUSIONS_OUTPUT = SCRIPT04_DIR / "04_manual_exclusions.csv"
WDI_COUNTRY_CONTEXT_OUTPUT = SCRIPT04_DIR / "04_wdi_country_context_2015_2019.csv"
COUNTRY_REFERENCE_RAW_OUTPUT = SCRIPT04_DIR / "04_country_reference_inputs_raw.csv"
POPULATION_CROSSCHECK_OUTPUT = SCRIPT04_DIR / "04_population_crosscheck_2015_2019.csv"
OFFICIAL_REGION_SET_OUTPUT = SCRIPT04_DIR / "04_official_region_set_512.csv"
OFFICIAL_FEATURE_SET_OUTPUT = SCRIPT04_DIR / "04_official_feature_set.csv"
OFFICIAL_DYNAMIC_PANEL_OUTPUT = SCRIPT04_DIR / "04_official_dynamic_panel_2015_2019.csv"
OFFICIAL_STRUCTURAL_PANEL_OUTPUT = SCRIPT04_DIR / "04_official_structural_panel.csv"
OFFICIAL_INTERSECTION_QA_OUTPUT = SCRIPT04_DIR / "04_official_intersection_qa.csv"
QA_WORKBOOK_OUTPUT = OUTPUTS_DIR / "04_analytical_preparation_qa.xlsx"
PREPROCESSING_AUDIT_OUTPUT = SCRIPT04_DIR / "04_preprocessing_audit.csv"
BOD_WEIGHTS_OUTPUT = SCRIPT04_DIR / "04_bod_common_weights.csv"
BOD_FAMILY_SUMMARY_OUTPUT = SCRIPT04_DIR / "04_bod_family_summary.csv"
BOD_PROCESSED_LONG_OUTPUT = SCRIPT04_DIR / "04_bod_processed_indicators_long.csv"
BOD_WEIGHTED_CONTRIBUTIONS_OUTPUT = SCRIPT04_DIR / "04_bod_weighted_contributions_long.csv"
BOD_DYNAMIC_DECOMPOSITION_OUTPUT = SCRIPT04_DIR / "04_bod_dynamic_decomposition_2015_2019.csv"
BOD_STRUCTURAL_DECOMPOSITION_OUTPUT = SCRIPT04_DIR / "04_bod_structural_decomposition.csv"
BOD_DYNAMIC_SCORES_OUTPUT = SCRIPT04_DIR / "04_bod_dynamic_scores.csv"
BOD_STRUCTURAL_SCORES_OUTPUT = SCRIPT04_DIR / "04_bod_structural_scores.csv"
FINAL_DYNAMIC_MODEL_OUTPUT = SCRIPT04_DIR / "04_final_dynamic_model_panel_2015_2019.csv"
FINAL_STRUCTURAL_MODEL_OUTPUT = SCRIPT04_DIR / "04_final_structural_model_panel.csv"
FINAL_FEATURE_WEIGHTS_OUTPUT = SCRIPT04_DIR / "04_final_feature_weights.csv"
FINAL_DYNAMIC_LONG_OUTPUT = SCRIPT04_DIR / "04_mexp_fcmd_dynamic_long.csv"
FINAL_STRUCTURAL_LONG_OUTPUT = SCRIPT04_DIR / "04_mexp_fcmd_structural_long.csv"
COUNTRY_REFERENCE_DYNAMIC_OUTPUT = SCRIPT04_DIR / "04_country_reference_dynamic_final.csv"
COUNTRY_REFERENCE_STRUCTURAL_OUTPUT = SCRIPT04_DIR / "04_country_reference_structural_final.csv"
FINAL_HANDOFF_WORKBOOK_OUTPUT = OUTPUTS_DIR / "04_final_handoff_script05.xlsx"


# =============================================================================
# ACTIVE FEATURE REGISTRY
# =============================================================================

# Dynamic candidates are observed annually. "dea_direction" is only populated
# when the variable has a defensible normative direction. Values:
# DESEJAVEL = larger values represent higher performance;
# INDESEJAVEL = smaller values represent higher performance;
# NAO_NORMATIVA = descriptive variable without a universal performance direction.
DYNAMIC_FEATURES: dict[str, dict[str, Any]] = {
    # Productive demographic potential — normative family for BoD.
    "share_15_64": {
        "family": "demographic_productive_potential",
        "source": "WORLDPOP_AGESEX_DERIVED",
        "dea_direction": "DESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Share of the regional population aged 15–64, derived as 1 - share_0_14 - share_65_plus.",
    },
    "share_65_plus": {
        "family": "demographic_productive_potential",
        "source": "WORLDPOP_AGESEX",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Share of the regional population aged 65 or older; treated as an undesirable dependency-pressure indicator before BoD.",
    },

    # Socioeconomic deprivation — normative family for common-weight BoD.
    "poor420": {
        "family": "socioeconomic_deprivation",
        "source": "SPID",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Population share below the SPID 4.20 poverty threshold.",
    },
    "gini": {
        "family": "socioeconomic_deprivation",
        "source": "SPID",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Regional inequality measured by the Gini coefficient.",
    },
    "prosgap2021": {
        "family": "socioeconomic_deprivation",
        "source": "SPID",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Prosperity Gap; larger values imply a larger average shortfall from the prosperity standard.",
    },

    # Regional economic activity — single-indicator proxy, therefore no BoD.
    "ntl_per_capita": {
        "family": "regional_economic_activity_proxy",
        "source": "SPACE2STATS+WORLDPOP",
        "dea_direction": "DESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Nighttime-light intensity per resident, used as a proxy for regional economic activity.",
    },
}


STRUCTURAL_FEATURES: dict[str, dict[str, Any]] = {
    # Non-normative structural urban/built-environment block.
    "built_area_share_2015": {
        "family": "urbanization_built_environment",
        "source": "SPACE2STATS",
        "dea_direction": "NAO_NORMATIVA",
        "manual_status": "CANDIDATE",
        "description": "Share of regional land area covered by built-up surface in 2015.",
    },
    "ghs_urban_pop_share": {
        "family": "urbanization_built_environment",
        "source": "SPACE2STATS_GHSL",
        "dea_direction": "NAO_NORMATIVA",
        "manual_status": "CANDIDATE",
        "description": "Share of population in GHS urban classes 21, 22, 23 and 30.",
    },
    "ghs_urban_centre_pop_share": {
        "family": "urbanization_built_environment",
        "source": "SPACE2STATS_GHSL",
        "dea_direction": "NAO_NORMATIVA",
        "manual_status": "CANDIDATE",
        "description": "Share of population in GHS urban-centre class 30.",
    },

    # Environmental risk — structural normative family for a separate BoD.
    "drought_spei_1_5_rp100_mean": {
        "family": "environmental_risk",
        "source": "SPACE2STATS",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Regional structural drought-risk indicator based on the Space2Stats SPEI field.",
    },
    "fires_density_mean": {
        "family": "environmental_risk",
        "source": "SPACE2STATS",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE",
        "description": "Mean regional fire-density indicator.",
    },
    "landslide_susceptibility_mean_2023": {
        "family": "environmental_risk",
        "source": "SPACE2STATS",
        "dea_direction": "INDESEJAVEL",
        "manual_status": "CANDIDATE_EXTERNAL_SNAPSHOT",
        "description": "Mean landslide susceptibility in 2023; treated as structural context rather than a 2015–2019 trajectory.",
    },
}


# Variables intentionally removed from the active design.
MANUAL_EXCLUSIONS = [
    {
        "variable": "wp_age_population,population_density_per_km2",
        "reason": "DEMOGRAPHIC_SCALE_DISTRIBUTION_REMOVED_FROM_CLUSTERING",
        "notes": (
            "Population scale and population density are excluded from the final clustering feature space. "
            "WorldPop population remains available upstream only as a denominator for NTL per capita and "
            "for population cross-checks; density may remain in intermediate audit data but is not an active feature."
        ),
    },
    {
        "variable": "share_0_14",
        "reason": "REDUNDANT_WITH_PRODUCTIVE_AGE_SPECIFICATION",
        "notes": "Excluded from the official matrix because share_15_64 + share_65_plus already define the selected demographic BoD; retain only for sensitivity analysis.",
    },
    {
        "variable": "age_dependency_ratio",
        "reason": "REDUNDANT_WITH_SELECTED_AGE_STRUCTURE",
        "notes": "Derived from the same age structure represented by share_15_64 and share_65_plus.",
    },
    {
        "variable": "theil",
        "reason": "HIGH_REDUNDANCY_WITH_GINI",
        "notes": "Script 04 v1.0 found Spearman correlation approximately 0.985 with Gini.",
    },
    {
        "variable": "poor300",
        "reason": "POVERTY_THRESHOLD_REDUNDANCY",
        "notes": "Removed to avoid multiple highly related poverty-threshold indicators in the same family.",
    },
    {
        "variable": "poor830",
        "reason": "HIGH_REDUNDANCY_WITH_PROSPERITY_GAP",
        "notes": "Script 04 v1.0 found strong redundancy with prosgap2021.",
    },
    {
        "variable": "viirs_ntl_sum",
        "reason": "SIZE_EFFECT_REPLACED_BY_PER_CAPITA",
        "notes": "NTL per capita is retained to reduce mechanical population-scale effects.",
    },
    {
        "variable": "pop_flood_pct",
        "reason": "UNIT_SEMANTICS_UNCERTAIN",
        "notes": "Removed from the official environmental-risk family because values above 100 conflict with the percentage label; exclusion avoids introducing a measurement-risk variable into BoD.",
    },
    {
        "variable": "population_growth_yoy_pct",
        "reason": "NOT_AVAILABLE_FOR_FULL_2015_2019_WINDOW",
        "notes": "2015 is structurally undefined because no t-1 observation exists inside the analytical window.",
    },
    {
        "variable": "ntl_per_capita_growth_yoy_pct",
        "reason": "NOT_AVAILABLE_FOR_FULL_2015_2019_WINDOW",
        "notes": "2015 is structurally undefined because no t-1 observation exists inside the analytical window.",
    },
    {
        "variable": "dose_grp_pc_usd_2015,dose_ag_share,dose_man_share,dose_serv_share",
        "reason": "BELOW_512_COMPLETE_REGIONS",
        "notes": "DOSE remains auditable upstream but is excluded from the official P1 matrix under the >=512-region rule.",
    },
    {
        "variable": "niva_net_migration",
        "reason": "BELOW_512_COMPLETE_REGIONS",
        "notes": "Niva remains auditable upstream but is excluded from the official P1 matrix under the >=512-region rule.",
    },
    {
        "variable": "kummu_gdp_pc_ppp",
        "reason": "BELOW_512_COMPLETE_REGIONS",
        "notes": "Kummu GDP is excluded from the official P1 matrix under the >=512-region rule.",
    },
    {
        "variable": "gdl_shdi_and_components",
        "reason": "BELOW_512_COMPLETE_REGIONS",
        "notes": "GDL SHDI/components are excluded from the official P1 matrix under the >=512-region rule.",
    },
    {
        "variable": "oecd_unemployment_rate_15_64",
        "reason": "BELOW_512_COMPLETE_REGIONS",
        "notes": "OECD regional unemployment remains context/audit only.",
    },
    {
        "variable": "ookla_fixed_2019,ookla_mobile_2019",
        "reason": "BELOW_512_OR_UNRESOLVED_COVERAGE",
        "notes": "Ookla is excluded from the official P1 matrix under the >=512-region rule.",
    },
    {
        "variable": "WDI_COUNTRY_CONTEXT",
        "reason": "NATIONAL_CONTEXT_NOT_REGIONAL_FEATURE",
        "notes": "WDI is retained only as a national-context layer and is never copied to regional clustering rows.",
    },
]



# =============================================================================
# GENERIC HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_country(value: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(value).upper())


def normalize_code(value: Any) -> str:
    text = normalize_text(value).upper()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text.strip()


def source_geo_key(country: Any, region_code: Any) -> str:
    c = normalize_country(country)
    code = normalize_code(region_code)
    if not c or not code:
        return ""
    return f"{c}__CODE__{code}"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    exact = {str(c).casefold(): str(c) for c in frame.columns}
    for candidate in candidates:
        hit = exact.get(candidate.casefold())
        if hit is not None:
            return hit
    return None


def persist_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    temp_name = f"_tmp_{table_name}"
    connection.register(temp_name, frame)
    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    connection.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM "{temp_name}"')
    connection.unregister(temp_name)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def validate_required_inputs() -> None:
    missing: list[str] = []
    for label, path in REQUIRED_INPUTS.items():
        if path.exists():
            print(f"[FOUND]   {label:<24} {path.name}")
        else:
            print(f"[MISSING] {label:<24} {path}")
            missing.append(label)
    if missing:
        raise FileNotFoundError(
            "Required Script 04 inputs are missing: " + ", ".join(missing)
        )


def load_script03_raw(table_name: str) -> pd.DataFrame:
    con = duckdb.connect(str(SCRIPT03_DB), read_only=True)
    try:
        return con.execute(f'SELECT * FROM "{table_name}"').df()
    finally:
        con.close()


# =============================================================================
# REGION AREA
# =============================================================================

def build_region_area() -> pd.DataFrame:
    gdf = gpd.read_file(TARGET_GEOMETRIES)
    required = {"code", "geo_code", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(
            "Target geometry is missing columns required by Script 04: "
            f"{sorted(missing)}"
        )

    gdf = gdf[["code", "geo_code", "geometry"]].copy()
    gdf["code"] = gdf["code"].astype("string").str.strip().str.upper()
    gdf["geo_code"] = gdf["geo_code"].astype("string").str.strip()
    gdf = gdf.drop_duplicates(["code", "geo_code"]).copy()

    if gdf.crs is None:
        raise ValueError("Target geometry has no CRS; region area cannot be calculated safely.")

    projected = gdf.to_crs(AREA_EQUAL_CRS)
    gdf["region_area_km2"] = projected.geometry.area / 1_000_000.0
    gdf["region_id"] = gdf["code"].astype(str) + "__" + gdf["geo_code"].astype(str)

    if len(gdf) != EXPECTED_REGIONS:
        raise ValueError(
            f"Region-area table has {len(gdf)} regions; expected {EXPECTED_REGIONS}."
        )
    if (gdf["region_area_km2"] <= 0).any():
        raise ValueError("At least one canonical region has non-positive area.")

    return gdf[["region_id", "code", "geo_code", "region_area_km2"]].copy()


# =============================================================================
# CORE 2015–2019 PANEL — DUCKDB SQL
# =============================================================================

def build_core_panel_sql(
    region_area: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the canonical 2015–2019 core and Space2Stats structural layer."""

    con = duckdb.connect(str(SCRIPT03_DB), read_only=True)
    try:
        sql = f"""
        SELECT
            m.region_id,
            m.region_year_id,
            m.code,
            m.geo_code,
            m.geo_name,
            CAST(m.year AS INTEGER) AS year,

            -- SPID: active non-redundant candidates
            s.poor420,
            s.prosgap2021,
            s.gini,

            -- WorldPop
            w.wp_age_population,
            w.share_0_14,
            w.share_65_plus,

            -- Space2Stats annual NTL
            CASE CAST(m.year AS INTEGER)
                WHEN 2015 THEN z.sum_viirs_ntl_2015
                WHEN 2016 THEN z.sum_viirs_ntl_2016
                WHEN 2017 THEN z.sum_viirs_ntl_2017
                WHEN 2018 THEN z.sum_viirs_ntl_2018
                WHEN 2019 THEN z.sum_viirs_ntl_2019
            END AS viirs_ntl_sum,

            CASE CAST(m.year AS INTEGER)
                WHEN 2015 THEN z.sum_pop_2015
                WHEN 2016 THEN z.sum_pop_2016
                WHEN 2017 THEN z.sum_pop_2017
                WHEN 2018 THEN z.sum_pop_2018
                WHEN 2019 THEN z.sum_pop_2019
            END AS space2stats_population

        FROM master_region_year AS m
        LEFT JOIN raw_spid AS s
          ON m.code = s.code
         AND m.geo_code = s.geo_code
         AND CAST(m.year AS INTEGER) = CAST(s.year AS INTEGER)

        LEFT JOIN raw_worldpop_agesex AS w
          ON m.code = w.code
         AND m.geo_code = w.geo_code
         AND CAST(m.year AS INTEGER) = CAST(w.year AS INTEGER)

        LEFT JOIN raw_space2stats AS z
          ON m.code = z.code
         AND m.geo_code = z.geo_code

        WHERE CAST(m.year AS INTEGER) BETWEEN {START_YEAR} AND {END_YEAR}
        ORDER BY m.code, m.geo_code, CAST(m.year AS INTEGER)
        """
        core = con.execute(sql).df()

        static_sql = """
        SELECT
            m.region_id,
            m.code,
            m.geo_code,
            m.geo_name,
            z.drought_spei_1_5_rp100_mean,
            z.fires_density_mean,
            z.landslide_susceptibility_mean_2023,
            z.pop_flood,
            z.pop_flood_pct,
            z.ghs_11_pop,
            z.ghs_12_pop,
            z.ghs_13_pop,
            z.ghs_21_pop,
            z.ghs_22_pop,
            z.ghs_23_pop,
            z.ghs_30_pop,
            z.ghs_total_pop,
            z.sum_built_area_m_2015
        FROM dim_region AS m
        LEFT JOIN raw_space2stats AS z
          ON m.code = z.code
         AND m.geo_code = z.geo_code
        ORDER BY m.code, m.geo_code
        """
        structural = con.execute(static_sql).df()
    finally:
        con.close()

    if len(core) != EXPECTED_REGION_YEARS:
        raise ValueError(
            f"SQL core panel has {len(core):,} rows; expected {EXPECTED_REGION_YEARS:,}."
        )
    if core.duplicated(["region_id", "year"]).any():
        raise ValueError("SQL core panel contains duplicated region-year keys.")

    core = core.merge(
        region_area[["region_id", "region_area_km2"]],
        on="region_id",
        how="left",
        validate="many_to_one",
    )
    structural = structural.merge(
        region_area[["region_id", "region_area_km2"]],
        on="region_id",
        how="left",
        validate="one_to_one",
    )

    # Dynamic derived indicators retained under the full-period rule.
    core["population_density_per_km2"] = safe_divide(
        core["wp_age_population"], core["region_area_km2"]
    )
    core["ntl_per_capita"] = safe_divide(
        core["viirs_ntl_sum"], core["wp_age_population"]
    )
    # Productive-age share derived from mutually exclusive age shares.
    core["share_15_64"] = (
        1.0
        - pd.to_numeric(core["share_0_14"], errors="coerce")
        - pd.to_numeric(core["share_65_plus"], errors="coerce")
    )

    # Structural features are kept one row per region and never replicated over years.
    structural["built_area_share_2015"] = safe_divide(
        structural["sum_built_area_m_2015"],
        structural["region_area_km2"] * 1_000_000.0,
    )

    urban_pop = (
        pd.to_numeric(structural["ghs_21_pop"], errors="coerce").fillna(0)
        + pd.to_numeric(structural["ghs_22_pop"], errors="coerce").fillna(0)
        + pd.to_numeric(structural["ghs_23_pop"], errors="coerce").fillna(0)
        + pd.to_numeric(structural["ghs_30_pop"], errors="coerce").fillna(0)
    )
    structural["ghs_urban_pop_share"] = safe_divide(
        urban_pop, structural["ghs_total_pop"]
    )
    structural["ghs_urban_centre_pop_share"] = safe_divide(
        structural["ghs_30_pop"], structural["ghs_total_pop"]
    )

    return (
        core.sort_values(["region_id", "year"]).reset_index(drop=True),
        structural.sort_values(["code", "geo_code"]).reset_index(drop=True),
    )


# =============================================================================
# CROSSWALK
# =============================================================================

def load_matched_crosswalk() -> pd.DataFrame:
    cw = pd.read_csv(CROSSWALK_INPUT, low_memory=False, encoding="utf-8-sig")
    matched = cw[cw["match_status"].astype(str).eq("MATCHED")].copy()
    keep = [
        "source_id",
        "source_geo_key",
        "source_country",
        "source_region_code",
        "source_region_name",
        "target_region_id",
        "target_geo_code",
        "target_geo_name",
        "match_method",
        "match_score",
        "relation",
        "source_units_per_target",
    ]
    keep = [column for column in keep if column in matched.columns]
    return matched[keep].copy()


# =============================================================================
# FLAGGED DYNAMIC SOURCES
# =============================================================================

def harmonize_kummu(cw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit-only Kummu GDP; DOSE is the active economy source."""
    raw = load_script03_raw("raw_kummu_gdp")
    xw = cw[cw["source_id"].eq("KUMMU_GDP")].copy()
    xw = xw[xw["relation"].eq("ONE_SOURCE_TO_ONE_TARGET")].copy()

    raw["source_geo_key"] = [
        source_geo_key(c, r)
        for c, r in zip(raw["iso3"], raw["GID_nmbr"], strict=False)
    ]
    value_columns = [str(year) for year in YEARS if str(year) in raw.columns]
    long = raw[["source_geo_key", "iso3", "Subnat"] + value_columns].melt(
        id_vars=["source_geo_key", "iso3", "Subnat"],
        value_vars=value_columns,
        var_name="year",
        value_name="value",
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    joined = long.merge(
        xw[["source_geo_key", "target_region_id", "match_method", "relation"]],
        on="source_geo_key",
        how="inner",
        validate="many_to_one",
    )

    out = joined.rename(columns={"target_region_id": "region_id"})[
        ["region_id", "year", "value"]
    ].copy()
    out["variable"] = "kummu_gdp_pc_ppp"
    out["source_id"] = "KUMMU_GDP"

    qa = {
        "source_id": "KUMMU_GDP",
        "method": "ONE_TO_ONE_ONLY_AUDIT",
        "source_rows": len(raw),
        "matched_source_units_used": joined["source_geo_key"].nunique(),
        "target_regions_covered": out["region_id"].nunique(),
        "region_year_values": out["value"].notna().sum(),
        "active_in_current_design": False,
        "notes": "Retained for audit comparison; DOSE is preferred for the active regional-economy branch.",
    }
    return out, qa


def harmonize_dose(cw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_script03_raw("raw_dose")
    xw = cw[cw["source_id"].eq("DOSE")].copy()

    raw["source_geo_key"] = [
        source_geo_key(c, r)
        for c, r in zip(raw["GID_0"], raw["GID_1"], strict=False)
    ]

    joined = raw.merge(
        xw[["source_geo_key", "target_region_id", "match_method", "relation"]],
        on="source_geo_key",
        how="inner",
        validate="many_to_one",
    ).copy()
    joined = joined[pd.to_numeric(joined["year"], errors="coerce").isin(YEARS)].copy()

    for column in [
        "pop",
        "grp_pc_usd_2015",
        "ag_grp_pc_usd_2015",
        "man_grp_pc_usd_2015",
        "serv_grp_pc_usd_2015",
    ]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")

    joined["_grp_total"] = joined["grp_pc_usd_2015"] * joined["pop"]
    joined["_ag_total"] = joined["ag_grp_pc_usd_2015"] * joined["pop"]
    joined["_man_total"] = joined["man_grp_pc_usd_2015"] * joined["pop"]
    joined["_serv_total"] = joined["serv_grp_pc_usd_2015"] * joined["pop"]

    grouped = (
        joined.groupby(["target_region_id", "year"], as_index=False)
        .agg(
            population_weight=("pop", "sum"),
            grp_total=("_grp_total", "sum"),
            ag_total=("_ag_total", "sum"),
            man_total=("_man_total", "sum"),
            serv_total=("_serv_total", "sum"),
            source_units=("source_geo_key", "nunique"),
        )
    )

    grouped["dose_grp_pc_usd_2015"] = safe_divide(
        grouped["grp_total"], grouped["population_weight"]
    )
    grouped["dose_ag_share"] = safe_divide(grouped["ag_total"], grouped["grp_total"])
    grouped["dose_man_share"] = safe_divide(grouped["man_total"], grouped["grp_total"])
    grouped["dose_serv_share"] = safe_divide(grouped["serv_total"], grouped["grp_total"])

    long = grouped.melt(
        id_vars=["target_region_id", "year", "source_units"],
        value_vars=[
            "dose_grp_pc_usd_2015",
            "dose_ag_share",
            "dose_man_share",
            "dose_serv_share",
        ],
        var_name="variable",
        value_name="value",
    ).rename(columns={"target_region_id": "region_id"})
    long["source_id"] = "DOSE"

    qa = {
        "source_id": "DOSE",
        "method": "POPULATION_WEIGHTED_AGGREGATION",
        "source_rows": len(raw),
        "matched_source_units_used": joined["source_geo_key"].nunique(),
        "target_regions_covered": grouped["target_region_id"].nunique(),
        "region_year_values": long["value"].notna().sum(),
        "active_in_current_design": False,
        "notes": "Auditable upstream; excluded from the official >=512-region analytical design.",
    }
    return long[["region_id", "year", "variable", "value", "source_id"]], qa


def harmonize_niva(cw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_script03_raw("raw_niva_migration")
    xw = cw[cw["source_id"].eq("NIVA_MIGRATION")].copy()
    xw = xw[xw["relation"].eq("ONE_SOURCE_TO_ONE_TARGET")].copy()

    raw["source_geo_key"] = [
        source_geo_key(c, r)
        for c, r in zip(raw["iso3"], raw["GID_1"], strict=False)
    ]
    cols = [f"netMgr_{year}" for year in YEARS if f"netMgr_{year}" in raw.columns]
    long = raw[["source_geo_key", "iso3", "GID_1", "NAME_1"] + cols].melt(
        id_vars=["source_geo_key", "iso3", "GID_1", "NAME_1"],
        value_vars=cols,
        var_name="year_field",
        value_name="value",
    )
    long["year"] = pd.to_numeric(
        long["year_field"].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    ).astype("Int64")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    joined = long.merge(
        xw[["source_geo_key", "target_region_id", "match_method", "relation"]],
        on="source_geo_key",
        how="inner",
        validate="many_to_one",
    )
    out = joined.rename(columns={"target_region_id": "region_id"})[
        ["region_id", "year", "value"]
    ].copy()
    out["variable"] = "niva_net_migration"
    out["source_id"] = "NIVA_MIGRATION"

    qa = {
        "source_id": "NIVA_MIGRATION",
        "method": "ONE_TO_ONE_ONLY_NO_UNWEIGHTED_AGGREGATION",
        "source_rows": len(raw),
        "matched_source_units_used": joined["source_geo_key"].nunique(),
        "target_regions_covered": out["region_id"].nunique(),
        "region_year_values": out["value"].notna().sum(),
        "active_in_current_design": False,
        "notes": "Auditable upstream; excluded from the official >=512-region analytical design.",
    }
    return out, qa


def harmonize_gdl_shdi(cw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retain interpretable GDL components for coverage QA; the >=512 rule excludes them."""
    raw = load_script03_raw("raw_gdl_shdi")
    raw = raw[raw["level"].astype(str).str.strip().str.casefold().eq("subnat")].copy()
    xw = cw[cw["source_id"].eq("GDL_SHDI")].copy()
    xw = xw[xw["relation"].eq("ONE_SOURCE_TO_ONE_TARGET")].copy()

    raw["source_geo_key"] = [
        source_geo_key(c, r)
        for c, r in zip(raw["isocode3"], raw["gdlcode"], strict=False)
    ]
    variables = ["lifexp", "esch", "msch", "lgnic"]
    available = [column for column in variables if column in raw.columns]

    joined = raw.merge(
        xw[["source_geo_key", "target_region_id", "match_method", "relation"]],
        on="source_geo_key",
        how="inner",
        validate="many_to_one",
    )
    joined = joined[pd.to_numeric(joined["year"], errors="coerce").isin(YEARS)].copy()

    long = joined.melt(
        id_vars=["target_region_id", "year"],
        value_vars=available,
        var_name="gdl_variable",
        value_name="value",
    )
    long["variable"] = "gdl_" + long["gdl_variable"].astype(str)
    long = long.rename(columns={"target_region_id": "region_id"})
    long["source_id"] = "GDL_SHDI"

    qa = {
        "source_id": "GDL_SHDI",
        "method": "ONE_TO_ONE_MATCHED_SUBNATIONAL_ONLY",
        "source_rows": len(raw),
        "matched_source_units_used": joined["source_geo_key"].nunique(),
        "target_regions_covered": joined["target_region_id"].nunique(),
        "region_year_values": long["value"].notna().sum(),
        "active_in_current_design": "ELIGIBILITY_RULE_DECIDES",
        "notes": "Only interpretable GDL components are retained for QA; the official >=512-region rule excludes this source from the model.",
    }
    return long[["region_id", "year", "variable", "value", "source_id"]], qa


def build_flagged_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    cw = load_matched_crosswalk()
    parts: list[pd.DataFrame] = []
    qa_rows: list[dict[str, Any]] = []

    for builder in [harmonize_kummu, harmonize_dose, harmonize_niva, harmonize_gdl_shdi]:
        part, qa = builder(cw)
        parts.append(part)
        qa_rows.append(qa)

    flagged = pd.concat(parts, ignore_index=True)
    flagged["year"] = pd.to_numeric(flagged["year"], errors="coerce").astype("Int64")
    flagged["value"] = pd.to_numeric(flagged["value"], errors="coerce")
    flagged = flagged[flagged["year"].isin(YEARS)].copy()

    dup = flagged.duplicated(["region_id", "year", "variable"], keep=False)
    if dup.any():
        sample = flagged.loc[dup, ["region_id", "year", "variable", "source_id"]].head(20)
        raise ValueError(
            "Flagged harmonization produced duplicated region-year-variable keys.\n"
            + sample.to_string(index=False)
        )

    return flagged, pd.DataFrame(qa_rows)


# =============================================================================
# OOKLA STRUCTURAL SNAPSHOT
# =============================================================================

def build_ookla_structural(cw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover 2019 Ookla structural candidates only where crosswalk is safely 1:1."""
    try:
        raw = load_script03_raw("raw_ookla_wb")
    except Exception as exc:
        return pd.DataFrame(columns=["region_id", "ookla_fixed_2019", "ookla_mobile_2019"]), {
            "source_id": "OOKLA_WB",
            "method": "NOT_AVAILABLE",
            "target_regions_covered": 0,
            "notes": f"Could not read raw_ookla_wb: {exc}",
        }

    country_col = find_column(raw, ["REF_AREA_0", "ref_area_0", "country_code", "code"])
    region_col = find_column(raw, ["REF_AREA_1", "ref_area_1", "region_code", "geo_code"])
    year_col = find_column(raw, ["year", "TIME_PERIOD", "time_period"])
    fixed_col = find_column(raw, ["OOKLA_ST_FIX_PER_mean", "ookla_st_fix_per_mean"])
    mobile_col = find_column(raw, ["OOKLA_ST_MOB_PER_mean", "ookla_st_mob_per_mean"])

    if not all([country_col, region_col, year_col]) or not any([fixed_col, mobile_col]):
        return pd.DataFrame(columns=["region_id", "ookla_fixed_2019", "ookla_mobile_2019"]), {
            "source_id": "OOKLA_WB",
            "method": "COLUMN_DETECTION_FAILED",
            "target_regions_covered": 0,
            "notes": "Expected country/region/year/Ookla metric columns were not detected; no structural values were forced.",
        }

    raw = raw[pd.to_numeric(raw[year_col], errors="coerce").eq(2019)].copy()
    raw["source_geo_key"] = [
        source_geo_key(c, r)
        for c, r in zip(raw[country_col], raw[region_col], strict=False)
    ]

    xw = cw[cw["source_id"].eq("OOKLA_WB")].copy()
    if "relation" in xw.columns:
        xw = xw[xw["relation"].eq("ONE_SOURCE_TO_ONE_TARGET")].copy()
    if "source_units_per_target" in xw.columns:
        xw = xw[pd.to_numeric(xw["source_units_per_target"], errors="coerce").eq(1)].copy()

    keep_cols = ["source_geo_key", "target_region_id"]
    xw = xw[keep_cols].drop_duplicates("source_geo_key")
    joined = raw.merge(xw, on="source_geo_key", how="inner", validate="many_to_one")

    out = pd.DataFrame({"region_id": joined["target_region_id"]})
    if fixed_col:
        out["ookla_fixed_2019"] = pd.to_numeric(joined[fixed_col], errors="coerce")
    if mobile_col:
        out["ookla_mobile_2019"] = pd.to_numeric(joined[mobile_col], errors="coerce")

    value_cols = [c for c in ["ookla_fixed_2019", "ookla_mobile_2019"] if c in out.columns]
    if value_cols:
        # One target should now map to at most one safe source unit; if duplicates
        # still occur because raw has duplicate records, collapse identical/near-identical
        # annual records by mean and flag in QA rather than duplicating regions.
        out = out.groupby("region_id", as_index=False)[value_cols].mean()

    qa = {
        "source_id": "OOKLA_WB",
        "method": "SAFE_ONE_TO_ONE_2019_STRUCTURAL_SNAPSHOT",
        "target_regions_covered": out["region_id"].nunique() if not out.empty else 0,
        "notes": "2019 snapshot only; retained as structural digital-infrastructure context. Exact metric semantics/units require metadata review before final BoD/scaling.",
    }
    return out, qa


# =============================================================================
# WDI COUNTRY CONTEXT
# =============================================================================

def build_wdi_country_context() -> pd.DataFrame:
    """Preserve national WDI data separately; never replicate them to regions."""
    try:
        raw = load_script03_raw("raw_wdi")
    except Exception:
        return pd.DataFrame()

    year_col = find_column(raw, ["year", "TIME_PERIOD", "time_period"])
    out = raw.copy()
    if year_col is not None:
        years = pd.to_numeric(out[year_col], errors="coerce")
        out = out[years.isin(YEARS)].copy()
    out["p1_role"] = "COUNTRY_CONTEXT_ONLY"
    out["regional_feature"] = False
    return out.reset_index(drop=True)


# =============================================================================
# EXPANDED DYNAMIC PANEL
# =============================================================================

def build_expanded_panel(core: pd.DataFrame, flagged_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        flagged_long.pivot(index=["region_id", "year"], columns="variable", values="value")
        .reset_index()
    )
    wide.columns.name = None
    expanded = core.merge(wide, on=["region_id", "year"], how="left", validate="one_to_one")
    if len(expanded) != EXPECTED_REGION_YEARS:
        raise ValueError(
            f"Expanded candidate panel has {len(expanded):,} rows; expected {EXPECTED_REGION_YEARS:,}."
        )
    return expanded


# =============================================================================
# DIAGNOSTICS / ELIGIBILITY
# =============================================================================

def infer_transform_hint(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return "NO_DATA"
    minimum = float(x.min())
    maximum = float(x.max())
    skew = float(x.skew()) if len(x) >= 3 else np.nan
    if minimum >= 0 and maximum <= 1.000001:
        return "PROPORTION_OR_INDEX_REVIEW"
    if minimum >= 0 and np.isfinite(skew) and skew > 2.0:
        return "LOG1P_VS_ROBUST_SCALING_REVIEW"
    if minimum > 0 and maximum / max(minimum, 1e-12) > 1000:
        return "LOG_VS_ROBUST_SCALING_REVIEW"
    if minimum < 0:
        return "SIGNED_SCALING_REVIEW"
    return "STANDARDIZATION_OPTIONS_REVIEW"


def dynamic_diagnostics(
    panel: pd.DataFrame,
    registry: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    complete_long_parts: list[pd.DataFrame] = []

    region_meta = panel[["region_id", "code", "geo_code", "geo_name"]].drop_duplicates("region_id")

    for variable, meta in registry.items():
        if variable not in panel.columns:
            rows.append({
                "variable": variable,
                "family": meta["family"],
                "source": meta["source"],
                "manual_status": meta["manual_status"],
                "dea_direction": meta["dea_direction"],
                "description": meta["description"],
                "regions_complete_trajectory": 0,
                "countries_with_complete_region": 0,
                "eligible_by_coverage": False,
                "eligibility_status": "VARIABLE_NOT_AVAILABLE_IN_PANEL",
            })
            continue

        temp = panel[["region_id", "code", "year", variable]].copy()
        temp[variable] = pd.to_numeric(temp[variable], errors="coerce")
        counts = temp.groupby("region_id")[variable].apply(lambda s: int(s.notna().sum()))
        complete_regions = counts[counts.eq(N_YEARS)].index
        complete_meta = region_meta[region_meta["region_id"].isin(complete_regions)]

        x = temp[variable]
        non_missing = x.dropna()
        coverage_ok = len(complete_regions) >= MIN_COMPLETE_REGIONS
        manual = meta["manual_status"]
        manual_allowed = not str(manual).startswith("EXCLUDE_")
        eligible = bool(coverage_ok and manual_allowed)

        if not coverage_ok:
            status = f"EXCLUDE_LT_{MIN_COMPLETE_REGIONS}_COMPLETE_REGIONS"
        elif not manual_allowed:
            status = manual
        else:
            status = "ELIGIBLE_DYNAMIC_FULL_PERIOD"

        rows.append({
            "variable": variable,
            "family": meta["family"],
            "source": meta["source"],
            "type": "DYNAMIC",
            "manual_status": manual,
            "dea_direction": meta["dea_direction"],
            "description": meta["description"],
            "rows_total": len(panel),
            "non_missing": int(x.notna().sum()),
            "missing_pct": float(x.isna().mean()),
            "regions_with_any_data": int(temp.loc[x.notna(), "region_id"].nunique()),
            "regions_complete_trajectory": int(len(complete_regions)),
            "countries_with_complete_region": int(complete_meta["code"].nunique()),
            "complete_region_threshold": MIN_COMPLETE_REGIONS,
            "eligible_by_coverage": coverage_ok,
            "eligible_final": eligible,
            "eligibility_status": status,
            "years_required": ",".join(map(str, YEARS)),
            "min": float(non_missing.min()) if not non_missing.empty else np.nan,
            "median": float(non_missing.median()) if not non_missing.empty else np.nan,
            "mean": float(non_missing.mean()) if not non_missing.empty else np.nan,
            "max": float(non_missing.max()) if not non_missing.empty else np.nan,
            "std": float(non_missing.std()) if len(non_missing) > 1 else np.nan,
            "skew": float(non_missing.skew()) if len(non_missing) > 2 else np.nan,
            "transform_hint": infer_transform_hint(x),
        })

        if eligible and len(complete_regions) > 0:
            long = temp[temp["region_id"].isin(complete_regions)].rename(
                columns={variable: "value"}
            )
            long["variable"] = variable
            complete_long_parts.append(long[["region_id", "code", "year", "variable", "value"]])

    diag = pd.DataFrame(rows).sort_values(["eligible_final", "family", "variable"], ascending=[False, True, True])
    complete_long = (
        pd.concat(complete_long_parts, ignore_index=True)
        if complete_long_parts
        else pd.DataFrame(columns=["region_id", "code", "year", "variable", "value"])
    )
    return diag.reset_index(drop=True), complete_long


def structural_diagnostics(
    structural: pd.DataFrame,
    registry: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, meta in registry.items():
        if variable not in structural.columns:
            rows.append({
                "variable": variable,
                "family": meta["family"],
                "source": meta["source"],
                "type": "STRUCTURAL",
                "manual_status": meta["manual_status"],
                "dea_direction": meta["dea_direction"],
                "description": meta["description"],
                "regions_non_missing": 0,
                "countries_non_missing": 0,
                "eligible_final": False,
                "eligibility_status": "VARIABLE_NOT_AVAILABLE_IN_PANEL",
            })
            continue

        x = pd.to_numeric(structural[variable], errors="coerce")
        mask = x.notna()
        non_missing = x[mask]
        regions = int(mask.sum())
        countries = int(structural.loc[mask, "code"].nunique())
        coverage_ok = regions >= MIN_STRUCTURAL_REGIONS
        manual = meta["manual_status"]
        manual_block = str(manual).startswith("EXCLUDE_")
        eligible = bool(coverage_ok and not manual_block)

        if not coverage_ok:
            status = f"EXCLUDE_LT_{MIN_STRUCTURAL_REGIONS}_REGIONS"
        elif manual_block:
            status = manual
        elif "REVIEW" in str(manual) or "SNAPSHOT" in str(manual):
            status = "ELIGIBLE_WITH_REVIEW"
        else:
            status = "ELIGIBLE_STRUCTURAL"

        rows.append({
            "variable": variable,
            "family": meta["family"],
            "source": meta["source"],
            "type": "STRUCTURAL",
            "manual_status": manual,
            "dea_direction": meta["dea_direction"],
            "description": meta["description"],
            "regions_non_missing": regions,
            "countries_non_missing": countries,
            "structural_region_threshold": MIN_STRUCTURAL_REGIONS,
            "eligible_by_coverage": coverage_ok,
            "eligible_final": eligible,
            "eligibility_status": status,
            "min": float(non_missing.min()) if not non_missing.empty else np.nan,
            "median": float(non_missing.median()) if not non_missing.empty else np.nan,
            "mean": float(non_missing.mean()) if not non_missing.empty else np.nan,
            "max": float(non_missing.max()) if not non_missing.empty else np.nan,
            "std": float(non_missing.std()) if len(non_missing) > 1 else np.nan,
            "skew": float(non_missing.skew()) if len(non_missing) > 2 else np.nan,
            "transform_hint": infer_transform_hint(x),
            "unit_review_flag": bool(variable == "pop_flood_pct" and (non_missing > 100).any()),
            "outside_analytical_window_flag": bool(variable == "landslide_susceptibility_mean_2023"),
        })
    return pd.DataFrame(rows).sort_values(["eligible_final", "family", "variable"], ascending=[False, True, True]).reset_index(drop=True)


# =============================================================================
# CORRELATION / REDUNDANCY
# =============================================================================

def build_pairwise_correlations(
    panel: pd.DataFrame,
    variables: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = [v for v in variables if v in panel.columns]
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(existing):
        for b in existing[i + 1 :]:
            pair = panel[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(pair)
            if n < MIN_CORRELATION_PAIRS:
                continue
            pearson = pair[a].corr(pair[b], method="pearson")
            spearman = pair[a].corr(pair[b], method="spearman")
            rows.append({
                "variable_a": a,
                "variable_b": b,
                "paired_observations": n,
                "pearson": pearson,
                "spearman": spearman,
                "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                "family_a": DYNAMIC_FEATURES.get(a, {}).get("family", "unknown"),
                "family_b": DYNAMIC_FEATURES.get(b, {}).get("family", "unknown"),
            })
    corr = pd.DataFrame(rows)
    if corr.empty:
        return corr, corr.copy()
    corr = corr.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    high = corr[corr["abs_spearman"].ge(HIGH_REDUNDANCY_THRESHOLD)].copy()
    high["review_reason"] = "ABS_SPEARMAN_GE_0_90"
    return corr, high


# =============================================================================
# SAMPLE TRADE-OFF
# =============================================================================

def complete_region_set(panel: pd.DataFrame, variable: str) -> set[str]:
    if variable not in panel.columns:
        return set()
    counts = (
        panel[["region_id", variable]]
        .assign(_present=lambda d: pd.to_numeric(d[variable], errors="coerce").notna())
        .groupby("region_id")["_present"]
        .sum()
    )
    return set(counts[counts.eq(N_YEARS)].index)


def build_sample_tradeoff(panel: pd.DataFrame) -> pd.DataFrame:
    """Report the official >=512-region dynamic specification only."""
    scenarios: dict[str, list[str]] = {
        "OFFICIAL_512_DYNAMIC_CORE": list(DYNAMIC_FEATURES.keys()),
    }

    region_meta = panel[["region_id", "code"]].drop_duplicates("region_id")
    rows: list[dict[str, Any]] = []
    for name, vars_ in scenarios.items():
        available = [v for v in vars_ if v in panel.columns]
        missing_vars = sorted(set(vars_) - set(available))
        if missing_vars:
            region_set: set[str] = set()
        else:
            sets = [complete_region_set(panel, v) for v in available]
            region_set = set.intersection(*sets) if sets else set()
        country_count = int(region_meta[region_meta["region_id"].isin(region_set)]["code"].nunique())
        rows.append({
            "scenario": name,
            "variables": ",".join(vars_),
            "n_variables": len(vars_),
            "complete_regions": len(region_set),
            "countries_represented": country_count,
            "complete_region_year_rows": len(region_set) * N_YEARS,
            "missing_required_variables": ",".join(missing_vars),
        })
    return pd.DataFrame(rows)


def build_official_common_region_set(
    panel: pd.DataFrame,
    structural: pd.DataFrame,
    dynamic_diag: pd.DataFrame,
    structural_diag: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and validate the exact common region set for all official active features.

    The script never silently removes a feature to reach the coverage target. If the
    intersection across all eligible dynamic and structural features falls below 512,
    execution stops and the user must decide which feature to reconsider.
    """
    dyn_vars = dynamic_diag.loc[
        dynamic_diag["eligible_final"].fillna(False), "variable"
    ].astype(str).tolist()
    struct_vars = structural_diag.loc[
        structural_diag["eligible_final"].fillna(False), "variable"
    ].astype(str).tolist()

    sets: list[set[str]] = []
    qa_rows: list[dict[str, Any]] = []

    for variable in dyn_vars:
        s = complete_region_set(panel, variable)
        sets.append(s)
        qa_rows.append({
            "block": "DYNAMIC",
            "variable": variable,
            "regions_available": len(s),
            "threshold": MIN_COMPLETE_REGIONS,
        })

    for variable in struct_vars:
        if variable not in structural.columns:
            s = set()
        else:
            mask = pd.to_numeric(structural[variable], errors="coerce").notna()
            s = set(structural.loc[mask, "region_id"].astype(str))
        sets.append(s)
        qa_rows.append({
            "block": "STRUCTURAL",
            "variable": variable,
            "regions_available": len(s),
            "threshold": MIN_STRUCTURAL_REGIONS,
        })

    common = set.intersection(*sets) if sets else set()
    if len(common) < 512:
        raise ValueError(
            f"Official active-feature intersection has {len(common)} regions; "
            "the project rule requires at least 512. No feature was dropped automatically."
        )

    region_meta_cols = [c for c in ["region_id", "code", "geo_code", "geo_name"] if c in panel.columns]
    region_set = (
        panel[region_meta_cols]
        .drop_duplicates("region_id")
        .loc[lambda d: d["region_id"].astype(str).isin(common)]
        .sort_values([c for c in ["code", "geo_code", "region_id"] if c in region_meta_cols])
        .reset_index(drop=True)
    )

    dyn_cols = ["region_id", "code", "geo_code", "geo_name", "year"] + dyn_vars
    dyn_cols = [c for c in dyn_cols if c in panel.columns]
    official_dynamic = (
        panel.loc[panel["region_id"].astype(str).isin(common), dyn_cols]
        .sort_values(["region_id", "year"])
        .reset_index(drop=True)
    )

    struct_cols = ["region_id", "code", "geo_code", "geo_name"] + struct_vars
    struct_cols = [c for c in struct_cols if c in structural.columns]
    official_structural = (
        structural.loc[structural["region_id"].astype(str).isin(common), struct_cols]
        .sort_values([c for c in ["code", "geo_code", "region_id"] if c in struct_cols])
        .reset_index(drop=True)
    )

    feature_rows = []
    for _, r in dynamic_diag[dynamic_diag["eligible_final"].fillna(False)].iterrows():
        feature_rows.append({
            "block": "DYNAMIC", "variable": r["variable"], "family": r["family"],
            "dea_direction": r["dea_direction"], "source": r["source"],
            "coverage_regions": int(r["regions_complete_trajectory"]),
            "official_common_regions": len(common),
        })
    for _, r in structural_diag[structural_diag["eligible_final"].fillna(False)].iterrows():
        feature_rows.append({
            "block": "STRUCTURAL", "variable": r["variable"], "family": r["family"],
            "dea_direction": r["dea_direction"], "source": r["source"],
            "coverage_regions": int(r["regions_non_missing"]),
            "official_common_regions": len(common),
        })

    qa = pd.DataFrame(qa_rows)
    qa["official_common_regions"] = len(common)
    qa["passes_individual_threshold"] = qa["regions_available"].ge(512)

    return region_set, pd.DataFrame(feature_rows), official_dynamic, official_structural, qa


# =============================================================================
# BoD FAMILY DESIGN
# =============================================================================

def build_bod_design(
    dynamic_diag: pd.DataFrame,
    structural_diag: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the official BoD family design metadata."""

    all_diag = pd.concat(
        [
            dynamic_diag.assign(block="DYNAMIC"),
            structural_diag.assign(block="STRUCTURAL"),
        ],
        ignore_index=True,
        sort=False,
    )

    # BoD candidate families under the current official 512-region design.
    bod_family_policy = {
        "demographic_productive_potential": {
            "normative": True,
            "family_role": "DYNAMIC_NORMATIVE",
            "notes": "share_15_64 is DESEJAVEL; share_65_plus is INDESEJAVEL and will be converted by subtraction before BoD.",
        },
        "socioeconomic_deprivation": {
            "normative": True,
            "family_role": "DYNAMIC_NORMATIVE",
            "notes": "Poverty, inequality and prosperity shortfall are INDESEJAVEL and will be converted by subtraction before BoD.",
        },
        "environmental_risk": {
            "normative": True,
            "family_role": "STRUCTURAL_NORMATIVE",
            "notes": "Environmental risks are INDESEJAVEL and will be converted by subtraction before a separate structural BoD.",
        },
    }

    design_rows: list[dict[str, Any]] = []
    input_rows: list[pd.DataFrame] = []

    # Dynamic raw data source for BoD family staging.
    # Structural raw values are staged separately in main() because they are one row per region.
    for family, policy in bod_family_policy.items():
        subset = all_diag[all_diag["family"].eq(family)].copy()
        variables = subset["variable"].astype(str).tolist()
        eligible = subset[subset["eligible_final"].fillna(False)].copy()
        eligible_variables = eligible["variable"].astype(str).tolist()
        orientations = eligible.get("dea_direction", pd.Series(dtype="object")).dropna().astype(str).tolist()

        # A family needs at least two eligible indicators for BoD to add value.
        if len(eligible_variables) < 2:
            status = "NOT_READY_LT_2_ELIGIBLE_INDICATORS"
        elif any(d == "NAO_NORMATIVA" for d in orientations):
            status = "NOT_READY_NAO_NORMATIVA_DIRECTION_PRESENT"
        else:
            status = "READY_FOR_FINAL_COMMON_WEIGHT_BOD"

        design_rows.append({
            "family": family,
            "family_role": policy["family_role"],
            "normative": policy["normative"],
            "all_family_variables": ",".join(variables),
            "eligible_variables": ",".join(eligible_variables),
            "eligible_indicator_count": len(eligible_variables),
            "dea_directions": ",".join(orientations),
            "bod_method": BOD_METHOD,
            "weight_scope": "COMMON_ACROSS_REGIONS_AND_2015_2019" if policy["family_role"] == "DYNAMIC_NORMATIVE" else "COMMON_ACROSS_REGIONS_STRUCTURAL",
            "undesirable_treatment": "SUBTRACAO_ALPHA_MINUS_X",
            "normalization": "DIVIDE_BY_MAX_AFTER_DIRECTION_AND_POSITIVITY_TREATMENT",
            "zero_negative_protocol": "THESIS_EPSILON_1PCT_MIN_POSITIVE_POOLED_BY_VARIABLE",
            "weights_estimated_now": ESTIMATE_BOD_WEIGHTS,
            "status": status,
            "notes": policy["notes"],
        })

    return pd.DataFrame(design_rows), pd.concat(input_rows, ignore_index=True) if input_rows else pd.DataFrame()


def build_bod_raw_input_long(
    dynamic_complete_long: pd.DataFrame,
    structural: pd.DataFrame,
    dynamic_diag: pd.DataFrame,
    structural_diag: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    dyn_bod = dynamic_diag[
        dynamic_diag["eligible_final"].fillna(False)
        & dynamic_diag["family"].isin(["demographic_productive_potential", "socioeconomic_deprivation"])
    ][["variable", "family", "dea_direction"]]

    if not dyn_bod.empty and not dynamic_complete_long.empty:
        d = dynamic_complete_long.merge(dyn_bod, on="variable", how="inner")
        d["block"] = "DYNAMIC"
        parts.append(d[["region_id", "code", "year", "variable", "value", "family", "dea_direction", "block"]])

    struct_bod = structural_diag[
        structural_diag["eligible_final"].fillna(False)
        & structural_diag["family"].isin(["environmental_risk"])
    ][["variable", "family", "dea_direction"]]

    for _, row in struct_bod.iterrows():
        var = row["variable"]
        if var not in structural.columns:
            continue
        s = structural[["region_id", "code", var]].copy()
        s = s.rename(columns={var: "value"})
        s["year"] = pd.NA
        s["variable"] = var
        s["family"] = row["family"]
        s["dea_direction"] = row["dea_direction"]
        s["block"] = "STRUCTURAL"
        parts.append(s[["region_id", "code", "year", "variable", "value", "family", "dea_direction", "block"]])

    if not parts:
        return pd.DataFrame(columns=["region_id", "code", "year", "variable", "value", "family", "dea_direction", "block"])
    return pd.concat(parts, ignore_index=True)



# =============================================================================
# FINAL PREPROCESSING + COMMON-WEIGHT BoD
# =============================================================================

def thesis_epsilon(series: pd.Series) -> float:
    """Return 1% of the smallest strictly positive value, as in the thesis protocol."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    positive = x[x > 0]
    if positive.empty:
        raise ValueError(
            "Thesis epsilon cannot be calculated because the variable has no strictly positive values."
        )
    return float(positive.min()) * THESIS_EPSILON_SHARE


def preprocess_bod_variable(
    series: pd.Series,
    variable: str,
    family: str,
    dea_direction: str,
    block: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply direction, thesis positivity protocol and max normalization.

    Order of operations:
    1) INDESEJAVEL -> subtraction alpha - x, with alpha = max(x);
    2) if negatives remain, translate all values by |min| + epsilon;
       if only legitimate zeros remain and strict positivity is required, replace zeros by epsilon;
    3) divide by the pooled maximum.

    The epsilon is calculated on the post-direction series using 1% of its
    smallest strictly positive value. This preserves the thesis protocol while
    adapting it to the pooled 2015-2019 common-weight estimation universe.
    """
    raw = pd.to_numeric(series, errors="coerce")
    if raw.isna().any():
        raise ValueError(f"BoD variable {variable} contains missing values in the official common sample.")

    raw_min = float(raw.min())
    raw_max = float(raw.max())
    negative_raw = int((raw < 0).sum())
    zero_raw = int((raw == 0).sum())

    alpha = np.nan
    if dea_direction == "INDESEJAVEL":
        alpha = raw_max
        directed = alpha - raw
        direction_action = "SUBTRACAO_ALPHA_MINUS_X"
    elif dea_direction == "DESEJAVEL":
        directed = raw.copy()
        direction_action = "KEEP_ORIGINAL_DIRECTION"
    else:
        raise ValueError(
            f"BoD variable {variable} has non-normative DEA direction: {dea_direction}."
        )

    directed = pd.to_numeric(directed, errors="coerce")
    min_directed_before = float(directed.min())
    zero_directed_before = int((directed == 0).sum())
    negative_directed_before = int((directed < 0).sum())

    epsilon = np.nan
    positivity_action = "NONE"
    processed = directed.copy()

    if (processed < 0).any():
        epsilon = thesis_epsilon(processed)
        shift = abs(float(processed.min())) + epsilon
        processed = processed + shift
        positivity_action = "TRANSLATION_ABS_MIN_PLUS_EPSILON"
    elif (processed == 0).any():
        epsilon = thesis_epsilon(processed)
        processed = processed.mask(processed == 0, epsilon)
        positivity_action = "ZERO_TO_EPSILON"

    if (processed <= 0).any():
        raise ValueError(
            f"Strict positivity failed for BoD variable {variable} after the thesis protocol."
        )

    denominator = float(processed.max())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Invalid max-normalization denominator for {variable}: {denominator}")

    normalized = processed / denominator

    audit = {
        "block": block,
        "family": family,
        "variable": variable,
        "dea_direction_original": dea_direction,
        "direction_action": direction_action,
        "alpha_subtraction": alpha,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_zero_count": zero_raw,
        "raw_negative_count": negative_raw,
        "directed_min_before_positivity": min_directed_before,
        "directed_zero_count_before_positivity": zero_directed_before,
        "directed_negative_count_before_positivity": negative_directed_before,
        "epsilon": epsilon,
        "epsilon_share_of_min_positive": THESIS_EPSILON_SHARE,
        "positivity_action": positivity_action,
        "normalization": "DIVIDE_BY_MAX",
        "normalization_denominator": denominator,
        "processed_min": float(processed.min()),
        "processed_max": float(processed.max()),
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
    }
    return normalized.astype(float), audit


def solve_common_weight_bod(
    normalized: pd.DataFrame,
    family: str,
    block: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Estimate one common weight vector for a BoD family.

    The common vector maximizes the mean family score subject to every observation
    having score <= 1. The unrestricted solution is estimated first. If any
    indicator contributes less than BOD_MIN_CONTRIBUTION at the mean reference
    vector, the thesis-style minimum proportional contribution restriction
    (phi=0.05) is activated for the official solution.
    """
    variables = list(normalized.columns)
    Y = normalized.to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError(f"Non-finite values found in normalized BoD family {family}.")
    if (Y <= 0).any():
        raise ValueError(f"BoD family {family} contains non-positive values after preprocessing.")

    means = Y.mean(axis=0)
    if (means <= 0).any():
        raise ValueError(f"BoD family {family} has non-positive indicator means.")

    # score_i = Y_i * w <= 1 for every observation
    A_base = Y.copy()
    b_base = np.ones(Y.shape[0], dtype=float)
    c = -means  # maximize mean score
    bounds = [(0.0, None)] * len(variables)

    unrestricted = linprog(
        c=c,
        A_ub=A_base,
        b_ub=b_base,
        bounds=bounds,
        method="highs",
    )
    if not unrestricted.success:
        raise RuntimeError(
            f"Unrestricted common-weight BoD failed for {family}: {unrestricted.message}"
        )

    w_unres = unrestricted.x.astype(float)
    mean_weighted_unres = w_unres * means
    denom_unres = float(mean_weighted_unres.sum())
    contrib_unres = (
        mean_weighted_unres / denom_unres
        if denom_unres > 0
        else np.zeros_like(mean_weighted_unres)
    )

    restriction_needed = bool((contrib_unres < (BOD_MIN_CONTRIBUTION - 1e-10)).any())

    if restriction_needed:
        # phi * sum_h(w_h*ybar_h) - w_j*ybar_j <= 0
        A_contrib = []
        b_contrib = []
        for j in range(len(variables)):
            row = BOD_MIN_CONTRIBUTION * means.copy()
            row[j] -= means[j]
            A_contrib.append(row)
            b_contrib.append(0.0)
        A_final = np.vstack([A_base, np.asarray(A_contrib, dtype=float)])
        b_final = np.concatenate([b_base, np.asarray(b_contrib, dtype=float)])
        restricted = linprog(
            c=c,
            A_ub=A_final,
            b_ub=b_final,
            bounds=bounds,
            method="highs",
        )
        if not restricted.success:
            raise RuntimeError(
                f"Restricted common-weight BoD failed for {family}: {restricted.message}"
            )
        w_final = restricted.x.astype(float)
        solver_message = restricted.message
        final_model = "COMMON_WEIGHT_BOD_WITH_MIN_CONTRIBUTION"
    else:
        w_final = w_unres.copy()
        solver_message = unrestricted.message
        final_model = "COMMON_WEIGHT_BOD_UNRESTRICTED"

    scores = Y @ w_final
    # Numerical tolerance only; the LP itself defines the scale.
    scores = np.clip(scores, 0.0, 1.0 + 1e-9)

    mean_weighted_final = w_final * means
    denom_final = float(mean_weighted_final.sum())
    contrib_final = (
        mean_weighted_final / denom_final
        if denom_final > 0
        else np.zeros_like(mean_weighted_final)
    )

    weight_rows = []
    for j, variable in enumerate(variables):
        weight_rows.append({
            "block": block,
            "family": family,
            "variable": variable,
            "weight_unrestricted": float(w_unres[j]),
            "mean_contribution_unrestricted": float(contrib_unres[j]),
            "weight_final": float(w_final[j]),
            "mean_contribution_final": float(contrib_final[j]),
            "minimum_contribution_phi": BOD_MIN_CONTRIBUTION if restriction_needed else 0.0,
            "restriction_applied": restriction_needed,
            "final_model": final_model,
        })

    summary = {
        "block": block,
        "family": family,
        "n_indicators": len(variables),
        "n_observations": len(normalized),
        "restriction_applied": restriction_needed,
        "minimum_contribution_phi": BOD_MIN_CONTRIBUTION if restriction_needed else 0.0,
        "final_model": final_model,
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_max": float(scores.max()),
        "solver_message": solver_message,
    }
    return scores, pd.DataFrame(weight_rows), summary


def estimate_all_bod_families(
    official_dynamic: pd.DataFrame,
    official_structural: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess and estimate all official normative families."""
    preprocessing_rows: list[dict[str, Any]] = []
    processed_long_parts: list[pd.DataFrame] = []
    weight_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    dynamic_scores = official_dynamic[["region_id", "code", "geo_code", "geo_name", "year"]].copy()
    structural_scores = official_structural[["region_id", "code", "geo_code", "geo_name"]].copy()

    family_specs = [
        ("DYNAMIC", "demographic_productive_potential", official_dynamic, DYNAMIC_FEATURES),
        ("DYNAMIC", "socioeconomic_deprivation", official_dynamic, DYNAMIC_FEATURES),
        ("STRUCTURAL", "environmental_risk", official_structural, STRUCTURAL_FEATURES),
    ]

    for block, family, frame, registry in family_specs:
        variables = [
            variable
            for variable, meta in registry.items()
            if meta["family"] == family and variable in frame.columns
        ]
        if len(variables) < 2:
            raise ValueError(
                f"Official BoD family {family} has fewer than two available indicators: {variables}"
            )

        normalized = pd.DataFrame(index=frame.index)
        for variable in variables:
            direction = registry[variable]["dea_direction"]
            norm, audit = preprocess_bod_variable(
                frame[variable], variable, family, direction, block
            )
            normalized[variable] = norm
            preprocessing_rows.append(audit)

            ids = ["region_id", "code"] + (["year"] if block == "DYNAMIC" else [])
            temp = frame[ids].copy()
            temp["block"] = block
            temp["family"] = family
            temp["variable"] = variable
            temp["value_raw"] = pd.to_numeric(frame[variable], errors="raise").to_numpy(dtype=float)
            temp["value_normalized"] = norm.to_numpy()
            processed_long_parts.append(temp)

        scores, weights, summary = solve_common_weight_bod(normalized, family, block)
        weight_parts.append(weights)
        summaries.append(summary)
        score_col = f"bod_{family}"
        if block == "DYNAMIC":
            dynamic_scores[score_col] = scores
        else:
            structural_scores[score_col] = scores

    return (
        dynamic_scores,
        structural_scores,
        pd.concat(weight_parts, ignore_index=True),
        pd.DataFrame(summaries),
        pd.concat(processed_long_parts, ignore_index=True),
        pd.DataFrame(preprocessing_rows),
    )



def build_bod_weighted_decomposition(
    bod_processed_long: pd.DataFrame,
    bod_weights: pd.DataFrame,
    bod_dynamic_scores: pd.DataFrame,
    bod_structural_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build auditable variable-level BoD decomposition without re-normalizing scores.

    For every normative family observation:
        weighted_contribution_{i,t,j} = value_normalized_{i,t,j} * common_weight_j

    The family score is reconstructed as the sum of weighted contributions.
    Dynamic common weights are constant across all regions and all years in 2015-2019.
    No second normalization is applied after the BoD aggregation.
    """
    required_processed = {
        "region_id", "code", "block", "family", "variable",
        "value_raw", "value_normalized",
    }
    missing_processed = required_processed - set(bod_processed_long.columns)
    if missing_processed:
        raise ValueError(
            "BoD processed-long table is missing columns required for decomposition: "
            f"{sorted(missing_processed)}"
        )

    required_weights = {"block", "family", "variable", "weight_final"}
    missing_weights = required_weights - set(bod_weights.columns)
    if missing_weights:
        raise ValueError(
            "BoD weights table is missing columns required for decomposition: "
            f"{sorted(missing_weights)}"
        )

    if bod_weights.duplicated(["block", "family", "variable"]).any():
        raise ValueError("BoD common weights contain duplicated block-family-variable keys.")

    weights = bod_weights[
        ["block", "family", "variable", "weight_final", "final_model", "restriction_applied"]
    ].copy()
    weights = weights.rename(columns={"weight_final": "common_weight"})

    contributions = bod_processed_long.merge(
        weights,
        on=["block", "family", "variable"],
        how="left",
        validate="many_to_one",
    )
    if contributions["common_weight"].isna().any():
        bad = (
            contributions.loc[
                contributions["common_weight"].isna(),
                ["block", "family", "variable"],
            ]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Missing common BoD weights for decomposition keys: {bad}")

    contributions["weighted_contribution"] = (
        pd.to_numeric(contributions["value_normalized"], errors="raise")
        * pd.to_numeric(contributions["common_weight"], errors="raise")
    )

    dynamic_score_cols = [
        c for c in bod_dynamic_scores.columns if str(c).startswith("bod_")
    ]
    structural_score_cols = [
        c for c in bod_structural_scores.columns if str(c).startswith("bod_")
    ]

    dyn_score_long = bod_dynamic_scores.melt(
        id_vars=["region_id", "code", "geo_code", "geo_name", "year"],
        value_vars=dynamic_score_cols,
        var_name="score_feature",
        value_name="family_score",
    )
    dyn_score_long["family"] = dyn_score_long["score_feature"].str.replace(
        r"^bod_", "", regex=True
    )
    dyn_score_long["block"] = "DYNAMIC"

    struct_score_long = bod_structural_scores.melt(
        id_vars=["region_id", "code", "geo_code", "geo_name"],
        value_vars=structural_score_cols,
        var_name="score_feature",
        value_name="family_score",
    )
    struct_score_long["family"] = struct_score_long["score_feature"].str.replace(
        r"^bod_", "", regex=True
    )
    struct_score_long["block"] = "STRUCTURAL"

    dyn_contrib = contributions[contributions["block"].eq("DYNAMIC")].copy()
    dyn_contrib = dyn_contrib.merge(
        dyn_score_long[
            ["region_id", "code", "geo_code", "geo_name", "year", "family", "family_score"]
        ],
        on=["region_id", "code", "year", "family"],
        how="left",
        validate="many_to_one",
    )
    dyn_contrib["weight_scope"] = "COMMON_ACROSS_REGIONS_AND_2015_2019"

    struct_contrib = contributions[contributions["block"].eq("STRUCTURAL")].copy()
    struct_contrib = struct_contrib.merge(
        struct_score_long[
            ["region_id", "code", "geo_code", "geo_name", "family", "family_score"]
        ],
        on=["region_id", "code", "family"],
        how="left",
        validate="many_to_one",
    )
    struct_contrib["weight_scope"] = "COMMON_ACROSS_REGIONS_STRUCTURAL"

    contributions = pd.concat(
        [dyn_contrib, struct_contrib],
        ignore_index=True,
        sort=False,
    )

    if contributions["family_score"].isna().any():
        raise ValueError("At least one BoD weighted contribution could not be linked to its family score.")

    contributions["contribution_share_of_family_score"] = np.where(
        pd.to_numeric(contributions["family_score"], errors="coerce").abs() > 1e-15,
        contributions["weighted_contribution"] / contributions["family_score"],
        np.nan,
    )

    # IMPORTANT: contribution_share_of_family_score is created only after the
    # dynamic and structural contribution tables are concatenated. Rebuild the
    # block-specific views from the enriched table so that every downstream
    # reconciliation and wide export sees the complete decomposition schema.
    dyn_contrib = contributions[contributions["block"].eq("DYNAMIC")].copy()
    struct_contrib = contributions[contributions["block"].eq("STRUCTURAL")].copy()

    # Reconciliation: family score must equal the exact sum of weighted components.
    dyn_check = (
        dyn_contrib.groupby(["region_id", "year", "family"], as_index=False)
        .agg(
            reconstructed_family_score=("weighted_contribution", "sum"),
            family_score=("family_score", "first"),
        )
    )
    struct_check = (
        struct_contrib.groupby(["region_id", "family"], as_index=False)
        .agg(
            reconstructed_family_score=("weighted_contribution", "sum"),
            family_score=("family_score", "first"),
        )
    )
    all_errors = np.concatenate([
        (
            pd.to_numeric(dyn_check["reconstructed_family_score"], errors="raise")
            - pd.to_numeric(dyn_check["family_score"], errors="raise")
        ).abs().to_numpy(dtype=float),
        (
            pd.to_numeric(struct_check["reconstructed_family_score"], errors="raise")
            - pd.to_numeric(struct_check["family_score"], errors="raise")
        ).abs().to_numpy(dtype=float),
    ])
    max_abs_error = float(all_errors.max()) if len(all_errors) else 0.0
    if max_abs_error > 1e-10:
        raise ValueError(
            "BoD decomposition does not reconcile with family scores; "
            f"maximum absolute error = {max_abs_error:.3e}."
        )

    def _wide(
        frame: pd.DataFrame,
        score_frame: pd.DataFrame,
        id_cols: list[str],
    ) -> pd.DataFrame:
        base = score_frame.copy()
        metrics = {
            "value_raw": "raw",
            "value_normalized": "normalized",
            "common_weight": "common_weight",
            "weighted_contribution": "weighted_contribution",
            "contribution_share_of_family_score": "share_of_family_score",
        }
        for metric, suffix in metrics.items():
            piv = frame.pivot_table(
                index=id_cols,
                columns=["family", "variable"],
                values=metric,
                aggfunc="first",
            )
            if piv.empty:
                continue
            piv.columns = [
                f"{family}__{variable}__{suffix}"
                for family, variable in piv.columns.to_flat_index()
            ]
            piv = piv.reset_index()
            base = base.merge(
                piv,
                on=id_cols,
                how="left",
                validate="one_to_one",
            )
        return base

    dynamic_wide = _wide(
        dyn_contrib,
        bod_dynamic_scores,
        ["region_id", "code", "year"],
    )
    structural_wide = _wide(
        struct_contrib,
        bod_structural_scores,
        ["region_id", "code"],
    )

    # Stable sort for audit and downstream Power BI preparation.
    sort_cols = [c for c in ["block", "family", "region_id", "year", "variable"] if c in contributions.columns]
    contributions = contributions.sort_values(sort_cols).reset_index(drop=True)
    dynamic_wide = dynamic_wide.sort_values(["region_id", "year"]).reset_index(drop=True)
    structural_wide = structural_wide.sort_values(["region_id"]).reset_index(drop=True)

    return contributions, dynamic_wide, structural_wide


def transform_non_bod_variable(
    series: pd.Series,
    variable: str,
    family: str,
    block: str,
    transformation: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Transform descriptive features and normalize by their maximum.

    No Min-Max and no Z-score are used. LOG1P is applied only to active
    positive descriptive/single-indicator features when specified by the final design.
    """
    raw = pd.to_numeric(series, errors="coerce")
    if raw.isna().any():
        raise ValueError(f"Non-BoD variable {variable} contains missing values in the official sample.")

    if transformation == "LOG1P":
        if (raw < 0).any():
            raise ValueError(f"LOG1P requested for {variable}, but negative values are present.")
        transformed = np.log1p(raw)
    elif transformation == "NONE":
        transformed = raw.copy()
    else:
        raise ValueError(f"Unknown transformation for {variable}: {transformation}")

    denominator = float(transformed.max())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Invalid max-normalization denominator for {variable}: {denominator}")
    normalized = transformed / denominator

    audit = {
        "block": block,
        "family": family,
        "variable": variable,
        "dea_direction_original": "NAO_NORMATIVA" if family != "regional_economic_activity_proxy" else "DESEJAVEL_SINGLE_INDICATOR_NO_BOD",
        "direction_action": "NO_BOD_DIRECTION_TRANSFORMATION",
        "alpha_subtraction": np.nan,
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_zero_count": int((raw == 0).sum()),
        "raw_negative_count": int((raw < 0).sum()),
        "epsilon": np.nan,
        "positivity_action": "NOT_REQUIRED_OUTSIDE_BOD",
        "distribution_transformation": transformation,
        "normalization": "DIVIDE_BY_MAX",
        "normalization_denominator": denominator,
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
    }
    return pd.Series(normalized, index=series.index, dtype=float), audit


def build_final_model_representation(
    official_dynamic: pd.DataFrame,
    official_structural: pd.DataFrame,
    bod_dynamic_scores: pd.DataFrame,
    bod_structural_scores: pd.DataFrame,
    bod_preprocessing_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct the final Script-05 handoff and family-balanced feature metadata."""
    dynamic = official_dynamic[["region_id", "code", "geo_code", "geo_name", "year"]].copy()
    structural = official_structural[["region_id", "code", "geo_code", "geo_name"]].copy()

    dynamic = dynamic.merge(
        bod_dynamic_scores,
        on=["region_id", "code", "geo_code", "geo_name", "year"],
        how="left",
        validate="one_to_one",
    )
    structural = structural.merge(
        bod_structural_scores,
        on=["region_id", "code", "geo_code", "geo_name"],
        how="left",
        validate="one_to_one",
    )

    audit_rows = bod_preprocessing_audit.to_dict("records")

    non_bod_dynamic_specs = {
        "ntl_per_capita": ("regional_economic_activity_proxy", "LOG1P"),
    }
    for variable, (family, transformation) in non_bod_dynamic_specs.items():
        values, audit = transform_non_bod_variable(
            official_dynamic[variable], variable, family, "DYNAMIC", transformation
        )
        out_col = f"{variable}_norm"
        dynamic[out_col] = values.to_numpy()
        audit_rows.append(audit)

    non_bod_structural_specs = {
        "built_area_share_2015": ("urbanization_built_environment", "NONE"),
        "ghs_urban_pop_share": ("urbanization_built_environment", "NONE"),
        "ghs_urban_centre_pop_share": ("urbanization_built_environment", "NONE"),
    }
    for variable, (family, transformation) in non_bod_structural_specs.items():
        values, audit = transform_non_bod_variable(
            official_structural[variable], variable, family, "STRUCTURAL", transformation
        )
        out_col = f"{variable}_norm"
        structural[out_col] = values.to_numpy()
        audit_rows.append(audit)

    dynamic_features = [
        ("bod_demographic_productive_potential", "demographic_productive_potential", "BOD_FAMILY_SCORE"),
        ("bod_socioeconomic_deprivation", "socioeconomic_deprivation", "BOD_FAMILY_SCORE"),
        ("ntl_per_capita_norm", "regional_economic_activity_proxy", "SINGLE_INDICATOR"),
    ]
    structural_features = [
        ("bod_environmental_risk", "environmental_risk", "BOD_FAMILY_SCORE"),
        ("built_area_share_2015_norm", "urbanization_built_environment", "BALANCED_NON_NORMATIVE"),
        ("ghs_urban_pop_share_norm", "urbanization_built_environment", "BALANCED_NON_NORMATIVE"),
        ("ghs_urban_centre_pop_share_norm", "urbanization_built_environment", "BALANCED_NON_NORMATIVE"),
    ]

    feature_rows = []
    for block, specs in [("DYNAMIC", dynamic_features), ("STRUCTURAL", structural_features)]:
        family_counts: dict[str, int] = {}
        for _, family, _ in specs:
            family_counts[family] = family_counts.get(family, 0) + 1
        for feature, family, role in specs:
            n = family_counts[family]
            feature_rows.append({
                "block": block,
                "feature": feature,
                "family": family,
                "feature_role": role,
                "features_in_family": n,
                "within_family_distance_weight": 1.0 / n,
                "distance_scaling_factor_sqrt_weight": float(np.sqrt(1.0 / n)),
                "family_distance_weight": 1.0,
                "distance_rule": "MEAN_SQUARED_CONTRIBUTION_WITHIN_FAMILY",
            })
    feature_weights = pd.DataFrame(feature_rows)

    dyn_value_cols = [f for f, _, _ in dynamic_features]
    struct_value_cols = [f for f, _, _ in structural_features]
    if dynamic[dyn_value_cols].isna().any().any():
        raise ValueError("Final dynamic model panel contains missing analytical features.")
    if structural[struct_value_cols].isna().any().any():
        raise ValueError("Final structural model panel contains missing analytical features.")

    dyn_long = dynamic.melt(
        id_vars=["region_id", "code", "geo_code", "geo_name", "year"],
        value_vars=dyn_value_cols,
        var_name="feature",
        value_name="value",
    ).merge(
        feature_weights[feature_weights["block"].eq("DYNAMIC")],
        on="feature",
        how="left",
        validate="many_to_one",
    )

    struct_long = structural.melt(
        id_vars=["region_id", "code", "geo_code", "geo_name"],
        value_vars=struct_value_cols,
        var_name="feature",
        value_name="value",
    ).merge(
        feature_weights[feature_weights["block"].eq("STRUCTURAL")],
        on="feature",
        how="left",
        validate="many_to_one",
    )

    return dynamic, structural, feature_weights, dyn_long, struct_long, pd.DataFrame(audit_rows)


def build_final_country_references(
    dynamic_model: pd.DataFrame,
    structural_model: pd.DataFrame,
    feature_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dyn_features = feature_weights.loc[
        feature_weights["block"].eq("DYNAMIC"), "feature"
    ].tolist()
    struct_features = feature_weights.loc[
        feature_weights["block"].eq("STRUCTURAL"), "feature"
    ].tolist()

    dyn = (
        dynamic_model.groupby(["code", "year"], as_index=False)[dyn_features]
        .mean()
    )
    counts = (
        dynamic_model.groupby(["code", "year"], as_index=False)
        .agg(contributing_regions=("region_id", "nunique"))
    )
    dyn = dyn.merge(counts, on=["code", "year"], how="left", validate="one_to_one")

    struct = (
        structural_model.groupby("code", as_index=False)[struct_features]
        .mean()
    )
    scounts = (
        structural_model.groupby("code", as_index=False)
        .agg(contributing_regions=("region_id", "nunique"))
    )
    struct = struct.merge(scounts, on="code", how="left", validate="one_to_one")
    return dyn, struct

# =============================================================================
# COUNTRY REFERENCE INPUTS — RAW, PRE-SCALING
# =============================================================================

def build_country_reference_inputs(
    panel: pd.DataFrame,
    dynamic_diag: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare equal-weight country means for eligible dynamic variables.

    These are RAW PRE-SCALING reference inputs, not the final country profiles
    used for distance analysis. Final country profiles must be recomputed after
    the preprocessing/scaling decision so that region and country vectors live
    in exactly the same analytical space.
    """
    eligible_vars = dynamic_diag.loc[
        dynamic_diag["eligible_final"].fillna(False), "variable"
    ].astype(str).tolist()
    eligible_vars = [v for v in eligible_vars if v in panel.columns]
    if not eligible_vars:
        return pd.DataFrame()

    long = panel[["code", "year"] + eligible_vars].melt(
        id_vars=["code", "year"],
        value_vars=eligible_vars,
        var_name="variable",
        value_name="value",
    )
    out = (
        long.groupby(["code", "year", "variable"], as_index=False)
        .agg(
            equal_weight_region_mean=("value", "mean"),
            contributing_regions=("value", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
    )
    out["status"] = "RAW_PRE_SCALING_REFERENCE_INPUT"
    return out


# =============================================================================
# POPULATION CROSS-CHECK
# =============================================================================

def build_population_crosscheck(core: pd.DataFrame) -> pd.DataFrame:
    check = core[[
        "region_id",
        "code",
        "geo_code",
        "year",
        "wp_age_population",
        "space2stats_population",
    ]].copy()
    check["absolute_difference"] = check["space2stats_population"] - check["wp_age_population"]
    check["relative_difference_pct"] = (
        safe_divide(check["absolute_difference"], check["wp_age_population"]) * 100.0
    )
    return check


# =============================================================================
# EXCEL OUTPUT
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
            ws.column_dimensions[letter].width = min(max(max_length + 2, 10), 58)
    wb.save(path)


def write_qa_workbook(
    dynamic_diag: pd.DataFrame,
    structural_diag: pd.DataFrame,
    feature_eligibility: pd.DataFrame,
    sample_tradeoff: pd.DataFrame,
    bod_design: pd.DataFrame,
    bod_raw: pd.DataFrame,
    bod_processed_long: pd.DataFrame,
    bod_weighted_contributions: pd.DataFrame,
    flagged_qa: pd.DataFrame,
    correlations: pd.DataFrame,
    redundancy: pd.DataFrame,
    structural: pd.DataFrame,
    wdi_context: pd.DataFrame,
    manual_exclusions: pd.DataFrame,
    source_decisions: pd.DataFrame,
    script03_variable_qa: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(QA_WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        dynamic_diag.to_excel(writer, sheet_name="dynamic_diagnostics", index=False)
        structural_diag.to_excel(writer, sheet_name="structural_diagnostics", index=False)
        feature_eligibility.to_excel(writer, sheet_name="feature_eligibility", index=False)
        sample_tradeoff.to_excel(writer, sheet_name="sample_tradeoff", index=False)
        bod_design.to_excel(writer, sheet_name="bod_family_design", index=False)
        bod_raw.head(1_000_000).to_excel(writer, sheet_name="bod_input_raw", index=False)
        bod_processed_long.head(1_000_000).to_excel(writer, sheet_name="bod_processed", index=False)
        bod_weighted_contributions.head(1_000_000).to_excel(
            writer, sheet_name="bod_contributions", index=False
        )
        flagged_qa.to_excel(writer, sheet_name="flagged_source_qa", index=False)
        redundancy.to_excel(writer, sheet_name="high_redundancy", index=False)
        correlations.to_excel(writer, sheet_name="correlations", index=False)
        structural.to_excel(writer, sheet_name="structural_panel", index=False)
        wdi_context.head(1_000_000).to_excel(writer, sheet_name="wdi_country_context", index=False)
        manual_exclusions.to_excel(writer, sheet_name="manual_exclusions", index=False)
        source_decisions.to_excel(writer, sheet_name="script03_source_decisions", index=False)
        script03_variable_qa.to_excel(writer, sheet_name="script03_variable_qa", index=False)
    autofit_workbook(QA_WORKBOOK_OUTPUT)


# =============================================================================
# MAIN
# =============================================================================

def write_final_handoff_workbook(
    dynamic_model: pd.DataFrame,
    structural_model: pd.DataFrame,
    feature_weights: pd.DataFrame,
    preprocessing_audit: pd.DataFrame,
    bod_weights: pd.DataFrame,
    bod_summary: pd.DataFrame,
    bod_weighted_contributions: pd.DataFrame,
    bod_dynamic_decomposition: pd.DataFrame,
    bod_structural_decomposition: pd.DataFrame,
    country_dynamic: pd.DataFrame,
    country_structural: pd.DataFrame,
    official_region_set: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(FINAL_HANDOFF_WORKBOOK_OUTPUT, engine="openpyxl") as writer:
        dynamic_model.to_excel(writer, sheet_name="dynamic_model", index=False)
        structural_model.to_excel(writer, sheet_name="structural_model", index=False)
        feature_weights.to_excel(writer, sheet_name="feature_weights", index=False)
        preprocessing_audit.to_excel(writer, sheet_name="preprocessing_audit", index=False)
        bod_weights.to_excel(writer, sheet_name="bod_weights", index=False)
        bod_summary.to_excel(writer, sheet_name="bod_summary", index=False)
        bod_weighted_contributions.to_excel(writer, sheet_name="bod_contributions", index=False)
        bod_dynamic_decomposition.to_excel(writer, sheet_name="bod_dynamic_decomp", index=False)
        bod_structural_decomposition.to_excel(writer, sheet_name="bod_struct_decomp", index=False)
        country_dynamic.to_excel(writer, sheet_name="country_dynamic", index=False)
        country_structural.to_excel(writer, sheet_name="country_structural", index=False)
        official_region_set.to_excel(writer, sheet_name="official_regions", index=False)
    autofit_workbook(FINAL_HANDOFF_WORKBOOK_OUTPUT)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print_header("SCRIPT 04 v4.1 — FINAL ANALYTICAL PREPARATION + COMMON-WEIGHT BoD + ANNUAL DECOMPOSITION")
    print(f"Project root: {ROOT}")
    print(f"Analytical period: {START_YEAR}–{END_YEAR}")
    print(
        f"Canonical scope before feature-specific completeness filters: "
        f"{EXPECTED_COUNTRIES} countries | {EXPECTED_REGIONS} regions | "
        f"{EXPECTED_REGION_YEARS:,} region-year observations"
    )
    print(f"Dynamic eligibility rule: >= {MIN_COMPLETE_REGIONS} regions with all {N_YEARS} years")
    print(f"Structural eligibility rule: >= {MIN_STRUCTURAL_REGIONS} regions with non-missing values")
    print(f"BoD weights estimated now: {ESTIMATE_BOD_WEIGHTS}")
    print(f"BoD minimum contribution when needed: {BOD_MIN_CONTRIBUTION:.0%}")

    # ------------------------------------------------------------------
    # 1. Validate Script 03 handoff
    # ------------------------------------------------------------------
    print_header("1/10 — VALIDATE SCRIPT 03 HANDOFF")
    validate_required_inputs()
    source_decisions = pd.read_csv(SOURCE_DECISIONS_INPUT, low_memory=False, encoding="utf-8-sig")
    script03_variable_qa = pd.read_csv(VARIABLE_QA_INPUT, low_memory=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 2. Canonical panel and structural layer
    # ------------------------------------------------------------------
    print_header("2/10 — BUILD CANONICAL 2015–2019 CORE + STRUCTURAL BASE")
    region_area = build_region_area()
    core, structural = build_core_panel_sql(region_area)
    pop_crosscheck = build_population_crosscheck(core)
    print(
        f"Core panel: {len(core):,} rows | {core['region_id'].nunique():,} regions | "
        f"{core['year'].nunique()} years"
    )
    print(f"Structural base: {len(structural):,} regions")

    # ------------------------------------------------------------------
    # 3. Audit lower-coverage sources; they remain excluded from official model
    # ------------------------------------------------------------------
    print_header("3/10 — AUDIT LOWER-COVERAGE SOURCES")
    cw = load_matched_crosswalk()
    flagged_long, flagged_qa = build_flagged_candidates()
    expanded = build_expanded_panel(core, flagged_long)

    ookla_structural, ookla_qa = build_ookla_structural(cw)
    if not ookla_structural.empty:
        structural = structural.merge(
            ookla_structural,
            on="region_id",
            how="left",
            validate="one_to_one",
        )
    flagged_qa = pd.concat([flagged_qa, pd.DataFrame([ookla_qa])], ignore_index=True, sort=False)
    print(flagged_qa.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Eligibility under the official 512-region rule
    # ------------------------------------------------------------------
    print_header("4/10 — OFFICIAL FEATURE ELIGIBILITY")
    dynamic_diag, dynamic_complete_long = dynamic_diagnostics(expanded, DYNAMIC_FEATURES)
    structural_diag = structural_diagnostics(structural, STRUCTURAL_FEATURES)
    feature_eligibility = pd.concat(
        [dynamic_diag, structural_diag], ignore_index=True, sort=False
    )

    print("\nDYNAMIC FEATURES")
    print(dynamic_diag[[
        "variable", "family", "regions_complete_trajectory",
        "countries_with_complete_region", "eligible_final", "eligibility_status"
    ]].to_string(index=False))
    print("\nSTRUCTURAL FEATURES")
    print(structural_diag[[
        "variable", "family", "regions_non_missing",
        "countries_non_missing", "eligible_final", "eligibility_status"
    ]].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Official common sample + redundancy QA
    # ------------------------------------------------------------------
    print_header("5/10 — OFFICIAL COMMON SAMPLE + REDUNDANCY QA")
    eligible_dynamic = dynamic_diag.loc[
        dynamic_diag["eligible_final"].fillna(False), "variable"
    ].astype(str).tolist()
    correlations, redundancy = build_pairwise_correlations(expanded, eligible_dynamic)
    sample_tradeoff = build_sample_tradeoff(expanded)
    (
        official_region_set,
        official_feature_set,
        official_dynamic_panel,
        official_structural_panel,
        official_intersection_qa,
    ) = build_official_common_region_set(
        expanded, structural, dynamic_diag, structural_diag
    )

    print(sample_tradeoff.to_string(index=False))
    print(
        f"\nOFFICIAL COMMON REGION SET: {len(official_region_set)} regions | "
        f"{official_region_set['code'].nunique()} countries"
    )
    if redundancy.empty:
        print("No eligible dynamic pair has |Spearman| >= 0.90.")
    else:
        print("\nHIGH REDUNDANCY PAIRS")
        print(redundancy.head(30).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. BoD design and raw staging
    # ------------------------------------------------------------------
    print_header("6/10 — BoD DESIGN + THESIS PREPROCESSING PROTOCOL")
    bod_design, _ = build_bod_design(dynamic_diag, structural_diag)
    bod_raw = build_bod_raw_input_long(
        dynamic_complete_long,
        structural,
        dynamic_diag,
        structural_diag,
    )
    print(bod_design.to_string(index=False))
    print(
        "\nOfficial BoD preprocessing: INDESEJAVEL -> subtraction alpha-x; "
        "thesis epsilon protocol when strict positivity is required; "
        "normalization by division by the maximum; no Min-Max and no Z-score."
    )

    # ------------------------------------------------------------------
    # 7. Estimate common-weight BoD families
    # ------------------------------------------------------------------
    print_header("7/10 — ESTIMATE COMMON-WEIGHT BoD FAMILIES")
    (
        bod_dynamic_scores,
        bod_structural_scores,
        bod_weights,
        bod_summary,
        bod_processed_long,
        bod_preprocessing_audit,
    ) = estimate_all_bod_families(
        official_dynamic_panel,
        official_structural_panel,
    )
    print("\nBoD FAMILY SUMMARY")
    print(bod_summary.to_string(index=False))
    print("\nFINAL COMMON WEIGHTS")
    print(bod_weights[[
        "block", "family", "variable", "weight_final",
        "mean_contribution_final", "restriction_applied", "final_model"
    ]].to_string(index=False))

    (
        bod_weighted_contributions,
        bod_dynamic_decomposition,
        bod_structural_decomposition,
    ) = build_bod_weighted_decomposition(
        bod_processed_long,
        bod_weights,
        bod_dynamic_scores,
        bod_structural_scores,
    )
    print(
        "\nBoD decomposition: "
        f"{len(bod_weighted_contributions):,} variable-level contributions | "
        f"{len(bod_dynamic_decomposition):,} dynamic region-year rows | "
        f"{len(bod_structural_decomposition):,} structural rows"
    )
    print(
        "Dynamic BoD rule: same common weight vector for every region and year; "
        "annual family scores are exact sums of weighted normalized components; "
        "no post-BoD re-normalization."
    )

    # ------------------------------------------------------------------
    # 8. Build final dynamic/structural representation for M-Exp-FCMd
    # ------------------------------------------------------------------
    print_header("8/10 — BUILD FINAL M-Exp-FCMd ANALYTICAL REPRESENTATION")
    (
        final_dynamic_model,
        final_structural_model,
        final_feature_weights,
        final_dynamic_long,
        final_structural_long,
        preprocessing_audit,
    ) = build_final_model_representation(
        official_dynamic_panel,
        official_structural_panel,
        bod_dynamic_scores,
        bod_structural_scores,
        bod_preprocessing_audit,
    )

    print(
        f"Final dynamic panel: {len(final_dynamic_model):,} rows | "
        f"{final_dynamic_model['region_id'].nunique()} regions | "
        f"{final_dynamic_model['year'].nunique()} years"
    )
    print(
        f"Final structural panel: {len(final_structural_model):,} rows | "
        f"{final_structural_model['region_id'].nunique()} regions"
    )
    print("\nFEATURE / FAMILY DISTANCE WEIGHTS")
    print(final_feature_weights.to_string(index=False))

    # ------------------------------------------------------------------
    # 9. Final country-reference profiles in the exact model space
    # ------------------------------------------------------------------
    print_header("9/10 — FINAL COUNTRY REFERENCES + WDI CONTEXT")
    country_reference_dynamic, country_reference_structural = build_final_country_references(
        final_dynamic_model,
        final_structural_model,
        final_feature_weights,
    )
    wdi_context = build_wdi_country_context()
    country_reference_raw = build_country_reference_inputs(expanded, dynamic_diag)
    print(f"Dynamic country-reference rows: {len(country_reference_dynamic):,}")
    print(f"Structural country-reference rows: {len(country_reference_structural):,}")
    print(f"WDI country-context rows retained: {len(wdi_context):,}")
    print("WDI remains a separate national-context layer and is not copied to regional features.")

    # ------------------------------------------------------------------
    # 10. Persist final Script 04 outputs
    # ------------------------------------------------------------------
    print_header("10/10 — WRITE FINAL SCRIPT 04 OUTPUTS")
    manual_exclusions = pd.DataFrame(MANUAL_EXCLUSIONS)

    # Existing QA / lineage outputs
    write_csv(core, CORE_PANEL_OUTPUT)
    write_csv(expanded, EXPANDED_PANEL_OUTPUT)
    write_csv(dynamic_complete_long, DYNAMIC_COMPLETE_LONG_OUTPUT)
    write_csv(structural, STRUCTURAL_CONTEXT_OUTPUT)
    write_csv(flagged_long, FLAGGED_LONG_OUTPUT)
    write_csv(flagged_qa, FLAGGED_QA_OUTPUT)
    write_csv(dynamic_diag, DYNAMIC_DIAGNOSTICS_OUTPUT)
    write_csv(structural_diag, STRUCTURAL_DIAGNOSTICS_OUTPUT)
    write_csv(feature_eligibility, FEATURE_ELIGIBILITY_OUTPUT)
    write_csv(sample_tradeoff, SAMPLE_TRADEOFF_OUTPUT)
    write_csv(correlations, CORRELATIONS_OUTPUT)
    write_csv(redundancy, REDUNDANCY_OUTPUT)
    write_csv(bod_design, BOD_FAMILY_DESIGN_OUTPUT)
    write_csv(bod_raw, BOD_INPUT_RAW_OUTPUT)
    write_csv(manual_exclusions, MANUAL_EXCLUSIONS_OUTPUT)
    write_csv(wdi_context, WDI_COUNTRY_CONTEXT_OUTPUT)
    write_csv(country_reference_raw, COUNTRY_REFERENCE_RAW_OUTPUT)
    write_csv(pop_crosscheck, POPULATION_CROSSCHECK_OUTPUT)
    write_csv(official_region_set, OFFICIAL_REGION_SET_OUTPUT)
    write_csv(official_feature_set, OFFICIAL_FEATURE_SET_OUTPUT)
    write_csv(official_dynamic_panel, OFFICIAL_DYNAMIC_PANEL_OUTPUT)
    write_csv(official_structural_panel, OFFICIAL_STRUCTURAL_PANEL_OUTPUT)
    write_csv(official_intersection_qa, OFFICIAL_INTERSECTION_QA_OUTPUT)

    # Final analytical outputs
    write_csv(preprocessing_audit, PREPROCESSING_AUDIT_OUTPUT)
    write_csv(bod_weights, BOD_WEIGHTS_OUTPUT)
    write_csv(bod_summary, BOD_FAMILY_SUMMARY_OUTPUT)
    write_csv(bod_processed_long, BOD_PROCESSED_LONG_OUTPUT)
    write_csv(bod_weighted_contributions, BOD_WEIGHTED_CONTRIBUTIONS_OUTPUT)
    write_csv(bod_dynamic_decomposition, BOD_DYNAMIC_DECOMPOSITION_OUTPUT)
    write_csv(bod_structural_decomposition, BOD_STRUCTURAL_DECOMPOSITION_OUTPUT)
    write_csv(bod_dynamic_scores, BOD_DYNAMIC_SCORES_OUTPUT)
    write_csv(bod_structural_scores, BOD_STRUCTURAL_SCORES_OUTPUT)
    write_csv(final_dynamic_model, FINAL_DYNAMIC_MODEL_OUTPUT)
    write_csv(final_structural_model, FINAL_STRUCTURAL_MODEL_OUTPUT)
    write_csv(final_feature_weights, FINAL_FEATURE_WEIGHTS_OUTPUT)
    write_csv(final_dynamic_long, FINAL_DYNAMIC_LONG_OUTPUT)
    write_csv(final_structural_long, FINAL_STRUCTURAL_LONG_OUTPUT)
    write_csv(country_reference_dynamic, COUNTRY_REFERENCE_DYNAMIC_OUTPUT)
    write_csv(country_reference_structural, COUNTRY_REFERENCE_STRUCTURAL_OUTPUT)

    out_conn = duckdb.connect(str(DUCKDB_OUTPUT))
    try:
        tables = {
            "core_analytical_panel_2015_2019": core,
            "expanded_candidate_panel_2015_2019": expanded,
            "dynamic_complete_trajectories_long": dynamic_complete_long,
            "structural_candidate_panel": structural,
            "flagged_candidates_long_2015_2019": flagged_long,
            "qa_flagged_sources": flagged_qa,
            "qa_dynamic_diagnostics": dynamic_diag,
            "qa_structural_diagnostics": structural_diag,
            "feature_eligibility": feature_eligibility,
            "sample_tradeoff": sample_tradeoff,
            "qa_pairwise_correlations": correlations,
            "qa_high_redundancy_pairs": redundancy,
            "bod_family_design": bod_design,
            "bod_input_raw_long": bod_raw,
            "bod_processed_indicators_long": bod_processed_long,
            "bod_weighted_contributions_long": bod_weighted_contributions,
            "bod_dynamic_decomposition_2015_2019": bod_dynamic_decomposition,
            "bod_structural_decomposition": bod_structural_decomposition,
            "bod_common_weights": bod_weights,
            "bod_family_summary": bod_summary,
            "bod_dynamic_scores": bod_dynamic_scores,
            "bod_structural_scores": bod_structural_scores,
            "preprocessing_audit": preprocessing_audit,
            "manual_exclusions": manual_exclusions,
            "wdi_country_context_2015_2019": wdi_context,
            "country_reference_inputs_raw": country_reference_raw,
            "country_reference_dynamic_final": country_reference_dynamic,
            "country_reference_structural_final": country_reference_structural,
            "qa_population_crosscheck": pop_crosscheck,
            "official_region_set_512": official_region_set,
            "official_feature_set": official_feature_set,
            "official_dynamic_panel_2015_2019": official_dynamic_panel,
            "official_structural_panel": official_structural_panel,
            "final_dynamic_model_panel_2015_2019": final_dynamic_model,
            "final_structural_model_panel": final_structural_model,
            "final_feature_weights": final_feature_weights,
            "mexp_fcmd_dynamic_long": final_dynamic_long,
            "mexp_fcmd_structural_long": final_structural_long,
            "official_intersection_qa": official_intersection_qa,
            "script03_source_decisions": source_decisions,
            "script03_variable_qa": script03_variable_qa,
        }
        for name, frame in tables.items():
            persist_table(out_conn, name, frame)
    finally:
        out_conn.close()

    # Existing QA workbook + focused final handoff workbook.
    write_qa_workbook(
        dynamic_diag,
        structural_diag,
        feature_eligibility,
        sample_tradeoff,
        bod_design,
        bod_raw,
        bod_processed_long,
        bod_weighted_contributions,
        flagged_qa,
        correlations,
        redundancy,
        structural,
        wdi_context,
        manual_exclusions,
        source_decisions,
        script03_variable_qa,
    )
    write_final_handoff_workbook(
        final_dynamic_model,
        final_structural_model,
        final_feature_weights,
        preprocessing_audit,
        bod_weights,
        bod_summary,
        bod_weighted_contributions,
        bod_dynamic_decomposition,
        bod_structural_decomposition,
        country_reference_dynamic,
        country_reference_structural,
        official_region_set,
    )

    print("\nFINAL SCRIPT 04 SUMMARY")
    print("-" * 92)
    print(f"Analytical window:              {START_YEAR}–{END_YEAR}")
    print(f"Official common regions:        {len(official_region_set)}")
    print(f"Countries represented:          {official_region_set['code'].nunique()}")
    print(f"Raw eligible dynamic variables: {int(dynamic_diag['eligible_final'].fillna(False).sum())}")
    print(f"Raw eligible structural vars:   {int(structural_diag['eligible_final'].fillna(False).sum())}")
    print(f"Final dynamic model features:    {len(final_feature_weights[final_feature_weights['block'].eq('DYNAMIC')])}")
    print(f"Final structural model features: {len(final_feature_weights[final_feature_weights['block'].eq('STRUCTURAL')])}")
    print(f"BoD families estimated:          {len(bod_summary)}")
    print(f"BoD weights output:              {BOD_WEIGHTS_OUTPUT}")
    print(f"BoD annual decomposition:        {BOD_DYNAMIC_DECOMPOSITION_OUTPUT}")
    print(f"BoD contribution-level output:   {BOD_WEIGHTED_CONTRIBUTIONS_OUTPUT}")
    print(f"Dynamic Script-05 handoff:       {FINAL_DYNAMIC_MODEL_OUTPUT}")
    print(f"Structural Script-05 handoff:    {FINAL_STRUCTURAL_MODEL_OUTPUT}")
    print(f"Feature-weight metadata:         {FINAL_FEATURE_WEIGHTS_OUTPUT}")
    print(f"Final handoff workbook:          {FINAL_HANDOFF_WORKBOOK_OUTPUT}")
    print(f"DuckDB:                          {DUCKDB_OUTPUT}")

    dynamic_features_final = final_feature_weights.loc[
        final_feature_weights["block"].eq("DYNAMIC"), "feature"
    ].astype(str).tolist()
    removed_features = {"wp_age_population_norm", "population_density_per_km2_norm"}
    if removed_features.intersection(dynamic_features_final):
        raise ValueError(
            "Removed demographic scale/distribution features unexpectedly remain in the final handoff."
        )
    expected_final_dynamic = {
        "bod_demographic_productive_potential",
        "bod_socioeconomic_deprivation",
        "ntl_per_capita_norm",
    }
    if set(dynamic_features_final) != expected_final_dynamic:
        raise ValueError(
            "Unexpected final dynamic feature set. "
            f"Expected {sorted(expected_final_dynamic)}, found {sorted(dynamic_features_final)}."
        )

    print("\nSCRIPT 04 COMPLETE — ONE-CLICK HANDOFF VALIDATED; READY FOR SCRIPT 05 (M-Exp-FCMd)")


if __name__ == "__main__":
    main()
