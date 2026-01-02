"""
Q2 Australia Mapping Pipeline - Step by Step
==============================================
This script builds interactive Australia maps for Q2 analysis (Early Adopters vs Slow-Burn)
incrementally, allowing you to see results at each step.

Steps:
  1. Load and explore SA4 boundaries
  2. Extract and prepare Q2 cinema data
  3. Geocode cinemas to SA4 regions
  4. Aggregate Q2 metrics by SA4
  5. Create SA4 choropleth map
  6. Create state-level choropleth map
  7. Create cinema marker map
  8. Generate final outputs
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
print("=" * 80)
print("Q2 Australia Mapping Pipeline - Configuration")
print("=" * 80)

OUTPUTS_DIR = Path("outputs_locationquestions")
OUTPUTS_DIR.mkdir(exist_ok=True)

print(f"\n✓ Output directory: {OUTPUTS_DIR}")

# ==============================================================================
# STEP 1: Load and Explore SA4 Boundaries
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 1: Load and Explore SA4 Boundaries")
print("=" * 80)

sa4_path = "data/SA4_2021_AUST_GDA94.shp"
sa4_gdf = gpd.read_file(sa4_path)

print(f"\n✓ Loaded {len(sa4_gdf)} SA4 regions")
print(f"✓ Coordinate System: {sa4_gdf.crs}")
print(f"✓ Columns: {sa4_gdf.columns.tolist()}")

# Clean invalid geometries
valid_geoms = sa4_gdf.geometry.is_valid.sum()
if valid_geoms < len(sa4_gdf):
    print(f"⚠ Removing {len(sa4_gdf) - valid_geoms} invalid geometries")
    sa4_gdf = sa4_gdf[sa4_gdf.geometry.is_valid].copy()
    print(f"✓ Now have {len(sa4_gdf)} valid regions")

# Reproject to WGS84 for mapping
sa4_gdf = sa4_gdf.to_crs(epsg=4326)
print(f"✓ Reprojected to EPSG:4326 (WGS84)")

# Save cleaned version
sa4_gdf.to_file("data/SA4_2021_AUST_GDA94_cleaned.geojson", driver="GeoJSON")
print(f"✓ Saved cleaned SA4 to: data/SA4_2021_AUST_GDA94_cleaned.geojson")

# ==============================================================================
# STEP 2: Extract and Prepare Q2 Cinema Data
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 2: Extract and Prepare Q2 Cinema Data")
print("=" * 80)

# Connect to database
db_path = "data/numero_data.sqlite"
conn = sqlite3.connect(db_path)

sales_raw = pd.read_sql("SELECT * FROM sales_raw_data", conn)
film_meta = pd.read_sql("SELECT * FROM film_metadata", conn)
indian_titles = pd.read_sql("SELECT * FROM indian_titles", conn)
conn.close()

print(f"\n✓ Loaded sales_raw: {sales_raw.shape[0]} records")
print(f"✓ Loaded film_meta: {film_meta.shape[0]} films")
print(f"✓ Loaded indian_titles: {indian_titles.shape[0]} Indian films")

# Parse JSON and extract weekly sales
def weekly_gross_from_boxoffice(box_office):
    total = 0.0
    if not isinstance(box_office, dict):
        return 0.0
    for i in range(1, 8):
        di = box_office.get(f"day{i}", {})
        if isinstance(di, dict):
            v = di.get("today", 0) or 0
            total += float(v)
    return total

print("\nParsing raw JSON sales data...")
rows_out = []

for idx, r in sales_raw.iterrows():
    if idx % 5 == 0:
        print(f"  Processing record {idx+1}/{len(sales_raw)}...")
    
    film_id = r["numero_film_id"]
    j = json.loads(r["raw_json"])

    for week_start, payload in j.items():
        week_rows = payload.get("rows", []) if isinstance(payload, dict) else []
        for rr in week_rows:
            rows_out.append({
                "numero_film_id": film_id,
                "week_start": pd.to_datetime(week_start),
                "state": rr.get("state"),
                "city": rr.get("city"),
                "theatre_name": rr.get("theatre"),
                "region": rr.get("region"),
                "circuit": rr.get("circuit"),
                "weekly_gross": weekly_gross_from_boxoffice(rr.get("boxOffice", {})),
            })

sales_weekly = pd.DataFrame(rows_out)
sales_weekly["weekly_gross"] = sales_weekly["weekly_gross"].fillna(0).astype(float)
sales_weekly = sales_weekly.dropna(subset=["state", "city", "theatre_name", "week_start"])

print(f"✓ Parsed {len(sales_weekly)} weekly sales records")

# Filter to Indian titles
sales_wt = sales_weekly.merge(
    film_meta[["numero_film_id", "title"]],
    on="numero_film_id",
    how="left"
)

sales_wt["title"] = sales_wt["title"].astype(str).str.strip()
indian_titles["title"] = indian_titles["title"].astype(str).str.strip()

indian_only = sales_wt.merge(
    indian_titles[["title"]].drop_duplicates(),
    on="title",
    how="inner"
)

indian_only = indian_only[indian_only["weekly_gross"] > 0].copy()

print(f"✓ Filtered to Indian titles: {len(indian_only)} records")
print(f"✓ Unique cities: {indian_only['city'].nunique()}")
print(f"✓ Unique cinemas: {indian_only['theatre_name'].nunique()}")

# Add relative weeks
indian_only["week_start"] = pd.to_datetime(indian_only["week_start"])

def add_relative_week(df, group_cols):
    df = df.copy()
    first_week = (
        df.groupby(group_cols, as_index=False)["week_start"]
          .min()
          .rename(columns={"week_start": "first_week_start"})
    )
    out = df.merge(first_week, on=group_cols, how="left")
    out["rel_week"] = ((out["week_start"] - out["first_week_start"]).dt.days // 7) + 1
    return out

indian_city_rw = add_relative_week(
    indian_only,
    ["numero_film_id", "state", "city"]
)

indian_cinema_rw = add_relative_week(
    indian_only,
    ["numero_film_id", "state", "city", "theatre_name"]
)

# Build timing analysis
def build_film_location_timing(df, group_cols):
    df = df.copy()
    df["early_gross"] = np.where(df["rel_week"] <= 2, df["weekly_gross"], 0.0)
    df["late_gross"]  = np.where(df["rel_week"] >= 3, df["weekly_gross"], 0.0)

    out = (
        df.groupby(group_cols, as_index=False)
          .agg(
              total_gross=("weekly_gross", "sum"),
              early_gross=("early_gross", "sum"),
              late_gross=("late_gross", "sum"),
              weeks_active=("rel_week", "max")
          )
    )

    out["early_share"] = np.where(out["total_gross"] > 0, out["early_gross"] / out["total_gross"], np.nan)
    return out

film_city_timing = build_film_location_timing(
    indian_city_rw,
    ["numero_film_id", "title", "state", "city"]
)

film_cinema_timing = build_film_location_timing(
    indian_cinema_rw,
    ["numero_film_id", "title", "state", "city", "theatre_name"]
)

# Build place summaries
def safe_weighted_avg(values, weights):
    values = pd.Series(values)
    weights = pd.Series(weights).fillna(0)
    wsum = weights.sum()
    if wsum <= 0:
        return np.nan
    return np.average(values, weights=weights)

def build_place_summary(film_location_timing, place_cols):
    df = film_location_timing[film_location_timing["total_gross"] > 0].copy()
    out = (
        df.groupby(place_cols, as_index=False)
          .apply(lambda g: pd.Series({
              "total_gross": g["total_gross"].sum(),
              "n_films": g["numero_film_id"].nunique(),
              "weighted_early_share": safe_weighted_avg(g["early_share"], g["total_gross"])
          }))
          .reset_index(drop=True)
    )
    return out

city_summary = build_place_summary(film_city_timing, ["state", "city"])
cinema_summary = build_place_summary(film_cinema_timing, ["state", "city", "theatre_name"])

# Classify cities and cinemas
TOP_N = 36
city_plot = (
    city_summary[city_summary["total_gross"] > 0]
    .sort_values("total_gross", ascending=False)
    .head(TOP_N)
    .copy()
)

q25 = city_plot["weighted_early_share"].quantile(0.25)
q75 = city_plot["weighted_early_share"].quantile(0.75)

def timing_class(x):
    if x >= q75:
        return "EARLY_ADOPTER"
    elif x <= q25:
        return "SLOW_BURN"
    else:
        return "BALANCED"

city_plot["timing_class"] = city_plot["weighted_early_share"].apply(timing_class)

print(f"\nCity Classification (Top {TOP_N}):") 
print(city_plot["timing_class"].value_counts())

TOP_N_CINEMA = 60
cinema_plot = (
    cinema_summary[cinema_summary["total_gross"] > 0]
    .sort_values("total_gross", ascending=False)
    .head(TOP_N_CINEMA)
    .copy()
)

q25_c = cinema_plot["weighted_early_share"].quantile(0.25)
q75_c = cinema_plot["weighted_early_share"].quantile(0.75)

def timing_class_cinema(x):
    if x >= q75_c:
        return "EARLY_ADOPTER"
    elif x <= q25_c:
        return "SLOW_BURN"
    else:
        return "BALANCED"

cinema_plot["timing_class"] = cinema_plot["weighted_early_share"].apply(timing_class_cinema)

print(f"\nCinema Classification (Top {TOP_N_CINEMA}):")
print(cinema_plot["timing_class"].value_counts())

print("\n✓ STEP 2 COMPLETE - Q2 cinema data ready")

# Save intermediate data
city_plot.to_csv(OUTPUTS_DIR / "q2_cities_summary.csv", index=False)
cinema_plot.to_csv(OUTPUTS_DIR / "q2_cinemas_summary.csv", index=False)
print(f"✓ Saved city and cinema summaries to outputs")

# ==============================================================================
# NEXT STEPS
# ==============================================================================
print("\n" + "=" * 80)
print("PROGRESS SUMMARY")
print("=" * 80)
print(f"""
✓ STEP 1: Loaded SA4 boundaries (89 valid regions)
✓ STEP 2: Extracted Q2 cinema data
  - {len(city_plot)} cities analyzed
  - {len(cinema_plot)} cinemas analyzed
  - Early Adopters: {(city_plot['timing_class'] == 'EARLY_ADOPTER').sum()} cities

Ready to continue with:
  Step 3: Geocode cinemas to SA4 regions
  Step 4: Aggregate Q2 metrics by SA4
  Step 5: Create SA4 choropleth map
  Step 6: Create state-level choropleth  
  Step 7: Create cinema marker map
  Step 8: Generate final outputs
  
Generated files:
  - {OUTPUTS_DIR}/q2_cities_summary.csv
  - {OUTPUTS_DIR}/q2_cinemas_summary.csv
  - data/SA4_2021_AUST_GDA94_cleaned.geojson
""")

print("✓ Pipeline ready. Continue with Step 3? (run next script)")
