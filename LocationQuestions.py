# LocationQuestions.py
# This script performs location-based analysis for film box office performance
# It imports all preprocessed data and variables from DataExplorationMain

import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from DataExplorationMain import film_meta, indian_titles, sales, sales_indian
from project_paths import DATA_DIR, OUTPUTS_LOCATION_QUESTIONS, ensure_dir, find_database_path

OUTPUT_DIR = ensure_dir(OUTPUTS_LOCATION_QUESTIONS)

# ============================================================================
# QUESTION 1: Steady vs Inconsistent Box Office
# ============================================================================
# Which key cities and cinemas have steady weekly box office, and which are 
# more up-and-down (inconsistent)? How should Film Viet rank these locations 
# as safer vs higher-risk targets?

print("\n" + "="*80)
print("QUESTION 1: Steady vs Inconsistent Box Office")
print("="*80)

# Check the sales data shape from DataExplorationMain
print("\nData shape and columns from DataExplorationMain:")
print(sales.shape, sales.columns.tolist())

# Add week_start column
sales = sales.copy()
sales['week_start'] = (
    sales['actual_sales_date']
      .dt.to_period('W')
      .apply(lambda r: r.start_time)
      .dt.date
)

print("\nWeek start sample:")
print(sales[['actual_sales_date', 'week_start']].head(10))

# Aggregate by cinema and week
weekly_cinema = (
    sales
      .groupby(['state', 'city', 'theatre_name', 'week_start'], as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'weekly_gross'})
)

print("\nWeekly cinema sample:")
print(weekly_cinema.head())

# Aggregate by city and week
weekly_city = (
    sales
      .groupby(['state', 'city', 'week_start'], as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'weekly_gross'})
)

print("\nWeekly city sample:")
print(weekly_city.head())

print(f"\nWeekly aggregation shapes: cinema={weekly_cinema.shape}, city={weekly_city.shape}")

# Calculate statistics: mean, std, and CV for cinemas
cinema_stats = (
    weekly_cinema
      .groupby(['state', 'city', 'theatre_name'], as_index=False)
      .agg(
          mean_weekly_gross=('weekly_gross', 'mean'),
          std_weekly_gross=('weekly_gross', 'std'),
          weeks_count=('weekly_gross', 'count')
      )
)

cinema_stats['cv'] = cinema_stats['std_weekly_gross'] / cinema_stats['mean_weekly_gross']

print("\nCinema stats sample:")
print(cinema_stats.head())

# Calculate statistics for cities
city_stats = (
    weekly_city
      .groupby(['state', 'city'], as_index=False)
      .agg(
          mean_weekly_gross=('weekly_gross', 'mean'),
          std_weekly_gross=('weekly_gross', 'std'),
          weeks_count=('weekly_gross', 'count')
      )
)

city_stats['cv'] = city_stats['std_weekly_gross'] / city_stats['mean_weekly_gross']

print("\nCity stats sample:")
print(city_stats.head())

# Display statistics summary
print("\nCinema CV statistics:")
print(cinema_stats[['mean_weekly_gross', 'cv', 'weeks_count']].describe())

print("\nCity CV statistics:")
print(city_stats[['mean_weekly_gross', 'cv', 'weeks_count']].describe())

# Filter by minimum weeks requirement
MIN_WEEKS = 8

cinema_stats_f = cinema_stats.copy()
cinema_stats_f = cinema_stats_f[
    (cinema_stats_f['weeks_count'] >= MIN_WEEKS) &
    (cinema_stats_f['mean_weekly_gross'] > 0) &
    (cinema_stats_f['cv'].notna())
].copy()

city_stats_f = city_stats.copy()
city_stats_f = city_stats_f[
    (city_stats_f['weeks_count'] >= MIN_WEEKS) &
    (city_stats_f['mean_weekly_gross'] > 0) &
    (city_stats_f['cv'].notna())
].copy()

print(f"\nAfter MIN_WEEKS={MIN_WEEKS} filter:")
print(f"Cinemas: {cinema_stats_f.shape[0]} (from {cinema_stats.shape[0]})")
print(f"Cities: {city_stats_f.shape[0]} (from {city_stats.shape[0]})")

# Classify by risk category based on CV
def classify_cv(cv):
    if cv < 0.75:
        return 'Safer (Stable)'
    elif cv < 1.10:
        return 'Moderate'
    elif cv < 1.50:
        return 'Higher-Risk (Volatile)'
    else:
        return 'Highly Volatile'

cinema_stats_f['risk_category'] = cinema_stats_f['cv'].apply(classify_cv)
city_stats_f['risk_category'] = city_stats_f['cv'].apply(classify_cv)

print("\nRisk category distribution:")
print("Cinemas:")
print(cinema_stats_f['risk_category'].value_counts())
print("\nCities:")
print(city_stats_f['risk_category'].value_counts())

# Calculate risk-adjusted score
cinema_stats_f['risk_adjusted_score'] = cinema_stats_f['mean_weekly_gross'] / (1 + cinema_stats_f['cv'])
city_stats_f['risk_adjusted_score'] = city_stats_f['mean_weekly_gross'] / (1 + city_stats_f['cv'])

# Top key venues
key_cinemas = cinema_stats_f.sort_values('mean_weekly_gross', ascending=False).head(20).copy()
print("\nTop 10 key cinemas by revenue:")
print(key_cinemas[['state','city','theatre_name','mean_weekly_gross','cv','weeks_count','risk_category']].head(10))

key_cities = city_stats_f.sort_values('mean_weekly_gross', ascending=False).head(15).copy()
print("\nTop 10 key cities by revenue:")
print(key_cities[['state','city','mean_weekly_gross','cv','weeks_count','risk_category']].head(10))

# Safer venues
safer_key_cinemas = (
    key_cinemas[key_cinemas['risk_category'] == 'Safer (Stable)']
      .sort_values('mean_weekly_gross', ascending=False)
      .copy()
)

safer_key_cities = (
    key_cities[key_cities['risk_category'] == 'Safer (Stable)']
      .sort_values('mean_weekly_gross', ascending=False)
      .copy()
)

print("\nTop safer cinemas:")
print(safer_key_cinemas[['state','city','theatre_name','mean_weekly_gross','cv','weeks_count','risk_category']])

print("\nTop safer cities:")
print(safer_key_cities[['state','city','mean_weekly_gross','cv','weeks_count','risk_category']])

# Higher-risk venues
risk_key_cinemas2 = (
    cinema_stats_f[cinema_stats_f['risk_category'].isin(['Higher-Risk (Volatile)', 'Highly Volatile'])]
      .sort_values('mean_weekly_gross', ascending=False)
      .head(15)
)

risk_key_cities2 = (
    city_stats_f[city_stats_f['risk_category'].isin(['Higher-Risk (Volatile)', 'Highly Volatile'])]
      .sort_values('mean_weekly_gross', ascending=False)
      .head(10)
)

print("\nTop higher-risk cinemas:")
print(risk_key_cinemas2[['state','city','theatre_name','mean_weekly_gross','cv','weeks_count','risk_category']])

print("\nTop higher-risk cities:")
print(risk_key_cities2[['state','city','mean_weekly_gross','cv','weeks_count','risk_category']])

# Safer venues ranked by risk-adjusted score
safer_cinemas_ranked = (
    cinema_stats_f[cinema_stats_f['risk_category'] == 'Safer (Stable)']
      .sort_values('risk_adjusted_score', ascending=False)
      .head(10)
)

safer_cities_ranked = (
    city_stats_f[city_stats_f['risk_category'] == 'Safer (Stable)']
      .sort_values('risk_adjusted_score', ascending=False)
      .head(10)
)

print("\nTop 10 safer cinemas (by risk-adjusted score):")
print(safer_cinemas_ranked[['state','city','theatre_name','mean_weekly_gross','cv','weeks_count','risk_adjusted_score']])

print("\nTop 10 safer cities (by risk-adjusted score):")
print(safer_cities_ranked[['state','city','mean_weekly_gross','cv','weeks_count','risk_adjusted_score']])

# Create visualization: cinemas scatter plot
plot_df = pd.concat([
    safer_key_cinemas.assign(group='Safer key (revenue-first)'),
    risk_key_cinemas2.assign(group='Higher-risk key (opportunity-first)')
], ignore_index=True)

plt.figure(figsize=(12, 7))

for g, sub in plot_df.groupby('group'):
    plt.scatter(sub['mean_weekly_gross'], sub['cv'], label=g, s=120, alpha=0.85)

plt.xscale('log')

labels = pd.concat([
    safer_key_cinemas.sort_values('mean_weekly_gross', ascending=False).head(3),
    risk_key_cinemas2.sort_values('cv', ascending=False).head(3),
], ignore_index=True)

for _, r in labels.iterrows():
    plt.annotate(
        r['theatre_name'],
        (r['mean_weekly_gross'], r['cv']),
        textcoords="offset points",
        xytext=(6, 6),
        ha='left',
        fontsize=9
    )

plt.xlabel("Average Weekly Gross ($, log scale)")
plt.ylabel("CV (Volatility)")
plt.title("Key Cinemas: Stability vs Volatility (Log scale improves readability)")
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_cinemas_scatter.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: cities scatter plot
plot_city = pd.concat([
    safer_key_cities.assign(group='Safer key (revenue-first)'),
    risk_key_cities2.assign(group='Higher-risk key (opportunity-first)')
], ignore_index=True)

plt.figure(figsize=(12, 7))

for g, sub in plot_city.groupby('group'):
    plt.scatter(sub['mean_weekly_gross'], sub['cv'], label=g, s=140, alpha=0.85)

plt.xscale('log')

labels_city = pd.concat([
    safer_key_cities.sort_values('mean_weekly_gross', ascending=False).head(3),
    risk_key_cities2.sort_values('cv', ascending=False).head(3),
], ignore_index=True)

for _, r in labels_city.iterrows():
    plt.annotate(r['city'], (r['mean_weekly_gross'], r['cv']),
                 textcoords="offset points", xytext=(6, 6), fontsize=9)

plt.xlabel("Average Weekly Gross ($, log scale)")
plt.ylabel("CV (Volatility)")
plt.title("Key Cities: Stability vs Volatility (Filtered, MIN_WEEKS applied)")
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_cities_scatter.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: safer cinemas bar chart
safer_cinemas_ranked_plot = (
    cinema_stats_f[cinema_stats_f['risk_category'] == 'Safer (Stable)']
      .sort_values('risk_adjusted_score', ascending=False)
      .head(10)
      .copy()
)

safer_cinemas_ranked_plot['label'] = safer_cinemas_ranked_plot['theatre_name'] + " (" + safer_cinemas_ranked_plot['city'] + ")"
safer_cinemas_ranked_plot = safer_cinemas_ranked_plot.sort_values('risk_adjusted_score', ascending=True)

plt.figure(figsize=(12, 6))
plt.barh(safer_cinemas_ranked_plot['label'], safer_cinemas_ranked_plot['risk_adjusted_score'])
plt.xlabel("Risk-adjusted score (mean / (1 + CV))")
plt.title("Top 10 Safer Cinemas (Ranked)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_safer_cinemas_barh.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: higher-risk cinemas bar chart
risk_cinemas_ranked_plot = (
    cinema_stats_f[cinema_stats_f['risk_category'].isin(['Higher-Risk (Volatile)', 'Highly Volatile'])]
      .sort_values(['mean_weekly_gross','cv'], ascending=[False, False])
      .head(10)
      .copy()
)

risk_cinemas_ranked_plot['label'] = risk_cinemas_ranked_plot['theatre_name'] + " (" + risk_cinemas_ranked_plot['city'] + ")"
risk_cinemas_ranked_plot = risk_cinemas_ranked_plot.sort_values('mean_weekly_gross', ascending=True)

plt.figure(figsize=(12, 6))
plt.barh(risk_cinemas_ranked_plot['label'], risk_cinemas_ranked_plot['mean_weekly_gross'])
plt.xlabel("Average weekly gross ($)")
plt.title("Top 10 Higher-risk Cinemas (Ranked by Upside)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_higher_risk_cinemas_barh.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: safer cities bar chart
safer_cities_ranked_plot = (
    city_stats_f[city_stats_f['risk_category'] == 'Safer (Stable)']
      .sort_values('risk_adjusted_score', ascending=False)
      .head(10)
      .copy()
)

safer_cities_ranked_plot['label'] = safer_cities_ranked_plot['city'] + " (" + safer_cities_ranked_plot['state'] + ")"
safer_cities_ranked_plot = safer_cities_ranked_plot.sort_values('risk_adjusted_score', ascending=True)

plt.figure(figsize=(12, 6))
plt.barh(safer_cities_ranked_plot['label'], safer_cities_ranked_plot['risk_adjusted_score'])
plt.xlabel("Risk-adjusted score (mean / (1 + CV))")
plt.title("Top 10 Safer Cities (Ranked)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_safer_cities_barh.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: higher-risk cities bar chart
risk_cities_ranked_plot = (
    city_stats_f[city_stats_f['risk_category'].isin(['Higher-Risk (Volatile)', 'Highly Volatile'])]
      .sort_values(['mean_weekly_gross','cv'], ascending=[False, False])
      .head(10)
      .copy()
)

risk_cities_ranked_plot['label'] = risk_cities_ranked_plot['city'] + " (" + risk_cities_ranked_plot['state'] + ")"
risk_cities_ranked_plot = risk_cities_ranked_plot.sort_values('mean_weekly_gross', ascending=True)

plt.figure(figsize=(12, 6))
plt.barh(risk_cities_ranked_plot['label'], risk_cities_ranked_plot['mean_weekly_gross'])
plt.xlabel("Average weekly gross ($)")
plt.title("Top 10 Higher-risk Cities (Ranked by Upside)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_higher_risk_cities_barh.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: volatility heatmap for top cinemas
heat_cinemas = pd.concat([safer_cinemas_ranked_plot, risk_cinemas_ranked_plot], ignore_index=True).copy()

heat_cinemas['label'] = heat_cinemas['theatre_name'] + " (" + heat_cinemas['city'] + ")"
heat_cinemas_labels = heat_cinemas['label'].tolist()

weekly_cinema2 = weekly_cinema.copy()
weekly_cinema2['label'] = weekly_cinema2['theatre_name'] + " (" + weekly_cinema2['city'] + ")"

heat_source = weekly_cinema2[weekly_cinema2['label'].isin(heat_cinemas_labels)].copy()

heat_pivot = (
    heat_source
      .pivot_table(index='label', columns='week_start', values='weekly_gross', aggfunc='sum')
      .reindex(index=heat_cinemas_labels)
)

# Create z-score matrix for heatmap
X = np.log1p(heat_pivot)
row_mean = X.mean(axis=1)
row_std = X.std(axis=1).replace(0, np.nan)
Z = (X.sub(row_mean, axis=0)).div(row_std, axis=0)

plt.figure(figsize=(16, 8))

Z_mat = Z.to_numpy()
mask = np.isnan(Z_mat)
Z_plot = np.where(mask, 0, Z_mat)

im = plt.imshow(Z_plot, aspect='auto', interpolation='nearest')
alpha = np.where(mask, 0.0, 1.0)
im.set_alpha(alpha)

plt.colorbar(im, label="Relative weekly performance (z-score of log1p gross)")

plt.yticks(range(len(Z.index)), Z.index, fontsize=9)

weeks = list(Z.columns)
step = max(1, len(weeks)//12)
xticks = list(range(0, len(weeks), step))
plt.xticks(xticks, [str(weeks[i]) for i in xticks], rotation=45, ha='right', fontsize=9)

plt.title("Volatility Fingerprint: Weekly Gross Patterns (Top Safer vs Higher-risk Cinemas)")
plt.xlabel("Week start")
plt.ylabel("Cinema")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_volatility_heatmap.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nQ1 visualizations saved to outputs/:")
print("  - q1_cinemas_scatter.png")
print("  - q1_cities_scatter.png")
print("  - q1_safer_cinemas_barh.png")
print("  - q1_higher_risk_cinemas_barh.png")
print("  - q1_safer_cities_barh.png")
print("  - q1_higher_risk_cities_barh.png")
print("  - q1_volatility_heatmap.png")

# ============================================================================
# QUESTION 2: Early Adopter vs Slow Burn
# ============================================================================
# Which places watch Indian movies right away in the first 1-2 weeks, and 
# which places take longer to build up?

print("\n" + "="*80)
print("QUESTION 2: Early Adopter vs Slow Burn")
print("="*80)

# Find & connect to database
db_path = find_database_path(DATA_DIR / "numero_data.sqlite")
conn = sqlite3.connect(db_path)
print(f"\nDB: {db_path}")

# Load raw sales table
sales_raw = pd.read_sql("SELECT * FROM sales_raw_data", conn)
film_meta = pd.read_sql("SELECT * FROM film_metadata", conn)
indian_titles = pd.read_sql("SELECT * FROM indian_titles", conn)

print(f"sales_raw: {sales_raw.shape}")
print(f"film_meta: {film_meta.shape}")
print(f"indian_titles: {indian_titles.shape}")

# Helper function to extract weekly gross from boxOffice JSON
def weekly_gross_from_boxoffice(box_office: dict) -> float:
    total = 0.0
    if not isinstance(box_office, dict):
        return 0.0
    for i in range(1, 8):
        di = box_office.get(f"day{i}", {})
        if isinstance(di, dict):
            v = di.get("today", 0) or 0
            total += float(v)
    return total

# Parse raw JSON and extract weekly sales by location
rows_out = []

for _, r in sales_raw.iterrows():
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

# Basic cleanup
sales_weekly["weekly_gross"] = sales_weekly["weekly_gross"].fillna(0).astype(float)
sales_weekly = sales_weekly.dropna(subset=["state", "city", "theatre_name", "week_start"])

print(f"\nsales_weekly shape: {sales_weekly.shape}")
print(f"weekly_gross <= 0 rows: {(sales_weekly['weekly_gross'] <= 0).sum()}")

# Filter to Indian titles only
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

print(f"indian_only: {indian_only.shape}")
print(f"unique films: {indian_only['numero_film_id'].nunique()}")

# Add relative week from first week
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

indian_cinema_rw = add_relative_week(
    indian_only,
    ["numero_film_id", "state", "city", "theatre_name"]
)

indian_city_rw = add_relative_week(
    indian_only,
    ["numero_film_id", "state", "city"]
)

indian_state_rw = add_relative_week(
    indian_only,
    ["numero_film_id", "state"]
)

print(f"cinema_rw: {indian_cinema_rw.shape}")
print(f"city_rw: {indian_city_rw.shape}")
print(f"state_rw: {indian_state_rw.shape}")

# Build film-location timing analysis
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

film_cinema_timing = build_film_location_timing(
    indian_cinema_rw,
    ["numero_film_id", "title", "state", "city", "theatre_name"]
)

film_city_timing = build_film_location_timing(
    indian_city_rw,
    ["numero_film_id", "title", "state", "city"]
)

film_state_timing = build_film_location_timing(
    indian_state_rw,
    ["numero_film_id", "title", "state"]
)

print(f"film_cinema_timing: {film_cinema_timing.shape}")
print(f"film_city_timing: {film_city_timing.shape}")
print(f"film_state_timing: {film_state_timing.shape}")

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

state_summary = build_place_summary(film_state_timing, ["state"])
city_summary = build_place_summary(film_city_timing, ["state", "city"])
cinema_summary = build_place_summary(film_cinema_timing, ["state", "city", "theatre_name"])

print(f"state_summary: {state_summary.shape}")
print(f"city_summary: {city_summary.shape}")
print(f"cinema_summary: {cinema_summary.shape}")

# Classify cities by timing
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

print(f"\nCity timing classification (TOP_N={TOP_N}):")
print(f"q25={q25:.3f}, q75={q75:.3f}")
print(city_plot["timing_class"].value_counts())

# Classify cinemas by timing
TOP_N_CINEMA = 60

cinema_plot = (
    cinema_summary[cinema_summary["total_gross"] > 0]
    .sort_values("total_gross", ascending=False)
    .head(TOP_N_CINEMA)
    .copy()
)

q25_c = cinema_plot["weighted_early_share"].quantile(0.25)
q75_c = cinema_plot["weighted_early_share"].quantile(0.75)

def timing_class_cinema(x, q25, q75):
    if x >= q75:
        return "EARLY_ADOPTER"
    elif x <= q25:
        return "SLOW_BURN"
    else:
        return "BALANCED"

cinema_plot["timing_class"] = cinema_plot["weighted_early_share"].apply(lambda x: timing_class_cinema(x, q25_c, q75_c))

print(f"\nCinema timing classification (TOP_N={TOP_N_CINEMA}):")
print(f"q25_c={q25_c:.3f}, q75_c={q75_c:.3f}")
print(cinema_plot["timing_class"].value_counts())

# Calculate ramp-up speeds
city_rw = indian_city_rw.copy()

weekly_city_rel = (
    city_rw
    .groupby(['numero_film_id', 'title', 'state', 'city', 'rel_week'], as_index=False)['weekly_gross']
    .sum()
    .sort_values(['numero_film_id', 'state', 'city', 'rel_week'])
)

city_week = (
    weekly_city_rel
    .groupby(["state", "city", "rel_week"], as_index=False)["weekly_gross"]
    .sum()
)

city_totals = (
    city_week
    .groupby(["state", "city"], as_index=False)["weekly_gross"]
    .sum()
    .rename(columns={"weekly_gross": "city_total_gross"})
)

city_week = city_week.merge(city_totals, on=["state", "city"], how="left")
city_week["week_share"] = city_week["weekly_gross"] / city_week["city_total_gross"]

city_week = city_week.sort_values(["state", "city", "rel_week"])
city_week["cum_share"] = city_week.groupby(["state", "city"])["week_share"].cumsum()

def first_week_reach(df, threshold):
    hit = df.loc[df["cum_share"] >= threshold, "rel_week"]
    return int(hit.min()) if len(hit) else np.nan

city_speed = (
    city_week
    .sort_values(["state", "city", "rel_week"])
    .groupby(["state", "city"], as_index=False)
    .apply(lambda g: pd.Series({
        "weeks_to_80": first_week_reach(g, 0.80),
        "weeks_to_95": first_week_reach(g, 0.95),
        "final_week": int(g["rel_week"].max()),
    }))
    .reset_index(drop=True)
)

print(f"\nCity speed statistics:")
print(city_speed[["weeks_to_80","weeks_to_95","final_week"]].describe())

# Cinema ramp-up speeds
cinema_week = (
    indian_cinema_rw
    .groupby(["state", "city", "theatre_name", "rel_week"], as_index=False)["weekly_gross"]
    .sum()
    .sort_values(["state", "city", "theatre_name", "rel_week"])
)

cinema_week["cinema_total_gross"] = cinema_week.groupby(
    ["state", "city", "theatre_name"]
)["weekly_gross"].transform("sum")

cinema_week["week_share"] = cinema_week["weekly_gross"] / cinema_week["cinema_total_gross"]

cinema_week["cum_share"] = cinema_week.groupby(
    ["state", "city", "theatre_name"]
)["week_share"].cumsum()

cinema_speed = (
    cinema_week.sort_values(["state", "city", "theatre_name", "rel_week"])
    .groupby(["state", "city", "theatre_name"], as_index=False)
    .apply(lambda g: pd.Series({
        "weeks_to_80": first_week_reach(g, 0.80),
        "weeks_to_95": first_week_reach(g, 0.95),
        "final_week": int(g["rel_week"].max()),
        "total_gross": float(g["weekly_gross"].sum())
    }))
    .reset_index(drop=True)
)

print(f"\nCinema speed statistics:")
print(cinema_speed[["weeks_to_80","weeks_to_95","final_week","total_gross"]].describe())

# Merge timing class with speed data
city_speed_plot = city_speed.merge(
    city_plot[["state", "city", "timing_class"]],
    on=["state", "city"],
    how="left"
)

cinema_speed_plot = cinema_speed.merge(
    cinema_plot[["state", "city", "theatre_name", "timing_class"]],
    on=["state", "city", "theatre_name"],
    how="left"
)

print(f"\nCity speed plot shape: {city_speed_plot.shape}")
print(f"Cinema speed plot shape: {cinema_speed_plot.shape}")

# Print market value breakdown by timing type
print("\n" + "="*80)
print("Q2 MARKET VALUE ANALYSIS")
print("="*80)

timing_breakdown = city_plot.groupby("timing_class")["total_gross"].agg(['sum', 'count', 'mean', 'median', 'min', 'max']).reset_index()
total_market = city_plot['total_gross'].sum()

print(f"\nTotal Market Value (Top 36 Cities): ${total_market:,.0f}")
print(f"Average Revenue per City: ${total_market/len(city_plot):,.0f}\n")

print(f'{"Timing Type":<20} {"Total Revenue":>18} {"# Cities":>10} {"Avg Revenue":>18} {"Median":>18}')
print('-'*88)

for _, row in timing_breakdown.iterrows():
    timing_type = row['timing_class']
    total = row['sum']
    count = int(row['count'])
    avg = row['mean']
    median = row['median']
    pct = 100 * total / total_market
    print(f'{timing_type:<20} ${total:>17,.0f} {count:>10d} ${avg:>17,.0f} ${median:>17,.0f}')

print('-'*88)
print(f'\nMarket Share by Timing Type:')
for _, row in timing_breakdown.iterrows():
    timing_type = row['timing_class']
    total = row['sum']
    pct = 100 * total / total_market
    count = int(row['count'])
    print(f'  {timing_type:<18}: ${total:>17,.0f} ({pct:>5.1f}%)  [{count} cities]')

print(f'\n  {"TOTAL":<18}: ${total_market:>17,.0f} (100.0%)  [{len(city_plot)} cities]')

# ============== Q2 VISUALIZATIONS ==============

# Create visualization: City timing map (bubble chart)
size = 300 + 2200 * (city_plot["total_gross"] / city_plot["total_gross"].max())
gross_line = city_plot["total_gross"].median()

color_map = {
    "EARLY_ADOPTER": "tab:blue",
    "BALANCED": "tab:orange",
    "SLOW_BURN": "tab:green"
}

plt.figure(figsize=(14, 7))

for c, g in city_plot.groupby("timing_class"):
    plt.scatter(
        g["weighted_early_share"],
        g["total_gross"],
        s=size.loc[g.index],
        alpha=0.75,
        label=f"{c} (n={len(g)})",
        c=color_map[c],
        edgecolor="white",
        linewidth=0.7
    )

plt.axvline(q25, linestyle="--")
plt.axvline(q75, linestyle="--")
plt.axhline(gross_line, linestyle=":")

plt.yscale("log")
plt.title("Timing Map (Cities): Early-adopter vs Slow-burn demand for Indian titles (classification thresholds)")
plt.xlabel("Early Share (Week 1-2 / Total Gross)")
plt.ylabel("Total Gross (log scale)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.25)

for _, r in city_plot.iterrows():
    plt.text(
        r["weighted_early_share"],
        r["total_gross"],
        f"{r['state']} | {r['city']}",
        fontsize=9,
        ha="left",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q2_cities_timing_map.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: Cinema timing map (bubble chart)
size = 300 + 2200 * (cinema_plot["total_gross"] / cinema_plot["total_gross"].max())
gross_line = cinema_plot["total_gross"].median()

plt.figure(figsize=(14, 7))

for c, g in cinema_plot.groupby("timing_class"):
    plt.scatter(
        g["weighted_early_share"],
        g["total_gross"],
        s=size.loc[g.index],
        alpha=0.75,
        label=f"{c} (n={len(g)})",
        c=color_map[c],
        edgecolor="white",
        linewidth=0.7
    )

plt.axvline(q25_c, linestyle="--")
plt.axvline(q75_c, linestyle="--")
plt.axhline(gross_line, linestyle=":")

plt.yscale("log")
plt.title("Timing Map (Cinemas): Release first vs second wave (classification thresholds)")
plt.xlabel("Early Share (Week 1-2 / Total Gross)")
plt.ylabel("Total Gross (log scale)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.25)

# Label top 3 cinemas per timing class
top_per_class = 3
for timing_type in ['EARLY_ADOPTER', 'BALANCED', 'SLOW_BURN']:
    top_labels = (
        cinema_plot[cinema_plot['timing_class'] == timing_type]
        .sort_values("total_gross", ascending=False)
        .head(top_per_class)
    )
    for _, r in top_labels.iterrows():
        plt.text(
            r["weighted_early_share"],
            r["total_gross"],
            f"{r['state']} | {r['theatre_name']}",
            fontsize=9,
            ha="left",
            va="bottom"
        )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q2_cinemas_timing_map.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: Cinema speed distribution (stacked bar chart)
cinema_speed_plot_clean = cinema_speed_plot.copy()
cinema_speed_plot_clean["weeks_to_95"] = pd.to_numeric(cinema_speed_plot_clean["weeks_to_95"], errors="coerce").round()
cinema_speed_plot_clean["weeks_to_95"] = cinema_speed_plot_clean["weeks_to_95"].astype("Int64")
cinema_speed_plot_clean = cinema_speed_plot_clean.dropna(subset=["weeks_to_95"]).copy()
cinema_speed_plot_clean["weeks_to_95"] = cinema_speed_plot_clean["weeks_to_95"].astype(int)

counts = (
    cinema_speed_plot_clean
    .groupby(["timing_class", "weeks_to_95"])
    .size()
    .unstack(fill_value=0)
)

class_order = ["SLOW_BURN", "BALANCED", "EARLY_ADOPTER"]
counts = counts.reindex([c for c in class_order if c in counts.index])

week_cols = sorted(counts.columns.tolist())
counts = counts[week_cols]

x = np.arange(len(counts.index))
bottom = np.zeros(len(counts.index))

plt.figure(figsize=(12, 6))

for w in week_cols:
    vals = counts[w].values
    plt.bar(x, vals, bottom=bottom, label=str(w))
    bottom += vals

plt.xticks(x, [f"{c} (n={counts.loc[c].sum()})" for c in counts.index], rotation=0)
plt.ylabel("Number of cinemas")
plt.xlabel("Timing class")
plt.title("Cinemas: Distribution of weeks_to_95 by timing class (counts)")
plt.legend(title="weeks_to_95", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q2_cinemas_speed_distribution.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: City speed distribution (stacked bar chart)
counts_city = (
    city_speed_plot
    .groupby(["timing_class", "weeks_to_95"])
    .size()
    .unstack(fill_value=0)
)

class_order = ["SLOW_BURN", "BALANCED", "EARLY_ADOPTER"]
counts_city = counts_city.reindex([c for c in class_order if c in counts_city.index])

weeks_order = sorted(counts_city.columns, key=lambda x: float(x))
counts_city = counts_city[weeks_order]

ax = counts_city.plot(kind="bar", stacked=True, figsize=(12, 6))

ax.set_title("Cities: Number of cities by weeks_to_95 and timing class")
ax.set_xlabel("Timing class")
ax.set_ylabel("Number of cities")

totals = counts_city.sum(axis=1).values
for i, t in enumerate(totals):
    ax.text(i, t + 0.2, str(int(t)), ha="center", va="bottom", fontsize=10)

ax.legend(title="weeks_to_95", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q2_cities_speed_distribution.png", dpi=100, bbox_inches='tight')
plt.close()

# Create visualization: Revenue comparison by timing type (Cities)
# Merge timing classification with city-level aggregated revenue data
city_timing_revenue = city_summary.merge(
    city_plot[["state", "city", "timing_class"]],
    on=["state", "city"],
    how="left"
).dropna(subset=['timing_class', 'total_gross'])

fig = go.Figure()

timing_colors = {
    'EARLY_ADOPTER': 'rgb(31, 119, 180)',  # blue
    'BALANCED': 'rgb(255, 127, 14)',       # orange
    'SLOW_BURN': 'rgb(44, 160, 44)'        # green
}

for timing_type in ['EARLY_ADOPTER', 'BALANCED', 'SLOW_BURN']:
    # Show total gross per city (aggregated across all films)
    data = city_timing_revenue[city_timing_revenue['timing_class'] == timing_type]['total_gross']
    
    fig.add_trace(go.Box(
        y=data,
        name=timing_type,
        marker_color=timing_colors[timing_type],
        boxmean='sd',  # Show mean and standard deviation
        hovertemplate='<b>%{fullData.name}</b><br>Total Gross: $%{y:,.0f}<extra></extra>'
    ))

fig.update_layout(
    title='Revenue Distribution by Timing Type (Cities)',
    yaxis_title='Total Gross (log scale)',
    xaxis_title='Timing Type',
    yaxis_type='log',
    height=600,
    width=1000,
    showlegend=True,
    hovermode='closest',
    template='plotly_white'
)

fig.write_html(OUTPUT_DIR / "q2_cities_revenue_boxplot_interactive.html")
print("Saved: outputs_locationquestions/q2_cities_revenue_boxplot_interactive.html")

# Also save as static PNG for reports
fig.write_image(OUTPUT_DIR / "q2_cities_revenue_boxplot.png", width=1000, height=600)
print("Saved: outputs_locationquestions/q2_cities_revenue_boxplot.png")

# Print revenue statistics by timing type
print("\n" + "-"*80)
print("Revenue Statistics by Timing Type (Cities)")
print("-"*80)
for timing_type in ['EARLY_ADOPTER', 'BALANCED', 'SLOW_BURN']:
    data = city_timing_revenue[city_timing_revenue['timing_class'] == timing_type]['total_gross']
    print(f"\n{timing_type}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: ${data.mean():,.0f}")
    print(f"  Median: ${data.median():,.0f}")
    print(f"  Std Dev: ${data.std():,.0f}")
    print(f"  Min: ${data.min():,.0f}")
    print(f"  Max: ${data.max():,.0f}")
    print(f"  Q1 (25%): ${data.quantile(0.25):,.0f}")
    print(f"  Q3 (75%): ${data.quantile(0.75):,.0f}")

# Create visualization: Revenue comparison by timing type (Cinemas)
# Merge timing classification with cinema-level aggregated revenue data
cinema_timing_revenue = cinema_summary.merge(
    cinema_plot[["state", "city", "theatre_name", "timing_class"]],
    on=["state", "city", "theatre_name"],
    how="left"
).dropna(subset=['timing_class', 'total_gross'])

fig = go.Figure()

for timing_type in ['EARLY_ADOPTER', 'BALANCED', 'SLOW_BURN']:
    # Show total gross per cinema (aggregated across all films)
    data = cinema_timing_revenue[cinema_timing_revenue['timing_class'] == timing_type]['total_gross']
    
    fig.add_trace(go.Box(
        y=data,
        name=timing_type,
        marker_color=timing_colors[timing_type],
        boxmean='sd',  # Show mean and standard deviation
        hovertemplate='<b>%{fullData.name}</b><br>Total Gross: $%{y:,.0f}<extra></extra>'
    ))

fig.update_layout(
    title='Revenue Distribution by Timing Type (Cinemas)',
    yaxis_title='Total Gross (log scale)',
    xaxis_title='Timing Type',
    yaxis_type='log',
    height=600,
    width=1000,
    showlegend=True,
    hovermode='closest',
    template='plotly_white'
)

fig.write_html(OUTPUT_DIR / "q2_cinemas_revenue_boxplot_interactive.html")
print("\nSaved: outputs_locationquestions/q2_cinemas_revenue_boxplot_interactive.html")

# Also save as static PNG for reports
fig.write_image(OUTPUT_DIR / "q2_cinemas_revenue_boxplot.png", width=1000, height=600)
print("Saved: outputs_locationquestions/q2_cinemas_revenue_boxplot.png")

# Print revenue statistics by timing type
print("\n" + "-"*80)
print("Revenue Statistics by Timing Type (Cinemas)")
print("-"*80)
for timing_type in ['EARLY_ADOPTER', 'BALANCED', 'SLOW_BURN']:
    data = cinema_timing_revenue[cinema_timing_revenue['timing_class'] == timing_type]['total_gross']
    print(f"\n{timing_type}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: ${data.mean():,.0f}")
    print(f"  Median: ${data.median():,.0f}")
    print(f"  Std Dev: ${data.std():,.0f}")
    print(f"  Min: ${data.min():,.0f}")
    print(f"  Max: ${data.max():,.0f}")
    print(f"  Q1 (25%): ${data.quantile(0.25):,.0f}")
    print(f"  Q3 (75%): ${data.quantile(0.75):,.0f}")

print("\nQ2 visualizations saved to outputs/:")
print("  - q2_cities_timing_map.png")
print("  - q2_cinemas_timing_map.png")
print("  - q2_cinemas_speed_distribution.png")
print("  - q2_cities_speed_distribution.png")
print("  - q2_cities_revenue_boxplot.png")
print("  - q2_cinemas_revenue_boxplot.png")

# ============== QUESTION 3: SEASONALITY BY CALENDAR WEEK ==============

print("\n" + "="*80)
print("QUESTION 3: Seasonality by Calendar Week")
print("="*80)

# Prepare data for seasonality analysis
# Create week_start from actual_sales_date (Monday of the week)
sales_for_seasonality = sales_indian.copy()
sales_for_seasonality['actual_sales_date'] = pd.to_datetime(sales_for_seasonality['actual_sales_date'])
sales_for_seasonality['week_start'] = sales_for_seasonality['actual_sales_date'] - pd.to_timedelta(sales_for_seasonality['actual_sales_date'].dt.dayofweek, unit='d')

# Convert to ISO year/week
sales_for_seasonality['iso_year'] = sales_for_seasonality['week_start'].dt.isocalendar().year
sales_for_seasonality['iso_week'] = sales_for_seasonality['week_start'].dt.isocalendar().week

# Aggregate by state-week
state_week = (
    sales_for_seasonality
    .groupby(['iso_year', 'iso_week', 'state'], as_index=False)
    .agg({
        'gross_today': 'sum',
        'numero_film_id': 'nunique',
        'theatre_name': 'nunique',
        'city': 'nunique'
    })
    .rename(columns={
        'gross_today': 'total_gross',
        'numero_film_id': 'n_titles',
        'theatre_name': 'n_cinemas',
        'city': 'n_cities'
    })
)

# Seasonality by state-week (mean across years)
state_seasonality = (
    state_week
    .groupby(['state', 'iso_week'], as_index=False)
    .agg({
        'total_gross': ['mean', 'median'],
        'n_titles': 'mean',
        'n_cinemas': 'mean',
        'n_cities': 'mean'
    })
    .reset_index(drop=True)
)

state_seasonality.columns = ['state', 'iso_week', 'avg_gross', 'med_gross', 'avg_titles', 'avg_cinemas', 'avg_cities']

# Z-score within each state
state_seasonality['gross_z'] = state_seasonality.groupby('state')['avg_gross'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)

# Seasonality index
state_seasonality['seasonality_idx'] = (
    state_seasonality['avg_gross'] / 
    state_seasonality.groupby('state')['avg_gross'].transform('median')
)

print(f"State seasonality shape: {state_seasonality.shape}")
print(f"ISO weeks covered: {sorted(state_seasonality['iso_week'].unique())}")

# Create heatmap data for states
state_pivot_z = state_seasonality.pivot(index='state', columns='iso_week', values='gross_z')
state_pivot_titles = state_seasonality.pivot(index='state', columns='iso_week', values='avg_titles')
state_pivot_cinemas = state_seasonality.pivot(index='state', columns='iso_week', values='avg_cinemas')

# Plot state heatmaps
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Heatmap 1: Z-scored seasonality
sns.heatmap(state_pivot_z, cmap='RdBu_r', center=0, ax=axes[0], cbar_kws={'label': 'z-score'})
axes[0].set_title('State x ISO Week: Z-scored Seasonality (Red=Busy, Blue=Quiet)')
axes[0].set_xlabel('ISO Week')
axes[0].set_ylabel('State')

# Heatmap 2: Avg titles (competition)
sns.heatmap(state_pivot_titles, cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Avg # Titles'})
axes[1].set_title('State x ISO Week: Average # Active Indian Titles (Competition proxy)')
axes[1].set_xlabel('ISO Week')
axes[1].set_ylabel('State')

# Heatmap 3: Avg cinemas
sns.heatmap(state_pivot_cinemas, cmap='YlGn', ax=axes[2], cbar_kws={'label': 'Avg # Cinemas'})
axes[2].set_title('State x ISO Week: Average # Cinemas Screening Indian Titles')
axes[2].set_xlabel('ISO Week')
axes[2].set_ylabel('State')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q3_state_seasonality_heatmaps.png", dpi=100, bbox_inches='tight')
plt.close()

# Find peak and trough weeks per state
print("\nTop 5 Busiest and Quietest Weeks per State:")
for state in sorted(state_seasonality['state'].unique()):
    state_data = state_seasonality[state_seasonality['state'] == state].sort_values('avg_gross', ascending=False)
    print(f"\n{state}:")
    print("  Top 3 Busiest Weeks:")
    for i, row in state_data.head(3).iterrows():
        print(f"    Week {int(row['iso_week'])}: ${row['avg_gross']:,.0f} (avg {row['avg_titles']:.1f} titles, {row['avg_cinemas']:.1f} cinemas)")
    print("  Top 3 Quietest Weeks:")
    for i, row in state_data.tail(3).iterrows():
        print(f"    Week {int(row['iso_week'])}: ${row['avg_gross']:,.0f} (avg {row['avg_titles']:.1f} titles, {row['avg_cinemas']:.1f} cinemas)")

# Seasonality index line chart for top states
top_states = state_seasonality.groupby('state')['avg_gross'].sum().nlargest(4).index

plt.figure(figsize=(14, 6))
for state in top_states:
    state_data = state_seasonality[state_seasonality['state'] == state].sort_values('iso_week')
    plt.plot(state_data['iso_week'], state_data['seasonality_idx'], marker='o', label=state, linewidth=2)

plt.axhline(1, linestyle='--', color='gray', alpha=0.5)
plt.xlabel('ISO Week')
plt.ylabel('Seasonality Index (avg_gross / median_week)')
plt.title('State Seasonality Index by ISO Week (Top 4 States)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q3_state_seasonality_lines.png", dpi=100, bbox_inches='tight')
plt.close()

# City-level seasonality
city_week = (
    sales_for_seasonality
    .groupby(['iso_year', 'iso_week', 'state', 'city'], as_index=False)
    .agg({
        'gross_today': 'sum',
        'numero_film_id': 'nunique',
        'theatre_name': 'nunique'
    })
    .rename(columns={
        'gross_today': 'total_gross',
        'numero_film_id': 'n_titles',
        'theatre_name': 'n_cinemas'
    })
)

city_seasonality = (
    city_week
    .groupby(['state', 'city', 'iso_week'], as_index=False)
    .agg({
        'total_gross': ['mean', 'median'],
        'n_titles': 'mean',
        'n_cinemas': 'mean'
    })
    .reset_index(drop=True)
)

city_seasonality.columns = ['state', 'city', 'iso_week', 'avg_gross', 'med_gross', 'avg_titles', 'avg_cinemas']

# Seasonality index for cities
city_seasonality['seasonality_idx'] = (
    city_seasonality['avg_gross'] / 
    city_seasonality.groupby(['state', 'city'])['avg_gross'].transform('median')
)

# Select key cities (top by total average gross)
key_cities = (
    city_seasonality
    .groupby(['state', 'city'])['avg_gross']
    .sum()
    .nlargest(10)
    .index.tolist()
)

city_seasonality_key = city_seasonality[
    (city_seasonality['state'].isin([c[0] for c in key_cities])) &
    (city_seasonality['city'].isin([c[1] for c in key_cities]))
].copy()

# Create city heatmap (seasonality index)
city_pivot_idx = city_seasonality_key.pivot_table(
    index=['state', 'city'],
    columns='iso_week',
    values='seasonality_idx'
)

# Create city labels
city_pivot_idx.index = [f"{c[0][:10]} | {c[1][:15]}" for c in city_pivot_idx.index]

plt.figure(figsize=(16, 8))
sns.heatmap(city_pivot_idx, cmap='RdBu_r', center=1, cbar_kws={'label': 'Seasonality Index'}, vmin=0.5, vmax=1.5)
plt.title('Key Cities x ISO Week: Seasonality Index (Red=Over-index, Blue=Under-index)')
plt.xlabel('ISO Week')
plt.ylabel('City')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q3_city_seasonality_heatmap.png", dpi=100, bbox_inches='tight')
plt.close()

# Bump chart: city ranks by week
plt.figure(figsize=(14, 8))
for state, city in key_cities:
    city_data = city_seasonality[
        (city_seasonality['state'] == state) & 
        (city_seasonality['city'] == city)
    ].sort_values('iso_week')
    
    # Compute rank within each week
    city_data = city_data.copy()
    plt.plot(city_data['iso_week'], city_data['avg_gross'], marker='o', label=f"{city[:15]}", alpha=0.7, linewidth=2)

plt.xlabel('ISO Week')
plt.ylabel('Average Gross (in $10 Millions)')
plt.title('Key Cities: Revenue Trend by ISO Week (Bumps Show Seasonal Variation)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q3_city_revenue_trends.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nQ3 visualizations saved to outputs/:")
print("  - q3_state_seasonality_heatmaps.png (3-panel heatmaps)")
print("  - q3_state_seasonality_lines.png (seasonality index trends)")
print("  - q3_city_seasonality_heatmap.png (key cities by week)")
print("  - q3_city_revenue_trends.png (city revenue by week)")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nQ1 Summary:")
print(f"Total cinemas analyzed: {len(cinema_stats_f)}")
print(f"Safer (Stable) cinemas: {len(cinema_stats_f[cinema_stats_f['risk_category'] == 'Safer (Stable)'])}")
print(f"Total cities analyzed: {len(city_stats_f)}")
print(f"Safer (Stable) cities: {len(city_stats_f[city_stats_f['risk_category'] == 'Safer (Stable)'])}")

print(f"\nQ2 Summary:")
print(f"Cities analyzed: {len(city_plot)}")
print(f"Early adopter cities: {len(city_plot[city_plot['timing_class'] == 'EARLY_ADOPTER'])}")
print(f"Slow burn cities: {len(city_plot[city_plot['timing_class'] == 'SLOW_BURN'])}")
print(f"Balanced cities: {len(city_plot[city_plot['timing_class'] == 'BALANCED'])}")

print(f"\nCinemas analyzed: {len(cinema_plot)}")
print(f"Early adopter cinemas: {len(cinema_plot[cinema_plot['timing_class'] == 'EARLY_ADOPTER'])}")
print(f"Slow burn cinemas: {len(cinema_plot[cinema_plot['timing_class'] == 'SLOW_BURN'])}")
print(f"Balanced cinemas: {len(cinema_plot[cinema_plot['timing_class'] == 'BALANCED'])}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
