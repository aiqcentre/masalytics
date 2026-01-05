#!/usr/bin/env python
"""Extract total market values by timing type for Q2 analysis."""

import numpy as np
import pandas as pd

from DataExplorationMain import film_meta, indian_titles, sales

# Replicate LocationQuestions.py preprocessing
# First, aggregate daily gross_today to weekly_gross
sales_copy = sales.copy()
sales_copy['week_start'] = (
    sales_copy['actual_sales_date']
      .dt.to_period('W')
      .apply(lambda r: r.start_time)
)

# Aggregate to weekly by location
sales_weekly = (
    sales_copy
      .groupby(['numero_film_id', 'state', 'city', 'theatre_name', 'week_start'], as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'weekly_gross'})
)

sales_wt = sales_weekly.merge(
    film_meta[["numero_film_id", "title"]],
    on="numero_film_id",
    how="left"
)

sales_wt["title"] = sales_wt["title"].astype(str).str.strip()
indian_titles_copy = indian_titles.copy()
indian_titles_copy["title"] = indian_titles_copy["title"].astype(str).str.strip()

indian_only = sales_wt.merge(
    indian_titles_copy[["title"]].drop_duplicates(),
    on="title",
    how="inner"
)

indian_only = indian_only[indian_only["weekly_gross"] > 0].copy()
indian_only["week_start"] = pd.to_datetime(indian_only["week_start"])

# Add relative week from first week
def add_relative_week(df, group_cols):
    out = df.copy()
    grouped = df.groupby(group_cols)["week_start"].min().reset_index()
    grouped.columns = group_cols + ["first_week_start"]
    out = out.merge(grouped, on=group_cols, how="left")
    out["rel_week"] = ((out["week_start"] - out["first_week_start"]).dt.days // 7) + 1
    return out

# Build rel_week
indian_copy = add_relative_week(
    indian_only,
    ["numero_film_id", "state", "city"]
)

# Build city aggregation
def build_film_location_timing(df, group_cols):
    df_copy = df.copy()
    df_copy['early_gross'] = np.where(df_copy['rel_week'] <= 2, df_copy['weekly_gross'], 0.0)
    df_copy['late_gross'] = np.where(df_copy['rel_week'] >= 3, df_copy['weekly_gross'], 0.0)
    
    out = (
        df_copy.groupby(group_cols, as_index=False)
              .agg(
                  total_gross=('weekly_gross', 'sum'),
                  early_gross=('early_gross', 'sum'),
                  late_gross=('late_gross', 'sum'),
                  weeks_active=('rel_week', 'max')
              )
    )
    
    out['early_share'] = np.where(out['total_gross'] > 0, out['early_gross'] / out['total_gross'], np.nan)
    return out

# Build city timing
film_city_timing = build_film_location_timing(
    indian_copy,
    ['numero_film_id', 'title', 'state', 'city']
)

# Build city summary
def safe_weighted_avg(values, weights):
    values = pd.Series(values)
    weights = pd.Series(weights).fillna(0)
    wsum = weights.sum()
    if wsum <= 0:
        return np.nan
    return np.average(values, weights=weights)

def build_place_summary(film_location_timing, place_cols):
    df = film_location_timing[film_location_timing['total_gross'] > 0].copy()
    out = (
        df.groupby(place_cols, as_index=False)
          .apply(lambda g: pd.Series({
              'total_gross': g['total_gross'].sum(),
              'n_films': g['numero_film_id'].nunique(),
              'weighted_early_share': safe_weighted_avg(g['early_share'], g['total_gross'])
          }))
          .reset_index(drop=True)
    )
    return out

city_summary = build_place_summary(film_city_timing, ['state', 'city'])

# Get top 36 cities
TOP_N = 36
city_plot = (
    city_summary[city_summary['total_gross'] > 0]
    .sort_values('total_gross', ascending=False)
    .head(TOP_N)
    .copy()
)

q25 = city_plot['weighted_early_share'].quantile(0.25)
q75 = city_plot['weighted_early_share'].quantile(0.75)

def timing_class(x):
    if x >= q75:
        return 'EARLY_ADOPTER'
    elif x <= q25:
        return 'SLOW_BURN'
    else:
        return 'BALANCED'

city_plot['timing_class'] = city_plot['weighted_early_share'].apply(timing_class)

# Calculate totals by timing class
timing_breakdown = city_plot.groupby('timing_class')['total_gross'].agg(['sum', 'count', 'mean', 'median', 'min', 'max']).reset_index()
total_market = city_plot['total_gross'].sum()

print('='*80)
print('TOTAL MARKET VALUE BY TIMING TYPE (Cities)')
print('='*80)
print(f'\nTotal Market Value (Top 36 Cities): ${total_market:,.0f}')
print(f'Average per City: ${total_market/len(city_plot):,.0f}\n')

print(f'{"Timing Type":<20} {"Revenue":>15} {"# Cities":>10} {"Avg Revenue":>15} {"Median":>15}')
print('-'*80)
for _, row in timing_breakdown.iterrows():
    timing_type = row['timing_class']
    total = row['sum']
    count = int(row['count'])
    avg = row['mean']
    median = row['median']
    pct = 100 * total / total_market
    print(f'{timing_type:<20} ${total:>14,.0f} {count:>10d} ${avg:>14,.0f} ${median:>14,.0f}')

print('-'*80)
print(f'\nMarket Share by Timing Type:')
for _, row in timing_breakdown.iterrows():
    timing_type = row['timing_class']
    total = row['sum']
    pct = 100 * total / total_market
    count = int(row['count'])
    print(f'  {timing_type:<18}: ${total:>14,.0f} ({pct:>5.1f}%)  - {count} cities')

print(f'\n  TOTAL:              ${total_market:>14,.0f} (100.0%)  - {len(city_plot)} cities')
