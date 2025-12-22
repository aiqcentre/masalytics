"""
TitlesDistributors.py
======================
Distributor analysis and visualization for Indian cinema films.

This script:
1. Loads preprocessed data from DataExplorationMain
2. Retrieves distributor information from the database
3. Analyzes distributor performance and market share
4. Creates visualizations: treemap, Pareto chart, and state heatmap
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path

# ============================================================================
# Load preprocessed data from DataExplorationMain
# ============================================================================

print("\n" + "="*80)
print("TITLES & DISTRIBUTORS ANALYSIS")
print("="*80)

from DataExplorationMain import sales, film_meta, conn

print(f"\nData loaded:")
print(f"  sales: {sales.shape}")
print(f"  film_meta: {film_meta.shape}")

# ============================================================================
# 1. LOAD DISTRIBUTOR INFORMATION
# ============================================================================

print("\n" + "-"*80)
print("LOADING DISTRIBUTOR INFORMATION")
print("-"*80)

dist_info = pd.read_sql(
    "SELECT title, distributor FROM indian_titles;",
    conn
)

print(f"\nDistributor info loaded: {dist_info.shape}")
print(f"Missing distributors: {dist_info['distributor'].isna().mean():.1%}")
print("\nSample:")
print(dist_info.head())

# ============================================================================
# 2. MERGE DISTRIBUTOR WITH FILM METADATA AND SALES
# ============================================================================

print("\n" + "-"*80)
print("MERGING DISTRIBUTOR DATA WITH FILMS AND SALES")
print("-"*80)

titles_with_dist = film_meta.merge(
    dist_info,
    on='title',
    how='left'
)

print(f"\nTitles with distributor: {titles_with_dist.shape}")
print(f"Coverage: {(1 - titles_with_dist['distributor'].isna().mean()):.1%}")

sales_with_dist = sales.merge(
    titles_with_dist[['numero_film_id', 'distributor']],
    on='numero_film_id',
    how='left'
)

print(f"Sales with distributor: {sales_with_dist.shape}")
print(f"Coverage: {(1 - sales_with_dist['distributor'].isna().mean()):.1%}")

# ============================================================================
# 3. DISTRIBUTOR TOTALS
# ============================================================================

print("\n" + "-"*80)
print("DISTRIBUTOR REVENUE TOTALS")
print("-"*80)

dist_totals = (
    sales_with_dist[~sales_with_dist['distributor'].isna()]
      .groupby('distributor', as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'gross_per_dis'})
      .sort_values('gross_per_dis', ascending=False)
)

print(f"\nTotal distributors: {len(dist_totals)}")
print("\nTop 10 distributors:")
print(dist_totals.head(10))

# ============================================================================
# 4. VISUALIZATION 1: TREEMAP (TOP 15 DISTRIBUTORS)
# ============================================================================

print("\n" + "-"*80)
print("VISUALIZATION 1: TREEMAP")
print("-"*80)

top_dists = dist_totals.head(15).copy()

fig = px.treemap(
    top_dists,
    path=['distributor'],
    values='gross_per_dis',
    title='Distributor Share of Total Gross (Top 15)'
)

fig.write_html('outputs_titlesdistributors/distributor_treemap.html')
fig.write_image('outputs_titlesdistributors/distributor_treemap.png', width=1000, height=700)

print("Saved: outputs_titlesdistributors/distributor_treemap.html")
print("Saved: outputs_titlesdistributors/distributor_treemap.png")

# ============================================================================
# 5. VISUALIZATION 2: PARETO CHART (TOP 20 DISTRIBUTORS)
# ============================================================================

print("\n" + "-"*80)
print("VISUALIZATION 2: PARETO CHART")
print("-"*80)

pareto = dist_totals.copy()
pareto['cum_gross'] = pareto['gross_per_dis'].cumsum()
pareto['cum_share'] = pareto['cum_gross'] / pareto['gross_per_dis'].sum()

# Keep it readable
pareto_plot = pareto.head(20)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=pareto_plot['distributor'], y=pareto_plot['gross_per_dis'], name='Gross'),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(
        x=pareto_plot['distributor'],
        y=pareto_plot['cum_share'],
        name='Cumulative Share',
        mode='lines+markers'
    ),
    secondary_y=True
)

fig.update_layout(
    title="Pareto: Distributor Revenue Concentration (Top 20)",
    xaxis_title="Distributor",
    yaxis_title="Gross",
    width=1200,
    height=600
)
fig.update_yaxes(title_text="Cumulative Share", secondary_y=True, tickformat=".0%")
fig.update_xaxes(tickangle=45)

fig.write_html('outputs_titlesdistributors/distributor_pareto.html')
fig.write_image('outputs_titlesdistributors/distributor_pareto.png', width=1200, height=600)

print("Saved: outputs_titlesdistributors/distributor_pareto.html")
print("Saved: outputs_titlesdistributors/distributor_pareto.png")

# ============================================================================
# 6. VISUALIZATION 3: STATE HEATMAP (TOP 10 DISTRIBUTORS)
# ============================================================================

print("\n" + "-"*80)
print("VISUALIZATION 3: STATE HEATMAP")
print("-"*80)

dist_state = (
    sales_with_dist
      .dropna(subset=['distributor', 'state'])
      .groupby(['distributor', 'state'], as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'gross'})
)

# Focus on top distributors to keep the heatmap readable
top10_dist_names = dist_totals.head(10)['distributor'].tolist()
dist_state_top = dist_state[dist_state['distributor'].isin(top10_dist_names)].copy()

pivot = dist_state_top.pivot_table(
    index='distributor',
    columns='state',
    values='gross',
    fill_value=0
)

fig = px.imshow(
    pivot,
    aspect='auto',
    title='Gross by Distributor and State (Top 10 Distributors)',
    labels=dict(x="State", y="Distributor", color="Gross"),
    color_continuous_scale='Viridis'
)

fig.write_html('outputs_titlesdistributors/distributor_state_heatmap.html')
fig.write_image('outputs_titlesdistributors/distributor_state_heatmap.png', width=1000, height=600)

print("Saved: outputs_titlesdistributors/distributor_state_heatmap.html")
print("Saved: outputs_titlesdistributors/distributor_state_heatmap.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TITLES & DISTRIBUTORS ANALYSIS COMPLETE")
print("="*80)
print("\nVisualizations saved to outputs_titlesdistributors/:")
print("  - distributor_treemap.html / .png")
print("  - distributor_pareto.html / .png")
print("  - distributor_state_heatmap.html / .png")
print("\n" + "="*80)
