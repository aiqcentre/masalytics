"""
SalesOverview.py
================
Sales analysis and overview visualizations for Indian cinema box office data.

This script:
1. Loads preprocessed data from DataExplorationMain
2. Analyzes top-performing films
3. Tracks monthly and daily trends
4. Compares weekday performance
"""

import matplotlib.pyplot as plt
import pandas as pd

from project_paths import OUTPUTS_SALESOVERVIEW, ensure_dir

# ============================================================================
# Load preprocessed data from DataExplorationMain
# ============================================================================

print("\n" + "="*80)
print("SALES OVERVIEW ANALYSIS")
print("="*80)

OUTPUT_DIR = ensure_dir(OUTPUTS_SALESOVERVIEW)

# Import from DataExplorationMain
from DataExplorationMain import sales, film_meta, sales_indian

print(f"\nData loaded:")
print(f"  sales: {sales.shape}")
print(f"  film_meta: {film_meta.shape}")

# Compute film totals with titles
film_totals_with_titles = (
    sales_indian
      .groupby('numero_film_id', as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'total_gross'})
      .merge(film_meta[['numero_film_id', 'title']], on='numero_film_id', how='left')
)

print(f"  film_totals_with_titles: {film_totals_with_titles.shape}")

# ============================================================================
# 1. TOP 10 FILMS
# ============================================================================

print("\n" + "-"*80)
print("TOP 10 FILMS BY TOTAL GROSS")
print("-"*80)

top10_films = (
    film_totals_with_titles
      .sort_values("total_gross", ascending=False)
      .head(10)
)

print("\nTop 10 Films:")
print(top10_films[['numero_film_id', 'title', 'total_gross']])

print(f"\nData quality check:")
print(f"  Missing titles: {top10_films['title'].isna().sum()}")
print(f"  Min gross in top 10: {top10_films['total_gross'].min():,.0f}")

# Visualization: Top 10 films bar chart
plt.figure(figsize=(12, 6))
plt.barh(top10_films['title'], top10_films['total_gross'])
plt.gca().invert_yaxis()  # biggest at the top

plt.xlabel("Total Gross")
plt.ylabel("Title")
plt.title("Top 10 Films by Total Gross")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top10_films.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nSaved: outputs_salesoverview/top10_films.png")

# ============================================================================
# 2. MONTHLY TRENDS
# ============================================================================

print("\n" + "-"*80)
print("MONTHLY BOX OFFICE TRENDS")
print("-"*80)

monthly_gross = (
    sales
      .groupby('year_month', as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'total_gross'})
      .sort_values('year_month')
)

print("\nFirst 5 months:")
print(monthly_gross.head())

best_month = monthly_gross.loc[monthly_gross['total_gross'].idxmax()]
worst_month = monthly_gross.loc[monthly_gross['total_gross'].idxmin()]

print(f"\nBest month:  {best_month['year_month']} ({best_month['total_gross']:,.0f})")
print(f"Worst month: {worst_month['year_month']} ({worst_month['total_gross']:,.0f})")

# Visualization: Monthly trends bar chart
plt.figure(figsize=(12, 6))
plt.bar(monthly_gross['year_month'], monthly_gross['total_gross'])

plt.xlabel("Year-Month")
plt.ylabel("Total Gross")
plt.title("Monthly Box Office (Total Gross)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_trends.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nSaved: outputs_salesoverview/monthly_trends.png")

# ============================================================================
# 3. TOP FILM DAILY TREND
# ============================================================================

print("\n" + "-"*80)
print("TOP FILM DAILY BOX OFFICE TREND")
print("-"*80)

target_id = int(top10_films.iloc[0]['numero_film_id'])
target_title = top10_films.iloc[0]['title']

print(f"\nAnalyzing top film: {target_title} (ID: {target_id})")

film_daily = (
    sales[sales['numero_film_id'] == target_id]
      .groupby('actual_sales_date', as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'daily_gross'})
      .sort_values('actual_sales_date')
)

print(f"Daily records: {len(film_daily)}")
print("\nFirst 5 days:")
print(film_daily.head())

# Visualization: Daily trend line chart
plt.figure(figsize=(12, 6))
plt.plot(film_daily['actual_sales_date'], film_daily['daily_gross'], marker='o', linewidth=2, markersize=4)

plt.xlabel("Date")
plt.ylabel("Daily Gross")
plt.title(f"Daily Box Office Trend: {target_title}")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_film_daily_trend.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nSaved: outputs_salesoverview/top_film_daily_trend.png")

# ============================================================================
# 4. WEEKDAY PERFORMANCE
# ============================================================================

print("\n" + "-"*80)
print("WEEKDAY BOX OFFICE PERFORMANCE")
print("-"*80)

weekday_totals = (
    sales
      .groupby(['weekday_index', 'weekday'], as_index=False)['gross_today']
      .sum()
      .rename(columns={'gross_today': 'total_gross'})
      .sort_values('weekday_index')
)

print("\nTotal gross by weekday:")
print(weekday_totals)

# Visualization: Weekday bar chart
plt.figure(figsize=(8, 5))
plt.bar(weekday_totals['weekday'], weekday_totals['total_gross'])

plt.xlabel("Weekday")
plt.ylabel("Total Gross")
plt.title("Total Box Office by Weekday")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weekday_performance.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nSaved: outputs_salesoverview/weekday_performance.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SALES OVERVIEW COMPLETE")
print("="*80)
print("\nVisualizations saved to outputs_salesoverview/:")
print("  - top10_films.png")
print("  - monthly_trends.png")
print("  - top_film_daily_trend.png")
print("  - weekday_performance.png")
print("\n" + "="*80)
