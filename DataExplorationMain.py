# DataExplorationMain.py
# This module contains data preprocessing and loading code extracted from DataExplorationMain.ipynb
# It prepares the sales data and makes it available for downstream analysis

import datetime as dt
import json
import sqlite3

import pandas as pd

from project_paths import DATA_DIR, find_database_path

# Connect to database
DB_PATH = find_database_path(DATA_DIR / "numero_data.sqlite")
conn = sqlite3.connect(DB_PATH)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_columns(conn, table_name):
    """Get column names from a table"""
    try:
        df_schema = pd.read_sql(f"PRAGMA table_info({table_name});", conn)
        return df_schema['name'].tolist()
    except Exception as e:
        print(f"Error getting columns for {table_name}: {e}")
        return []

def flatten_one_film(raw_json_str, film_id):
    """
    Parse a single film's raw JSON and extract daily sales by cinema/location
    """
    rows_out = []

    data = json.loads(raw_json_str)

    for week_start_str, week_content in data.items():
        try:
            week_dt = dt.datetime.strptime(week_start_str, "%Y-%m-%d").date()
        except Exception:
            continue

        rows = week_content.get("rows", [])
        if not isinstance(rows, list):
            continue

        for cinema_row in rows:
            if not isinstance(cinema_row, dict):
                continue

            state = cinema_row.get("state")
            city = cinema_row.get("city")
            theatre_name = cinema_row.get("theatre")

            box_office = cinema_row.get("boxOffice", {}) or {}
            if not isinstance(box_office, dict):
                continue

            for day_key, sales_data in box_office.items():
                if not isinstance(day_key, str) or not day_key.startswith("day"):
                    continue

                day_num_str = day_key.replace("day", "")
                if not day_num_str.isdigit():
                    continue

                day_offset = int(day_num_str) - 1
                actual_date = week_dt + dt.timedelta(days=day_offset)

                if not isinstance(sales_data, dict):
                    continue

                gross_today = sales_data.get("today")

                rows_out.append({
                    "numero_film_id": film_id,
                    "actual_sales_date": actual_date,
                    "state": state,
                    "city": city,
                    "theatre_name": theatre_name,
                    "gross_today": gross_today,
                })

    return pd.DataFrame(rows_out)

def flatten_sales_json(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten all films' raw JSON into a single DataFrame with daily sales records
    """
    all_frames = []

    for _, row in df_raw.iterrows():
        film_id = int(row['numero_film_id'])
        raw = row['raw_json']

        df_one = flatten_one_film(raw, film_id)

        if not df_one.empty:
            all_frames.append(df_one)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=["numero_film_id", "actual_sales_date",
                     "state", "city", "theatre_name", "gross_today"]
        )

    return result

def norm_title(s: pd.Series) -> pd.Series:
    """Normalize titles for comparison (lowercase, strip whitespace)"""
    return (s.astype(str)
             .str.strip()
             .str.lower())

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Load raw data from database
print(f"Using database: {DB_PATH}")
print("Loading raw data from database...")
df_raw = pd.read_sql(
    "SELECT numero_film_id, raw_json FROM sales_raw_data;",
    conn
)

# Flatten JSON into daily sales records
print("Flattening JSON sales data...")
sales = flatten_sales_json(df_raw)
print(f"Flattened sales shape: {sales.shape}")

# Ensure correct data types
sales['actual_sales_date'] = pd.to_datetime(sales['actual_sales_date'], errors='coerce')
sales['gross_today'] = pd.to_numeric(sales['gross_today'], errors='coerce')

# Add derived columns
sales['year_month'] = (
    sales['actual_sales_date']
      .dt.to_period('M')
      .astype(str)
)
sales['weekday'] = sales['actual_sales_date'].dt.day_name()
sales['weekday_index'] = sales['actual_sales_date'].dt.dayofweek

# Load film metadata
print("Loading film metadata...")
film_meta = pd.read_sql(
    """
    SELECT numero_film_id, title
    FROM film_metadata;
    """,
    conn
)

# Load Indian titles list
print("Loading Indian titles...")
indian_titles = pd.read_sql("""
    SELECT title
    FROM indian_titles
""", conn)

# Normalize titles for matching
print("Matching Indian films...")
film_meta_norm = film_meta.copy()
film_meta_norm["title_norm"] = norm_title(film_meta_norm["title"])

indian_titles_norm = indian_titles.copy()
indian_titles_norm["title_norm"] = norm_title(indian_titles_norm["title"])

# Find Indian films
indian_film_ids = (
    film_meta_norm.merge(indian_titles_norm[["title_norm"]], on="title_norm", how="inner")
                   [["numero_film_id", "title"]]
                   .drop_duplicates()
)

print(f"Indian film_ids matched: {indian_film_ids['numero_film_id'].nunique()}")

# Filter sales to Indian films
sales_indian = sales.merge(indian_film_ids[["numero_film_id"]], on="numero_film_id", how="inner")

print(f"\nFinal sales (all films): {sales.shape}")
print(f"Final sales (Indian films only): {sales_indian.shape}")
print(f"Unique Indian films in sales: {sales_indian['numero_film_id'].nunique()}")

# Export main variables (used by downstream scripts)
print("\nData preprocessing complete. Variables available:")
print("  - sales: All films sales data")
print("  - sales_indian: Indian films only")
print("  - film_meta: Film metadata")
print("  - indian_titles: Indian titles reference")
print("  - conn: Database connection")
