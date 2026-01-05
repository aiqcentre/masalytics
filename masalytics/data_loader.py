"""Load and normalize core data tables for analysis scripts."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .paths import find_database


@dataclass
class DataBundle:
    """Container for the core data tables."""

    conn: sqlite3.Connection
    sales: pd.DataFrame
    sales_indian: pd.DataFrame
    film_meta: pd.DataFrame
    indian_titles: pd.DataFrame


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(message)


def get_all_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    """Return column names from a SQLite table."""
    try:
        df_schema = pd.read_sql(f"PRAGMA table_info({table_name});", conn)
        return df_schema["name"].tolist()
    except Exception as exc:
        _log(f"Error getting columns for {table_name}: {exc}", True)
        return []


def _flatten_one_film(raw_json_str: str | None, film_id: int) -> pd.DataFrame:
    """Parse a single film's raw JSON and extract daily sales rows."""
    if not raw_json_str:
        return pd.DataFrame(
            columns=[
                "numero_film_id",
                "actual_sales_date",
                "state",
                "city",
                "theatre_name",
                "gross_today",
            ]
        )

    try:
        data = json.loads(raw_json_str)
    except json.JSONDecodeError:
        return pd.DataFrame(
            columns=[
                "numero_film_id",
                "actual_sales_date",
                "state",
                "city",
                "theatre_name",
                "gross_today",
            ]
        )

    if not isinstance(data, dict):
        return pd.DataFrame(
            columns=[
                "numero_film_id",
                "actual_sales_date",
                "state",
                "city",
                "theatre_name",
                "gross_today",
            ]
        )

    rows_out: list[dict] = []

    for week_start_str, week_content in data.items():
        try:
            week_dt = dt.datetime.strptime(week_start_str, "%Y-%m-%d").date()
        except Exception:
            continue

        rows = week_content.get("rows", []) if isinstance(week_content, dict) else []
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

                rows_out.append(
                    {
                        "numero_film_id": film_id,
                        "actual_sales_date": actual_date,
                        "state": state,
                        "city": city,
                        "theatre_name": theatre_name,
                        "gross_today": gross_today,
                    }
                )

    return pd.DataFrame(rows_out)


def flatten_sales_json(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten all films' raw JSON into a daily sales DataFrame."""
    all_frames: list[pd.DataFrame] = []

    for _, row in df_raw.iterrows():
        try:
            film_id = int(row["numero_film_id"])
        except (TypeError, ValueError):
            continue

        df_one = _flatten_one_film(row.get("raw_json"), film_id)
        if not df_one.empty:
            all_frames.append(df_one)

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "numero_film_id",
            "actual_sales_date",
            "state",
            "city",
            "theatre_name",
            "gross_today",
        ]
    )


def normalize_title(series: pd.Series) -> pd.Series:
    """Normalize titles for comparison (lowercase, stripped)."""
    return series.astype(str).str.strip().str.lower()


def load_data(db_path: str | None = None, verbose: bool = True) -> DataBundle:
    """Load core tables and derive sales datasets."""
    resolved_db = find_database(db_path)
    _log(f"Using database: {resolved_db}", verbose)

    conn = sqlite3.connect(str(resolved_db))

    _log("Loading raw data from database...", verbose)
    df_raw = pd.read_sql(
        "SELECT numero_film_id, raw_json FROM sales_raw_data;",
        conn,
    )

    _log("Flattening JSON sales data...", verbose)
    sales = flatten_sales_json(df_raw)
    _log(f"Flattened sales shape: {sales.shape}", verbose)

    sales["actual_sales_date"] = pd.to_datetime(sales["actual_sales_date"], errors="coerce")
    sales["gross_today"] = pd.to_numeric(sales["gross_today"], errors="coerce")

    sales["year_month"] = sales["actual_sales_date"].dt.to_period("M").astype(str)
    sales["weekday"] = sales["actual_sales_date"].dt.day_name()
    sales["weekday_index"] = sales["actual_sales_date"].dt.dayofweek

    _log("Loading film metadata...", verbose)
    film_meta = pd.read_sql(
        """
        SELECT numero_film_id, title
        FROM film_metadata;
        """,
        conn,
    )

    _log("Loading Indian titles...", verbose)
    indian_titles = pd.read_sql(
        """
        SELECT title
        FROM indian_titles
        """,
        conn,
    )

    _log("Matching Indian films...", verbose)
    film_meta_norm = film_meta.copy()
    film_meta_norm["title_norm"] = normalize_title(film_meta_norm["title"])

    indian_titles_norm = indian_titles.copy()
    indian_titles_norm["title_norm"] = normalize_title(indian_titles_norm["title"])

    indian_film_ids = (
        film_meta_norm.merge(
            indian_titles_norm[["title_norm"]],
            on="title_norm",
            how="inner",
        )[["numero_film_id", "title"]]
        .drop_duplicates()
    )

    _log(
        f"Indian film_ids matched: {indian_film_ids['numero_film_id'].nunique()}",
        verbose,
    )

    sales_indian = sales.merge(
        indian_film_ids[["numero_film_id"]],
        on="numero_film_id",
        how="inner",
    )

    _log(f"Final sales (all films): {sales.shape}", verbose)
    _log(f"Final sales (Indian films only): {sales_indian.shape}", verbose)
    _log(
        f"Unique Indian films in sales: {sales_indian['numero_film_id'].nunique()}",
        verbose,
    )

    return DataBundle(
        conn=conn,
        sales=sales,
        sales_indian=sales_indian,
        film_meta=film_meta,
        indian_titles=indian_titles,
    )
