from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

try:
    from shapely.geometry import shape
except Exception:  # pragma: no cover - optional dependency
    shape = None


OUTPUT_DIR = Path("outputs_locationquestions")

STATE_CSV = OUTPUT_DIR / "q3_state_seasonality.csv"
CITY_CSV = OUTPUT_DIR / "q3_city_seasonality.csv"

STATE_GEOJSON = OUTPUT_DIR / "q3_state_geometry_simplified.geojson"
CITY_GEOJSON = OUTPUT_DIR / "q3_city_sa4_geometry_simplified.geojson"

BASE_SA4_GEOJSON = Path("data/SA4_2021_AUST_GDA94_cleaned.geojson")

INCLUDE_PLOTLYJS = "cdn"  # Smaller HTML; requires internet to load plotly.js.
CITY_LABEL_MAX_COUNT = 12
CITY_LABEL_MIN_DISTANCE_DEG = 1.4
AUSTRALIA_BORDER_COLOR = "#5F5F5F"
AUSTRALIA_BORDER_WIDTH = 1.2
AUSTRALIA_SIMPLIFY_TOLERANCE = 0.05
CITY_PEAK_QUANTILE = 0.9
CITY_LOW_QUANTILE = 0.1

COLOR_SCALES = {
    "RdBu_r": px.colors.diverging.RdBu[::-1],
    "YlOrRd": px.colors.sequential.YlOrRd,
    "YlGn": px.colors.sequential.YlGn,
}

METRIC_LABELS = {
    "gross_z": "Seasonality (Z-Score)",
    "avg_titles": "Avg # Titles",
    "avg_cinemas": "Avg # Cinemas",
    "avg_cities": "Avg # Cities",
    "avg_gross": "Average Gross",
    "seasonality_idx": "Seasonality Index",
    "opportunity_score": "Opportunity Score",
    "peak_low_score": "Peaks/Lows",
}

METRIC_DESCRIPTIONS = {
    "gross_z": "Z-score of weekly average gross versus each location's baseline (red = above average, blue = below).",
    "avg_titles": "Average number of Indian titles active in that week (competition proxy).",
    "avg_cinemas": "Average number of cinemas screening Indian titles in that week (screen availability).",
    "avg_gross": "Average weekly gross by location across all years.",
    "seasonality_idx": "Weekly gross versus each location's median week (red = above median, blue = below).",
    "opportunity_score": "Demand minus competition: gross_z - 0.8 * titles_z.",
    "peak_low_score": "Top/bottom weeks per city (top 10% = peak, bottom 10% = low).",
}


def load_geojson(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())


def normalize_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _flatten_coords(geom: dict) -> list[tuple[float, float]]:
    coords = geom.get("coordinates", [])
    gtype = geom.get("type")
    points: list[tuple[float, float]] = []
    if gtype == "Polygon":
        for ring in coords:
            points.extend(ring)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                points.extend(ring)
    return points


def label_points_from_geojson(geojson: dict, property_name: str) -> dict[str, tuple[float, float]]:
    labels: dict[str, tuple[float, float]] = {}
    for feature in geojson.get("features", []):
        props = feature.get("properties", {}) or {}
        name = props.get(property_name)
        geom = feature.get("geometry")
        if not name or not geom:
            continue

        lon_lat = None
        if shape is not None:
            try:
                geom_obj = shape(geom)
                if not geom_obj.is_empty:
                    point = geom_obj.representative_point()
                    lon_lat = (float(point.x), float(point.y))
            except Exception:
                lon_lat = None

        if lon_lat is None:
            points = _flatten_coords(geom)
            if points:
                lons = [p[0] for p in points]
                lats = [p[1] for p in points]
                lon_lat = ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)

        if lon_lat is not None:
            labels[str(name)] = lon_lat

    return labels


def add_week_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iso_week_str"] = df["iso_week"].astype(int).astype(str).str.zfill(2)
    return df


def hover_format(metric: str) -> str:
    if metric in {"avg_titles", "avg_cinemas", "avg_cities"}:
        return ":.1f"
    if metric in {"gross_z", "seasonality_idx"}:
        return ":.2f"
    if metric == "peak_low_score":
        return ":.0f"
    if metric == "avg_gross":
        return ":,.0f"
    return ":.2f"


def hover_template(metric: str, label: str) -> str:
    fmt = hover_format(metric)
    return f"%{{location}}<br>ISO Week: %{{customdata}}<br>{label}: %{{z{fmt}}}<extra></extra>"


def pick_geojson(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    return fallback


def select_spread_labels(
    candidates: list[str],
    label_points: dict[str, tuple[float, float]],
    max_count: int,
    min_distance: float,
) -> list[str]:
    selected: list[str] = []
    for name in candidates:
        if name not in label_points:
            continue
        lon, lat = label_points[name]
        if all(
            math.hypot(lon - label_points[other][0], lat - label_points[other][1])
            >= min_distance
            for other in selected
        ):
            selected.append(name)
        if len(selected) >= max_count:
            break

    if not selected:
        selected = [name for name in candidates if name in label_points][:max_count]

    return selected


def prepare_sales_for_seasonality() -> pd.DataFrame:
    from DataExplorationMain import sales_indian

    sales_for_seasonality = sales_indian.copy()
    sales_for_seasonality["actual_sales_date"] = pd.to_datetime(
        sales_for_seasonality["actual_sales_date"]
    )
    sales_for_seasonality["week_start"] = (
        sales_for_seasonality["actual_sales_date"]
        - pd.to_timedelta(
            sales_for_seasonality["actual_sales_date"].dt.dayofweek, unit="d"
        )
    )

    sales_for_seasonality["iso_year"] = (
        sales_for_seasonality["week_start"].dt.isocalendar().year
    )
    sales_for_seasonality["iso_week"] = (
        sales_for_seasonality["week_start"].dt.isocalendar().week
    )

    return sales_for_seasonality


def compute_seasonality_from_sales(
    sales_for_seasonality: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sales_for_seasonality is None:
        sales_for_seasonality = prepare_sales_for_seasonality()

    state_week = (
        sales_for_seasonality.groupby(["iso_year", "iso_week", "state"], as_index=False)
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
                "theatre_name": "nunique",
                "city": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
                "theatre_name": "n_cinemas",
                "city": "n_cities",
            }
        )
    )

    state_seasonality = (
        state_week.groupby(["state", "iso_week"], as_index=False)
        .agg(
            {
                "total_gross": ["mean", "median"],
                "n_titles": "mean",
                "n_cinemas": "mean",
                "n_cities": "mean",
            }
        )
        .reset_index(drop=True)
    )

    state_seasonality.columns = [
        "state",
        "iso_week",
        "avg_gross",
        "med_gross",
        "avg_titles",
        "avg_cinemas",
        "avg_cities",
    ]

    state_seasonality["gross_z"] = state_seasonality.groupby("state")[
        "avg_gross"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    state_seasonality["seasonality_idx"] = state_seasonality["avg_gross"] / (
        state_seasonality.groupby("state")["avg_gross"].transform("median")
    )

    city_week = (
        sales_for_seasonality.groupby(
            ["iso_year", "iso_week", "state", "city"], as_index=False
        )
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
                "theatre_name": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
                "theatre_name": "n_cinemas",
            }
        )
    )

    city_seasonality = (
        city_week.groupby(["state", "city", "iso_week"], as_index=False)
        .agg(
            {
                "total_gross": ["mean", "median"],
                "n_titles": "mean",
                "n_cinemas": "mean",
            }
        )
        .reset_index(drop=True)
    )

    city_seasonality.columns = [
        "state",
        "city",
        "iso_week",
        "avg_gross",
        "med_gross",
        "avg_titles",
        "avg_cinemas",
    ]

    city_seasonality["seasonality_idx"] = city_seasonality["avg_gross"] / (
        city_seasonality.groupby(["state", "city"])["avg_gross"].transform("median")
    )

    city_seasonality["gross_z"] = city_seasonality.groupby(["state", "city"])[
        "avg_gross"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_seasonality["titles_z"] = city_seasonality.groupby(["state", "city"])[
        "avg_titles"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_seasonality["opportunity_score"] = (
        city_seasonality["gross_z"] - 0.8 * city_seasonality["titles_z"]
    )

    return state_seasonality, city_seasonality


def compute_city_week_by_year(
    sales_for_seasonality: pd.DataFrame,
) -> pd.DataFrame:
    city_week = (
        sales_for_seasonality.groupby(
            ["iso_year", "iso_week", "state", "city"], as_index=False
        )
        .agg(
            {
                "gross_today": "sum",
                "numero_film_id": "nunique",
                "theatre_name": "nunique",
            }
        )
        .rename(
            columns={
                "gross_today": "total_gross",
                "numero_film_id": "n_titles",
                "theatre_name": "n_cinemas",
            }
        )
    )

    city_week["seasonality_idx"] = city_week["total_gross"] / city_week.groupby(
        ["state", "city", "iso_year"]
    )["total_gross"].transform("median")

    city_week["gross_z"] = city_week.groupby(["state", "city", "iso_year"])[
        "total_gross"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_week["titles_z"] = city_week.groupby(["state", "city", "iso_year"])[
        "n_titles"
    ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    city_week["opportunity_score"] = (
        city_week["gross_z"] - 0.8 * city_week["titles_z"]
    )

    return city_week


def build_state_geometry(sa4: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def map_state(name: str) -> str:
        if name in ["New South Wales", "Australian Capital Territory"]:
            return "New South Wales (inc ACT)"
        if name in ["Victoria", "Tasmania"]:
            return "Victoria (inc TAS)"
        return name

    state_geom = sa4.copy()
    state_geom["state"] = state_geom["STE_NAME21"].map(map_state)
    state_geom = state_geom.dissolve(by="state", as_index=False)
    state_geom = state_geom[["state", "geometry"]]
    state_geom["geometry"] = state_geom["geometry"].simplify(
        tolerance=0.05, preserve_topology=True
    )
    return state_geom


def build_city_geometry(
    sa4: gpd.GeoDataFrame, city_names: list[str]
) -> gpd.GeoDataFrame:
    sa4_names = set(sa4["SA4_NAME21"].dropna().astype(str).str.strip())
    sa4_name_map = {normalize_key(name): name for name in sa4_names}

    manual_map = {
        normalize_key("Canberra"): ["Australian Capital Territory"],
        normalize_key("Central and Far West"): ["Central West", "Far West and Orana"],
        normalize_key("City and Inner South"): ["Sydney - City and Inner South"],
        normalize_key("Eastern Suburbs"): ["Sydney - Eastern Suburbs"],
        normalize_key("Hills & Hawkesbury"): ["Sydney - Baulkham Hills and Hawkesbury"],
        normalize_key("Inner West"): ["Sydney - Inner West"],
        normalize_key("Murray and Riverina"): ["Murray", "Riverina"],
        normalize_key("North Sydney - Hornsby"): ["Sydney - North Sydney and Hornsby"],
        normalize_key("Northern Beaches"): ["Sydney - Northern Beaches"],
        normalize_key("Parramatta & Ryde"): ["Sydney - Parramatta", "Sydney - Ryde"],
        normalize_key("South West Sydney"): ["Sydney - South West"],
        normalize_key("Sutherland & St George"): ["Sydney - Sutherland"],
        normalize_key("West and Blue Mountains"): ["Sydney - Outer West and Blue Mountains"],
        normalize_key("N.T Outback"): ["Northern Territory - Outback"],
        normalize_key("Brisbane - Inner City"): ["Brisbane Inner City"],
        normalize_key("Cairns Region"): ["Cairns"],
        normalize_key("Ipswich Region"): ["Ipswich"],
        normalize_key("QLD - Outback"): ["Queensland - Outback"],
        normalize_key("Toowoomba - Darling Downs"): ["Toowoomba", "Darling Downs - Maranoa"],
        normalize_key("Townsville Region"): ["Townsville"],
        normalize_key("S.A  - Outback"): ["South Australia - Outback"],
        normalize_key("S.A - South East"): ["South Australia - South East"],
        normalize_key("Central Inner Melbourne"): ["Melbourne - Inner"],
        normalize_key("Inner East Melbourne"): ["Melbourne - Inner East"],
        normalize_key("Inner South Melbourne"): ["Melbourne - Inner South"],
        normalize_key("North East Melbourne"): ["Melbourne - North East"],
        normalize_key("North West Melbourne"): ["Melbourne - North West"],
        normalize_key("North West Victoria"): ["North West"],
        normalize_key("Outer East Melbourne"): ["Melbourne - Outer East"],
        normalize_key("South East Melbourne"): ["Melbourne - South East"],
        normalize_key("Tas - North East"): ["Launceston and North East"],
        normalize_key("Tas - North West"): ["West and North West"],
        normalize_key("West Melbourne"): ["Melbourne - West"],
        normalize_key("Mandurah - Bunbury"): ["Mandurah", "Bunbury"],
        normalize_key("W.A - Outback South"): ["Western Australia - Outback (South)"],
        normalize_key("W.A - Wheat Belt"): ["Western Australia - Wheat Belt"],
    }

    for targets in manual_map.values():
        for target in targets:
            if target not in sa4_names:
                raise SystemExit(f"Manual mapping target not in SA4 list: {target}")

    rows = []
    unmatched = []
    for city in city_names:
        key = normalize_key(city)
        if key in sa4_name_map:
            rows.append({"city": city, "sa4_name": sa4_name_map[key]})
        elif key in manual_map:
            for target in manual_map[key]:
                rows.append({"city": city, "sa4_name": target})
        else:
            unmatched.append(city)

    if unmatched:
        raise SystemExit(f"Unmatched city names: {', '.join(sorted(unmatched))}")

    crosswalk = pd.DataFrame(rows)
    merged = crosswalk.merge(
        sa4[["SA4_NAME21", "geometry"]],
        left_on="sa4_name",
        right_on="SA4_NAME21",
        how="left",
    )

    city_geom = gpd.GeoDataFrame(merged, geometry="geometry", crs=sa4.crs)
    city_geom = city_geom.dissolve(by="city", as_index=False)
    city_geom = city_geom[["city", "geometry"]]
    city_geom["geometry"] = city_geom["geometry"].simplify(
        tolerance=0.02, preserve_topology=True
    )
    return city_geom


def build_australia_outline(sa4: gpd.GeoDataFrame) -> dict:
    outline = sa4.dissolve()
    geom = outline.geometry.iloc[0]
    geom = geom.simplify(
        tolerance=AUSTRALIA_SIMPLIFY_TOLERANCE, preserve_topology=True
    )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Australia"},
                "geometry": geom.__geo_interface__,
            }
        ],
    }


def build_animation(
    df: pd.DataFrame,
    geojson: dict,
    location_col: str,
    feature_key: str,
    metric: str,
    title: str,
    color_scale: str,
    output_path: Path | None,
    midpoint: float | None = None,
    label_points: dict[str, tuple[float, float]] | None = None,
    label_names: list[str] | set[str] | None = None,
    label_font_size: int = 12,
    label_textposition: str = "middle center",
    range_percentiles: tuple[float, float] | None = None,
    border_geojson: dict | None = None,
    border_color: str = AUSTRALIA_BORDER_COLOR,
    border_width: float = AUSTRALIA_BORDER_WIDTH,
    write_html: bool = True,
) -> go.Figure:
    if metric not in df.columns:
        raise ValueError(f"Metric not found: {metric}")

    df = df[df[location_col].notna()].copy()
    df = add_week_str(df)

    weeks = sorted(df["iso_week_str"].unique(), key=lambda x: int(x))
    pivot = df.pivot_table(
        index=location_col,
        columns="iso_week_str",
        values=metric,
        aggfunc="mean",
    )

    locations = sorted(pivot.index.astype(str))
    pivot = pivot.reindex(index=locations, columns=weeks)

    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    metric_series = df[metric].dropna()
    if metric_series.empty:
        raise ValueError(f"No data for metric: {metric}")

    if range_percentiles:
        low, high = range_percentiles
        zmin = float(metric_series.quantile(low / 100))
        zmax = float(metric_series.quantile(high / 100))
    else:
        zmin = float(metric_series.min())
        zmax = float(metric_series.max())

    if midpoint is not None:
        span = max(abs(zmin - midpoint), abs(zmax - midpoint))
        zmin = midpoint - span
        zmax = midpoint + span

    base_week = weeks[0]
    base_z = pivot[base_week].tolist()

    data = [
        go.Choropleth(
            geojson=geojson,
            locations=locations,
            featureidkey=feature_key,
            z=base_z,
            colorscale=COLOR_SCALES[color_scale],
            zmin=zmin,
            zmax=zmax,
            zmid=midpoint,
            marker_line_width=0.5,
            marker_line_color="#FFFFFF",
            customdata=[base_week] * len(locations),
            hovertemplate=hover_template(metric, label),
        )
    ]

    if border_geojson:
        data.append(
            go.Choropleth(
                geojson=border_geojson,
                locations=["Australia"],
                featureidkey="properties.name",
                z=[1],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_line_color=border_color,
                marker_line_width=border_width,
                hoverinfo="skip",
            )
        )

    if label_points:
        if label_names:
            if isinstance(label_names, set):
                label_names = [loc for loc in locations if loc in label_names]
            else:
                label_names = [name for name in label_names if name in label_points]
        else:
            label_names = [loc for loc in locations if loc in label_points]

        label_lons = [label_points[name][0] for name in label_names]
        label_lats = [label_points[name][1] for name in label_names]
        data.append(
            go.Scattergeo(
                lon=label_lons,
                lat=label_lats,
                text=label_names,
                mode="text",
                textfont={"size": label_font_size, "color": "#1F1F1F"},
                textposition=label_textposition,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=data)

    frames = []
    for week in weeks:
        frames.append(
            go.Frame(
                name=week,
                data=[
                    go.Choropleth(
                        z=pivot[week].tolist(),
                        locations=locations,
                        customdata=[week] * len(locations),
                    )
                ],
                traces=[0],
            )
        )

    fig.frames = frames

    steps = [
        {
            "label": week,
            "method": "animate",
            "args": [
                [week],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for week in weeks
    ]

    fig.update_layout(
        title=title,
        height=650,
        width=1000,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        template="plotly_white",
        geo={"fitbounds": "locations", "visible": False, "projection_type": "mercator"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.01,
                "y": 0.05,
                "direction": "left",
                "pad": {"r": 10, "t": 0},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 350, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "ISO Week: "},
                "pad": {"t": 40},
                "steps": steps,
            }
        ],
    )

    if write_html and output_path is not None:
        fig.write_html(output_path, include_plotlyjs=INCLUDE_PLOTLYJS)
    return fig


def write_selector_html(
    figures: dict[str, go.Figure],
    output_path: Path,
    options: list[str],
    label_map: dict[str, str],
    descriptions: dict[str, str],
    title: str,
    select_label: str = "Metric",
) -> None:
    panels = []
    first = True
    for key in options:
        fig = figures.get(key)
        if fig is None:
            continue

        html = pio.to_html(
            fig,
            include_plotlyjs="cdn" if first else False,
            full_html=False,
            config={"responsive": True},
        )
        display = "block" if first else "none"
        panels.append(
            f'<div class="map-panel" id="panel-{key}" style="display:{display};">{html}</div>'
        )
        first = False

    option_html = "\n".join(
        [
            f'<option value="{key}">{label_map.get(key, key)}</option>'
            for key in options
            if key in figures
        ]
    )
    description_map = {
        key: descriptions.get(key, "")
        for key in options
        if key in figures
    }

    wrapper = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{ margin: 0; font-family: Arial, sans-serif; }}
      .toolbar {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-bottom: 1px solid #e5e5e5;
        background: #fafafa;
      }}
      .description {{
        padding: 0 16px 12px 16px;
        color: #333;
        font-size: 14px;
      }}
      .map-wrap {{ padding: 8px 0; }}
      .map-panel {{ width: 100%; }}
    </style>
  </head>
  <body>
    <div class="toolbar">
      <label for="metricSelect"><strong>{select_label}</strong></label>
      <select id="metricSelect">{option_html}</select>
    </div>
    <div class="description" id="metricDescription"></div>
    <div class="map-wrap">
      {''.join(panels)}
    </div>
    <script>
      const descriptions = {json.dumps(description_map)};
      function showPanel(metric) {{
        document.querySelectorAll('.map-panel').forEach(panel => {{
          panel.style.display = 'none';
        }});
        const panel = document.getElementById('panel-' + metric);
        if (!panel) return;
        panel.style.display = 'block';
        const plotDiv = panel.querySelector('.plotly-graph-div');
        if (plotDiv && window.Plotly) {{
          Plotly.Plots.resize(plotDiv);
        }}
        const desc = descriptions[metric] || '';
        const descEl = document.getElementById('metricDescription');
        if (descEl) descEl.textContent = desc;
      }}

      const select = document.getElementById('metricSelect');
      select.addEventListener('change', (event) => {{
        showPanel(event.target.value);
      }});

      // Default to the first option
      if (select.value) {{
        showPanel(select.value);
      }}
    </script>
  </body>
</html>
"""

    output_path.write_text(wrapper, encoding="utf-8")


def write_highlight_html(
    main_fig: go.Figure,
    highlight_fig: go.Figure,
    output_path: Path,
    highlight_heading: str = "HIGHLIGHT",
    highlight_note: str | None = None,
) -> None:
    main_html = pio.to_html(
        main_fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
    )
    highlight_html = pio.to_html(
        highlight_fig,
        include_plotlyjs=False,
        full_html=False,
        config={"responsive": True},
    )

    note_html = (
        f'<div class="highlight-note">{highlight_note}</div>' if highlight_note else ""
    )

    wrapper = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Q3 City Seasonality (Highlights)</title>
    <style>
      body {{ margin: 0; font-family: Arial, sans-serif; }}
      .section {{ padding: 12px 16px 0 16px; }}
      .section h2 {{ margin: 12px 0 6px 0; font-size: 18px; }}
      .highlight-note {{ margin: 0 0 12px 0; color: #444; font-size: 13px; }}
      .map-panel {{ width: 100%; }}
    </style>
  </head>
  <body>
    <div class="map-panel">{main_html}</div>
    <div class="section">
      <h2>{highlight_heading}</h2>
      {note_html}
    </div>
    <div class="map-panel">{highlight_html}</div>
  </body>
</html>
"""

    output_path.write_text(wrapper, encoding="utf-8")


def main() -> None:
    sales_for_seasonality = prepare_sales_for_seasonality()

    required_city_cols = {
        "seasonality_idx",
        "avg_gross",
        "gross_z",
        "avg_titles",
        "avg_cinemas",
        "opportunity_score",
    }

    if STATE_CSV.exists() and CITY_CSV.exists():
        state_df = pd.read_csv(STATE_CSV)
        city_df = pd.read_csv(CITY_CSV)
        if not required_city_cols.issubset(set(city_df.columns)):
            state_df, city_df = compute_seasonality_from_sales(sales_for_seasonality)
    else:
        state_df, city_df = compute_seasonality_from_sales(sales_for_seasonality)

    if not BASE_SA4_GEOJSON.exists():
        raise SystemExit("Missing base SA4 GeoJSON file")

    sa4 = gpd.read_file(BASE_SA4_GEOJSON)
    australia_outline = build_australia_outline(sa4)

    state_geojson_path = pick_geojson(
        STATE_GEOJSON, OUTPUT_DIR / "q3_state_geometry.geojson"
    )
    city_geojson_path = pick_geojson(
        CITY_GEOJSON, OUTPUT_DIR / "q3_city_sa4_geometry.geojson"
    )

    if state_geojson_path.exists():
        state_geojson = load_geojson(state_geojson_path)
    else:
        state_geom = build_state_geometry(sa4)
        state_geojson = gdf_to_geojson(state_geom)

    if city_geojson_path.exists():
        city_geojson = load_geojson(city_geojson_path)
    else:
        city_names = sorted(city_df["city"].dropna().astype(str).unique().tolist())
        city_geom = build_city_geometry(sa4, city_names)
        city_geojson = gdf_to_geojson(city_geom)

    state_labels = label_points_from_geojson(state_geojson, "state")

    state_maps = [
        {
            "metric": "gross_z",
            "title": "Q3 State Seasonality (Z-score) by ISO Week",
            "scale": "RdBu_r",
            "midpoint": 0.0,
            "output": OUTPUT_DIR / "q3_state_seasonality_gross_z_map.html",
        },
        {
            "metric": "avg_titles",
            "title": "Q3 State Average Titles by ISO Week",
            "scale": "YlOrRd",
            "midpoint": None,
            "output": OUTPUT_DIR / "q3_state_seasonality_avg_titles_map.html",
        },
        {
            "metric": "avg_cinemas",
            "title": "Q3 State Average Cinemas by ISO Week",
            "scale": "YlGn",
            "midpoint": None,
            "output": OUTPUT_DIR / "q3_state_seasonality_avg_cinemas_map.html",
        },
    ]

    state_figures: dict[str, go.Figure] = {}
    for spec in state_maps:
        fig = build_animation(
            state_df,
            state_geojson,
            location_col="state",
            feature_key="properties.state",
            metric=spec["metric"],
            title=spec["title"],
            color_scale=spec["scale"],
            output_path=spec["output"],
            midpoint=spec["midpoint"],
            label_points=state_labels,
        )
        state_figures[spec["metric"]] = fig

    write_selector_html(
        figures=state_figures,
        output_path=OUTPUT_DIR / "q3_state_seasonality_selector.html",
        options=["gross_z", "avg_titles", "avg_cinemas"],
        label_map=METRIC_LABELS,
        descriptions=METRIC_DESCRIPTIONS,
        title="Q3 State Seasonality Maps",
    )

    city_labels = label_points_from_geojson(city_geojson, "city")
    city_candidates = (
        city_df.groupby("city", as_index=False)["avg_gross"]
        .mean()
        .sort_values("avg_gross", ascending=False)
        ["city"]
        .tolist()
    )
    top_cities = select_spread_labels(
        city_candidates,
        city_labels,
        CITY_LABEL_MAX_COUNT,
        CITY_LABEL_MIN_DISTANCE_DEG,
    )

    city_peak_df = city_df.copy()
    peak_threshold = city_peak_df.groupby(["state", "city"])["seasonality_idx"].transform(
        lambda x: x.quantile(CITY_PEAK_QUANTILE)
    )
    low_threshold = city_peak_df.groupby(["state", "city"])["seasonality_idx"].transform(
        lambda x: x.quantile(CITY_LOW_QUANTILE)
    )
    city_peak_df["peak_low_score"] = 0.0
    city_peak_df.loc[
        city_peak_df["seasonality_idx"] >= peak_threshold, "peak_low_score"
    ] = 1.0
    city_peak_df.loc[
        city_peak_df["seasonality_idx"] <= low_threshold, "peak_low_score"
    ] = -1.0

    seasonality_fig = build_animation(
        city_df,
        city_geojson,
        location_col="city",
        feature_key="properties.city",
        metric="seasonality_idx",
        title=(
            "Q3 City Seasonality Index by ISO Week "
            f"(Top {len(top_cities)} labels)"
        ),
        color_scale="RdBu_r",
        output_path=None,
        midpoint=1.0,
        label_points=city_labels,
        label_names=top_cities,
        label_font_size=9,
        label_textposition="top center",
        range_percentiles=(5, 95),
        border_geojson=australia_outline,
        write_html=False,
    )

    peak_low_fig = build_animation(
        city_peak_df,
        city_geojson,
        location_col="city",
        feature_key="properties.city",
        metric="peak_low_score",
        title=(
            "Q3 City Peak/Low Weeks by ISO Week "
            f"(Top {len(top_cities)} labels)"
        ),
        color_scale="RdBu_r",
        output_path=None,
        midpoint=0.0,
        label_points=city_labels,
        label_names=top_cities,
        label_font_size=9,
        label_textposition="top center",
        range_percentiles=None,
        border_geojson=australia_outline,
        write_html=False,
    )

    write_highlight_html(
        main_fig=seasonality_fig,
        highlight_fig=peak_low_fig,
        output_path=OUTPUT_DIR / "q3_city_seasonality_index_map.html",
        highlight_heading="HIGHLIGHT",
        highlight_note=(
            "Peak/low weeks per city (top 10% = peak, bottom 10% = low)."
        ),
    )

    city_metric_specs = [
        {
            "metric": "seasonality_idx",
            "title": "Q3 City Seasonality Index by ISO Week",
            "scale": "RdBu_r",
            "midpoint": 1.0,
            "range_percentiles": (5, 95),
        },
        {
            "metric": "avg_gross",
            "title": "Q3 City Average Gross by ISO Week",
            "scale": "YlOrRd",
            "midpoint": None,
            "range_percentiles": (5, 95),
        },
        {
            "metric": "gross_z",
            "title": "Q3 City Gross Z-Score by ISO Week",
            "scale": "RdBu_r",
            "midpoint": 0.0,
            "range_percentiles": None,
        },
        {
            "metric": "avg_titles",
            "title": "Q3 City Average Titles by ISO Week",
            "scale": "YlOrRd",
            "midpoint": None,
            "range_percentiles": (5, 95),
        },
        {
            "metric": "avg_cinemas",
            "title": "Q3 City Average Cinemas by ISO Week",
            "scale": "YlGn",
            "midpoint": None,
            "range_percentiles": (5, 95),
        },
    ]

    city_figures: dict[str, go.Figure] = {}
    for spec in city_metric_specs:
        fig = build_animation(
            city_df,
            city_geojson,
            location_col="city",
            feature_key="properties.city",
            metric=spec["metric"],
            title=spec["title"],
            color_scale=spec["scale"],
            output_path=None,
            midpoint=spec["midpoint"],
            label_points=city_labels,
            label_names=top_cities,
            label_font_size=9,
            label_textposition="top center",
            range_percentiles=spec["range_percentiles"],
            border_geojson=australia_outline,
            write_html=False,
        )
        city_figures[spec["metric"]] = fig

    write_selector_html(
        figures=city_figures,
        output_path=OUTPUT_DIR / "q3_city_seasonality_selector.html",
        options=[spec["metric"] for spec in city_metric_specs],
        label_map=METRIC_LABELS,
        descriptions=METRIC_DESCRIPTIONS,
        title="Q3 City Seasonality Maps",
    )

    build_animation(
        city_df,
        city_geojson,
        location_col="city",
        feature_key="properties.city",
        metric="opportunity_score",
        title="Q3 City Opportunity Score by ISO Week",
        color_scale="RdBu_r",
        output_path=OUTPUT_DIR / "q3_city_opportunity_map.html",
        midpoint=0.0,
        label_points=city_labels,
        label_names=top_cities,
        label_font_size=9,
        label_textposition="top center",
        range_percentiles=(5, 95),
        border_geojson=australia_outline,
    )

    city_year_week = compute_city_week_by_year(sales_for_seasonality)
    years = sorted(city_year_week["iso_year"].dropna().unique().tolist())

    year_figures: dict[str, go.Figure] = {}
    for year in years:
        year_df = city_year_week[city_year_week["iso_year"] == year].copy()
        fig = build_animation(
            year_df,
            city_geojson,
            location_col="city",
            feature_key="properties.city",
            metric="seasonality_idx",
            title=f"Q3 City Seasonality Index by ISO Week ({year})",
            color_scale="RdBu_r",
            output_path=None,
            midpoint=1.0,
            label_points=city_labels,
            label_names=top_cities,
            label_font_size=9,
            label_textposition="top center",
            range_percentiles=(5, 95),
            border_geojson=australia_outline,
            write_html=False,
        )
        year_figures[str(year)] = fig

    year_labels = {str(year): f"ISO Year {year}" for year in years}
    year_descriptions = {
        str(year): "Seasonality index within the selected year."
        for year in years
    }

    write_selector_html(
        figures=year_figures,
        output_path=OUTPUT_DIR / "q3_city_seasonality_year_selector.html",
        options=[str(year) for year in years],
        label_map=year_labels,
        descriptions=year_descriptions,
        title="Q3 City Seasonality by Year",
        select_label="Year",
    )


if __name__ == "__main__":
    main()
