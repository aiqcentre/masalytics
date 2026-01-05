# Masalytics: AI Coding Instructions

**Masalytics** analyzes Indian film performance in Australia to help Film Viet Australia optimize scheduling and marketing for Vietnamese releases. This document guides AI agents on the codebase architecture, data flows, and project conventions.

## Architecture Overview

### Data Flow
```
data/numero_data.sqlite (raw source)
    ↓
DataExplorationMain.py (root ETL module)
    ├─ Loads sales_raw_data (JSON-serialized box office)
    ├─ Loads film_metadata & indian_titles
    ├─ Flattens JSON → daily sales records
    ├─ Matches Indian titles via normalized title comparison
    └─ Exports: sales, sales_indian, film_meta, conn
    ↓
[Analysis Scripts] (import * from DataExplorationMain)
    ├─ SalesOverview.py: Top films, monthly/daily trends
    ├─ LocationQuestions.py: Stable vs volatile locations (Q1, Q2, Q3)
    ├─ TitlesDistributors.py: Distributor networks & footprint
    └─ Q2_mapping_step*.py: Geographic SA4 mapping
    ↓
outputs_*/ (HTML dashboards, CSVs)
```

**Key principle**: `DataExplorationMain` is the single ETL gateway. All analysis scripts import from it via `from DataExplorationMain import *`. Never bypass this—it ensures consistent data preprocessing.

### Database Schema

SQLite database (`data/numero_data.sqlite`) contains:
- **`film_metadata`**: `numero_film_id`, `title` (all films in system)
- **`sales_raw_data`**: `numero_film_id`, `raw_json` (nested JSON by week + cinema)
- **`indian_titles`**: `title`, `distributor` (reference list of Indian releases)

Raw JSON structure (nested by week):
```json
{
  "2025-01-10": {
    "rows": [
      {
        "state": "NSW", "city": "Sydney", "theatre": "Cineplex Broadway",
        "boxOffice": {
          "day1": {"today": 15000}, "day2": {"today": 14000}, ...
        }
      }
    ]
  }
}
```

## Project-Specific Patterns

### 1. Data Matching & Normalization
**Pattern**: Titles are matched via normalized lowercase comparison to handle inconsistent capitalization.

```python
def norm_title(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()
```

**When to use**: Anytime you merge film titles across sources. Always normalize before comparing.

### 2. JSON Flattening to Daily Records
**Pattern**: Box office JSON is nested by week and cinema. `flatten_one_film()` and `flatten_sales_json()` convert this into a flat DataFrame with one row per cinema per day.

**Key fields in flattened output**:
- `numero_film_id`: Film ID (integer)
- `actual_sales_date`: Date of sales (datetime)
- `state`, `city`, `theatre_name`: Location hierarchy
- `gross_today`: Daily gross revenue

**When modifying**: If the JSON structure changes, edit `flatten_one_film()` and `flatten_sales_json()` in `DataExplorationMain.py`.

### 3. Temporal Aggregation
**Pattern**: Weekly aggregation using `pd.Period('W')` for consistency across analyses:

```python
sales['week_start'] = (
    sales['actual_sales_date']
      .dt.to_period('W')
      .apply(lambda r: r.start_time)
      .dt.date
)
weekly_gross = sales.groupby(['week_start', 'state', 'city'])['gross_today'].sum()
```

**Always use this pattern for weekly rollups** to ensure week boundaries are consistent.

### 4. Geographic Hierarchy: SA4 Regions
**Pattern**: Spatial analysis uses SA4 (Statistical Area Level 4) boundaries from shapefiles:
- `data/SA4_2021_AUST_GDA94.shp` (main shapefile)
- Loaded via `geopandas.read_file()` in mapping scripts
- Coordinate system: GDA94 (EPSG:4283)

**Q2 mapping workflow**:
1. `Q2_mapping_step1_explore_sa4.py`: Load and explore SA4 structure
2. `Q2_mapping_step2_extract_q2_data.py`: Map cinema locations to SA4, aggregate metrics
3. `Q2_Mapping_Step_by_Step.ipynb`: Interactive visualization

**When adding geographic analysis**: Keep cinema-to-SA4 mapping separate from temporal analysis for reusability.

### 5. Output Organization
**Pattern**: Results are organized by analysis type in output directories:

```
outputs_salesoverview/      → top10_films.png, monthly trends
outputs_locationquestions/  → Q1/Q2/Q3 stability analysis, heatmaps
outputs_titlesdistributors/ → distributor treemaps, Pareto charts
```

**Convention**: Use descriptive filenames like `q1_cinemas_revenue_boxplot_interactive.html`, not generic names.

### 6. Analysis Workflow: Questions-Driven
Each major analysis is a question (Q1, Q2, Q3):

- **Q1**: Stable vs volatile locations (by cinema and city)
- **Q2**: Temporal & geographic patterns (SA4 mapping, release timing)
- **Q3**: Distributor strategies and market control

Each has:
- A standalone Python script or notebook
- An explanation document (`Q*_ANALYSIS_EXPLANATION.md`) describing methodology and insights
- Output visualizations in the corresponding output directory

**When adding new analysis**: Create a new Qn_*, write an explanation, and save outputs to `outputs_qn/`.

## Critical Workflows

### Running Analysis End-to-End
1. Ensure `data/numero_data.sqlite` exists (not included in repo)
2. Run or import `DataExplorationMain.py` first—it validates data connectivity and exports global variables
3. Run downstream analysis scripts (e.g., `python SalesOverview.py`)
4. Check for errors in console output and validate output files are created

### Debugging Data Issues
- **Missing data**: Check `DataExplorationMain.py` output for shape/nulls
- **Title mismatches**: Verify normalization logic; check `indian_titles` table
- **Sales anomalies**: Inspect raw JSON in `sales_raw_data` (may need `json.loads()`)
- **Location gaps**: Cross-reference `state`, `city`, `theatre_name` columns; some cinemas may be missing metadata

### Adding New Analysis
1. Import data from `DataExplorationMain` (e.g., `from DataExplorationMain import sales, film_meta, conn`)
2. Aggregate as needed (weekly, by state, by cinema)
3. Save outputs to `outputs_<new_analysis_name>/` directory
4. Create `<NEW_QUESTION>_ANALYSIS_EXPLANATION.md` with methodology and findings

## Key Files Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `DataExplorationMain.py` | ETL gateway; loads DB, flattens JSON, exports common vars | Data source changes, new fields needed |
| `SalesOverview.py` | Top films, revenue trends | Adding new film-level metrics |
| `LocationQuestions.py` | Q1/Q2/Q3 location stability analysis | Changing risk metrics, output formats |
| `TitlesDistributors.py` | Distributor analysis & visualizations | Adding distributor insights |
| `Q2_mapping_step*.py` | Geographic (SA4) aggregation | Changing SA4 mapping logic |
| `data/numero_data.sqlite` | SQLite source database | (external; managed by data team) |
| `data/SA4_*.shp` | Spatial boundaries | (external; use as-is) |

## Dependencies & Environment

**Key packages**:
- `pandas`: Data manipulation
- `sqlite3`: Database access
- `geopandas`: Spatial analysis (SA4 mapping)
- `plotly`: Interactive visualizations
- `matplotlib`, `seaborn`: Static plots
- `numpy`: Numerical operations

**Database**: SQLite file at `data/numero_data.sqlite` (not versioned in Git).

## Common Pitfalls

1. **Forgetting to import from DataExplorationMain**: Always use `from DataExplorationMain import *` to ensure consistent preprocessing
2. **Not normalizing titles**: Title matching fails silently; always use `norm_title()` when comparing
3. **Inconsistent week boundaries**: Use the `pd.Period('W')` pattern, not custom aggregations
4. **Hardcoding file paths**: Use `Path()` or relative paths; maintain consistency with existing scripts
5. **Ignoring null values**: The JSON flattening may produce nulls; check with `.isnull().sum()` before analysis

## Questions or Patterns Not Clear?

Refer to existing analysis scripts (e.g., `SalesOverview.py`, `LocationQuestions.py`) for working examples. Check `*_ANALYSIS_EXPLANATION.md` files for methodology documentation.
