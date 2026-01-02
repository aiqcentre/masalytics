"""
STEP 1: Load and Explore SA4 Boundaries
========================================
This script loads the SA4 shapefile and explores its structure to understand
the data available for mapping Australian cinema adoption patterns.
"""

import geopandas as gpd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# STEP 1A: Load SA4 Shapefile
# ==============================================================================
print("=" * 80)
print("STEP 1: Loading SA4 Boundaries")
print("=" * 80)

sa4_path = "data/SA4_2021_AUST_GDA94.shp"
print(f"\nLoading shapefile from: {sa4_path}")

try:
    sa4_gdf = gpd.read_file(sa4_path)
    print("✓ Successfully loaded SA4 boundaries")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    exit(1)

# ==============================================================================
# STEP 1B: Explore Structure
# ==============================================================================
print("\n" + "=" * 80)
print("Basic Information")
print("=" * 80)

print(f"\nShape: {sa4_gdf.shape[0]} regions, {sa4_gdf.shape[1]} columns")
print(f"Coordinate System: {sa4_gdf.crs}")
print(f"Bounds: {sa4_gdf.total_bounds}")

# ==============================================================================
# STEP 1C: Column Names and Data Types
# ==============================================================================
print("\n" + "=" * 80)
print("Columns and Data Types")
print("=" * 80)

print(f"\nColumn Names ({len(sa4_gdf.columns)} total):")
for i, col in enumerate(sa4_gdf.columns, 1):
    print(f"  {i:2d}. {col:<20} ({sa4_gdf[col].dtype})")

# ==============================================================================
# STEP 1D: Sample Data
# ==============================================================================
print("\n" + "=" * 80)
print("First 5 Regions (Sample Data)")
print("=" * 80)

# Show key columns (exclude geometry for readability)
display_cols = [col for col in sa4_gdf.columns if col != 'geometry']
print(f"\n{sa4_gdf[display_cols].head(10).to_string()}")

# ==============================================================================
# STEP 1E: Identify Key Columns for Mapping
# ==============================================================================
print("\n" + "=" * 80)
print("Key Columns for Mapping")
print("=" * 80)

# List unique values for identification columns
print("\nUnique SA4 identification values:")
for col in display_cols:
    n_unique = sa4_gdf[col].nunique()
    if n_unique <= 120:  # Only show columns with reasonable number of unique values
        print(f"\n  {col}: {n_unique} unique values")
        if n_unique <= 20:
            print(f"    Examples: {sa4_gdf[col].unique()[:5].tolist()}")

# ==============================================================================
# STEP 1F: Check Geometry Validity
# ==============================================================================
print("\n" + "=" * 80)
print("Geometry Quality Check")
print("=" * 80)

valid_geoms = sa4_gdf.geometry.is_valid.sum()
total_geoms = len(sa4_gdf)
print(f"\nValid geometries: {valid_geoms}/{total_geoms}")

if valid_geoms < total_geoms:
    print("⚠ Warning: Some geometries are invalid")
    sa4_gdf = sa4_gdf[sa4_gdf.geometry.is_valid]
    print(f"  Removed {total_geoms - valid_geoms} invalid geometries")

# ==============================================================================
# STEP 1G: Summary Statistics
# ==============================================================================
print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

print(f"\nTotal SA4 regions: {len(sa4_gdf)}")
print(f"Total area: {sa4_gdf.geometry.area.sum():,.0f} square units")
print(f"Average area per region: {sa4_gdf.geometry.area.mean():,.0f} square units")

# ==============================================================================
# SAVE CLEANED DATA
# ==============================================================================
print("\n" + "=" * 80)
print("Saving Cleaned Data")
print("=" * 80)

output_file = "data/SA4_2021_AUST_GDA94_cleaned.geojson"
sa4_gdf.to_file(output_file, driver="GeoJSON")
print(f"\n✓ Saved cleaned SA4 data to: {output_file}")

# ==============================================================================
# NEXT STEPS
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 1 COMPLETE")
print("=" * 80)
print("""
Next steps:
  1. Review the columns above to identify the key SA4 identifier
  2. Run STEP 2 to extract Q2 cinema data
  3. We'll match cinemas to SA4 regions using location information
  
Key findings to remember:
  - SA4 has {} unique regions
  - Coordinate system: {}
  - All geometries are valid
""".format(len(sa4_gdf), sa4_gdf.crs))

print("\n✓ Ready for STEP 2: Extract Q2 cinema data\n")
