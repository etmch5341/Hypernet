"""
USGS Water Resources Data Loading Implementation

This module shows how to load and rasterize USGS water resources data
from shapefiles or GeoJSON files.

Supports:
- Aquifer zones
- Watershed boundaries
- Water quality/sensitivity zones
- Riparian buffers
- Wellhead protection areas
"""

from typing import Tuple, Optional
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pyproj import Transformer
from shapely.geometry import box


def load_usgs_water_resources(usgs_data_path: str,
                              bbox: Tuple[float, float, float, float],
                              resolution: float,
                              target_crs: str = "EPSG:3083",
                              sensitivity_field: Optional[str] = None) -> np.ndarray:
    """
    Load USGS water resources data from shapefile and rasterize to grid.
    
    Args:
        usgs_data_path: Path to USGS shapefile or GeoJSON
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution: Pixel resolution in meters
        target_crs: Target CRS for rasterization
        sensitivity_field: Name of field containing sensitivity values
                          (if None, will try common field names)
        
    Returns:
        2D array with sensitivity: 0=not_sensitive, 1=moderate, 2=high
    """
    print(f"Loading USGS water resources from {usgs_data_path}...")
    
    # Calculate grid dimensions
    width, height = _create_grid(bbox, resolution)
    
    # Create transformer
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    try:
        # Load shapefile
        print("  Reading shapefile...")
        gdf = gpd.read_file(usgs_data_path)
        
        print(f"  Loaded {len(gdf)} features")
        print(f"  Original CRS: {gdf.crs}")
        
        # Print available fields to help user
        print(f"  Available fields: {list(gdf.columns)}")
        
        # Reproject to target CRS if needed
        if gdf.crs != target_crs:
            print(f"  Reprojecting to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
        
        # Clip to bounding box
        bbox_geom = box(minx, miny, maxx, maxy)
        gdf = gdf[gdf.intersects(bbox_geom)]
        print(f"  {len(gdf)} features within bounding box")
        
        if len(gdf) == 0:
            print("  WARNING: No features found in bounding box!")
            return np.zeros((height, width), dtype=int)
        
        # Determine sensitivity field
        sensitivity_field = _identify_sensitivity_field(gdf, sensitivity_field)
        
        if sensitivity_field:
            print(f"  Using sensitivity field: '{sensitivity_field}'")
            # Classify geometries by sensitivity
            geometries_by_sensitivity = _classify_by_sensitivity(
                gdf, sensitivity_field
            )
        else:
            print("  No sensitivity field found, treating all as moderate sensitivity")
            # Treat all as moderate sensitivity
            geometries_by_sensitivity = {
                1: [(geom, 1) for geom in gdf.geometry]
            }
        
        # Create affine transform
        affine_transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        # Rasterize - start with no sensitivity
        sensitivity = np.zeros((height, width), dtype=np.uint8)
        
        # Rasterize each sensitivity level
        for sens_level in [0, 1, 2]:
            if sens_level in geometries_by_sensitivity:
                geoms = geometries_by_sensitivity[sens_level]
                if geoms:
                    print(f"    Rasterizing {len(geoms)} features at sensitivity {sens_level}...")
                    raster = rasterize(
                        geoms,
                        out_shape=(height, width),
                        transform=affine_transform,
                        fill=0,
                        dtype=np.uint8,
                        all_touched=True  # Include pixels touched by geometry
                    )
                    sensitivity[raster == sens_level] = sens_level
        
        # Statistics
        no_sens_pct = np.sum(sensitivity == 0) / sensitivity.size * 100
        mod_sens_pct = np.sum(sensitivity == 1) / sensitivity.size * 100
        high_sens_pct = np.sum(sensitivity == 2) / sensitivity.size * 100
        
        print(f"\nWater resource sensitivity distribution:")
        print(f"  Not sensitive: {no_sens_pct:.1f}%")
        print(f"  Moderate sensitivity: {mod_sens_pct:.1f}%")
        print(f"  High sensitivity: {high_sens_pct:.1f}%")
        
        return sensitivity
        
    except Exception as e:
        print(f"ERROR loading USGS data: {e}")
        print("Returning zeros (no sensitivity)")
        return np.zeros((height, width), dtype=int)


def _identify_sensitivity_field(gdf: gpd.GeoDataFrame, 
                                user_field: Optional[str]) -> Optional[str]:
    """
    Identify which field contains sensitivity information.
    
    Tries user-provided field first, then common field names.
    """
    if user_field and user_field in gdf.columns:
        return user_field
    
    # Common field names in USGS water resources data
    common_fields = [
        'sensitivity', 'SENSITIVITY', 'Sensitivity',
        'sens', 'SENS',
        'priority', 'PRIORITY', 'Priority',
        'class', 'CLASS', 'Class', 'classification', 'CLASSIFICATION',
        'category', 'CATEGORY', 'Category',
        'rank', 'RANK', 'Rank',
        'zone', 'ZONE', 'Zone',
        'type', 'TYPE', 'Type'
    ]
    
    for field in common_fields:
        if field in gdf.columns:
            return field
    
    return None


def _classify_by_sensitivity(gdf: gpd.GeoDataFrame, 
                             sensitivity_field: str) -> dict:
    """
    Classify geometries into sensitivity levels (0, 1, 2).
    
    Handles various field types: numeric, text categories, etc.
    """
    geometries = {0: [], 1: [], 2: []}
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        value = row[sensitivity_field]
        
        # Determine sensitivity level based on value type
        sens_level = _parse_sensitivity_value(value)
        
        geometries[sens_level].append((geom, sens_level))
    
    return geometries


def _parse_sensitivity_value(value) -> int:
    """
    Parse sensitivity value from various formats to 0, 1, or 2.
    
    Handles:
    - Numeric: 0, 1, 2 or 1-5 scale
    - Text: 'low', 'moderate', 'high', 'none', etc.
    - Boolean: True/False
    """
    # Handle None/NaN
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    
    # Convert to string for text matching
    value_str = str(value).lower().strip()
    
    # Text categories
    if any(word in value_str for word in ['high', 'critical', 'severe', 'priority 1']):
        return 2
    elif any(word in value_str for word in ['moderate', 'medium', 'priority 2']):
        return 1
    elif any(word in value_str for word in ['low', 'minimal', 'none', 'priority 3']):
        return 0
    
    # Try numeric
    try:
        num_value = float(value)
        
        # Scale 0-2 directly
        if 0 <= num_value <= 2:
            return int(round(num_value))
        
        # Scale 1-3
        elif 1 <= num_value <= 3:
            return int(round(num_value)) - 1
        
        # Scale 1-5 (common in USGS data)
        elif 1 <= num_value <= 5:
            if num_value <= 2:
                return 0  # Low (1-2)
            elif num_value <= 3.5:
                return 1  # Moderate (3)
            else:
                return 2  # High (4-5)
        
        # Boolean as number
        elif num_value == 0:
            return 0
        elif num_value == 1:
            return 1
        
    except (ValueError, TypeError):
        pass
    
    # Default to moderate if unclear
    return 1


def _create_grid(bbox: Tuple[float, float, float, float], 
                resolution: float) -> Tuple[int, int]:
    """Calculate grid dimensions from bbox and resolution."""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    lat_center = (min_lat + max_lat) / 2
    lon_distance = (max_lon - min_lon) * 111000 * np.cos(np.radians(lat_center))
    lat_distance = (max_lat - min_lat) * 111000
    
    width = int(np.ceil(lon_distance / resolution))
    height = int(np.ceil(lat_distance / resolution))
    
    return width, height


# =============================================================================
# INTEGRATION INTO extract_water_resources()
# =============================================================================

def extract_water_resources_COMPLETE(bbox: Tuple[float, float, float, float], 
                                    resolution: float,
                                    target_crs: str = "EPSG:3083",
                                    water_bodies_array: Optional[np.ndarray] = None,
                                    elevation_array: Optional[np.ndarray] = None,
                                    usgs_data_path: Optional[str] = None,
                                    sensitivity_field: Optional[str] = None) -> np.ndarray:
    """
    Complete implementation with USGS data loading.
    
    Replace the TODO section in your extract_water_resources() function with this.
    """
    print("Extracting water resource sensitivity...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Try USGS water resources data if provided
    if usgs_data_path is not None:
        sensitivity = load_usgs_water_resources(
            usgs_data_path,
            bbox,
            resolution,
            target_crs,
            sensitivity_field
        )
        
        # If USGS loading succeeded, return it
        if sensitivity is not None and sensitivity.max() > 0:
            return sensitivity
        else:
            print("    USGS data loading failed, falling back to estimation")
    
    # Fallback: Estimate from water proximity
    print("Estimating water resource sensitivity from proximity to water bodies...")
    
    from scipy.ndimage import distance_transform_edt
    
    # Get water bodies if not provided
    if water_bodies_array is None:
        water_bodies_array = np.zeros((height, width), dtype=np.uint8)
        print("    (No water bodies data provided, sensitivity will be low)")
    
    # Calculate distance from water bodies
    distance_from_water = distance_transform_edt(1 - water_bodies_array)
    distance_meters = distance_from_water * resolution
    
    # Define sensitivity zones
    sensitivity = np.zeros((height, width), dtype=int)
    sensitivity[distance_meters <= 200] = 1  # Moderate
    sensitivity[distance_meters <= 50] = 2   # High
    
    # Identify potential recharge areas if elevation available
    if elevation_array is not None:
        elev_range = elevation_array.max() - elevation_array.min()
        if elev_range > 0:
            normalized_elev = (elevation_array - elevation_array.min()) / elev_range
            
            high_elev_mask = normalized_elev > 0.7
            near_water_mask = distance_meters < 500
            recharge_zone = high_elev_mask & near_water_mask
            
            sensitivity[recharge_zone & (sensitivity == 0)] = 1
            sensitivity[recharge_zone & (sensitivity == 1)] = 2
    
    no_sens_pct = np.sum(sensitivity == 0) / sensitivity.size * 100
    mod_sens_pct = np.sum(sensitivity == 1) / sensitivity.size * 100
    high_sens_pct = np.sum(sensitivity == 2) / sensitivity.size * 100
    
    print(f"Water resource sensitivity distribution:")
    print(f"  Not sensitive: {no_sens_pct:.1f}%")
    print(f"  Moderate sensitivity: {mod_sens_pct:.1f}%")
    print(f"  High sensitivity: {high_sens_pct:.1f}%")
    print(f"  (Note: This is estimated. Provide usgs_data_path for accurate data)")
    
    return sensitivity


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """Example usage of USGS water resources loading."""
    
    print("="*80)
    print("USGS WATER RESOURCES DATA LOADING - EXAMPLES")
    print("="*80)
    print()
    
    # Example 1: Load from USGS shapefile
    print("Example 1: Loading from USGS shapefile")
    print("-" * 40)
    
    bbox = (-98.0, 29.0, -97.5, 29.5)
    resolution = 100
    
    # Path to USGS water resources shapefile
    # This could be:
    # - Aquifer boundaries
    # - Wellhead protection areas
    # - Watershed boundaries with sensitivity ratings
    usgs_path = "path/to/usgs_water_resources.shp"
    
    sensitivity = load_usgs_water_resources(
        usgs_data_path=usgs_path,
        bbox=bbox,
        resolution=resolution,
        target_crs="EPSG:3083",
        sensitivity_field="SENSITIVITY"  # Or None to auto-detect
    )
    
    print()
    
    # Example 2: Auto-detect sensitivity field
    print("Example 2: Auto-detecting sensitivity field")
    print("-" * 40)
    
    # If you don't know the field name, pass None
    # The function will try common names:
    # - 'sensitivity', 'SENSITIVITY'
    # - 'priority', 'PRIORITY'
    # - 'class', 'CLASS'
    # - etc.
    
    sensitivity = load_usgs_water_resources(
        usgs_data_path=usgs_path,
        bbox=bbox,
        resolution=resolution,
        target_crs="EPSG:3083",
        sensitivity_field=None  # Auto-detect
    )
    
    print()
    
    # Example 3: Handle different sensitivity formats
    print("Example 3: Sensitivity value formats")
    print("-" * 40)
    
    print("The function handles various formats:")
    print("  Numeric 0-2: 0, 1, 2 → 0, 1, 2")
    print("  Numeric 1-5: 1-2 → 0, 3 → 1, 4-5 → 2")
    print("  Text: 'Low' → 0, 'Moderate' → 1, 'High' → 2")
    print("  Text: 'Priority 1' → 2, 'Priority 2' → 1, 'Priority 3' → 0")
    
    # Test parsing
    test_values = [
        0, 1, 2,  # Numeric 0-2
        1, 3, 5,  # Numeric 1-5
        'Low', 'Moderate', 'High',  # Text
        'Priority 1', 'Priority 2', 'Priority 3',  # Priority
        None, np.nan  # Missing values
    ]
    
    print("\nParsing examples:")
    for val in test_values:
        parsed = _parse_sensitivity_value(val)
        print(f"  {val!r:20s} → {parsed}")
    
    print()
    print("="*80)