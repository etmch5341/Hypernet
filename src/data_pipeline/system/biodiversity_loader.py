"""
Biodiversity and Conservation Database Loading Implementation

Supports multiple biodiversity data formats:
1. Species richness rasters (continuous biodiversity scores)
2. Critical habitat shapefiles (USFWS, NatureServe)
3. Biodiversity hotspot zones (Conservation International)
4. Species occurrence points (GBIF, eBird) - converted to density
5. Multiple overlapping datasets (combined score)

Common sources:
- USFWS Critical Habitat: https://ecos.fws.gov/ecp/report/table/critical-habitat.html
- NatureServe: https://www.natureserve.org/
- GBIF (species occurrences): https://www.gbif.org/
- eBird: https://ebird.org/
- Conservation International: https://www.conservation.org/
"""

from typing import Tuple, Optional, List, Dict
import numpy as np
from pathlib import Path
import rasterio
import geopandas as gpd




def load_biodiversity_data(conservation_db_path: str,
                           bbox: Tuple[float, float, float, float],
                           resolution: float,
                           target_crs: str = "EPSG:3083",
                           data_type: Optional[str] = None) -> np.ndarray:
    """
    Load biodiversity data from various formats and return 0-1 biodiversity score.
    
    Auto-detects format from file extension if data_type not specified.
    
    Args:
        conservation_db_path: Path to biodiversity data file/directory
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution: Pixel resolution in meters
        target_crs: Target CRS
        data_type: Force specific type: 'raster', 'shapefile', 'points', 'directory'
        
    Returns:
        2D array with biodiversity score 0-1 (0=low, 1=high)
    """
    print(f"Loading biodiversity data from {conservation_db_path}...")
    
    # Auto-detect data type
    path = Path(conservation_db_path)
    
    if data_type is None:
        if path.is_dir():
            data_type = 'directory'
        elif path.suffix.lower() in ['.tif', '.tiff', '.img']:
            data_type = 'raster'
        elif path.suffix.lower() in ['.shp', '.geojson', '.gpkg']:
            data_type = 'shapefile'
        elif path.suffix.lower() in ['.csv', '.txt']:
            data_type = 'points'
        else:
            print(f"  WARNING: Unknown file type {path.suffix}")
            return None
    
    print(f"  Detected data type: {data_type}")
    
    # Load based on type
    if data_type == 'raster':
        return load_biodiversity_raster(
            conservation_db_path, bbox, resolution, target_crs
        )
    
    elif data_type == 'shapefile':
        return load_biodiversity_shapefile(
            conservation_db_path, bbox, resolution, target_crs
        )
    
    elif data_type == 'points':
        return load_biodiversity_points(
            conservation_db_path, bbox, resolution, target_crs
        )
    
    elif data_type == 'directory':
        return load_biodiversity_directory(
            conservation_db_path, bbox, resolution, target_crs
        )
    
    else:
        print(f"  ERROR: Unknown data type '{data_type}'")
        return None


# =============================================================================
# 1. RASTER DATA (Species Richness Maps)
# =============================================================================

def load_biodiversity_raster(raster_path: str,
                            bbox: Tuple[float, float, float, float],
                            resolution: float,
                            target_crs: str) -> np.ndarray:
    """
    Load species richness or biodiversity index raster.
    
    Common sources:
    - Global species richness maps
    - Regional biodiversity indices
    - Habitat suitability models
    
    Returns continuous 0-1 biodiversity score.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    
    print(f"  Loading biodiversity raster...")
    
    # Calculate grid dimensions
    width, height = _create_grid(bbox, resolution)
    
    # Transform bbox
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    with rasterio.open(raster_path) as src:
        print(f"    Source CRS: {src.crs}")
        print(f"    Source bounds: {src.bounds}")
        print(f"    Source resolution: {src.res}")
        
        # Read and reproject
        biodiversity_raw = np.empty((height, width), dtype=np.float32)
        
        dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=biodiversity_raw,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear  # Bilinear for continuous data
        )
        
        # Normalize to 0-1
        biodiversity = _normalize_biodiversity(biodiversity_raw)
        
        print(f"    Biodiversity range: {biodiversity.min():.3f} to {biodiversity.max():.3f}")
        print(f"    Mean: {biodiversity.mean():.3f}")
        
        return biodiversity


# =============================================================================
# 2. SHAPEFILE DATA (Critical Habitat, Conservation Zones)
# =============================================================================

def load_biodiversity_shapefile(shapefile_path: str,
                                bbox: Tuple[float, float, float, float],
                                resolution: float,
                                target_crs: str) -> np.ndarray:
    """
    Load critical habitat or biodiversity zones from shapefile.
    
    Common sources:
    - USFWS Critical Habitat designations
    - NatureServe biodiversity areas
    - State/local conservation priorities
    
    Returns 0-1 biodiversity score based on zone importance.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    
    print(f"  Loading biodiversity shapefile...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"    Loaded {len(gdf)} features")
    print(f"    Available fields: {list(gdf.columns)}")
    
    # Reproject
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    
    # Clip to bbox
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    bbox_geom = box(minx, miny, maxx, maxy)
    gdf = gdf[gdf.intersects(bbox_geom)]
    print(f"    {len(gdf)} features in bounding box")
    
    if len(gdf) == 0:
        print("    WARNING: No features in bounding box!")
        return np.zeros((height, width), dtype=float)
    
    # Identify importance/priority field
    importance_field = _identify_importance_field(gdf)
    
    if importance_field:
        print(f"    Using importance field: '{importance_field}'")
        # Classify by importance
        geometries = _classify_by_importance(gdf, importance_field)
    else:
        print("    No importance field found, using uniform high biodiversity")
        # Treat all as high biodiversity (0.8)
        geometries = [(geom, 0.8) for geom in gdf.geometry]
    
    # Rasterize
    dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # For continuous values, need to handle differently
    biodiversity = np.zeros((height, width), dtype=float)
    
    # Rasterize with max value (if overlapping features)
    if isinstance(geometries, list):
        # List of (geom, value) tuples
        for i in range(0, len(geometries), 100):  # Process in batches
            batch = geometries[i:i+100]
            raster = rasterize(
                batch,
                out_shape=(height, width),
                transform=dst_transform,
                fill=0,
                dtype=float,
                merge_alg=rasterio.enums.MergeAlg.add  # Add overlapping values
            )
            biodiversity = np.maximum(biodiversity, raster)
    else:
        # Dictionary by class
        for value in sorted(geometries.keys()):
            geoms = geometries[value]
            if geoms:
                raster = rasterize(
                    geoms,
                    out_shape=(height, width),
                    transform=dst_transform,
                    fill=0,
                    dtype=float
                )
                biodiversity[raster > 0] = value
    
    # Ensure 0-1 range
    biodiversity = np.clip(biodiversity, 0, 1)
    
    print(f"    Biodiversity range: {biodiversity.min():.3f} to {biodiversity.max():.3f}")
    print(f"    High biodiversity (>0.7): {np.sum(biodiversity > 0.7) / biodiversity.size * 100:.1f}%")
    
    return biodiversity


def _identify_importance_field(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Identify field containing biodiversity importance/priority."""
    common_fields = [
        # Generic
        'importance', 'IMPORTANCE', 'priority', 'PRIORITY',
        'value', 'VALUE', 'score', 'SCORE',
        
        # USFWS Critical Habitat
        'STATUS', 'status', 'CRITHABITAT', 'critical',
        
        # NatureServe
        'GRANK', 'SRANK', 'EORANKID', 'biodiv_rank',
        
        # Conservation priorities
        'conservation_priority', 'cons_prior', 'CONS_VALUE'
    ]
    
    for field in common_fields:
        if field in gdf.columns:
            return field
    
    return None


def _classify_by_importance(gdf: gpd.GeoDataFrame, 
                            importance_field: str) -> List[Tuple]:
    """
    Classify features by importance into 0-1 biodiversity scores.
    
    Handles:
    - Text: 'Critical', 'High', 'Medium', 'Low'
    - Numeric: 1-5 scales, etc.
    - Conservation ranks: G1-G5, S1-S5
    """
    geometries = []
    
    for idx, row in gdf.iterrows():
        importance = row[importance_field]
        geom = row.geometry
        
        # Parse importance to 0-1 score
        score = _parse_importance_value(importance)
        
        geometries.append((geom, score))
    
    return geometries


def _parse_importance_value(value) -> float:
    """Parse importance value to 0-1 biodiversity score."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5  # Default moderate
    
    value_str = str(value).lower().strip()
    
    # Text categories
    if 'critical' in value_str or 'essential' in value_str:
        return 0.95
    elif 'high' in value_str or 'very important' in value_str:
        return 0.8
    elif 'moderate' in value_str or 'medium' in value_str:
        return 0.6
    elif 'low' in value_str or 'minimal' in value_str:
        return 0.3
    
    # Conservation ranks (G1 = critically imperiled, G5 = secure)
    if value_str.startswith('g') or value_str.startswith('s'):
        try:
            rank = int(value_str[1])
            # G1/S1 = 0.95, G5/S5 = 0.2
            return 1.1 - (rank * 0.18)
        except (ValueError, IndexError):
            pass
    
    # Numeric scales
    try:
        num = float(value)
        
        # 1-5 scale (1=highest)
        if 1 <= num <= 5:
            return 1.1 - (num * 0.18)
        
        # 0-100 scale
        elif 0 <= num <= 100:
            return num / 100.0
        
        # 0-1 scale (already normalized)
        elif 0 <= num <= 1:
            return num
            
    except (ValueError, TypeError):
        pass
    
    # Default
    return 0.5


# =============================================================================
# 3. POINT DATA (Species Occurrences)
# =============================================================================

def load_biodiversity_points(points_path: str,
                            bbox: Tuple[float, float, float, float],
                            resolution: float,
                            target_crs: str,
                            kernel_size: int = 500) -> np.ndarray:
    """
    Load species occurrence points and convert to biodiversity density.
    
    Common sources:
    - GBIF (Global Biodiversity Information Facility)
    - eBird observations
    - iNaturalist records
    
    Args:
        points_path: CSV file with columns: latitude, longitude, species (optional)
        kernel_size: Kernel radius in meters for density estimation
        
    Returns 0-1 biodiversity score based on species density.
    """
    import pandas as pd
    from scipy.ndimage import gaussian_filter
    from pyproj import Transformer
    
    print(f"  Loading species occurrence points...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Load points
    df = pd.read_csv(points_path)
    print(f"    Loaded {len(df)} occurrence records")
    
    # Identify lat/lon columns
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'lat' in col_lower and lat_col is None:
            lat_col = col
        if 'lon' in col_lower or 'lng' in col_lower and lon_col is None:
            lon_col = col
    
    if lat_col is None or lon_col is None:
        print("    ERROR: Could not find latitude/longitude columns!")
        print(f"    Available columns: {list(df.columns)}")
        return np.zeros((height, width), dtype=float)
    
    print(f"    Using columns: lat='{lat_col}', lon='{lon_col}'")
    
    # Filter to bounding box
    min_lon, min_lat, max_lon, max_lat = bbox
    df = df[
        (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
        (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
    ]
    print(f"    {len(df)} records in bounding box")
    
    if len(df) == 0:
        return np.zeros((height, width), dtype=float)
    
    # Transform points to target CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    # Convert lat/lon to pixel coordinates
    x_coords, y_coords = transformer.transform(
        df[lon_col].values, df[lat_col].values
    )
    
    # Convert to pixel indices
    pixel_x = ((x_coords - minx) / (maxx - minx) * width).astype(int)
    pixel_y = ((maxy - y_coords) / (maxy - miny) * height).astype(int)
    
    # Remove out-of-bounds
    valid = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
    pixel_x = pixel_x[valid]
    pixel_y = pixel_y[valid]
    
    # Create occurrence density map
    density = np.zeros((height, width), dtype=float)
    
    for x, y in zip(pixel_x, pixel_y):
        density[y, x] += 1
    
    # Apply Gaussian kernel to spread influence
    sigma = kernel_size / resolution  # Convert meters to pixels
    density_smooth = gaussian_filter(density, sigma=sigma)
    
    # Normalize to 0-1
    if density_smooth.max() > 0:
        biodiversity = density_smooth / density_smooth.max()
    else:
        biodiversity = np.zeros_like(density_smooth)
    
    print(f"    Biodiversity range: {biodiversity.min():.3f} to {biodiversity.max():.3f}")
    print(f"    High biodiversity (>0.7): {np.sum(biodiversity > 0.7) / biodiversity.size * 100:.1f}%")
    
    return biodiversity


# =============================================================================
# 4. DIRECTORY (Multiple Datasets Combined)
# =============================================================================

def load_biodiversity_directory(directory_path: str,
                               bbox: Tuple[float, float, float, float],
                               resolution: float,
                               target_crs: str) -> np.ndarray:
    """
    Load and combine multiple biodiversity datasets from a directory.
    
    Useful when you have multiple data sources:
    - Critical habitat shapefiles
    - Species occurrence CSVs
    - Biodiversity rasters
    
    Takes the maximum biodiversity score from all sources.
    """
    from pathlib import Path
    
    print(f"  Loading multiple biodiversity datasets from directory...")
    
    width, height = _create_grid(bbox, resolution)
    combined_biodiversity = np.zeros((height, width), dtype=float)
    
    directory = Path(directory_path)
    
    # Find all relevant files
    files = []
    for pattern in ['*.tif', '*.tiff', '*.shp', '*.geojson', '*.csv']:
        files.extend(directory.glob(pattern))
    
    print(f"    Found {len(files)} data files")
    
    for file_path in files:
        print(f"    Processing {file_path.name}...")
        
        try:
            biodiversity = load_biodiversity_data(
                str(file_path),
                bbox,
                resolution,
                target_crs
            )
            
            if biodiversity is not None:
                # Take maximum (most important habitat wins)
                combined_biodiversity = np.maximum(combined_biodiversity, biodiversity)
                print(f"      ✓ Added to combined map")
        
        except Exception as e:
            print(f"      ✗ Error: {e}")
            continue
    
    print(f"    Combined biodiversity range: {combined_biodiversity.min():.3f} to {combined_biodiversity.max():.3f}")
    
    return combined_biodiversity


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _normalize_biodiversity(data: np.ndarray) -> np.ndarray:
    """
    Normalize biodiversity data to 0-1 range.
    
    Handles:
    - Already 0-1: no change
    - Percentage (0-100): divide by 100
    - Count data: normalize by max
    - Negative values: clip to 0
    """
    # Remove NaN and inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip negative values
    data = np.maximum(data, 0)
    
    # Check range
    data_min = data.min()
    data_max = data.max()
    
    # Already 0-1
    if data_min >= 0 and data_max <= 1:
        return data
    
    # Percentage (0-100)
    elif data_min >= 0 and data_max <= 100:
        return data / 100.0
    
    # Normalize by max
    elif data_max > 0:
        return data / data_max
    
    else:
        return data


def _create_grid(bbox: Tuple[float, float, float, float], 
                resolution: float) -> Tuple[int, int]:
    """Calculate grid dimensions."""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    lat_center = (min_lat + max_lat) / 2
    lon_distance = (max_lon - min_lon) * 111000 * np.cos(np.radians(lat_center))
    lat_distance = (max_lat - min_lat) * 111000
    
    width = int(np.ceil(lon_distance / resolution))
    height = int(np.ceil(lat_distance / resolution))
    
    return width, height


# =============================================================================
# INTEGRATION INTO extract_biodiversity_hotspots()
# =============================================================================

def extract_biodiversity_hotspots_COMPLETE(
    bbox: Tuple[float, float, float, float], 
    resolution: float,
    target_crs: str = "EPSG:3083",
    land_cover_array: Optional[np.ndarray] = None,
    protected_areas_array: Optional[np.ndarray] = None,
    conservation_db_path: Optional[str] = None,
    data_type: Optional[str] = None) -> np.ndarray:
    """
    Complete implementation with conservation database loading.
    
    Replace your extract_biodiversity_hotspots() function with this.
    """
    print("Extracting biodiversity hotspots...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Try conservation database if provided
    if conservation_db_path is not None:
        biodiversity = load_biodiversity_data(
            conservation_db_path,
            bbox,
            resolution,
            target_crs,
            data_type
        )
        
        if biodiversity is not None and biodiversity.max() > 0:
            print("  Using conservation database data")
            return biodiversity
        else:
            print("  Conservation database loading failed, falling back to estimation")
    
    # Fallback: Estimate from land cover
    print("Estimating biodiversity from land cover and protected areas...")
    
    biodiversity = np.zeros((height, width), dtype=float)
    
    if land_cover_array is not None:
        # Base scores by land cover
        biodiversity[land_cover_array == 1] = 0.1  # developed
        biodiversity[land_cover_array == 2] = 0.3  # agriculture
        biodiversity[land_cover_array == 3] = 0.5  # grassland
        biodiversity[land_cover_array == 4] = 0.7  # forest
        biodiversity[land_cover_array == 5] = 0.9  # wetland
        biodiversity[land_cover_array == 6] = 0.6  # water
    else:
        biodiversity[:] = 0.4
    
    # Boost in protected areas
    if protected_areas_array is not None:
        biodiversity[protected_areas_array == 1] = np.minimum(
            biodiversity[protected_areas_array == 1] + 0.2, 1.0
        )
    
    # Add heterogeneity bonus
    from scipy.ndimage import uniform_filter, gaussian_filter
    
    local_mean = uniform_filter(biodiversity, size=5, mode='reflect')
    local_sq_mean = uniform_filter(biodiversity**2, size=5, mode='reflect')
    local_variance = np.maximum(0, local_sq_mean - local_mean**2)
    
    if local_variance.max() > 0:
        heterogeneity = local_variance / local_variance.max()
        biodiversity = np.minimum(biodiversity + heterogeneity * 0.15, 1.0)
    
    biodiversity = gaussian_filter(biodiversity, sigma=1.0)
    biodiversity = np.clip(biodiversity, 0, 1)
    
    print(f"Biodiversity score statistics:")
    print(f"  Mean: {biodiversity.mean():.3f}")
    print(f"  High biodiversity (>0.7): {np.sum(biodiversity > 0.7) / biodiversity.size * 100:.2f}%")
    print(f"  (Note: This is estimated. Provide conservation_db_path for accurate data)")
    
    return biodiversity


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("BIODIVERSITY DATA LOADING - EXAMPLES")
    print("="*80)
    print()
    
    bbox = (-98.0, 29.0, -97.5, 29.5)
    resolution = 100
    
    # Example 1: Load species richness raster
    print("Example 1: Species Richness Raster")
    print("-" * 40)
    biodiversity = load_biodiversity_data(
        "path/to/species_richness.tif",
        bbox, resolution, "EPSG:3083"
    )
    print()
    
    # Example 2: Load critical habitat shapefile
    print("Example 2: Critical Habitat Shapefile")
    print("-" * 40)
    biodiversity = load_biodiversity_data(
        "path/to/critical_habitat.shp",
        bbox, resolution, "EPSG:3083"
    )
    print()
    
    # Example 3: Load species occurrence points
    print("Example 3: Species Occurrence Points (GBIF)")
    print("-" * 40)
    biodiversity = load_biodiversity_data(
        "path/to/species_occurrences.csv",
        bbox, resolution, "EPSG:3083"
    )
    print()
    
    # Example 4: Load directory of multiple datasets
    print("Example 4: Multiple Datasets (Combined)")
    print("-" * 40)
    biodiversity = load_biodiversity_data(
        "path/to/biodiversity_data/",
        bbox, resolution, "EPSG:3083"
    )
    print()