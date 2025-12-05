import geopandas as gpd
from pyproj import CRS
import numpy as np
from typing import Optional, Tuple

def detect_crs_from_geojson(geojson_path: str) -> str:
    """
    Detect CRS from GeoJSON file.
    
    Args:
        geojson_path: Path to GeoJSON file
        
    Returns:
        CRS string (e.g., "EPSG:4326")
    """
    try:
        gdf = gpd.read_file(geojson_path)
        
        if gdf.crs is None:
            print("Warning: No CRS found in GeoJSON. Assuming WGS84 (EPSG:4326)")
            return "EPSG:4326"
        
        # Get EPSG code if available
        if gdf.crs.to_epsg() is not None:
            detected_crs = f"EPSG:{gdf.crs.to_epsg()}"
            print(f"Detected CRS: {detected_crs}")
            return detected_crs
        else:
            # Return WKT or PROJ string if no EPSG code
            detected_crs = gdf.crs.to_string()
            print(f"Detected CRS (non-EPSG): {detected_crs}")
            return detected_crs
            
    except Exception as e:
        print(f"Error detecting CRS: {e}")
        print("Defaulting to WGS84 (EPSG:4326)")
        return "EPSG:4326"

def infer_target_crs_from_bbox(bbox: Tuple[float, float, float, float]) -> str:
    """
    Infer the best target CRS based on bounding box location.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Appropriate CRS string for the region
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    
    # Texas-specific detection
    if -107 <= center_lon <= -93 and 25 <= center_lat <= 37:
        print("Detected Texas region - using Texas Centric Albers (EPSG:3083)")
        return "EPSG:3083"
    
    # Otherwise, use appropriate UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # Determine hemisphere
    if center_lat >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
        hemisphere = "N"
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
        hemisphere = "S"
    
    target_crs = f"EPSG:{epsg_code}"
    print(f"Inferred UTM Zone {utm_zone}{hemisphere}: {target_crs}")
    
    return target_crs

def auto_detect_crs(bbox: Tuple[float, float, float, float] = None, 
                    geojson_path: str = None,
                    prefer_local_projection: bool = True) -> Tuple[str, str]:
    """
    Automatically detect source and target CRS.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        geojson_path: Path to GeoJSON file
        prefer_local_projection: If True, use region-specific projection; 
                                 if False, use UTM
        
    Returns:
        Tuple of (source_crs, target_crs)
    """
    source_crs = "EPSG:4326"  # Default assumption
    
    # Detect source CRS from GeoJSON if available
    if geojson_path:
        source_crs = detect_crs_from_geojson(geojson_path)
    
    # Infer target CRS from bbox
    if bbox:
        target_crs = infer_target_crs_from_bbox(bbox)
    else:
        # If no bbox, extract from GeoJSON
        if geojson_path:
            gdf = gpd.read_file(geojson_path)
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            bbox = tuple(bounds)
            target_crs = infer_target_crs_from_bbox(bbox)
        else:
            raise ValueError("Either bbox or geojson_path must be provided")
    
    return source_crs, target_crs