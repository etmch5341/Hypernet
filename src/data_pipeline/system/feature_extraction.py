"""
Feature extraction functions for hyperloop route optimization.

Each function takes a bounding box and resolution, and returns a 2D numpy array
representing that feature across the region.

Bounding box format: (min_lon, min_lat, max_lon, max_lat)
Resolution: meters per pixel
"""

from typing import Optional, Tuple
import numpy as np
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import heapq
import geopandas as gpd
import matplotlib.pyplot as plt

from esy.osmfilter import osm_colors as CC
from esy.osmfilter import run_filter, Node,Way,Relation 
from esy.osmfilter import export_geojson

from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform
from shapely.geometry import box

from scipy.ndimage import distance_transform_edt, uniform_filter

from water_resources_loader import load_usgs_water_resources
from biodiversity_loader import load_biodiversity_data

import os

DEFAULT_CRS_OSM = "EPSG:4326"  # WGS84 - EPSG:4326 is the default for pbf and osm data
DEFAULT_CRS_TX_ALBERS = "EPSG:3083"  # Texas Centric Albers Equal Area

def _create_grid(bbox: Tuple[float, float, float, float], resolution: float) -> Tuple[int, int]:
    """
    Helper function to calculate grid dimensions.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        resolution: meters per pixel
        
    Returns:
        (width, height) in pixels
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    lat_center = (min_lat + max_lat) / 2
    lon_distance = (max_lon - min_lon) * 111000 * np.cos(np.radians(lat_center))
    lat_distance = (max_lat - min_lat) * 111000
    
    width = int(np.ceil(lon_distance / resolution))
    height = int(np.ceil(lat_distance / resolution))
    
    return width, height

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def _display_raster_geographic(raster, minx, miny, maxx, maxy, title="Rasterized Data"):
    """
    Display raster with geographic coordinates.
    
    Args:
        raster: 2D numpy array
        minx, miny, maxx, maxy: Bounds in projected CRS (meters)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Display with extent to show geographic coordinates
    im = ax.imshow(raster, 
                   cmap='binary',  # or 'viridis', 'hot', 'RdYlGn_r'
                   origin='upper',
                   extent=[minx, maxx, miny, maxy],  # Set geographic bounds
                   interpolation='nearest')
    
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_xlabel('Easting (meters)')
    ax.set_ylabel('Northing (meters)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Format axis labels to show coordinates nicely
    ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout()
    plt.show()
    
def _get_road_colormap(style='neon_cyan'):
    """
    Get predefined colormaps for road visualization.
    
    Available styles:
    - 'neon_cyan': Bright cyan roads (great contrast on satellite)
    - 'hot_pink': Magenta/hot pink roads
    - 'electric_yellow': Bright yellow roads
    - 'lime': Bright lime green roads
    - 'orange': Orange roads
    - 'white': White roads (classic look)
    - 'gradient_red': Red gradient (thicker = more red)
    """
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    
    styles = {
        'neon_cyan': ['none', '#00FFFF'],
        'hot_pink': ['none', '#FF1493'],
        'electric_yellow': ['none', '#FFFF00'],
        'lime': ['none', '#00FF00'],
        'orange': ['none', '#FF8C00'],
        'white': ['none', '#FFFFFF'],
        'royal_blue': ['none', '#4169E1'],
        'red': ['none', '#FF0000'],
    }
    
    if style in styles:
        return ListedColormap(styles[style], name=style, N=2)
    else:
        # Default to cyan
        return ListedColormap(styles['neon_cyan'], name='default', N=2)
    
def _display_raster_with_satellite(raster, minx, miny, maxx, maxy, 
                                    target_crs=DEFAULT_CRS_TX_ALBERS,
                                    title="Rasterized Data with Satellite Background",
                                    raster_alpha=0.8,
                                    basemap_alpha=1.0,
                                    color_style='neon_cyan',
                                    basemap_style='light'):  # New parameter
    """
    Display raster overlaid on basemap background.
    
    Args:
        raster: 2D numpy array
        minx, miny, maxx, maxy: Bounds in projected CRS (meters)
        target_crs: CRS of the raster data
        title: Plot title
        raster_alpha: Transparency of raster overlay (0=transparent, 1=opaque)
        basemap_alpha: Transparency of basemap (0=transparent, 1=opaque)
        color_style: Color style for roads
        basemap_style: 'light', 'satellite', 'streets', or 'dark'
    """
    import contextily as ctx
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # Define basemap providers based on style
    basemap_options = {
        'light': [
            ("CartoDB Positron", ctx.providers.CartoDB.Positron),
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
        ],
        'satellite': [
            ("Esri WorldImagery", ctx.providers.Esri.WorldImagery),
        ],
        'streets': [
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
            ("CartoDB Voyager", ctx.providers.CartoDB.Voyager),
        ],
        'dark': [
            ("CartoDB Dark Matter", ctx.providers.CartoDB.DarkMatter),
        ]
    }
    
    # Get the provider list for the selected style
    basemap_providers = basemap_options.get(basemap_style, basemap_options['light'])
    
    basemap_loaded = False
    for provider_name, provider in basemap_providers:
        try:
            print(f"Attempting to load {provider_name} basemap...")
            ctx.add_basemap(ax, 
                           crs=target_crs,
                           source=provider,
                           alpha=basemap_alpha,
                           zoom='auto',
                           attribution=False)
            print(f"Successfully loaded {provider_name}")
            basemap_loaded = True
            break
        except Exception as e:
            print(f"Could not load {provider_name}: {e}")
            continue
    
    if not basemap_loaded:
        print("Warning: No basemap could be loaded. Displaying raster only with gray background.")
        ax.set_facecolor('#e0e0e0')
    
    # Get the colormap for the selected style
    cmap = _get_road_colormap(color_style)
    
    im = ax.imshow(raster, 
                   cmap=cmap,
                   origin='upper',
                   extent=[minx, maxx, miny, maxy],
                   interpolation='nearest',
                   alpha=raster_alpha,
                   vmin=-0.5,
                   vmax=1.5)
    
    plt.colorbar(im, ax=ax, label='Road Network', pad=0.02, ticks=[0, 1])
    ax.set_xlabel('Easting (meters)', fontsize=11)
    ax.set_ylabel('Northing (meters)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout()
    plt.show()
    
def _display_raster_with_goals(raster, goal_points, title="Raster with Goal Points"):
    """
    Display the raster with goal points highlighted.
    
    Parameters:
    - raster: numpy array representing the cost map
    - goal_points: list of tuples (x, y, name) in pixel coordinates
    - title: plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Display the raster
    # Use a colormap where goal points (value=2) will be clearly visible
    plt.imshow(raster, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cost Value')
    
    # Overlay goal points with red markers
    for x, y, name in goal_points:
        plt.plot(x, y, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        plt.annotate(name, (x, y), xytext=(10, 10), textcoords='offset points',
                    color='white', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7))
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def _display_raster_with_goals_and_basemap(raster, goal_points, minx, miny, maxx, maxy,
                                           target_crs=DEFAULT_CRS_TX_ALBERS,
                                           title="Road Network with Goal Points",
                                           raster_alpha=0.85,
                                           basemap_alpha=1.0,
                                           color_style='neon_cyan',
                                           basemap_style='light',
                                           goal_marker='*',
                                           goal_size=400,
                                           goal_color='red'):
    """
    Display raster with goal points overlaid on basemap background.
    
    Args:
        raster: 2D numpy array
        goal_points: list of tuples (pixel_x, pixel_y, name)
        minx, miny, maxx, maxy: Bounds in projected CRS (meters)
        target_crs: CRS of the raster data
        title: Plot title
        raster_alpha: Transparency of raster overlay (0=transparent, 1=opaque)
        basemap_alpha: Transparency of basemap (0=transparent, 1=opaque)
        color_style: Color style for roads
        basemap_style: 'light', 'satellite', 'streets', or 'dark'
        goal_marker: Marker style ('*', 'o', 's', 'D', '^', 'v')
        goal_size: Size of goal markers
        goal_color: Color of goal markers ('red', 'yellow', 'lime', 'cyan', etc.)
    """
    import contextily as ctx
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # Define basemap providers based on style
    basemap_options = {
        'light': [
            ("CartoDB Positron", ctx.providers.CartoDB.Positron),
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
        ],
        'satellite': [
            ("Esri WorldImagery", ctx.providers.Esri.WorldImagery),
        ],
        'streets': [
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
            ("CartoDB Voyager", ctx.providers.CartoDB.Voyager),
        ],
        'dark': [
            ("CartoDB Dark Matter", ctx.providers.CartoDB.DarkMatter),
        ]
    }
    
    # Load basemap
    basemap_providers = basemap_options.get(basemap_style, basemap_options['light'])
    basemap_loaded = False
    for provider_name, provider in basemap_providers:
        try:
            print(f"Attempting to load {provider_name} basemap...")
            ctx.add_basemap(ax, 
                           crs=target_crs,
                           source=provider,
                           alpha=basemap_alpha,
                           zoom='auto',
                           attribution=False)
            print(f"Successfully loaded {provider_name}")
            basemap_loaded = True
            break
        except Exception as e:
            print(f"Could not load {provider_name}: {e}")
            continue
    
    if not basemap_loaded:
        print("Warning: No basemap could be loaded. Displaying raster only with gray background.")
        ax.set_facecolor('#e0e0e0')
    
    # Get the colormap for roads
    cmap = _get_road_colormap(color_style)
    
    # Display raster
    im = ax.imshow(raster, 
                   cmap=cmap,
                   origin='upper',
                   extent=[minx, maxx, miny, maxy],
                   interpolation='nearest',
                   alpha=raster_alpha,
                   vmin=-0.5,
                   vmax=2.5)  # Extended range to accommodate goal points (value=2)
    
    # Convert pixel coordinates to geographic coordinates for plotting
    width = raster.shape[1]
    height = raster.shape[0]
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    
    # Plot goal points
    for pixel_x, pixel_y, name in goal_points:
        # Convert pixel coordinates back to geographic coordinates
        geo_x = minx + (pixel_x + 0.5) * pixel_width
        geo_y = maxy - (pixel_y + 0.5) * pixel_height
        
        # Plot marker
        ax.scatter(geo_x, geo_y, 
                  marker=goal_marker, 
                  s=goal_size, 
                  c=goal_color,
                  edgecolors='white', 
                  linewidths=2.5,
                  zorder=5,  # Ensure markers appear on top
                  alpha=0.9)
        
        # Add label
        ax.annotate(name, 
                   (geo_x, geo_y), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   color='white', 
                   fontsize=11, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=goal_color, 
                           edgecolor='white',
                           linewidth=2,
                           alpha=0.85),
                   zorder=6)
    
    plt.colorbar(im, ax=ax, label='Road Network', pad=0.02)
    ax.set_xlabel('Easting (meters)', fontsize=11)
    ax.set_ylabel('Northing (meters)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout()
    plt.show()
    
def display_raster_with_goals_and_basemap(raster, goal_points, bbox, target_crs,
                                          title="Raster with Goal Points",
                                          output_file=None,
                                          raster_alpha=0.85,
                                          basemap_alpha=1.0,
                                          color_style='neon_cyan',
                                          basemap_style='light',
                                          goal_marker='*',
                                          goal_size=400,
                                          goal_color='red'):
    """
    Display raster with goal points overlaid on basemap background.
    
    Parameters:
    - raster: numpy array representing the cost map
    - goal_points: list of tuples (pixel_x, pixel_y, name) in pixel coordinates
    - bbox: bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
    - target_crs: CRS of the raster data
    - title: plot title
    - output_file: path to save the plot (if None, shows interactively)
    - raster_alpha: Transparency of raster overlay (0=transparent, 1=opaque)
    - basemap_alpha: Transparency of basemap (0=transparent, 1=opaque)
    - color_style: Color style for roads
    - basemap_style: 'light', 'satellite', 'streets', or 'dark'
    - goal_marker: Marker style ('*', 'o', 's', 'D', '^', 'v')
    - goal_size: Size of goal markers
    - goal_color: Color of goal markers
    """
    import contextily as ctx
    
    # Transform bbox from WGS84 to target CRS
    transformer = Transformer.from_crs(DEFAULT_CRS_OSM, target_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # Define basemap providers based on style
    basemap_options = {
        'light': [
            ("CartoDB Positron", ctx.providers.CartoDB.Positron),
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
        ],
        'satellite': [
            ("Esri WorldImagery", ctx.providers.Esri.WorldImagery),
        ],
        'streets': [
            ("OpenStreetMap", ctx.providers.OpenStreetMap.Mapnik),
            ("CartoDB Voyager", ctx.providers.CartoDB.Voyager),
        ],
        'dark': [
            ("CartoDB Dark Matter", ctx.providers.CartoDB.DarkMatter),
        ]
    }
    
    # Load basemap
    basemap_providers = basemap_options.get(basemap_style, basemap_options['light'])
    basemap_loaded = False
    for provider_name, provider in basemap_providers:
        try:
            print(f"Attempting to load {provider_name} basemap...")
            ctx.add_basemap(ax, 
                           crs=target_crs,
                           source=provider,
                           alpha=basemap_alpha,
                           zoom='auto',
                           attribution=False)
            print(f"Successfully loaded {provider_name}")
            basemap_loaded = True
            break
        except Exception as e:
            print(f"Could not load {provider_name}: {e}")
            continue
    
    if not basemap_loaded:
        print("Warning: No basemap could be loaded. Displaying raster only with gray background.")
        ax.set_facecolor('#e0e0e0')
    
    # Define color schemes
    color_schemes = {
        'neon_cyan': ['none', '#00FFFF'],
        'hot_pink': ['none', '#FF1493'],
        'electric_yellow': ['none', '#FFFF00'],
        'lime': ['none', '#00FF00'],
        'orange': ['none', '#FF8C00'],
        'white': ['none', '#FFFFFF'],
        'royal_blue': ['none', '#4169E1'],
    }
    
    colors = color_schemes.get(color_style, color_schemes['neon_cyan'])
    cmap = ListedColormap(colors, name='road_cmap', N=2)
    
    # Display raster
    im = ax.imshow(raster, 
                   cmap=cmap,
                   origin='upper',
                   extent=[minx, maxx, miny, maxy],
                   interpolation='nearest',
                   alpha=raster_alpha,
                   vmin=-0.5,
                   vmax=2.5)
    
    # Convert pixel coordinates to geographic coordinates for plotting
    width = raster.shape[1]
    height = raster.shape[0]
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    
    # Plot goal points
    for pixel_x, pixel_y, name in goal_points:
        # Convert pixel coordinates back to geographic coordinates
        geo_x = minx + (pixel_x + 0.5) * pixel_width
        geo_y = maxy - (pixel_y + 0.5) * pixel_height
        
        # Plot marker
        ax.scatter(geo_x, geo_y, 
                  marker=goal_marker, 
                  s=goal_size, 
                  c=goal_color,
                  edgecolors='white', 
                  linewidths=2.5,
                  zorder=5,
                  alpha=0.9)
        
        # Add label
        ax.annotate(name, 
                   (geo_x, geo_y), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   color='white', 
                   fontsize=11, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=goal_color, 
                           edgecolor='white',
                           linewidth=2,
                           alpha=0.85),
                   zorder=6)
    
    plt.colorbar(im, ax=ax, label='Road Network', pad=0.02)
    ax.set_xlabel('Easting (meters)', fontsize=11)
    ax.set_ylabel('Northing (meters)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def rasterize_geometries(resolution: float, 
                          geojson_input_path="", 
                          target_crs: str = DEFAULT_CRS_TX_ALBERS,  # Texas Centric Albers Equal Area
                          bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """
    Helper function to rasterize geometries into a bitmap.
    
    NOTE: If bbox is None, bounds are calculated from the geometries. However, this can lead to high consumption of memory, so it is RECOMMENDED to provide a bbox.
    
    Args:
        resolution: Pixel resolution in meters
        geojson_input_path: Path to input GeoJSON file
        target_crs: Target CRS for rasterization (default: Texas Albers)
        bbox: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        2D numpy array raster
    """
    
    # --- Load and Rasterize Road Network ---
    with open(geojson_input_path) as f:
        geojson = json.load(f)

    # geometries = [(feature["geometry"], 1) for feature in geojson["features"]]
    
    # Create transformer from WGS84 to target CRS
    # TODO: Should update to handle automatic crs detection using the crs_detection.py module
    transformer = Transformer.from_crs(DEFAULT_CRS_OSM, target_crs, always_xy=True) #NOTE: EPSG:4326 is the default for pbf and osm data
    
    # Reproject geometries and prepare for rasterization
    geometries = []
    for feature in geojson["features"]:
        geom = shape(feature["geometry"])
        # Reproject geometry
        projected_geom = transform(transformer.transform, geom)
        geometries.append((mapping(projected_geom), 1))

    # Get bounds
    if bbox is None:
        # Calculate bounds from projected geometries
        all_geoms = [shape(geom_mapping) for geom_mapping, _ in geometries]
        bounds = [geom.bounds for geom in all_geoms]
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
    else:
        # Reproject the WGS84 bbox to target CRS
        min_lon, min_lat, max_lon, max_lat = bbox
        minx, miny = transformer.transform(min_lon, min_lat)
        maxx, maxy = transformer.transform(max_lon, max_lat)
    
    # Calculate width and height based on bounds and desired resolution
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    # Create affine transform
    affine_transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Rasterize
    raster = rasterize(geometries, out_shape=(height, width), transform=affine_transform, fill=0, dtype=np.uint8)

    # Verify pixel size
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height

    print(f"Bounds: ({minx}, {miny}) to ({maxx}, {maxy})")
    print(f"Raster dimensions: {width} x {height} pixels")
    print(f"Pixel width: {pixel_width} meters")
    print(f"Pixel height: {pixel_height} meters")
    
    # Display standard raster
    _display_raster_geographic(raster, minx, miny, maxx, maxy, 
                          title=f"Road Network ({target_crs} Coordinate System)")
    
    # Display raster with satellite background
    _display_raster_with_satellite(raster, minx, miny, maxx, maxy,
                               target_crs=target_crs,
                               title=f"Road Network with Light Background ({target_crs})",
                               basemap_style='satellite',  # Use light style
                               color_style='red',
                               raster_alpha=0.9,
                               basemap_alpha=0.9)
    
    return raster, width, height

def add_goal_points_to_raster(raster, point_map, width, height, bbox, target_crs=DEFAULT_CRS_TX_ALBERS):
    """
    Add goal points to the raster by converting lat/lon coordinates to pixel coordinates.
    
    Parameters:
    - raster: numpy array representing the cost map
    - point_map: dict mapping station names to [lon, lat] coordinates
    - width: raster width in pixels
    - height: raster height in pixels
    - bbox: bounding box (min_lon, min_lat, max_lon, max_lat)
    - target_crs: coordinate reference system for the raster
    
    Returns:
    - raster: modified raster with goal points marked as 2
    - goal_points: list of tuples (x, y, name) in pixel coordinates
    """
    from pyproj import Transformer
    
    goal_points = []
    
    # Create transformer from WGS84 to target CRS
    transformer = Transformer.from_crs(DEFAULT_CRS_OSM, target_crs, always_xy=True)
    
    # Transform bbox corners to get the extent in target CRS
    min_x, min_y = transformer.transform(bbox[0], bbox[1])
    max_x, max_y = transformer.transform(bbox[2], bbox[3])
    
    # Calculate pixel resolution
    pixel_width = (max_x - min_x) / width
    pixel_height = (max_y - min_y) / height
    
    # Process each goal point
    for name, coords in point_map.items():
        lon, lat = coords
        
        # Transform to target CRS
        x_crs, y_crs = transformer.transform(lon, lat)
        
        # Convert to pixel coordinates
        # Note: y is inverted because raster row 0 is at the top (max_y)
        pixel_x = int((x_crs - min_x) / pixel_width)
        pixel_y = int((max_y - y_crs) / pixel_height)
        
        # Check if point is within bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            raster[pixel_y, pixel_x] = 2  # Mark as goal point
            goal_points.append((pixel_x, pixel_y, name))
            print(f"Added {name} at pixel ({pixel_x}, {pixel_y})")
        else:
            print(f"Warning: {name} at ({lon}, {lat}) is outside raster bounds")
            
    _display_raster_with_goals(raster, goal_points)
    
    _display_raster_with_goals_and_basemap(raster, goal_points, 
                                           min_x, min_y, max_x, max_y,
                                           target_crs=target_crs,
                                           basemap_style='light',
                                           color_style='royal_blue',
                                           raster_alpha=0.85,
                                           goal_color='red',
                                           goal_marker='*',
                                           goal_size=400)
    
    return raster, goal_points

# ============================================================================
# BASE FEATURES
# ============================================================================

def extract_road_network(bbox: Tuple[float, float, float, float], resolution: float, target_crs: str = DEFAULT_CRS_TX_ALBERS, geojson_input_path = "", pbf_input_path= "", output_path="") -> np.ndarray:
    """
    Extract road and rail network bitmap.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) - Set to None if using geojson_input_path
        resolution: Pixel resolution in meters
        target_crs: Target CRS for rasterization (default: Texas Albers)
        geojson_input_path: Path to input GeoJSON file - must include filename in path
        pbf_input_path: Path to input data source (PBF format) - must include filename in path
        output_path: Path to save geojson output - must include filename in path
    Returns:
        2D binary array (1 = road/rail exists, 0 = no road/rail)
    """
    
    # If GeoJSON input is provided, rasterize directly
    if geojson_input_path != "":
        raster, width, height = rasterize_geometries(bbox=bbox, resolution=resolution, geojson_input_path=geojson_input_path, target_crs=target_crs)
        return raster, width, height
    
    # If GeoJSON input not provided, process from PBF input
    # Validate input paths
    if pbf_input_path == "":
        raise ValueError("pbf_input_path must be provided.")
    
    if output_path == "":
        raise ValueError("output_path must be provided.")
    
    # Process into GeoJSON using osmfilter
    prefilter = {
        Node: {},
        Way: {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential']},
        Relation: {}
    }
    
    blackfilter = []
    
    whitefilter = [(
        ('highway', 'motorway'),
        ('highway', 'trunk'),
        ('highway', 'primary'),
        ('highway', 'secondary'),
        ('highway', 'tertiary'),
        ('highway', 'unclassified'),
        ('highway', 'residential')
    )]
    
    # Json file for the Data dictionary
    JSON_outputfile = os.path.join(os.getcwd(), output_path)
    
    # Json file for the Elements dictionary
    PBF_inputfile = os.path.join(os.getcwd(), pbf_input_path)
    
    element_name = "filtered-roadways"
    [Data,Elements]=run_filter(element_name, 
                               PBF_inputfile, 
                               JSON_outputfile, 
                               prefilter,whitefilter,blackfilter, 
                               NewPreFilterData=True, 
                               CreateElements=True, 
                               LoadElements=False, 
                               verbose=True, 
                               multiprocess=False)
    
    # Debugging outputs to check the number of elements and tags
    print(f"Data for 'Way' elements: {len(Data['Way'])}")
    print(f"Elements after filtering: {len(Elements[element_name]['Way'])}")
    
    # Export to GeoJSON for visualization
    export_geojson(
        Elements['filtered-roadways']['Way'],
        Data,
        filename= output_path,
        jsontype='Line'
    )
    
    # Rasterize the exported GeoJSON
    road_bitmap, width, height = rasterize_geometries(bbox=bbox, resolution=resolution, geojson_input_path=output_path, target_crs=target_crs)
    
    return road_bitmap, width, height
   

# ============================================================================
# PATH GEOMETRY / OPERATIONAL EFFICIENCY FEATURES
# ============================================================================

def extract_elevation(bbox: Tuple[float, float, float, float], 
                     resolution: float,
                     target_crs: str = DEFAULT_CRS_TX_ALBERS,
                     srtm_data_path: Optional[str] = None) -> np.ndarray:
    """
    Extract elevation data (Digital Elevation Model) from SRTM/ASTER GDEM.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution: Pixel resolution in meters
        target_crs: Target CRS for output raster (default: Texas Albers)
        srtm_data_path: Path to SRTM .tif file (if None, will attempt to download)
        
    Returns:
        2D array of elevation values in meters
    """
    import rasterio
    from rasterio.warp import reproject, calculate_default_transform, Resampling
    from rasterio.mask import mask
    from shapely.geometry import box
    
    print(f"Extracting elevation data for bbox {bbox}")
    
    # Create transformer from WGS84 to target CRS
    transformer = Transformer.from_crs(DEFAULT_CRS_OSM, target_crs, always_xy=True)
    
    # Transform bbox to target CRS
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    # Calculate output dimensions
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    
    print(f"Output dimensions: {width} x {height} pixels")
    print(f"Target resolution: {resolution} meters/pixel")
    pad = 0.1  # ~10km buffer in degrees
    download_bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    
    # If no SRTM data path provided, try to download from OpenTopography
    if srtm_data_path is None:
        print("No SRTM data path provided. Attempting to use elevation-data library...")
        try:
            import elevation
            
            # Create temporary output path
            import tempfile
            temp_dir = tempfile.mkdtemp()
            srtm_data_path = os.path.join(temp_dir, 'srtm_temp.tif')
            
            # Download SRTM data for the bbox
            elevation.clip(bounds=download_bbox, output=srtm_data_path, product='SRTM3')
            print(f"Downloaded SRTM data to {srtm_data_path}")
            
        except ImportError:
            print("ERROR: elevation library not installed.")
            print("Install with: pip install elevation")
            print("Returning placeholder elevation data.")
            return np.random.uniform(0, 500, (height, width))
        except Exception as e:
            print(f"ERROR downloading SRTM data: {e}")
            print("Returning placeholder elevation data.")
            return np.random.uniform(0, 500, (height, width))
    
    # Load and process the SRTM data
    try:
        with rasterio.open(srtm_data_path) as src:
            print(f"Loaded SRTM data: {src.width} x {src.height}, CRS: {src.crs}")
            
            # # Create bounding box polygon in WGS84
            # bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
            
            # # Mask the raster to our bbox
            # out_image, out_transform = mask(src, [bbox_polygon], crop=True, all_touched=True)
            # out_image = out_image[0]  # Get first band
            
            # # Replace nodata values
            # nodata = src.nodata
            # if nodata is not None:
            #     out_image = np.where(out_image == nodata, 0, out_image)
            
            # print(f"Cropped to bbox: {out_image.shape}")
            
            # # Reproject to target CRS and resolution
            # src_crs = src.crs
            
            # # Calculate transform for target CRS
            # dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
            
            # # Create destination array
            # elevation_array = np.zeros((height, width), dtype=np.float32)
            
            # # Reproject
            # reproject(
            #     source=out_image,
            #     destination=elevation_array,
            #     src_transform=out_transform,
            #     src_crs=src_crs,
            #     dst_transform=dst_transform,
            #     dst_crs=target_crs,
            #     resampling=Resampling.bilinear
            # )
            
            # Below fix fixes and uses the buffer to not have any missing areas of the bbox
            dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)
            
            elevation_array = np.zeros((height, width), dtype=np.float32)
            
            reproject(
                source=rasterio.band(src, 1),
                destination=elevation_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            
            print(f"Reprojected to {target_crs}")
            print(f"Elevation range: {elevation_array.min():.1f}m to {elevation_array.max():.1f}m")
            
            # Display the elevation map
            _display_elevation_map(elevation_array, minx, miny, maxx, maxy, target_crs)
            
            return elevation_array
            
    except Exception as e:
        print(f"ERROR processing elevation data: {e}")
        print("Returning placeholder elevation data.")
        return np.random.uniform(0, 500, (height, width))


def extract_slope(bbox: Tuple[float, float, float, float], 
                 resolution: float,
                 target_crs: str = DEFAULT_CRS_TX_ALBERS,
                 elevation_array: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract slope/gradient data (derived from elevation or provided elevation array).
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution: Pixel resolution in meters
        target_crs: Target CRS (default: Texas Albers)
        elevation_array: Pre-computed elevation array (if None, will extract)
        
    Returns:
        2D array of slope values (as decimal, e.g., 0.06 = 6%)
    """
    print("Calculating slope from elevation data...")
    
    # If no elevation array provided, extract it
    if elevation_array is None:
        elevation_array = extract_elevation(bbox, resolution, target_crs)
    
    # Calculate gradients (change in elevation per pixel)
    # dy is gradient in y-direction (rows), dx is gradient in x-direction (cols)
    dy, dx = np.gradient(elevation_array)
    
    # Convert to slope in meters per meter
    # Since our pixels are in meters (resolution), the gradient gives us meters of elevation per pixel
    # Divide by resolution to get meters per meter (slope)
    slope_y = dy / resolution
    slope_x = dx / resolution
    
    # Calculate total slope magnitude
    slope = np.sqrt(slope_x**2 + slope_y**2)
    
    # Clip extreme values (sometimes gradients at edges can be noisy)
    slope = np.clip(slope, 0, 1.0)  # Max 100% slope
    
    print(f"Slope statistics:")
    print(f"  Mean slope: {np.mean(slope)*100:.2f}%")
    print(f"  Max slope: {np.max(slope)*100:.2f}%")
    print(f"  Pixels with slope > 6%: {np.sum(slope > 0.06) / slope.size * 100:.2f}%")
    
    # Display the slope map
    transformer = Transformer.from_crs(DEFAULT_CRS_OSM, target_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    _display_slope_map(slope, minx, miny, maxx, maxy, target_crs)
    
    return slope


def extract_existing_corridors(bbox: Tuple[float, float, float, float], 
                               resolution: float,
                               target_crs: str = DEFAULT_CRS_TX_ALBERS,
                               geojson_input_path: str = "",
                               pbf_input_path: str = "",
                               output_path: str = "",
                               include_railways: bool = True) -> np.ndarray:
    """
    Extract existing transportation corridors (railways, highways) from OSM.
    This is essentially the same as extract_road_network but with railway support.
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
        resolution: Pixel resolution in meters
        target_crs: Target CRS for rasterization (default: Texas Albers)
        geojson_input_path: Path to input GeoJSON file (if available)
        pbf_input_path: Path to input PBF file
        output_path: Path to save output GeoJSON
        include_railways: Whether to include railway lines in addition to roads
        
    Returns:
        2D binary array (1 = corridor exists, 0 = no corridor)
    """
    print("Extracting existing transportation corridors (roads and railways)...")
    
    # If GeoJSON input is provided, rasterize directly
    if geojson_input_path != "":
        raster, width, height = rasterize_geometries(
            bbox=bbox, 
            resolution=resolution, 
            geojson_input_path=geojson_input_path, 
            target_crs=target_crs
        )
        return raster
    
    # Validate input paths
    if pbf_input_path == "":
        raise ValueError("pbf_input_path must be provided if geojson_input_path is not provided.")
    
    if output_path == "":
        raise ValueError("output_path must be provided.")
    
    # Define filters for roads and optionally railways
    prefilter = {
        Node: {},
        Way: {
            'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 
                       'unclassified', 'residential', 'motorway_link', 'trunk_link']
        },
        Relation: {}
    }
    
    whitefilter = [
        (
            ('highway', 'motorway'),
            ('highway', 'trunk'),
            ('highway', 'primary'),
            ('highway', 'secondary'),
            ('highway', 'tertiary'),
            ('highway', 'unclassified'),
            ('highway', 'residential'),
            ('highway', 'motorway_link'),
            ('highway', 'trunk_link')
        )
    ]
    
    # Add railway filter if requested
    if include_railways:
        prefilter[Way]['railway'] = ['rail', 'light_rail', 'subway']
        whitefilter.append((
            ('railway', 'rail'),
            ('railway', 'light_rail'),
            ('railway', 'subway')
        ))
    
    blackfilter = []
    
    # Set up file paths
    JSON_outputfile = os.path.join(os.getcwd(), output_path)
    PBF_inputfile = os.path.join(os.getcwd(), pbf_input_path)
    
    element_name = "filtered-corridors"
    
    # Run the filter
    print("Filtering OSM data for roads and railways...")
    [Data, Elements] = run_filter(
        element_name, 
        PBF_inputfile, 
        JSON_outputfile, 
        prefilter, 
        whitefilter, 
        blackfilter, 
        NewPreFilterData=True, 
        CreateElements=True, 
        LoadElements=False, 
        verbose=True, 
        multiprocess=False
    )
    
    # Debug output
    print(f"Data for 'Way' elements: {len(Data['Way'])}")
    print(f"Elements after filtering: {len(Elements[element_name]['Way'])}")
    
    # Export to GeoJSON
    export_geojson(
        Elements[element_name]['Way'],
        Data,
        filename=output_path,
        jsontype='Line'
    )
    
    print(f"Exported corridors to {output_path}")
    
    # Rasterize the exported GeoJSON
    corridor_bitmap, width, height = rasterize_geometries(
        bbox=bbox, 
        resolution=resolution, 
        geojson_input_path=output_path, 
        target_crs=target_crs
    )
    
    print(f"Corridor coverage: {np.sum(corridor_bitmap) / corridor_bitmap.size * 100:.2f}%")
    
    return corridor_bitmap

def _display_elevation_map(elevation, minx, miny, maxx, maxy, target_crs):
    """Display elevation map with terrain colormap."""
    import contextily as ctx
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Display elevation with terrain colormap
    im = ax.imshow(elevation,
                   cmap='terrain',
                   origin='upper',
                   extent=[minx, maxx, miny, maxy],
                   interpolation='bilinear')
    
    plt.colorbar(im, ax=ax, label='Elevation (meters)', pad=0.02)
    ax.set_xlabel('Easting (meters)', fontsize=11)
    ax.set_ylabel('Northing (meters)', fontsize=11)
    ax.set_title('Elevation Map (SRTM/ASTER GDEM)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    ax.ticklabel_format(style='plain', useOffset=False)
    
    plt.tight_layout()
    plt.show()


def _display_slope_map(slope, minx, miny, maxx, maxy, target_crs):
    """Display slope map with appropriate colormap."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Display slope as percentage
    slope_percent = slope * 100
    
    # Use a colormap where red indicates steep slopes (>6%)
    im = ax.imshow(slope_percent,
                   cmap='YlOrRd',
                   origin='upper',
                   extent=[minx, maxx, miny, maxy],
                   interpolation='bilinear',
                   vmin=0,
                   vmax=15)  # Cap display at 15% for better visualization
    
    plt.colorbar(im, ax=ax, label='Slope (%)', pad=0.02)
    ax.set_xlabel('Easting (meters)', fontsize=11)
    ax.set_ylabel('Northing (meters)', fontsize=11)
    ax.set_title('Slope Map (derived from elevation)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    ax.ticklabel_format(style='plain', useOffset=False)
    
    # Add a horizontal line indicating 6% slope threshold
    from matplotlib.patches import Rectangle
    legend_text = ax.text(0.02, 0.98, 'Red areas: >6% slope\n(Hyperloop limit)', 
                         transform=ax.transAxes,
                         fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
# def extract_geometry_efficiency_features(bbox: Tuple[float, float, float, float],
#                                         resolution: float,
#                                         target_crs: str = DEFAULT_CRS_TX_ALBERS,
#                                         pbf_input_path: str = "",
#                                         corridors_output_path: str = "",
#                                         srtm_data_path: Optional[str] = None) -> dict:
#     """
#     Extract all geometry/operational efficiency features at once.
    
#     Args:
#         bbox: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
#         resolution: Pixel resolution in meters
#         target_crs: Target CRS (default: Texas Albers)
#         pbf_input_path: Path to OSM PBF file
#         corridors_output_path: Path to save corridors GeoJSON
#         srtm_data_path: Path to SRTM data (if None, will download)
        
#     Returns:
#         Dictionary containing:
#         - 'elevation': elevation array (meters)
#         - 'slope': slope array (decimal)
#         - 'corridors': binary corridor array
#         - 'efficiency_score': composite efficiency score (0-1)
#         - 'alignment_quality': alignment quality score (0-1)
#         - 'terrain_curvature': terrain curvature (normalized 0-1)
#     """
#     print("="*80)
#     print("EXTRACTING GEOMETRY/OPERATIONAL EFFICIENCY FEATURES")
#     print("="*80)
    
#     features = {}
    
#     # 1. Extract elevation
#     print("\n[1/3] Extracting elevation data...")
#     features['elevation'] = extract_elevation(bbox, resolution, target_crs, srtm_data_path)
    
#     # 2. Calculate slope from elevation
#     print("\n[2/3] Calculating slope from elevation...")
#     features['slope'] = extract_slope(bbox, resolution, target_crs, features['elevation'])
    
#     # 3. Extract existing corridors
#     print("\n[3/3] Extracting existing transportation corridors...")
#     features['corridors'] = extract_existing_corridors(
#         bbox, resolution, target_crs,
#         pbf_input_path=pbf_input_path,
#         output_path=corridors_output_path,
#         include_railways=True
#     )
    
#     # Calculate derived features
#     print("\nCalculating derived efficiency features...")
    
#     # Terrain curvature (Gaussian curvature of elevation surface)
#     elevation = features['elevation']
#     dy, dx = np.gradient(elevation)
#     dyy, dyx = np.gradient(dy)
#     dxy, dxx = np.gradient(dx)
#     terrain_curvature = np.abs(dxx * dyy - dxy * dyx)
#     terrain_curvature = terrain_curvature / (terrain_curvature.max() + 1e-6)  # Normalize
#     features['terrain_curvature'] = terrain_curvature
    
#     # Alignment quality (low curvature + existing corridor = high quality)
#     alignment_quality = np.ones_like(elevation)
#     alignment_quality -= terrain_curvature * 0.5
#     alignment_quality[features['corridors'] == 1] += 0.3
#     alignment_quality = np.clip(alignment_quality, 0, 1)
#     features['alignment_quality'] = alignment_quality
    
#     # Slope penalty (0 = no penalty, 1 = at/exceeds 6% limit)
#     slope_penalty = np.clip(features['slope'] / 0.06, 0, 1)
    
#     # Composite efficiency score
#     efficiency_score = (
#         alignment_quality * 0.4 +
#         (1 - slope_penalty) * 0.3 +
#         (1 - terrain_curvature) * 0.3
#     )
#     features['efficiency_score'] = efficiency_score
    
#     print(f"\nEfficiency score statistics:")
#     print(f"  Mean: {np.mean(efficiency_score):.3f}")
#     print(f"  Min: {np.min(efficiency_score):.3f}")
#     print(f"  Max: {np.max(efficiency_score):.3f}")
    
#     print("\n" + "="*80)
#     print("GEOMETRY/EFFICIENCY FEATURE EXTRACTION COMPLETE")
#     print("="*80)
    
#     return features

# ============================================================================
# CONSTRUCTION COST FEATURES
# ============================================================================

def extract_terrain_classification(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract terrain type classification.
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with terrain codes: 1=flat, 2=rolling, 3=mountainous
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Derive from slope/elevation or use existing terrain classification
    print("    TODO: Classify terrain from elevation/slope data")
    
    # Placeholder: random terrain types
    terrain = np.random.choice([1, 2, 3], size=(height, width), p=[0.5, 0.3, 0.2])
    
    return terrain


def extract_geology(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract soil/geology type for excavation cost estimation.
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with geology codes: 1=soil, 2=soft_rock, 3=hard_rock
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from USGS geology data
    print("    TODO: Extract geology from USGS geological survey data")
    
    # Placeholder: random geology types
    geology = np.random.choice([1, 2, 3], size=(height, width), p=[0.6, 0.25, 0.15])
    
    return geology


def extract_land_use(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract land use type (agricultural, developed, forest, etc.).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with land use codes: 1=agricultural, 2=forest, 3=developed, 4=barren
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from OSM landuse tags or NLCD data
    print("    TODO: Extract land use from OSM or NLCD")
    
    # Placeholder: random land use
    land_use = np.random.choice([1, 2, 3, 4], size=(height, width), p=[0.4, 0.2, 0.3, 0.1])
    
    return land_use


def extract_urban_density(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract urban density (building footprint density).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with density values 0-1 (0=rural, 1=dense urban)
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Calculate from OSM building density or census data
    print("    TODO: Calculate urban density from OSM buildings or census")
    
    # Placeholder: random density with some urban centers
    density = np.random.beta(2, 5, (height, width))  # Skewed toward low density
    
    return density


def extract_water_bodies(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract water bodies (rivers, lakes).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D binary array (1 = water, 0 = land)
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from OSM (natural=water, waterway=*)
    print("    TODO: Extract water bodies from OSM")
    
    # Placeholder: some water features
    water = np.random.choice([0, 1], size=(height, width), p=[0.97, 0.03])
    
    return water


def extract_flood_zones(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract flood risk zones.
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with flood risk: 0=no_risk, 1=moderate, 2=high
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from FEMA flood maps
    print("    TODO: Extract flood zones from FEMA data")
    
    # Placeholder: mostly no risk
    flood_risk = np.random.choice([0, 1, 2], size=(height, width), p=[0.85, 0.10, 0.05])
    
    return flood_risk


# ============================================================================
# ENVIRONMENTAL IMPACT FEATURES
# ============================================================================

def extract_protected_areas(bbox: Tuple[float, float, float, float], 
                           resolution: float,
                           target_crs: str = "EPSG:3083",
                           pbf_input_path: str = "",
                           output_path: str = "",
                           padus_data_path: Optional[str] = None) -> np.ndarray:
    """
    Extract protected areas (national/state parks, wildlife refuges, conservation zones).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        target_crs: Target CRS
        pbf_input_path: Path to OSM PBF file
        output_path: Path to save protected areas GeoJSON
        padus_data_path: Path to USGS PAD-US database shapefile (optional)
        
    Returns:
        2D binary array (1 = protected, 0 = not protected)
    """
    print("Extracting protected areas...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Try PAD-US data first if provided
    if padus_data_path is not None:
        print(f"Loading PAD-US data from {padus_data_path}...")
        # TODO: Implement PAD-US shapefile loading and rasterization
        print("    PAD-US loading not yet implemented, falling back to OSM")
        padus_data_path = None
    
    # Extract from OSM
    if pbf_input_path != "" and output_path != "":
        print("Extracting protected areas from OSM...")
        
        from esy.osmfilter import run_filter, Node, Way, Relation, export_geojson
        
        # Filter for protected areas and leisure parks
        prefilter = {
            Node: {},
            Way: {
                'leisure': ['nature_reserve', 'park'],
                'boundary': ['protected_area', 'national_park'],
                'protect_class': True  # Any protection class
            },
            Relation: {
                'leisure': ['nature_reserve'],
                'boundary': ['protected_area', 'national_park']
            }
        }
        
        whitefilter = [
            (
                ('leisure', 'nature_reserve'),
                ('leisure', 'park'),
                ('boundary', 'protected_area'),
                ('boundary', 'national_park'),
                ('protect_class', None)  # Any value
            )
        ]
        
        blackfilter = []
        
        JSON_outputfile = os.path.join(os.getcwd(), output_path)
        PBF_inputfile = os.path.join(os.getcwd(), pbf_input_path)
        element_name = "protected_areas"
        
        print("Filtering OSM data for protected areas...")
        [Data, Elements] = run_filter(
            element_name,
            PBF_inputfile,
            JSON_outputfile,
            prefilter,
            whitefilter,
            blackfilter,
            NewPreFilterData=True,
            CreateElements=True,
            LoadElements=False,
            verbose=True,
            multiprocess=False
        )
        
        print(f"Found {len(Elements[element_name]['Way'])} protected areas")
        
        # Export polygons
        export_geojson(
            Elements[element_name]['Way'],
            Data,
            filename=output_path,
            jsontype='Polygon'
        )
        
        # Rasterize
        from feature_extraction import rasterize_geometries
        protected_bitmap = rasterize_geometries(
            resolution=resolution,
            geojson_input_path=output_path,
            target_crs=target_crs,
            bbox=bbox
        )
        
    else:
        print("No data provided, creating zero-protection placeholder...")
        protected_bitmap = np.zeros((height, width), dtype=np.uint8)
    
    protected_pct = np.sum(protected_bitmap) / protected_bitmap.size * 100
    print(f"Protected area coverage: {protected_pct:.2f}%")
    
    return protected_bitmap


# def extract_habitat_classification(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
#     """
#     Extract habitat/ecosystem classification.
    
#     Args:
#         bbox: Bounding box
#         resolution: Pixel resolution in meters
        
#     Returns:
#         2D array with habitat codes: 0=developed, 1=agricultural, 2=grassland, 
#                                       3=forest, 4=wetland, 5=critical_habitat
#     """
#     width, height = _create_grid(bbox, resolution)
    
#     # TODO: Extract from USFWS critical habitat data or GAP analysis
#     print("    TODO: Extract habitat data from USFWS or GAP")
    
#     # Placeholder: various habitat types
#     habitat = np.random.choice([0, 1, 2, 3, 4, 5], size=(height, width), 
#                                p=[0.25, 0.35, 0.15, 0.15, 0.05, 0.05])
    
#     return habitat


def extract_land_cover(bbox: Tuple[float, float, float, float], 
                      resolution: float,
                      target_crs: str = "EPSG:3083",
                      pbf_input_path: str = "",
                      output_path: str = "",
                      nlcd_data_path: Optional[str] = None) -> np.ndarray:
    """
    Extract natural land cover type for environmental impact assessment.
    
    This is the PRIMARY ecosystem indicator (prefer this over habitat_classification).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        target_crs: Target CRS
        pbf_input_path: Path to OSM PBF file
        output_path: Path to save land cover GeoJSON
        nlcd_data_path: Path to NLCD raster (optional)
        
    Returns:
        2D array with cover codes: 1=developed, 2=agriculture, 3=grassland,
                                    4=forest, 5=wetland, 6=water
    """
    print("Extracting land cover...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Try NLCD first if provided
    if nlcd_data_path is not None:
        print(f"Loading NLCD data from {nlcd_data_path}...")
        # TODO: Implement NLCD raster loading with rasterio
        print("    NLCD loading not yet implemented, falling back to OSM")
        nlcd_data_path = None
    
    # Extract from OSM (same as land_use but with different classification)
    if pbf_input_path != "" and output_path != "":
        print("Extracting land cover from OSM...")
        
        from esy.osmfilter import run_filter, Node, Way, Relation, export_geojson
        
        prefilter = {
            Node: {},
            Way: {
                'natural': ['water', 'wood', 'wetland', 'grassland', 'scrub', 'heath'],
                'landuse': ['forest', 'farmland', 'farm', 'grass', 'meadow',
                           'residential', 'commercial', 'industrial', 'retail'],
                'waterway': ['river', 'stream']
            },
            Relation: {}
        }
        
        whitefilter = [
            (
                ('natural', 'water'),
                ('natural', 'wood'),
                ('natural', 'wetland'),
                ('natural', 'grassland'),
                ('natural', 'scrub'),
                ('natural', 'heath'),
                ('landuse', 'forest'),
                ('landuse', 'farmland'),
                ('landuse', 'farm'),
                ('landuse', 'grass'),
                ('landuse', 'meadow'),
                ('landuse', 'residential'),
                ('landuse', 'commercial'),
                ('landuse', 'industrial'),
                ('landuse', 'retail'),
                ('waterway', 'river'),
                ('waterway', 'stream')
            )
        ]
        
        blackfilter = []
        
        JSON_outputfile = os.path.join(os.getcwd(), output_path)
        PBF_inputfile = os.path.join(os.getcwd(), pbf_input_path)
        element_name = "land_cover"
        
        [Data, Elements] = run_filter(
            element_name,
            PBF_inputfile,
            JSON_outputfile,
            prefilter,
            whitefilter,
            blackfilter,
            NewPreFilterData=True,
            CreateElements=True,
            LoadElements=False,
            verbose=True,
            multiprocess=False
        )
        
        print(f"Found {len(Elements[element_name]['Way'])} land cover polygons")
        
        export_geojson(
            Elements[element_name]['Way'],
            Data,
            filename=output_path,
            jsontype='Polygon'
        )
        
        # Rasterize with land cover codes
        land_cover = _rasterize_land_cover_polygons(
            output_path, bbox, resolution, target_crs, width, height
        )
        
    else:
        print("No OSM data provided, creating agricultural placeholder...")
        land_cover = np.full((height, width), 2, dtype=int)  # Default to agriculture
    
    # Statistics
    dev_pct = np.sum(land_cover == 1) / land_cover.size * 100
    ag_pct = np.sum(land_cover == 2) / land_cover.size * 100
    grass_pct = np.sum(land_cover == 3) / land_cover.size * 100
    forest_pct = np.sum(land_cover == 4) / land_cover.size * 100
    wetland_pct = np.sum(land_cover == 5) / land_cover.size * 100
    water_pct = np.sum(land_cover == 6) / land_cover.size * 100
    
    print(f"Land cover distribution:")
    print(f"  Developed: {dev_pct:.1f}%")
    print(f"  Agriculture: {ag_pct:.1f}%")
    print(f"  Grassland: {grass_pct:.1f}%")
    print(f"  Forest: {forest_pct:.1f}%")
    print(f"  Wetland: {wetland_pct:.1f}%")
    print(f"  Water: {water_pct:.1f}%")
    
    return land_cover

def _rasterize_land_cover_polygons(geojson_path: str,
                                   bbox: Tuple[float, float, float, float],
                                   resolution: float,
                                   target_crs: str,
                                   width: int,
                                   height: int) -> np.ndarray:
    """Helper to rasterize land cover polygons with appropriate codes."""
    import json
    from shapely.geometry import shape, mapping
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    
    with open(geojson_path) as f:
        geojson = json.load(f)
    
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    
    # Classify geometries
    geometries_by_class = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    
    for feature in geojson["features"]:
        props = feature.get("properties", {})
        natural = props.get("natural", "")
        landuse = props.get("landuse", "")
        waterway = props.get("waterway", "")
        
        # Classify - codes: 1=developed, 2=agriculture, 3=grassland, 4=forest, 5=wetland, 6=water
        if landuse in ['residential', 'commercial', 'industrial', 'retail']:
            code = 1  # developed
        elif landuse in ['farmland', 'farm']:
            code = 2  # agriculture
        elif natural in ['grassland', 'heath'] or landuse in ['grass', 'meadow']:
            code = 3  # grassland
        elif natural in ['wood', 'scrub'] or landuse == 'forest':
            code = 4  # forest
        elif natural == 'wetland':
            code = 5  # wetland
        elif natural == 'water' or waterway in ['river', 'stream']:
            code = 6  # water
        else:
            continue
        
        geom = shape(feature["geometry"])
        projected_geom = shapely_transform(transformer.transform, geom)
        geometries_by_class[code].append((mapping(projected_geom), code))
    
    affine_transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Start with agriculture default
    land_cover = np.full((height, width), 2, dtype=np.uint8)
    
    # Rasterize each class
    for code in [1, 2, 3, 4, 5, 6]:
        if geometries_by_class[code]:
            raster = rasterize(
                geometries_by_class[code],
                out_shape=(height, width),
                transform=affine_transform,
                fill=0,
                dtype=np.uint8
            )
            land_cover[raster == code] = code
    
    return land_cover


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

def extract_water_resources(bbox: Tuple[float, float, float, float], 
                           resolution: float,
                           target_crs: str = "EPSG:3083",
                           water_bodies_array: Optional[np.ndarray] = None,
                           elevation_array: Optional[np.ndarray] = None,
                           usgs_data_path: Optional[str] = None,
                           sensitivity_field: Optional[str] = None) -> np.ndarray:
    """
    Extract sensitive water resources (aquifers, watersheds, riparian zones).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        target_crs: Target CRS
        water_bodies_array: Pre-computed water bodies array
        elevation_array: Pre-computed elevation array (for watershed delineation)
        usgs_data_path: Path to USGS water resources shapefile (optional)
        
    Returns:
        2D array with sensitivity: 0=not_sensitive, 1=moderate, 2=high
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

def extract_biodiversity_hotspots(bbox: Tuple[float, float, float, float], 
                                  resolution: float,
                                  target_crs: str = "EPSG:3083",
                                  land_cover_array: Optional[np.ndarray] = None,
                                  protected_areas_array: Optional[np.ndarray] = None,
                                  conservation_db_path: Optional[str] = None,
                                  data_type: Optional[str] = None) -> np.ndarray:
    """
    Extract biodiversity hotspots and areas of high ecological value.
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        target_crs: Target CRS
        land_cover_array: Pre-computed land cover (used for estimation)
        protected_areas_array: Pre-computed protected areas (used for estimation)
        conservation_db_path: Path to conservation/species richness database (optional)
        
    Returns:
        2D array with biodiversity score 0-1 (0=low, 1=high)
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
            data_type  # Optional: 'raster', 'shapefile', 'points', 'directory'
        )
        
        if biodiversity is not None and biodiversity.max() > 0:
            return biodiversity
        else:
            print("    Falling back to estimation")
    
    if conservation_db_path is None:
        print("Estimating biodiversity from land cover and protected areas...")
        
        # Initialize with low biodiversity
        biodiversity = np.zeros((height, width), dtype=float)
        
        # Get land cover and protected areas if available
        if land_cover_array is not None:
            # Assign base biodiversity scores by land cover type
            # Higher diversity in natural, undisturbed ecosystems
            biodiversity[land_cover_array == 1] = 0.1  # developed - low
            biodiversity[land_cover_array == 2] = 0.3  # agriculture - low-moderate
            biodiversity[land_cover_array == 3] = 0.5  # grassland - moderate
            biodiversity[land_cover_array == 4] = 0.7  # forest - high
            biodiversity[land_cover_array == 5] = 0.9  # wetland - very high
            biodiversity[land_cover_array == 6] = 0.6  # water - moderate-high
        else:
            # No land cover data - use moderate baseline
            biodiversity[:] = 0.4
            print("    (No land cover data, using uniform baseline)")
        
        # Boost biodiversity in protected areas
        if protected_areas_array is not None:
            protected_mask = protected_areas_array == 1
            biodiversity[protected_mask] = np.minimum(biodiversity[protected_mask] + 0.2, 1.0)
        
        # Add some natural variation (habitat heterogeneity)
        # More varied landscapes support more biodiversity
        from scipy.ndimage import gaussian_filter
        
        # Calculate local variance as a proxy for heterogeneity
        local_mean = uniform_filter(biodiversity, size=5, mode='reflect')
        local_sq_mean = uniform_filter(biodiversity**2, size=5, mode='reflect')
        local_variance = np.maximum(0, local_sq_mean - local_mean**2)
        
        # Normalize variance
        if local_variance.max() > 0:
            heterogeneity = local_variance / local_variance.max()
        else:
            heterogeneity = np.zeros_like(biodiversity)
        
        # Add heterogeneity bonus (up to +0.15)
        biodiversity = np.minimum(biodiversity + heterogeneity * 0.15, 1.0)
        
        # Smooth slightly to avoid noise
        biodiversity = gaussian_filter(biodiversity, sigma=1.0)
        
        # Ensure 0-1 range
        biodiversity = np.clip(biodiversity, 0, 1)
        
        print(f"Biodiversity score statistics:")
        print(f"  Mean: {biodiversity.mean():.3f}")
        print(f"  Min: {biodiversity.min():.3f}")
        print(f"  Max: {biodiversity.max():.3f}")
        print(f"  High biodiversity (>0.7): {np.sum(biodiversity > 0.7) / biodiversity.size * 100:.2f}%")
        print(f"  (Note: This is estimated. Provide conservation_db_path for accurate data)")
    
    return biodiversity

# TODO: Implement NRCS soil survey loading
def extract_agricultural_preservation(bbox: Tuple[float, float, float, float], 
                                     resolution: float,
                                     target_crs: str = "EPSG:3083",
                                     land_cover_array: Optional[np.ndarray] = None,
                                     elevation_array: Optional[np.ndarray] = None,
                                     nrcs_data_path: Optional[str] = None) -> np.ndarray:
    """
    Extract agricultural preservation zones (prime farmland).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        target_crs: Target CRS
        land_cover_array: Pre-computed land cover (used for estimation)
        elevation_array: Pre-computed elevation (used for estimation)
        nrcs_data_path: Path to USDA NRCS soil survey shapefile (optional)
        
    Returns:
        2D binary array (1 = prime farmland, 0 = not prime)
    """
    print("Extracting agricultural preservation zones...")
    
    width, height = _create_grid(bbox, resolution)
    
    # Try NRCS data if provided
    if nrcs_data_path is not None:
        print(f"Loading NRCS prime farmland data from {nrcs_data_path}...")
        # TODO: Implement NRCS soil survey loading
        print("    NRCS data loading not yet implemented")
        nrcs_data_path = None
    
    if nrcs_data_path is None:
        print("Estimating prime farmland from land cover and elevation...")
        
        prime_farmland = np.zeros((height, width), dtype=np.uint8)
        
        # Prime farmland criteria (simplified):
        # 1. Currently agricultural or grassland
        # 2. Relatively flat terrain (slope < 8%)
        # 3. Not too high elevation (productive soil zones)
        
        if land_cover_array is not None:
            # Start with agricultural and grassland areas
            ag_mask = (land_cover_array == 2) | (land_cover_array == 3)
            
            if elevation_array is not None:
                # Calculate slope
                from scipy.ndimage import sobel
                dy, dx = np.gradient(elevation_array)
                slope = np.sqrt(dx**2 + dy**2) / resolution
                
                # Prime farmland: agricultural + gentle slope + moderate elevation
                gentle_slope = slope < 0.08  # Less than 8% slope
                
                # Moderate elevation (middle 60% of elevation range)
                elev_range = elevation_array.max() - elevation_array.min()
                if elev_range > 0:
                    normalized_elev = (elevation_array - elevation_array.min()) / elev_range
                    moderate_elev = (normalized_elev > 0.2) & (normalized_elev < 0.8)
                else:
                    moderate_elev = np.ones_like(elevation_array, dtype=bool)
                
                prime_farmland = (ag_mask & gentle_slope & moderate_elev).astype(np.uint8)
            else:
                # No elevation data - just use agricultural areas
                prime_farmland = ag_mask.astype(np.uint8)
                print("    (No elevation data, using only land cover)")
        else:
            # No land cover data - estimate 30% as prime farmland
            print("    (No land cover data, using random 30% baseline)")
            prime_farmland = np.random.choice([0, 1], size=(height, width), p=[0.70, 0.30])
        
        prime_pct = np.sum(prime_farmland) / prime_farmland.size * 100
        print(f"Prime farmland coverage: {prime_pct:.1f}%")
        print(f"  (Note: This is estimated. Provide nrcs_data_path for accurate data)")
    
    return prime_farmland
