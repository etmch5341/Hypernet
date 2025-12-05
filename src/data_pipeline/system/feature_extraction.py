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

def extract_elevation(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract elevation data (Digital Elevation Model).
    
    Args:
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        resolution: Pixel resolution in meters
        
    Returns:
        2D array of elevation values in meters
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Implement actual SRTM/ASTER GDEM data extraction
    # For now, return placeholder data
    print("    TODO: Implement SRTM/ASTER elevation extraction")
    
    # Placeholder: random elevation between 0-500m
    elevation = np.random.uniform(0, 500, (height, width))
    
    return elevation


def extract_slope(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract slope/gradient data (can be derived from elevation).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array of slope percentages
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Calculate from elevation DEM using gradient
    print("    TODO: Calculate slope from elevation data")
    
    # Placeholder: random slope 0-30%
    slope = np.random.uniform(0, 30, (height, width))
    
    return slope


def extract_existing_corridors(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract existing transportation corridors (railways, highways).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D binary array (1 = corridor exists, 0 = no corridor)
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from OSM (highway=*, railway=*)
    print("    TODO: Extract corridors from OpenStreetMap")
    
    # Placeholder: some random corridors
    corridors = np.random.choice([0, 1], size=(height, width), p=[0.95, 0.05])
    
    return corridors


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

def extract_protected_areas(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract protected areas (national/state parks, wildlife refuges).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D binary array (1 = protected, 0 = not protected)
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from USGS Protected Areas Database (PAD-US)
    print("    TODO: Extract protected areas from PAD-US database")
    
    # Placeholder: some protected areas
    protected = np.random.choice([0, 1], size=(height, width), p=[0.92, 0.08])
    
    return protected


def extract_habitat_classification(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract habitat/ecosystem classification.
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with habitat codes: 0=developed, 1=agricultural, 2=grassland, 
                                      3=forest, 4=wetland, 5=critical_habitat
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from USFWS critical habitat data or GAP analysis
    print("    TODO: Extract habitat data from USFWS or GAP")
    
    # Placeholder: various habitat types
    habitat = np.random.choice([0, 1, 2, 3, 4, 5], size=(height, width), 
                               p=[0.25, 0.35, 0.15, 0.15, 0.05, 0.05])
    
    return habitat


def extract_land_cover(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract natural land cover type (for environmental impact).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with cover codes: 1=developed, 2=agriculture, 3=grassland,
                                    4=forest, 5=wetland, 6=water
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from NLCD (National Land Cover Database)
    print("    TODO: Extract land cover from NLCD")
    
    # Placeholder: various cover types
    cover = np.random.choice([1, 2, 3, 4, 5, 6], size=(height, width),
                            p=[0.20, 0.35, 0.15, 0.20, 0.05, 0.05])
    
    return cover


def extract_water_resources(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract sensitive water resources (aquifers, watersheds, riparian zones).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with sensitivity: 0=not_sensitive, 1=moderate, 2=high
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from USGS water resources data
    print("    TODO: Extract water resources from USGS")
    
    # Placeholder: varying sensitivity
    sensitivity = np.random.choice([0, 1, 2], size=(height, width), p=[0.75, 0.15, 0.10])
    
    return sensitivity


def extract_biodiversity_hotspots(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract biodiversity hotspots and areas of high ecological value.
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D array with biodiversity score 0-1 (0=low, 1=high)
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from conservation databases or species richness data
    print("    TODO: Extract biodiversity data from conservation databases")
    
    # Placeholder: mostly low biodiversity with some hotspots
    biodiversity = np.random.beta(2, 8, (height, width))
    
    return biodiversity


def extract_agricultural_preservation(bbox: Tuple[float, float, float, float], resolution: float) -> np.ndarray:
    """
    Extract agricultural preservation zones (prime farmland).
    
    Args:
        bbox: Bounding box
        resolution: Pixel resolution in meters
        
    Returns:
        2D binary array (1 = prime farmland, 0 = not prime)
    """
    width, height = _create_grid(bbox, resolution)
    
    # TODO: Extract from USDA NRCS soil survey (prime farmland classification)
    print("    TODO: Extract prime farmland from USDA NRCS")
    
    # Placeholder: some prime farmland
    prime_farmland = np.random.choice([0, 1], size=(height, width), p=[0.70, 0.30])
    
    return prime_farmland
