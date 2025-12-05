'''
Load .npz raster test sets and visualize with goal points on basemap background.
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import ListedColormap
from pyproj import Transformer

DEFAULT_CRS_OSM = "EPSG:4326"  # WGS84

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


# Load and visualize Austin test set
print("\n=== Processing Austin Test Set ===")
data = np.load('./src/sample-test-set/austin_test_raster.npz', allow_pickle=True)
austin_raster = data['raster']
austin_goal_points = data['goal_points'].tolist()
austin_bbox = data['bbox'].tolist()
austin_crs = str(data['target_crs'])

display_raster_with_goals_and_basemap(
    austin_raster, austin_goal_points, austin_bbox, austin_crs,
    title="Austin Test Raster with Goal Points",
    output_file='/workspace/outputs/austin_test_raster_basemap.png',
    basemap_style='light',
    color_style='hot_pink',
    raster_alpha=0.85,
    goal_color='red',
    goal_marker='*',
    goal_size=400
)

# Load and visualize Seattle test set
print("\n=== Processing Seattle Test Set ===")
data = np.load('./src/sample-test-set/seattle_test_raster.npz', allow_pickle=True)
seattle_raster = data['raster']
seattle_goal_points = data['goal_points'].tolist()
seattle_bbox = data['bbox'].tolist()
seattle_crs = str(data['target_crs'])

display_raster_with_goals_and_basemap(
    seattle_raster, seattle_goal_points, seattle_bbox, seattle_crs,
    title="Seattle Test Raster with Goal Points",
    output_file='/workspace/outputs/seattle_test_raster_basemap.png',
    basemap_style='light',
    color_style='royal_blue',
    raster_alpha=0.85,
    goal_color='orange',
    goal_marker='*',
    goal_size=400
)

# Load and visualize Portland test set
print("\n=== Processing Portland Test Set ===")
data = np.load('./src/sample-test-set/portland_test_raster.npz', allow_pickle=True)
portland_raster = data['raster']
portland_goal_points = data['goal_points'].tolist()
portland_bbox = data['bbox'].tolist()
portland_crs = str(data['target_crs'])

display_raster_with_goals_and_basemap(
    portland_raster, portland_goal_points, portland_bbox, portland_crs,
    title="Portland Test Raster with Goal Points",
    output_file='/workspace/outputs/portland_test_raster_basemap.png',
    basemap_style='satellite',
    color_style='neon_cyan',
    raster_alpha=0.8,
    goal_color='yellow',
    goal_marker='*',
    goal_size=400
)

print("\n=== All visualizations complete! ===")