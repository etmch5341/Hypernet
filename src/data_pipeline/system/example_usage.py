"""
Example usage of the cost map generation system.

This script demonstrates how to:
1. Define a bounding box and resolution
2. Create instances of each cost map type
3. Extract features
4. Generate cost maps
5. Save and visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import feature_extraction as features
from geometry_cost_map import GeometryCostMap
from construction_cost_map import ConstructionCostMap
from environmental_cost_map import EnvironmentalCostMap


def main():
    """
    Generate all three cost maps for the Texas Triangle region.
    """
    
    # Define bounding box for Texas Triangle
    # (min_lon, min_lat, max_lon, max_lat)
    # This is a rough approximation covering Houston-Dallas-San Antonio-Austin
    texas_triangle_bbox = (-99.0, 29.0, -95.0, 33.0)
    
    # Set resolution (meters per pixel)
    # Lower resolution = more detail but larger computation
    # For initial testing, use 1km resolution
    resolution = 1000.0  # 1000 meters = 1 km per pixel
    
    print("="*70)
    print("TEXAS TRIANGLE HYPERLOOP COST MAP GENERATION")
    print("="*70)
    print(f"\nBounding box: {texas_triangle_bbox}")
    print(f"Resolution: {resolution}m per pixel")
    print()
    
    # ========================================================================
    # 1. GEOMETRY COST MAP
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING GEOMETRY COST MAP")
    print("="*70)
    
    geometry_map = GeometryCostMap(texas_triangle_bbox, resolution)
    print(f"Grid dimensions: {geometry_map.width} x {geometry_map.height} pixels")
    
    # Extract features
    geometry_map.extract_features(features)
    
    # Generate cost map
    geometry_cost = geometry_map.generate_cost_map()
    
    # Save
    geometry_map.save_cost_map('/home/claude/geometry_cost_map.npy')
    
    # ========================================================================
    # 2. CONSTRUCTION COST MAP
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING CONSTRUCTION COST MAP")
    print("="*70)
    
    construction_map = ConstructionCostMap(texas_triangle_bbox, resolution)
    print(f"Grid dimensions: {construction_map.width} x {construction_map.height} pixels")
    
    # Extract features
    construction_map.extract_features(features)
    
    # Generate cost map
    construction_cost = construction_map.generate_cost_map()
    
    # Save
    construction_map.save_cost_map('/home/claude/construction_cost_map.npy')
    
    # ========================================================================
    # 3. ENVIRONMENTAL COST MAP
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING ENVIRONMENTAL COST MAP")
    print("="*70)
    
    environmental_map = EnvironmentalCostMap(texas_triangle_bbox, resolution)
    print(f"Grid dimensions: {environmental_map.width} x {environmental_map.height} pixels")
    
    # Extract features
    environmental_map.extract_features(features)
    
    # Generate cost map
    environmental_cost = environmental_map.generate_cost_map()
    
    # Save
    environmental_map.save_cost_map('/home/claude/environmental_cost_map.npy')
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Geometry cost map
    im1 = axes[0].imshow(geometry_cost, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('Geometry Cost Map\n(Path Alignment & Operational Efficiency)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='Cost Multiplier')
    
    # Construction cost map
    im2 = axes[1].imshow(construction_cost, cmap='YlOrRd', aspect='auto')
    axes[1].set_title('Construction Cost Map\n(Terrain, Urban, Infrastructure)')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='Cost Multiplier')
    
    # Environmental cost map
    im3 = axes[2].imshow(environmental_cost, cmap='YlOrRd', aspect='auto')
    axes[2].set_title('Environmental Cost Map\n(Protected Areas, Habitats, Ecosystems)')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2], label='Cost Multiplier')
    
    plt.tight_layout()
    plt.savefig('/home/claude/cost_maps_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to cost_maps_comparison.png")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("COST MAP STATISTICS")
    print("="*70)
    
    print("\nGeometry Cost Map:")
    print(f"  Min: {geometry_cost.min():.3f}")
    print(f"  Max: {geometry_cost.max():.3f}")
    print(f"  Mean: {geometry_cost.mean():.3f}")
    print(f"  Std: {geometry_cost.std():.3f}")
    
    print("\nConstruction Cost Map:")
    print(f"  Min: {construction_cost.min():.3f}")
    print(f"  Max: {construction_cost.max():.3f}")
    print(f"  Mean: {construction_cost.mean():.3f}")
    print(f"  Std: {construction_cost.std():.3f}")
    
    print("\nEnvironmental Cost Map:")
    print(f"  Min: {environmental_cost.min():.3f}")
    print(f"  Max: {environmental_cost.max():.3f}")
    print(f"  Mean: {environmental_cost.mean():.3f}")
    print(f"  Std: {environmental_cost.std():.3f}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Implement real data extraction in feature_extraction.py")
    print("2. Integrate cost maps with A-MHA* pathfinding algorithm")
    print("3. Define start/goal points for Texas Triangle routes")
    print("4. Run pathfinding with different objective weightings")


if __name__ == "__main__":
    main()
