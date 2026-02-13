"""
Test script for environmental impact feature extraction functions.

This demonstrates how to use all 6 implemented functions with your existing data.
"""

import numpy as np

# Mock imports for demonstration
from feature_extraction import (
    extract_protected_areas,
    # extract_habitat_classification,
    extract_land_cover,
    extract_water_resources,
    extract_biodiversity_hotspots,
    extract_agricultural_preservation
)

def test_environmental_features():
    """Test all environmental impact features."""
    
    print("="*80)
    print("TESTING ENVIRONMENTAL IMPACT FEATURE EXTRACTION")
    print("="*80)
    
    # Define test parameters
    bbox = (-98.0, 29.0, -97.5, 29.5)  # Austin, TX area
    resolution = 100  # 100 meters per pixel
    target_crs = "EPSG:3083"  # Texas Albers
    
    # Your OSM data path
    pbf_path = "./data/input/texas-latest.osm.pbf"
    
    print(f"\nTest Parameters:")
    print(f"  Bounding Box: {bbox}")
    print(f"  Resolution: {resolution}m per pixel")
    print(f"  CRS: {target_crs}")
    print()
    
    # We'll store extracted features to reuse
    features_cache = {}
    
    # ========================================================================
    # TEST 1: Protected Areas (from OSM)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: PROTECTED AREAS")
    print("="*80)
    
    try:
        protected = extract_protected_areas(
            bbox, resolution, target_crs,
            pbf_input_path=pbf_path,
            output_path="test_protected.geojson"
        )
        
        print("\n✓ Protected areas extraction successful!")
        print(f"  Shape: {protected.shape}")
        print(f"  Unique values: {np.unique(protected)}")
        print(f"  Protected coverage: {np.sum(protected==1)/protected.size*100:.2f}%")
        
        features_cache['protected'] = protected
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Skipping - provide pbf_input_path to test")
    
    # ========================================================================
    # TEST 2: Land Cover (from OSM) - PRIMARY ECOSYSTEM INDICATOR
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: LAND COVER (Primary Ecosystem Indicator)")
    print("="*80)
    
    try:
        land_cover = extract_land_cover(
            bbox, resolution, target_crs,
            pbf_input_path=pbf_path,
            output_path="test_land_cover.geojson"
        )
        
        print("\n✓ Land cover extraction successful!")
        print(f"  Shape: {land_cover.shape}")
        print(f"  Unique values: {np.unique(land_cover)}")
        print(f"  Developed: {np.sum(land_cover==1)/land_cover.size*100:.1f}%")
        print(f"  Agriculture: {np.sum(land_cover==2)/land_cover.size*100:.1f}%")
        print(f"  Grassland: {np.sum(land_cover==3)/land_cover.size*100:.1f}%")
        print(f"  Forest: {np.sum(land_cover==4)/land_cover.size*100:.1f}%")
        print(f"  Wetland: {np.sum(land_cover==5)/land_cover.size*100:.1f}%")
        print(f"  Water: {np.sum(land_cover==6)/land_cover.size*100:.1f}%")
        
        features_cache['land_cover'] = land_cover
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Skipping - provide pbf_input_path to test")
    
    # ========================================================================
    # TEST 3: Habitat Classification (OPTIONAL - redundant with land_cover)
    # ========================================================================
    # print("\n" + "="*80)
    # print("TEST 3: HABITAT CLASSIFICATION (Redundant - Skip in Production)")
    # print("="*80)
    
    # print("⚠️  WARNING: habitat_classification is redundant with land_cover")
    # print("   Recommended: Use land_cover instead to avoid double-counting")
    # print("   This test is included for completeness only.")
    
    # try:
    #     # If we have land_cover, derive habitat from it (efficient)
    #     if 'land_cover' in features_cache:
    #         habitat = extract_habitat_classification(
    #             bbox, resolution, target_crs,
    #             land_cover_array=features_cache['land_cover']  # Reuse!
    #         )
    #     else:
    #         habitat = extract_habitat_classification(
    #             bbox, resolution, target_crs,
    #             pbf_input_path=pbf_path,
    #             output_path="test_habitat.geojson"
    #         )
        
    #     print("\n✓ Habitat classification successful (but not recommended)!")
    #     print(f"  Shape: {habitat.shape}")
    #     print(f"  Unique values: {np.unique(habitat)}")
        
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    
    # ========================================================================
    # TEST 4: Water Resources (estimated from water proximity + elevation)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 4: WATER RESOURCE SENSITIVITY")
    print("="*80)
    
    try:
        # Get water bodies if we have them from construction tests
        water_bodies = features_cache.get('water_bodies', None)
        elevation = features_cache.get('elevation', None)
        
        water_resources = extract_water_resources(
            bbox, resolution, target_crs,
            water_bodies_array=water_bodies,
            elevation_array=elevation
        )
        
        print("\n✓ Water resources sensitivity calculated!")
        print(f"  Shape: {water_resources.shape}")
        print(f"  Unique values: {np.unique(water_resources)}")
        print(f"  Not sensitive: {np.sum(water_resources==0)/water_resources.size*100:.1f}%")
        print(f"  Moderate sensitivity: {np.sum(water_resources==1)/water_resources.size*100:.1f}%")
        print(f"  High sensitivity: {np.sum(water_resources==2)/water_resources.size*100:.1f}%")
        
        features_cache['water_resources'] = water_resources
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # ========================================================================
    # TEST 5: Biodiversity Hotspots (estimated from land cover + protected)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 5: BIODIVERSITY HOTSPOTS")
    print("="*80)
    
    try:
        # Use previously extracted features
        land_cover = features_cache.get('land_cover', None)
        protected = features_cache.get('protected', None)
        
        biodiversity = extract_biodiversity_hotspots(
            bbox, resolution, target_crs,
            land_cover_array=land_cover,
            protected_areas_array=protected
        )
        
        print("\n✓ Biodiversity hotspots calculated!")
        print(f"  Shape: {biodiversity.shape}")
        print(f"  Range: {biodiversity.min():.3f} to {biodiversity.max():.3f}")
        print(f"  Mean score: {biodiversity.mean():.3f}")
        print(f"  Low (<0.3): {np.sum(biodiversity<0.3)/biodiversity.size*100:.1f}%")
        print(f"  Moderate (0.3-0.7): {np.sum((biodiversity>=0.3)&(biodiversity<0.7))/biodiversity.size*100:.1f}%")
        print(f"  High (>0.7): {np.sum(biodiversity>=0.7)/biodiversity.size*100:.1f}%")
        
        # Check wetlands have high biodiversity
        if land_cover is not None and 5 in land_cover:
            wetland_bio = biodiversity[land_cover == 5].mean()
            print(f"  Wetland biodiversity: {wetland_bio:.3f} (should be high)")
        
        features_cache['biodiversity'] = biodiversity
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # ========================================================================
    # TEST 6: Agricultural Preservation (estimated from land cover + slope)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 6: AGRICULTURAL PRESERVATION ZONES")
    print("="*80)
    
    try:
        land_cover = features_cache.get('land_cover', None)
        elevation = features_cache.get('elevation', None)
        
        ag_preservation = extract_agricultural_preservation(
            bbox, resolution, target_crs,
            land_cover_array=land_cover,
            elevation_array=elevation
        )
        
        print("\n✓ Agricultural preservation zones calculated!")
        print(f"  Shape: {ag_preservation.shape}")
        print(f"  Unique values: {np.unique(ag_preservation)}")
        print(f"  Prime farmland: {np.sum(ag_preservation==1)/ag_preservation.size*100:.1f}%")
        print(f"  Not prime: {np.sum(ag_preservation==0)/ag_preservation.size*100:.1f}%")
        
        features_cache['ag_preservation'] = ag_preservation
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # ========================================================================
    # INTEGRATION TEST: Environmental Cost Map
    # ========================================================================
    print("\n" + "="*80)
    print("INTEGRATION TEST: ENVIRONMENTAL COST MAP")
    print("="*80)
    
    try:
        from environmental_cost_map import EnvironmentalCostMap
        import feature_extraction
        
        # Use CORRECTED version (without habitat_classification)
        print("Creating Environmental Cost Map (using corrected version)...")
        cost_map = EnvironmentalCostMap(bbox, resolution)
        
        print("\nExtracting all features...")
        cost_map.extract_features(feature_extraction)
        
        print("\nGenerating combined cost map...")
        final_cost_map = cost_map.generate_cost_map()
        
        print("\n✓ Environmental cost map generation successful!")
        print(f"  Shape: {final_cost_map.shape}")
        print(f"  Normalized cost range: {final_cost_map.min():.3f} to {final_cost_map.max():.3f}")
        print(f"  Mean cost: {final_cost_map.mean():.3f}")
        
        # Check that it's normalized
        assert final_cost_map.min() >= 0 and final_cost_map.max() <= 1, \
            "Cost map should be normalized to 0-1 range!"
        print("  ✓ Verified: Cost map is properly normalized to 0-1 range")
        
        # Analyze high-impact areas
        high_impact_pct = np.sum(final_cost_map > 0.7) / final_cost_map.size * 100
        print(f"  High-impact areas (>0.7): {high_impact_pct:.1f}%")
        
    except ImportError:
        print("✗ Could not import EnvironmentalCostMap")
        print("  Make sure environmental_cost_map.py is in your path")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # ========================================================================
    # VALIDATION TESTS
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION TESTS")
    print("="*80)
    
    print("\nChecking data quality...")
    
    # Check biodiversity in different land covers
    if 'biodiversity' in features_cache and 'land_cover' in features_cache:
        biodiversity = features_cache['biodiversity']
        land_cover = features_cache['land_cover']
        
        print("\nBiodiversity by land cover type:")
        for code, name in [(1, 'Developed'), (2, 'Agriculture'), 
                          (3, 'Grassland'), (4, 'Forest'), 
                          (5, 'Wetland'), (6, 'Water')]:
            if code in land_cover:
                bio_mean = biodiversity[land_cover == code].mean()
                print(f"  {name}: {bio_mean:.3f}")
        
        # Validation: Wetlands should have highest biodiversity
        if 5 in land_cover:
            wetland_bio = biodiversity[land_cover == 5].mean()
            developed_bio = biodiversity[land_cover == 1].mean() if 1 in land_cover else 0
            
            if wetland_bio > developed_bio:
                print("\n✓ PASS: Wetlands have higher biodiversity than developed areas")
            else:
                print("\n✗ FAIL: Wetlands should have higher biodiversity!")
    
    # Check water resources near water
    if 'water_resources' in features_cache and 'water_bodies' in features_cache:
        water_resources = features_cache['water_resources']
        water_bodies = features_cache['water_bodies']
        
        # Near water should have higher sensitivity
        near_water = water_resources[water_bodies == 1]
        away_water = water_resources[water_bodies == 0]
        
        if near_water.size > 0 and away_water.size > 0:
            if near_water.mean() > away_water.mean():
                print("✓ PASS: Water resources more sensitive near water bodies")
            else:
                print("✗ FAIL: Water resources should be more sensitive near water!")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print()


def test_feature_reuse():
    """Demonstrate efficient feature reuse."""
    
    print("="*80)
    print("FEATURE REUSE EFFICIENCY TEST")
    print("="*80)
    print()
    
    bbox = (-98.0, 29.0, -97.5, 29.5)
    resolution = 100
    target_crs = "EPSG:3083"
    
    print("INEFFICIENT approach (extracting multiple times):")
    print("  1. extract_land_cover() - extracts from OSM")
    print("  2. extract_habitat_classification() - extracts from OSM again! ❌")
    print("  3. extract_biodiversity_hotspots() - no land_cover input")
    print("     Result: OSM parsed 3 times, slow!")
    print()
    
    print("EFFICIENT approach (extract once, reuse):")
    print("  1. land_cover = extract_land_cover() - extracts from OSM")
    print("  2. habitat = extract_habitat_classification(")
    print("        land_cover_array=land_cover) - derives from existing! ✓")
    print("  3. biodiversity = extract_biodiversity_hotspots(")
    print("        land_cover_array=land_cover) - reuses existing! ✓")
    print("     Result: OSM parsed once, fast!")
    print()
    
    print("Code example:")
    print("""
    # Extract base features once
    elevation = extract_elevation(bbox, resolution)
    water_bodies = extract_water_bodies(bbox, resolution, pbf_path=pbf_path)
    land_cover = extract_land_cover(bbox, resolution, pbf_path=pbf_path)
    protected = extract_protected_areas(bbox, resolution, pbf_path=pbf_path)
    
    # Reuse them efficiently
    water_resources = extract_water_resources(
        bbox, resolution,
        water_bodies_array=water_bodies,  # ✓ Reuse
        elevation_array=elevation          # ✓ Reuse
    )
    
    biodiversity = extract_biodiversity_hotspots(
        bbox, resolution,
        land_cover_array=land_cover,      # ✓ Reuse
        protected_areas_array=protected    # ✓ Reuse
    )
    
    ag_preservation = extract_agricultural_preservation(
        bbox, resolution,
        land_cover_array=land_cover,      # ✓ Reuse
        elevation_array=elevation          # ✓ Reuse
    )
    """)
    print()


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║      ENVIRONMENTAL IMPACT FEATURES - TEST SUITE                        ║
    ║                                                                        ║
    ║  This script tests all 6 environmental impact feature extraction      ║
    ║  functions and demonstrates integration with EnvironmentalCostMap.    ║
    ║                                                                        ║
    ║  Before running:                                                       ║
    ║  1. Add functions from environmental_impact_features.py to            ║
    ║     feature_extraction.py                                              ║
    ║  2. Update pbf_path variable with your OSM data file                  ║
    ║  3. Use CORRECTED EnvironmentalCostMap (removes habitat redundancy)   ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Uncomment to run tests
    # test_feature_reuse()
    test_environmental_features()
    
    print("To run tests, uncomment the function calls at the end of this script.")
    print("Make sure to integrate the functions into feature_extraction.py first!")