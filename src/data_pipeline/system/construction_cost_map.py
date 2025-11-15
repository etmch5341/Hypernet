"""
Construction Cost Map for minimizing construction expenses.
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
from cost_map_base import CostMapBase
import feature_extraction as features


class ConstructionCostMap(CostMapBase):
    """
    Cost map for construction cost optimization.
    
    Considers:
    - Terrain difficulty (flat vs mountainous)
    - Elevation changes (tunneling, cut-and-fill)
    - Geology (soil vs rock excavation)
    - Land use (clearing costs)
    - Urban density (land acquisition, construction complexity)
    - Water bodies (bridge construction)
    - Flood zones (elevated construction)
    """
    
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define features needed for construction cost calculation.
        """
        return [
            ('elevation', features.extract_elevation),
            ('terrain_classification', features.extract_terrain_classification),
            ('geology', features.extract_geology),
            ('land_use', features.extract_land_use),
            ('urban_density', features.extract_urban_density),
            ('water_bodies', features.extract_water_bodies),
            ('flood_zones', features.extract_flood_zones),
        ]
    
    def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features into construction cost map.
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Cost map with multipliers (1.0 = baseline)
        """
        terrain = features_dict['terrain_classification']
        geology = features_dict['geology']
        land_use = features_dict['land_use']
        urban_density = features_dict['urban_density']
        water = features_dict['water_bodies']
        flood = features_dict['flood_zones']
        
        # Initialize with baseline cost
        cost_map = np.ones_like(terrain, dtype=float)
        
        # Terrain classification multipliers
        # 1=flat: 1.0x, 2=rolling: 2.0x, 3=mountainous: 4.5x
        terrain_multiplier = np.ones_like(terrain, dtype=float)
        terrain_multiplier[terrain == 2] = 2.0
        terrain_multiplier[terrain == 3] = 4.5
        cost_map *= terrain_multiplier
        
        # Geology multipliers (excavation difficulty)
        # 1=soil: 1.0x, 2=soft_rock: 1.8x, 3=hard_rock: 3.0x
        geology_multiplier = np.ones_like(geology, dtype=float)
        geology_multiplier[geology == 2] = 1.8
        geology_multiplier[geology == 3] = 3.0
        cost_map *= geology_multiplier
        
        # Land use multipliers (clearing and preparation costs)
        # 1=agricultural: 1.2x, 2=forest: 1.8x, 3=developed: 2.5x, 4=barren: 1.0x
        land_use_multiplier = np.ones_like(land_use, dtype=float)
        land_use_multiplier[land_use == 1] = 1.2
        land_use_multiplier[land_use == 2] = 1.8
        land_use_multiplier[land_use == 3] = 2.5
        land_use_multiplier[land_use == 4] = 1.0
        cost_map *= land_use_multiplier
        
        # Urban density multiplier (land acquisition and complexity)
        # Linear scale: 0=1.0x to 1.0=3.5x
        urban_multiplier = 1.0 + (urban_density * 2.5)
        cost_map *= urban_multiplier
        
        # Water bodies (bridges, special foundations)
        # Add 3.0x where water exists
        water_multiplier = np.where(water == 1, 3.0, 1.0)
        cost_map *= water_multiplier
        
        # Flood zones (elevated construction required)
        # 0=no_risk: 1.0x, 1=moderate: 1.5x, 2=high: 2.2x
        flood_multiplier = np.ones_like(flood, dtype=float)
        flood_multiplier[flood == 1] = 1.5
        flood_multiplier[flood == 2] = 2.2
        cost_map *= flood_multiplier
        
        print(f"  Construction cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        return cost_map
