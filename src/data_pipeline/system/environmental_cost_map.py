"""
Environmental Cost Map for minimizing environmental impact.
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
from cost_map_base import CostMapBase
import feature_extraction as features


class EnvironmentalCostMap(CostMapBase):
    """
    Cost map for environmental impact minimization.
    
    Considers:
    - Protected areas (parks, refuges)
    - Habitat classification (critical habitats)
    - Land cover (natural vegetation, wetlands)
    - Water resources (aquifers, watersheds)
    - Biodiversity hotspots
    - Agricultural preservation zones
    """
    
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define features needed for environmental impact calculation.
        """
        return [
            ('protected_areas', features.extract_protected_areas),
            ('habitat_classification', features.extract_habitat_classification),
            ('land_cover', features.extract_land_cover),
            ('water_resources', features.extract_water_resources),
            ('biodiversity_hotspots', features.extract_biodiversity_hotspots),
            ('agricultural_preservation', features.extract_agricultural_preservation),
        ]
    
    def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features into environmental impact cost map.
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Cost map with multipliers (1.0 = baseline, higher = more impact)
        """
        protected = features_dict['protected_areas']
        habitat = features_dict['habitat_classification']
        land_cover = features_dict['land_cover']
        water_resources = features_dict['water_resources']
        biodiversity = features_dict['biodiversity_hotspots']
        ag_preservation = features_dict['agricultural_preservation']
        
        # Initialize with baseline cost
        cost_map = np.ones_like(protected, dtype=float)
        
        # Protected areas - very high penalty
        # Protected areas should be strongly avoided
        protected_multiplier = np.where(protected == 1, 10.0, 1.0)
        cost_map *= protected_multiplier
        
        # Habitat classification multipliers
        # 0=developed: 1.0x, 1=agricultural: 1.3x, 2=grassland: 2.0x,
        # 3=forest: 2.5x, 4=wetland: 4.0x, 5=critical_habitat: 8.0x
        habitat_multiplier = np.ones_like(habitat, dtype=float)
        habitat_multiplier[habitat == 1] = 1.3
        habitat_multiplier[habitat == 2] = 2.0
        habitat_multiplier[habitat == 3] = 2.5
        habitat_multiplier[habitat == 4] = 4.0
        habitat_multiplier[habitat == 5] = 8.0
        cost_map *= habitat_multiplier
        
        # Land cover multipliers
        # 1=developed: 1.0x, 2=agriculture: 1.2x, 3=grassland: 2.0x,
        # 4=forest: 2.8x, 5=wetland: 5.0x, 6=water: 3.0x
        land_cover_multiplier = np.ones_like(land_cover, dtype=float)
        land_cover_multiplier[land_cover == 2] = 1.2
        land_cover_multiplier[land_cover == 3] = 2.0
        land_cover_multiplier[land_cover == 4] = 2.8
        land_cover_multiplier[land_cover == 5] = 5.0
        land_cover_multiplier[land_cover == 6] = 3.0
        cost_map *= land_cover_multiplier
        
        # Water resources sensitivity
        # 0=not_sensitive: 1.0x, 1=moderate: 2.5x, 2=high: 4.5x
        water_multiplier = np.ones_like(water_resources, dtype=float)
        water_multiplier[water_resources == 1] = 2.5
        water_multiplier[water_resources == 2] = 4.5
        cost_map *= water_multiplier
        
        # Biodiversity hotspots (0-1 scale)
        # Apply additional multiplier: 1.0x to 3.0x based on biodiversity score
        biodiversity_multiplier = 1.0 + (biodiversity * 2.0)
        cost_map *= biodiversity_multiplier
        
        # Agricultural preservation (prime farmland)
        # Prime farmland should have moderate avoidance
        ag_multiplier = np.where(ag_preservation == 1, 1.8, 1.0)
        cost_map *= ag_multiplier
        
        print(f"  Environmental cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        return cost_map
