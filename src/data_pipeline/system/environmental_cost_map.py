# """
# Environmental Cost Map for minimizing environmental impact.
# """

# from typing import List, Tuple, Callable, Dict
# import numpy as np
# from cost_map_base import CostMapBase
# import feature_extraction as features


# class EnvironmentalCostMap(CostMapBase):
#     """
#     Cost map for environmental impact minimization.
    
#     Considers:
#     - Protected areas (parks, refuges)
#     - Habitat classification (critical habitats)
#     - Land cover (natural vegetation, wetlands)
#     - Water resources (aquifers, watersheds)
#     - Biodiversity hotspots
#     - Agricultural preservation zones
#     """
    
#     def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
#         """
#         Define features needed for environmental impact calculation.
#         """
#         return [
#             ('protected_areas', features.extract_protected_areas),
#             ('habitat_classification', features.extract_habitat_classification),
#             ('land_cover', features.extract_land_cover),
#             ('water_resources', features.extract_water_resources),
#             ('biodiversity_hotspots', features.extract_biodiversity_hotspots),
#             ('agricultural_preservation', features.extract_agricultural_preservation),
#         ]
    
#     def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
#         """
#         Combine features into environmental impact cost map.
        
#         Args:
#             features_dict: Dictionary of extracted features
            
#         Returns:
#             Cost map with multipliers (1.0 = baseline, higher = more impact)
#         """
#         protected = features_dict['protected_areas']
#         habitat = features_dict['habitat_classification']
#         land_cover = features_dict['land_cover']
#         water_resources = features_dict['water_resources']
#         biodiversity = features_dict['biodiversity_hotspots']
#         ag_preservation = features_dict['agricultural_preservation']
        
#         # Initialize with baseline cost
#         cost_map = np.ones_like(protected, dtype=float)
        
#         # Protected areas - very high penalty
#         # Protected areas should be strongly avoided
#         protected_multiplier = np.where(protected == 1, 10.0, 1.0)
#         cost_map *= protected_multiplier
        
#         # Habitat classification multipliers
#         # 0=developed: 1.0x, 1=agricultural: 1.3x, 2=grassland: 2.0x,
#         # 3=forest: 2.5x, 4=wetland: 4.0x, 5=critical_habitat: 8.0x
#         habitat_multiplier = np.ones_like(habitat, dtype=float)
#         habitat_multiplier[habitat == 1] = 1.3
#         habitat_multiplier[habitat == 2] = 2.0
#         habitat_multiplier[habitat == 3] = 2.5
#         habitat_multiplier[habitat == 4] = 4.0
#         habitat_multiplier[habitat == 5] = 8.0
#         cost_map *= habitat_multiplier
        
#         # Land cover multipliers
#         # 1=developed: 1.0x, 2=agriculture: 1.2x, 3=grassland: 2.0x,
#         # 4=forest: 2.8x, 5=wetland: 5.0x, 6=water: 3.0x
#         land_cover_multiplier = np.ones_like(land_cover, dtype=float)
#         land_cover_multiplier[land_cover == 2] = 1.2
#         land_cover_multiplier[land_cover == 3] = 2.0
#         land_cover_multiplier[land_cover == 4] = 2.8
#         land_cover_multiplier[land_cover == 5] = 5.0
#         land_cover_multiplier[land_cover == 6] = 3.0
#         cost_map *= land_cover_multiplier
        
#         # Water resources sensitivity
#         # 0=not_sensitive: 1.0x, 1=moderate: 2.5x, 2=high: 4.5x
#         water_multiplier = np.ones_like(water_resources, dtype=float)
#         water_multiplier[water_resources == 1] = 2.5
#         water_multiplier[water_resources == 2] = 4.5
#         cost_map *= water_multiplier
        
#         # Biodiversity hotspots (0-1 scale)
#         # Apply additional multiplier: 1.0x to 3.0x based on biodiversity score
#         biodiversity_multiplier = 1.0 + (biodiversity * 2.0)
#         cost_map *= biodiversity_multiplier
        
#         # Agricultural preservation (prime farmland)
#         # Prime farmland should have moderate avoidance
#         ag_multiplier = np.where(ag_preservation == 1, 1.8, 1.0)
#         cost_map *= ag_multiplier
        
#         print(f"  Environmental cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
#         return cost_map

"""
Environmental Cost Map for minimizing environmental impact - CORRECTED VERSION

FIXES APPLIED:
- Removed habitat_classification (redundant with land_cover)
- Reduced protected area penalty from 10x to 5x (more realistic)
- Added normalization to 0-1 scale
- Reduced from 6 factors to 5 factors
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
from cost_map_base import CostMapBase
import feature_extraction as features


class EnvironmentalCostMap(CostMapBase):
    """
    Cost map for environmental impact minimization.
    
    Considers (5 factors, each counted once):
    - Protected areas (parks, refuges, conservation zones)
    - Land cover (natural vegetation, wetlands, forests)
    - Water resources (aquifers, watersheds, riparian zones)
    - Biodiversity hotspots (species richness)
    - Agricultural preservation (prime farmland)
    
    REMOVED habitat_classification (was redundant with land_cover)
    """
    
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define features needed for environmental impact calculation.
        
        NOTE: habitat_classification removed - it was redundant with land_cover
        """
        return [
            ('protected_areas', features.extract_protected_areas),
            # REMOVED: ('habitat_classification', ...) - redundant with land_cover
            ('land_cover', features.extract_land_cover),
            ('water_resources', features.extract_water_resources),
            ('biodiversity_hotspots', features.extract_biodiversity_hotspots),
            ('agricultural_preservation', features.extract_agricultural_preservation),
        ]
    
    def _normalize(self, cost_map: np.ndarray) -> np.ndarray:
        """
        Normalize cost map to 0-1 range for consistent scaling.
        
        Args:
            cost_map: Raw cost map with multipliers
            
        Returns:
            Normalized cost map (0-1 scale)
        """
        cost_min = cost_map.min()
        cost_max = cost_map.max()
        
        if cost_max > cost_min:
            normalized = (cost_map - cost_min) / (cost_max - cost_min)
        else:
            normalized = np.ones_like(cost_map) * 0.5
        
        print(f"    Normalization: {cost_min:.2f}-{cost_max:.2f} → 0.00-1.00")
        
        return normalized
    
    def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features into environmental impact cost map.
        
        This version:
        - Uses 5 factors (removed habitat_classification)
        - Reduced protected area penalty to 5x (was 10x)
        - Normalizes to 0-1 scale
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Normalized cost map (0-1 scale, higher = more impact)
        """
        protected = features_dict['protected_areas']
        land_cover = features_dict['land_cover']
        water_resources = features_dict['water_resources']
        biodiversity = features_dict['biodiversity_hotspots']
        ag_preservation = features_dict['agricultural_preservation']
        
        # Initialize with baseline
        cost_map = np.ones_like(protected, dtype=float)
        
        # === FACTOR 1: PROTECTED AREAS ===
        # Reduced from 10x to 5x - still high penalty but more realistic
        # Some protected areas may allow infrastructure with permits
        protected_multiplier = np.where(protected == 1, 5.0, 1.0)
        cost_map *= protected_multiplier
        
        # === FACTOR 2: LAND COVER (replaces both habitat and land_cover) ===
        # This now serves as the primary habitat/ecosystem indicator
        # 1=developed: 1.0x, 2=agriculture: 1.2x, 3=grassland: 2.0x,
        # 4=forest: 3.0x, 5=wetland: 5.0x, 6=water: 3.5x
        land_cover_multiplier = np.ones_like(land_cover, dtype=float)
        land_cover_multiplier[land_cover == 2] = 1.2
        land_cover_multiplier[land_cover == 3] = 2.0
        land_cover_multiplier[land_cover == 4] = 3.0  # Increased from 2.8
        land_cover_multiplier[land_cover == 5] = 5.0
        land_cover_multiplier[land_cover == 6] = 3.5  # Increased from 3.0
        cost_map *= land_cover_multiplier
        
        # === FACTOR 3: WATER RESOURCES ===
        # Sensitive water resources (aquifers, watersheds)
        # 0=not_sensitive: 1.0x, 1=moderate: 2.5x, 2=high: 4.5x
        water_multiplier = np.ones_like(water_resources, dtype=float)
        water_multiplier[water_resources == 1] = 2.5
        water_multiplier[water_resources == 2] = 4.5
        cost_map *= water_multiplier
        
        # === FACTOR 4: BIODIVERSITY HOTSPOTS ===
        # Biodiversity score 0-1, apply as 1.0x to 3.0x multiplier
        biodiversity_multiplier = 1.0 + (biodiversity * 2.0)
        cost_map *= biodiversity_multiplier
        
        # === FACTOR 5: AGRICULTURAL PRESERVATION ===
        # Prime farmland protection
        ag_multiplier = np.where(ag_preservation == 1, 1.8, 1.0)
        cost_map *= ag_multiplier
        
        print(f"  Raw environmental cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        # Cap maximum to prevent extreme values
        MAX_MULTIPLIER = 100.0
        cost_map = np.minimum(cost_map, MAX_MULTIPLIER)
        print(f"  After capping at {MAX_MULTIPLIER}: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        # Normalize to 0-1
        normalized_cost = self._normalize(cost_map)
        
        return normalized_cost