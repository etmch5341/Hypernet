"""
Geometry Cost Map for path geometry and operational efficiency optimization.
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
from cost_map_base import CostMapBase
import feature_extraction as features


class GeometryCostMap(CostMapBase):
    """
    Cost map for path geometry and operational efficiency.
    
    Considers:
    - Elevation changes (affects grades)
    - Terrain that affects alignment and curvature
    - Existing transportation corridors (easier right-of-way)
    """
    
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define features needed for geometry cost calculation.
        """
        return [
            ('elevation', features.extract_elevation),
            ('slope', features.extract_slope),
            ('existing_corridors', features.extract_existing_corridors),
        ]
    
    def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features into geometry cost map.
        
        Cost multipliers based on:
        - Slope: Higher slopes make it harder to maintain efficient alignment
        - Existing corridors: Lower cost (easier right-of-way)
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Cost map with multipliers (1.0 = baseline)
        """
        elevation = features_dict['elevation']
        slope = features_dict['slope']
        corridors = features_dict['existing_corridors']
        
        # Initialize with baseline cost
        cost_map = np.ones_like(slope, dtype=float)
        
        # Slope penalty: Higher slopes require more careful alignment
        # 0-5% slope: 1.0x (flat terrain)
        # 5-15% slope: 1.5x (rolling terrain)
        # 15-30% slope: 2.5x (hilly terrain)
        # >30% slope: 4.0x (mountainous terrain)
        slope_multiplier = np.ones_like(slope)
        slope_multiplier[slope > 5] = 1.5
        slope_multiplier[slope > 15] = 2.5
        slope_multiplier[slope > 30] = 4.0
        
        cost_map *= slope_multiplier
        
        # Corridor bonus: Existing corridors are easier for alignment
        # Reduce cost by 30% along existing corridors
        corridor_multiplier = np.where(corridors == 1, 0.7, 1.0)
        cost_map *= corridor_multiplier
        
        print(f"  Geometry cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        return cost_map
