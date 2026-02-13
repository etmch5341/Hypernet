# """
# Construction Cost Map for minimizing construction expenses.
# """

# from typing import List, Tuple, Callable, Dict
# import numpy as np
# from cost_map_base import CostMapBase
# import feature_extraction as features


# class ConstructionCostMap(CostMapBase):
#     """
#     Cost map for construction cost optimization.
    
#     Considers:
#     - Terrain difficulty (flat vs mountainous)
#     - Elevation changes (tunneling, cut-and-fill)
#     - Geology (soil vs rock excavation)
#     - Land use (clearing costs)
#     - Urban density (land acquisition, construction complexity)
#     - Water bodies (bridge construction)
#     - Flood zones (elevated construction)
#     """
    
#     def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
#         """
#         Define features needed for construction cost calculation.
#         """
#         return [
#             ('elevation', features.extract_elevation),
#             ('terrain_classification', features.extract_terrain_classification),
#             ('geology', features.extract_geology),
#             ('land_use', features.extract_land_use),
#             ('urban_density', features.extract_urban_density),
#             ('water_bodies', features.extract_water_bodies),
#             ('flood_zones', features.extract_flood_zones),
#         ]
    
#     def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
#         """
#         Combine features into construction cost map.
        
#         Args:
#             features_dict: Dictionary of extracted features
            
#         Returns:
#             Cost map with multipliers (1.0 = baseline)
#         """
#         terrain = features_dict['terrain_classification']
#         geology = features_dict['geology']
#         land_use = features_dict['land_use']
#         urban_density = features_dict['urban_density']
#         water = features_dict['water_bodies']
#         flood = features_dict['flood_zones']
        
#         # Initialize with baseline cost
#         cost_map = np.ones_like(terrain, dtype=float)
        
#         # Terrain classification multipliers
#         # 1=flat: 1.0x, 2=rolling: 2.0x, 3=mountainous: 4.5x
#         terrain_multiplier = np.ones_like(terrain, dtype=float)
#         terrain_multiplier[terrain == 2] = 2.0
#         terrain_multiplier[terrain == 3] = 4.5
#         cost_map *= terrain_multiplier
        
#         # Geology multipliers (excavation difficulty)
#         # 1=soil: 1.0x, 2=soft_rock: 1.8x, 3=hard_rock: 3.0x
#         geology_multiplier = np.ones_like(geology, dtype=float)
#         geology_multiplier[geology == 2] = 1.8
#         geology_multiplier[geology == 3] = 3.0
#         cost_map *= geology_multiplier
        
#         # Land use multipliers (clearing and preparation costs)
#         # 1=agricultural: 1.2x, 2=forest: 1.8x, 3=developed: 2.5x, 4=barren: 1.0x
#         land_use_multiplier = np.ones_like(land_use, dtype=float)
#         land_use_multiplier[land_use == 1] = 1.2
#         land_use_multiplier[land_use == 2] = 1.8
#         land_use_multiplier[land_use == 3] = 2.5
#         land_use_multiplier[land_use == 4] = 1.0
#         cost_map *= land_use_multiplier
        
#         # Urban density multiplier (land acquisition and complexity)
#         # Linear scale: 0=1.0x to 1.0=3.5x
#         urban_multiplier = 1.0 + (urban_density * 2.5)
#         cost_map *= urban_multiplier
        
#         # Water bodies (bridges, special foundations)
#         # Add 3.0x where water exists
#         water_multiplier = np.where(water == 1, 3.0, 1.0)
#         cost_map *= water_multiplier
        
#         # Flood zones (elevated construction required)
#         # 0=no_risk: 1.0x, 1=moderate: 1.5x, 2=high: 2.2x
#         flood_multiplier = np.ones_like(flood, dtype=float)
#         flood_multiplier[flood == 1] = 1.5
#         flood_multiplier[flood == 2] = 2.2
#         cost_map *= flood_multiplier
        
#         print(f"  Construction cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
#         return cost_map
"""
Construction Cost Map for minimizing construction expenses - CORRECTED VERSION

FIXES APPLIED:
- Removed terrain_classification (redundant with slope)
- Added normalization to 0-1 scale
- Reduced from 7 factors to 6 factors
- Capped maximum multiplier at 100x
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
from cost_map_base import CostMapBase
import feature_extraction as features
from scipy.ndimage import sobel


class ConstructionCostMap(CostMapBase):
    """
    Cost map for construction cost optimization.
    
    Considers (6 factors, each counted once):
    - Slope/elevation changes (tunneling, cut-and-fill) - from elevation
    - Geology (soil vs rock excavation)
    - Land use (clearing costs)
    - Urban density (land acquisition, construction complexity)
    - Water bodies (bridge construction)
    - Flood zones (elevated construction)
    
    REMOVED terrain_classification (was redundant with slope)
    """
    
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define features needed for construction cost calculation.
        
        NOTE: terrain_classification removed - it was redundant with slope
        """
        return [
            ('elevation', features.extract_elevation),
            # REMOVED: ('terrain_classification', ...) - redundant with slope
            ('geology', features.extract_geology),
            ('land_use', features.extract_land_use),
            ('urban_density', features.extract_urban_density),
            ('water_bodies', features.extract_water_bodies),
            ('flood_zones', features.extract_flood_zones),
        ]
    
    def _calculate_slope(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate slope from elevation data using gradient magnitude.
        
        Args:
            elevation: 2D elevation array in meters
            
        Returns:
            2D slope array (rise/run as decimal, 0-1 scale)
        """
        # Calculate gradients using Sobel operator
        dx = sobel(elevation, axis=1, mode='constant') / (8 * self.resolution)
        dy = sobel(elevation, axis=0, mode='constant') / (8 * self.resolution)
        
        # Calculate gradient magnitude
        slope = np.sqrt(dx**2 + dy**2)
        
        return slope
    
    def _apply_slope_penalty(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate cost multiplier based on slope (steepness).
        
        This replaces BOTH the old slope penalty AND terrain classification.
        
        Args:
            elevation: 2D elevation array
            
        Returns:
            Slope multiplier array
        """
        slope = self._calculate_slope(elevation)
        
        # Combined slope/terrain multipliers (more gradual than before)
        # 0-2% slope: 1.0x (flat, minimal earthwork)
        # 2-5% slope: 1.5x (gentle, moderate earthwork)
        # 5-10% slope: 2.5x (moderate, significant earthwork)
        # 10-20% slope: 4.0x (steep, major earthwork)
        # >20% slope: 6.0x (very steep, tunneling required)
        slope_multiplier = np.ones_like(slope, dtype=float)
        slope_multiplier[slope > 0.02] = 1.5
        slope_multiplier[slope > 0.05] = 2.5
        slope_multiplier[slope > 0.10] = 4.0
        slope_multiplier[slope > 0.20] = 6.0
        
        print(f"    Slope range: {slope.min()*100:.2f}% to {slope.max()*100:.2f}%")
        print(f"    Slope multiplier range: {slope_multiplier.min():.2f} to {slope_multiplier.max():.2f}")
        
        return slope_multiplier
    
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
            # All values the same
            normalized = np.ones_like(cost_map) * 0.5
        
        print(f"    Normalization: {cost_min:.2f}-{cost_max:.2f} → 0.00-1.00")
        
        return normalized
    
    def get_elevation_statistics(self) -> Dict[str, float]:
        """Calculate elevation statistics for the region."""
        if 'elevation' not in self.features:
            raise RuntimeError("Elevation not extracted. Call extract_features() first.")
        
        elevation = self.features['elevation']
        slope = self._calculate_slope(elevation)
        
        return {
            'min_elevation': float(elevation.min()),
            'max_elevation': float(elevation.max()),
            'elevation_range': float(elevation.max() - elevation.min()),
            'mean_elevation': float(elevation.mean()),
            'std_elevation': float(elevation.std()),
            'max_slope_percent': float(slope.max() * 100),
            'mean_slope_percent': float(slope.mean() * 100),
        }
    
    def get_cost_breakdown(self, x: int, y: int) -> Dict[str, float]:
        """
        Get detailed breakdown of cost factors at a specific position.
        
        Args:
            x: Grid x-coordinate
            y: Grid y-coordinate
            
        Returns:
            Dictionary with individual cost multipliers
        """
        if not self.features:
            raise RuntimeError("Features not extracted. Call extract_features() first.")
        
        elevation = self.features['elevation']
        geology = self.features['geology']
        land_use = self.features['land_use']
        urban_density = self.features['urban_density']
        water = self.features['water_bodies']
        flood = self.features['flood_zones']
        
        # Calculate slope multiplier
        slope = self._calculate_slope(elevation)
        slope_mult = 1.0
        if slope[y, x] > 0.02: slope_mult = 1.5
        if slope[y, x] > 0.05: slope_mult = 2.5
        if slope[y, x] > 0.10: slope_mult = 4.0
        if slope[y, x] > 0.20: slope_mult = 6.0
        
        # Get other multipliers
        geology_codes = {1: 1.0, 2: 1.8, 3: 3.0}
        land_use_codes = {1: 1.2, 2: 1.8, 3: 2.5, 4: 1.0}
        flood_codes = {0: 1.0, 1: 1.5, 2: 2.2}
        
        return {
            'slope_percent': float(slope[y, x] * 100),
            'slope_multiplier': float(slope_mult),
            'geology_multiplier': float(geology_codes.get(int(geology[y, x]), 1.0)),
            'land_use_multiplier': float(land_use_codes.get(int(land_use[y, x]), 1.0)),
            'urban_multiplier': float(1.0 + (urban_density[y, x] * 2.5)),
            'water_multiplier': float(3.0 if water[y, x] == 1 else 1.0),
            'flood_multiplier': float(flood_codes.get(int(flood[y, x]), 1.0)),
            'normalized_cost': float(self.cost_map[y, x]) if self.cost_map is not None else 0.0
        }
    
    def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features into construction cost map.
        
        This version:
        - Uses 6 factors (removed terrain_classification)
        - Normalizes to 0-1 scale
        - Caps maximum multiplier
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Normalized cost map (0-1 scale)
        """
        elevation = features_dict['elevation']
        geology = features_dict['geology']
        land_use = features_dict['land_use']
        urban_density = features_dict['urban_density']
        water = features_dict['water_bodies']
        flood = features_dict['flood_zones']
        
        # Initialize with baseline
        cost_map = np.ones_like(geology, dtype=float)
        
        # === FACTOR 1: SLOPE (replaces both slope and terrain) ===
        print("  Calculating slope penalties from elevation...")
        slope_multiplier = self._apply_slope_penalty(elevation)
        cost_map *= slope_multiplier
        
        # === FACTOR 2: GEOLOGY ===
        geology_multiplier = np.ones_like(geology, dtype=float)
        geology_multiplier[geology == 2] = 1.8
        geology_multiplier[geology == 3] = 3.0
        cost_map *= geology_multiplier
        
        # === FACTOR 3: LAND USE ===
        land_use_multiplier = np.ones_like(land_use, dtype=float)
        land_use_multiplier[land_use == 1] = 1.2
        land_use_multiplier[land_use == 2] = 1.8
        land_use_multiplier[land_use == 3] = 2.5
        land_use_multiplier[land_use == 4] = 1.0
        cost_map *= land_use_multiplier
        
        # === FACTOR 4: URBAN DENSITY ===
        urban_multiplier = 1.0 + (urban_density * 2.5)
        cost_map *= urban_multiplier
        
        # === FACTOR 5: WATER BODIES ===
        water_multiplier = np.where(water == 1, 3.0, 1.0)
        cost_map *= water_multiplier
        
        # === FACTOR 6: FLOOD ZONES ===
        flood_multiplier = np.ones_like(flood, dtype=float)
        flood_multiplier[flood == 1] = 1.5
        flood_multiplier[flood == 2] = 2.2
        cost_map *= flood_multiplier
        
        print(f"  Raw cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        # Cap maximum to prevent extreme values
        MAX_MULTIPLIER = 100.0
        cost_map = np.minimum(cost_map, MAX_MULTIPLIER)
        print(f"  After capping at {MAX_MULTIPLIER}: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        # Normalize to 0-1 for consistent combining
        normalized_cost = self._normalize(cost_map)
        
        return normalized_cost