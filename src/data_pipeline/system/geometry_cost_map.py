# """
# Geometry Cost Map for path geometry and operational efficiency optimization.
# """

# from typing import List, Tuple, Callable, Dict
# import numpy as np
# from cost_map_base import CostMapBase
# import feature_extraction as features


# class GeometryCostMap(CostMapBase):
#     """
#     Cost map for path geometry and operational efficiency.
    
#     Considers:
#     - Elevation changes (affects grades)
#     - Terrain that affects alignment and curvature
#     - Existing transportation corridors (easier right-of-way)
#     """
    
#     def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
#         """
#         Define features needed for geometry cost calculation.
#         """
#         return [
#             ('elevation', features.extract_elevation),
#             ('slope', features.extract_slope),
#             ('existing_corridors', features.extract_existing_corridors),
#         ]
    
#     def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
#         """
#         Combine features into geometry cost map.
        
#         Cost multipliers based on:
#         - Slope: Higher slopes make it harder to maintain efficient alignment
#         - Existing corridors: Lower cost (easier right-of-way)
        
#         Args:
#             features_dict: Dictionary of extracted features
            
#         Returns:
#             Cost map with multipliers (1.0 = baseline)
#         """
#         elevation = features_dict['elevation']
#         slope = features_dict['slope']
#         corridors = features_dict['existing_corridors']
        
#         # Initialize with baseline cost
#         cost_map = np.ones_like(slope, dtype=float)
        
#         # Slope penalty: Higher slopes require more careful alignment
#         # 0-5% slope: 1.0x (flat terrain)
#         # 5-15% slope: 1.5x (rolling terrain)
#         # 15-30% slope: 2.5x (hilly terrain)
#         # >30% slope: 4.0x (mountainous terrain)
#         slope_multiplier = np.ones_like(slope)
#         slope_multiplier[slope > 0.05] = 1.5 * 100
#         slope_multiplier[slope > 0.15] = 2.5 * 100
#         slope_multiplier[slope > 0.30] = 4.0 * 100
        
#         cost_map *= slope_multiplier
        
#         # Corridor bonus: Existing corridors are easier for alignment
#         # Reduce cost by 30% along existing corridors
#         corridor_multiplier = np.where(corridors == 1, 0.7, 1.0)
#         cost_map *= corridor_multiplier
        
#         print(f"  Geometry cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
#         return cost_map
"""
Geometry Cost Map for path geometry and operational efficiency - CORRECTED VERSION

FIXES APPLIED:
- Removed slope (already in Construction map)
- Added terrain curvature calculation (geometry-specific)
- Added normalization to 0-1 scale
- Better focus on alignment quality vs construction difficulty
"""

from typing import List, Tuple, Callable, Dict
import numpy as np
from cost_map_base import CostMapBase
import feature_extraction as features


class GeometryCostMap(CostMapBase):
    """
    Cost map for path geometry and operational efficiency.
    
    Focuses on factors that affect route GEOMETRY and OPERATIONS,
    not construction difficulty (that's in ConstructionCostMap).
    
    Considers:
    - Terrain curvature (affects alignment smoothness, turn radius)
    - Existing transportation corridors (easier right-of-way, proven routes)
    
    REMOVED slope (already counted in ConstructionCostMap)
    ADDED terrain curvature (geometry-specific feature)
    """
    
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define features needed for geometry cost calculation.
        
        NOTE: Slope removed - it's already in ConstructionCostMap
        """
        return [
            ('elevation', features.extract_elevation),
            # REMOVED: ('slope', ...) - already in ConstructionCostMap
            ('existing_corridors', features.extract_existing_corridors),
        ]
    
    def _calculate_terrain_curvature(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate terrain curvature (Gaussian curvature approximation).
        
        High curvature areas require more frequent direction changes,
        making it harder to maintain smooth, efficient alignments.
        
        Args:
            elevation: 2D elevation array
            
        Returns:
            Curvature array (normalized 0-1, 0=flat/linear, 1=highly curved)
        """
        # Calculate first derivatives (slope)
        dy, dx = np.gradient(elevation)
        
        # Calculate second derivatives (curvature)
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        
        # Gaussian curvature approximation
        # K = (dxx * dyy - dxy * dyx) / (1 + dx^2 + dy^2)^2
        # Simplified: just use numerator as proxy
        curvature = np.abs(dxx * dyy - dxy * dyx)
        
        # Normalize to 0-1
        if curvature.max() > 0:
            curvature = curvature / (curvature.max() + 1e-6)
        else:
            curvature = np.zeros_like(elevation)
        
        print(f"    Curvature statistics:")
        print(f"      Mean: {curvature.mean():.4f}")
        print(f"      Max: {curvature.max():.4f}")
        print(f"      High curvature (>0.5): {np.sum(curvature > 0.5) / curvature.size * 100:.2f}%")
        
        return curvature
    
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
    
    def get_alignment_quality(self) -> np.ndarray:
        """
        Calculate alignment quality score (higher = better for hyperloop).
        
        Returns:
            Quality score 0-1 (0=poor alignment, 1=excellent alignment)
        """
        if self.cost_map is None:
            raise RuntimeError("Cost map not generated. Call generate_cost_map() first.")
        
        # Invert cost map: low cost = high quality
        quality = 1.0 - self.cost_map
        
        return quality
    
    def combine_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine features into geometry cost map.
        
        This version:
        - Uses terrain curvature instead of slope
        - Focuses on alignment quality
        - Normalizes to 0-1 scale
        
        Args:
            features_dict: Dictionary of extracted features
            
        Returns:
            Normalized cost map (0-1 scale, higher = worse geometry)
        """
        elevation = features_dict['elevation']
        corridors = features_dict['existing_corridors']
        
        # Initialize with baseline
        cost_map = np.ones_like(corridors, dtype=float)
        
        # === FACTOR 1: TERRAIN CURVATURE ===
        # High curvature makes it hard to maintain smooth alignment
        # Hyperloop needs gentle curves for high-speed operation
        print("  Calculating terrain curvature...")
        curvature = self._calculate_terrain_curvature(elevation)
        
        # Convert curvature (0-1) to cost multiplier (1.0-4.0)
        # Low curvature (flat/linear) = 1.0x (good for alignment)
        # High curvature (complex terrain) = 4.0x (bad for alignment)
        curvature_multiplier = 1.0 + (curvature * 3.0)
        cost_map *= curvature_multiplier
        
        # === FACTOR 2: EXISTING CORRIDORS ===
        # Following existing corridors provides:
        # - Proven route alignment
        # - Easier right-of-way acquisition
        # - Potentially smoother grades (already engineered)
        # 
        # Reduce cost by 30% along corridors
        corridor_multiplier = np.where(corridors == 1, 0.7, 1.0)
        cost_map *= corridor_multiplier
        
        print(f"  Raw geometry cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")
        
        # Normalize to 0-1
        normalized_cost = self._normalize(cost_map)
        
        return normalized_cost