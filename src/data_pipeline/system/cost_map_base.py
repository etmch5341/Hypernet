"""
Base class for cost map generation.
Provides infrastructure for extracting features and combining them into cost maps.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict
import numpy as np


class CostMapBase(ABC):
    """
    Abstract base class for generating cost maps from multiple feature layers.
    
    Child classes should define which features to extract and how to combine them.
    """
    
    def __init__(self, bbox: Tuple[float, float, float, float], resolution: float):
        """
        Initialize the cost map generator.
        
        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            resolution: Pixel resolution in meters
        """
        self.bbox = bbox
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.width, self.height = self._calculate_grid_dimensions()
        
        # Storage for extracted features
        self.features: Dict[str, np.ndarray] = {}
        
        # The final compiled cost map
        self.cost_map: np.ndarray = None
    
    def _calculate_grid_dimensions(self) -> Tuple[int, int]:
        """
        Calculate grid dimensions based on bounding box and resolution.
        
        Returns:
            (width, height) in pixels
        """
        min_lon, min_lat, max_lon, max_lat = self.bbox
        
        # Approximate conversion (more accurate methods exist for production)
        # At ~30°N latitude, 1 degree ≈ 111km for latitude, ~96km for longitude
        lat_center = (min_lat + max_lat) / 2
        lon_distance = (max_lon - min_lon) * 111000 * np.cos(np.radians(lat_center))
        lat_distance = (max_lat - min_lat) * 111000
        
        width = int(np.ceil(lon_distance / self.resolution))
        height = int(np.ceil(lat_distance / self.resolution))
        
        return width, height
    
    @abstractmethod
    def get_feature_extractors(self) -> List[Tuple[str, Callable]]:
        """
        Define which features need to be extracted for this cost map type.
        
        Returns:
            List of (feature_name, extraction_function) tuples
            Each extraction_function should take (bbox, resolution) and return a 2D numpy array
        """
        pass
    
    @abstractmethod
    def combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine extracted features into a single cost map.
        
        Args:
            features: Dictionary mapping feature names to their 2D arrays
            
        Returns:
            Combined cost map as 2D numpy array (cost multipliers)
        """
        pass
    
    def extract_features(self, feature_module):
        """
        Extract all required features using functions from the feature module.
        
        Args:
            feature_module: Python module containing feature extraction functions
        """
        extractors = self.get_feature_extractors()
        
        for feature_name, extractor_func in extractors:
            print(f"Extracting feature: {feature_name}")
            
            # Call the extraction function with bbox and resolution
            feature_data = extractor_func(self.bbox, self.resolution)
            
            # Validate dimensions
            if feature_data.shape != (self.height, self.width):
                raise ValueError(
                    f"Feature '{feature_name}' has shape {feature_data.shape}, "
                    f"expected ({self.height}, {self.width})"
                )
            
            self.features[feature_name] = feature_data
            print(f"  ✓ Extracted {feature_name}: shape {feature_data.shape}")
    
    def generate_cost_map(self) -> np.ndarray:
        """
        Generate the final cost map by combining all extracted features.
        
        Returns:
            Cost map as 2D numpy array
        """
        if not self.features:
            raise RuntimeError("No features extracted. Call extract_features() first.")
        
        print(f"Combining features for {self.__class__.__name__}")
        self.cost_map = self.combine_features(self.features)
        
        print(f"  ✓ Generated cost map: shape {self.cost_map.shape}")
        return self.cost_map
    
    def get_cost_at_position(self, x: int, y: int) -> float:
        """
        Get the cost multiplier at a specific grid position.
        
        Args:
            x: Grid x-coordinate (column)
            y: Grid y-coordinate (row)
            
        Returns:
            Cost multiplier at that position
        """
        if self.cost_map is None:
            raise RuntimeError("Cost map not generated. Call generate_cost_map() first.")
        
        if not (0 <= y < self.height and 0 <= x < self.width):
            raise ValueError(f"Position ({x}, {y}) outside grid bounds")
        
        return self.cost_map[y, x]
    
    def save_cost_map(self, filepath: str):
        """
        Save the cost map to a file.
        
        Args:
            filepath: Path to save the numpy array
        """
        if self.cost_map is None:
            raise RuntimeError("Cost map not generated. Call generate_cost_map() first.")
        
        np.save(filepath, self.cost_map)
        print(f"Cost map saved to {filepath}")
    
    def load_cost_map(self, filepath: str):
        """
        Load a previously saved cost map.
        
        Args:
            filepath: Path to the saved numpy array
        """
        self.cost_map = np.load(filepath)
        print(f"Cost map loaded from {filepath}")