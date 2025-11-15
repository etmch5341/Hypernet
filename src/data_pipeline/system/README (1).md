# Hyperloop Route Optimization: Cost Map Generation System

This system generates three separate cost maps for multi-objective hyperloop route optimization across the Texas Triangle corridor.

## Architecture Overview

### Core Components

1. **`cost_map_base.py`** - Abstract base class providing infrastructure
   - Grid dimension calculation
   - Feature extraction orchestration
   - Cost map generation pipeline
   - Save/load functionality

2. **`feature_extraction.py`** - Individual feature extraction functions
   - Each function takes `(bbox, resolution)` and returns a 2D numpy array
   - Organized by objective category
   - Currently uses placeholder data (TODO: implement real data sources)

3. **Child Cost Map Classes** - Implement specific objective functions
   - `geometry_cost_map.py` - Path geometry and operational efficiency
   - `construction_cost_map.py` - Construction cost minimization
   - `environmental_cost_map.py` - Environmental impact mitigation

4. **`example_usage.py`** - Demonstrates complete workflow

## Three Optimization Objectives

### 1. Path Geometry / Operational Efficiency
**Features:**
- Elevation (for grade calculations)
- Slope (affects alignment difficulty)
- Existing corridors (easier right-of-way)

**Cost Factors:**
- Steeper slopes → harder to maintain efficient alignment
- Existing corridors → reduced cost (easier construction)

### 2. Construction Cost
**Features:**
- Terrain classification (flat/rolling/mountainous)
- Geology (soil vs rock)
- Land use (clearing costs)
- Urban density (acquisition complexity)
- Water bodies (bridge construction)
- Flood zones (elevated construction)

**Cost Factors:**
- Mountainous terrain → highest cost
- Hard rock geology → expensive excavation
- Dense urban areas → complex acquisition
- Water crossings → bridge construction

### 3. Environmental Impact
**Features:**
- Protected areas (parks, refuges)
- Habitat classification (critical habitats)
- Land cover (natural vegetation)
- Water resources (aquifers, watersheds)
- Biodiversity hotspots
- Agricultural preservation zones

**Cost Factors:**
- Protected areas → very high penalty (avoid)
- Critical habitats → high penalty
- Wetlands → substantial penalty
- Natural forests → moderate-high penalty

## Usage

### Basic Usage

```python
import feature_extraction as features
from geometry_cost_map import GeometryCostMap
from construction_cost_map import ConstructionCostMap
from environmental_cost_map import EnvironmentalCostMap

# Define bounding box (min_lon, min_lat, max_lon, max_lat)
bbox = (-99.0, 29.0, -95.0, 33.0)  # Texas Triangle
resolution = 1000.0  # 1km per pixel

# Create cost map instance
geo_map = GeometryCostMap(bbox, resolution)

# Extract features (uses functions from feature_extraction.py)
geo_map.extract_features(features)

# Generate cost map
cost_map = geo_map.generate_cost_map()

# Save for later use
geo_map.save_cost_map('geometry_cost.npy')

# Access cost at specific position
cost = geo_map.get_cost_at_position(x=100, y=200)
```

### Running the Example

```bash
python example_usage.py
```

This will:
1. Generate all three cost maps
2. Save them as `.npy` files
3. Create visualization comparing all three
4. Print statistics

## Integration with A-MHA*

Each cost map provides terrain-weighted multipliers for A-MHA* heuristics:

```python
# In your A-MHA* implementation:
def heuristic_geometry(current, goal):
    euclidean_dist = distance(current, goal)
    terrain_multiplier = geometry_cost_map.get_cost_at_position(current.x, current.y)
    return euclidean_dist * terrain_multiplier

def heuristic_construction(current, goal):
    euclidean_dist = distance(current, goal)
    terrain_multiplier = construction_cost_map.get_cost_at_position(current.x, current.y)
    return euclidean_dist * terrain_multiplier

def heuristic_environmental(current, goal):
    euclidean_dist = distance(current, goal)
    terrain_multiplier = environmental_cost_map.get_cost_at_position(current.x, current.y)
    return euclidean_dist * terrain_multiplier
```

## Extending the System

### Adding New Features

1. Add extraction function to `feature_extraction.py`:
```python
def extract_my_feature(bbox: Tuple[float, float, float, float], 
                      resolution: float) -> np.ndarray:
    width, height = _create_grid(bbox, resolution)
    # Your extraction logic here
    return feature_array
```

2. Add to appropriate cost map's `get_feature_extractors()`:
```python
def get_feature_extractors(self):
    return [
        # existing features...
        ('my_feature', features.extract_my_feature),
    ]
```

3. Update `combine_features()` to use the new feature:
```python
def combine_features(self, features_dict):
    my_feature = features_dict['my_feature']
    # Apply appropriate multiplier
    cost_map *= calculate_multiplier(my_feature)
    return cost_map
```

### Creating a New Cost Map Type

1. Create new file (e.g., `safety_cost_map.py`)
2. Inherit from `CostMapBase`
3. Implement `get_feature_extractors()` and `combine_features()`

## Data Sources (To Be Implemented)

### Elevation
- **SRTM** (Shuttle Radar Topography Mission) - 30m resolution
- **ASTER GDEM** - 30m resolution
- **OpenTopography** API

### Land Use / Land Cover
- **OpenStreetMap** - Various tags
- **NLCD** (National Land Cover Database) - 30m resolution
- **USGS Land Cover**

### Protected Areas
- **PAD-US** (Protected Areas Database) - USGS
- **USFWS Critical Habitat** data

### Urban / Infrastructure
- **OpenStreetMap** - Building footprints, roads, railways
- **Census data** - Population density
- **TIGER** (Topologically Integrated Geographic Encoding and Referencing)

### Geology
- **USGS Geological Survey** data
- **State geological surveys** (Texas Bureau of Economic Geology)

### Environmental
- **USFWS** - Species and habitat data
- **EPA** - Water resources, wetlands
- **FEMA** - Flood zones
- **USDA NRCS** - Soil surveys, prime farmland

## File Structure

```
.
├── cost_map_base.py              # Abstract base class
├── feature_extraction.py          # Feature extraction functions
├── geometry_cost_map.py          # Geometry cost map
├── construction_cost_map.py      # Construction cost map
├── environmental_cost_map.py     # Environmental cost map
├── example_usage.py              # Usage example
└── README.md                     # This file
```

## Next Steps

1. **Implement real data extraction** in `feature_extraction.py`
   - Start with elevation (SRTM/ASTER)
   - Add OSM data (urban, corridors, water)
   - Integrate protected areas (PAD-US)

2. **Integrate with A-MHA* pathfinding**
   - Use cost maps as heuristic multipliers
   - Define start/goal points for Texas Triangle
   - Implement weighted combination for anchor search

3. **Validation and calibration**
   - Compare with industry standards
   - Adjust multipliers based on domain expertise
   - Validate against known infrastructure costs

4. **Optimization**
   - Consider spatial indexing for large grids
   - Cache frequently accessed regions
   - Parallelize feature extraction

## Cost Multiplier Ranges

Based on current implementation:

| Cost Map | Typical Min | Typical Max | Extreme Max |
|----------|-------------|-------------|-------------|
| Geometry | 0.7 | 4.0 | ~6.0 |
| Construction | 1.0 | 50+ | 100+ |
| Environmental | 1.0 | 100+ | 1000+ |

Higher values indicate areas that should be strongly avoided for that objective.
