# Geometry/Efficiency Feature Map

elevation(x, y): from SRTM/ASTER GDEM data

terrain_curvature(x, y) = |∂²x/∂x² × ∂²y/∂y² - ∂²x/∂y² × ∂²y/∂x²|
    computed via Gaussian curvature on elevation surface

alignment_quality(x, y) = clip(1.0 - terrain_curvature(x,y) × 0.5 
                              + existing_corridor(x,y) × 0.3, 0, 1)

slope_penalty(x, y) = clip(|∇elevation(x,y)| / pixel_size_m / 0.06, 0, 1)

efficiency_score(x, y) = 0.4 × alignment_quality(x,y)
                       + 0.3 × (1 - slope_penalty(x,y))
                       + 0.3 × (1 - terrain_curvature(x,y))

# Construction Cost Feature Map
existing_corridor(x, y) = rasterize(OSM roads + OSM rails)

urban_density(x, y) = building_area_within_radius(x, y, 500m) / π × 500²

terrain_difficulty(x, y) = 1.0 + clip(slope(x,y) / 0.06, 0, 2) × 0.5

geology_multiplier(x, y): from geology database (e.g., 0.8-1.5 based on rock type)

cost_multiplier(x, y) = {
    1.0                                           if corridor(x,y) = 1
    terrain_difficulty(x,y) × geology_multiplier(x,y) 
        × (1 + urban_density(x,y) × 0.5) × water_penalty(x,y)   otherwise
}

# Environment Impact Feature Map
protected_areas(x, y) = IUCN_category_score(x, y) / 6
    where IUCN categories: Ia=1.0, Ib=0.95, II=0.9, III=0.8, IV=0.6, V=0.4, VI=0.2

habitat_quality(x, y) = land_cover_biodiversity_score(x, y)
    derived from land cover classification (e.g., forest=0.9, grassland=0.6, urban=0.1)

endangered_species(x, y) = species_occurrence_density(x, y)
    from species observation databases

sensitive_ecosystems(x, y) = wetland_score(x, y) + critical_habitat_score(x, y)

agricultural_land(x, y) = crop_productivity_score(x, y)

impact_score(x, y) = (protected_areas(x,y) × 1.0
                    + habitat_quality(x,y) × 0.6
                    + endangered_species(x,y) × 0.8
                    + sensitive_ecosystems(x,y) × 1.0
                    + agricultural_land(x,y) × 0.3) / 3.8
