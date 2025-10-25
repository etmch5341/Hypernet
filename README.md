# Texas HyperNet — Hyperloop route optimization using A* for the Texas Triangle
## Research Project under Texas Guadaloop (https://www.guadaloop.com/)

<!-- ![alt text](Title.png) -->
<p align="center">
  <picture>
    <img alt="LangChain Logo" src="Title.png">
  </picture>
</p>

Overview
--------
Hypernet builds an optimal hyperloop route across the Texas Triangle using a customized A* search algorithm with a multi-objective cost function. Instead of optimizing only distance, our approach evaluates candidate routes by combining real-world constraints such as track curvature, land grade, construction cost, and environmental impact. We design and tune a domain-specific heuristic so the A* search can balance these competing objectives and produce practical, low-cost, low-impact alignments.

Key goals
---------
- Produce hyperloop routes that balance travel distance with engineering, financial, and environmental costs.
- Extend the A* algorithm with a multi-objective cost function and a specialized heuristic tailored to hyperloop constraints.
- Use OSM and GIS datasets to ground route evaluation in real-world terrain, land use, and infrastructure data.

Methodology
-----------
- Data collection: extract and preprocess OpenStreetMap (OSM) and GIS layers (elevation, protected areas, land use).
- Graph construction: convert relevant OSM features to a routable graph and augment it with terrain and cost attributes.
- Cost modeling: compute per-segment costs from curvature, grade, land type, and estimated construction difficulty.
- Pathfinding: run a customized A* that uses a composite cost function and a domain heuristic to explore trade-offs.
- Analysis: compare candidate solutions on distance, monetary cost, and environmental impact metrics.

Data
----
Source data used in the repository (examples):
- OpenStreetMap extracts (.osm.pbf) stored in `data/input/`
- Processed GeoJSON outputs in `data/output/` and `LI_data/`, `outputs/`
- Project scripts that call filters and preprocessors are in `src/` and `scripts/`

Quick start
-----------
1. Create a Python virtual environment and install dependencies (this repository uses a `.venv` which is ignored by git):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Preprocess a geographic extract (example paths depend on your local files):

```bash
# examples - adapt paths as needed
python3 src/lichtenstein_roads.py
```

3. Run the pathfinding prototype (example):

```bash
python3 src/pathfinding_texas.py
```

Replace the example script names and arguments with the actual scripts you use in the `src/` folder.

Repository structure (high level)
--------------------------------
- `src/` — main scripts for preprocessing, route generation, and analysis
- `data/` — input and output datasets and intermediate files
- `LI_data/` - Lichtenstein data (osm and geojson)
- `tests/` — test scripts for simple experiments
- `docker-envr/` — Docker setup for reproducible environments
- `docker-envr/scripts/` — helper scripts for data extraction and filtering

How to contribute
-----------------
- Open an issue describing the feature or bug.
- Create feature branches from `main` and open a pull request with tests where applicable.
- Keep changes small and document assumptions in PR descriptions.

Notes and next steps
--------------------
- We are actively improving the heuristic and experimenting with optimization strategies (multi-objective A*, pareto-front approaches, and metaheuristics).
- Recommended next work items: add an example end-to-end notebook that runs preprocessing → routing → visualization, and include a small sample dataset for reproducibility.

License
-------
This project is licensed under the MIT License — see the included `LICENSE` file in the repository root for the full text.

Copyright (c) 2025 etmch5341

Contact
-------
For questions or collaboration, open an issue or contact the repository owner.