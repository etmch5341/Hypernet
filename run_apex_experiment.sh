#!/bin/bash
# Quick runner for A*pex experiment

echo "========================================"
echo "Hypernet A*pex Experiment Runner"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "src/lichtenstein_roads.py" ]; then
    echo "Error: Please run this from the Hypernet repository root"
    exit 1
fi

# Check if data exists
if [ ! -f "data/output/LI/LI_roads.geojson" ]; then
    echo "Liechtenstein data not found. Running preprocessing..."
    python3 src/lichtenstein_roads.py
    
    if [ $? -ne 0 ]; then
        echo "Error: Preprocessing failed"
        exit 1
    fi
fi

# Check for protected areas (may not exist, but that's ok)
if [ ! -f "data/output/LI/protected_areas.geojson" ]; then
    echo "Warning: Protected areas data not found"
    echo "The experiment will run, but environmental constraints will be limited"
    echo ""
fi

# Copy the experiment script to src if not already there
if [ ! -f "src/apex_emo_test.py" ]; then
    if [ -f "apex_emo_test.py" ]; then
        echo "Copying apex_emo_test.py to src/..."
        cp apex_emo_test.py src/
    else
        echo "Error: apex_emo_test.py not found"
        echo "Please make sure the script is in the repository root"
        exit 1
    fi
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import pymoo" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing pymoo..."
    pip install pymoo==0.6.1.1
fi

# Run the experiment
echo ""
echo "Running A*pex experiment..."
echo ""
python3 src/apex_emo_test.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Experiment completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: data/output/apex_results/"
    echo ""
    echo "View results:"
    echo "  - all_seed_paths.png: Path visualization"
    echo "  - pareto_projections.png: Objective space"
    echo "  - apex_seeds.json: Seed data"
else
    echo ""
    echo "❌ Experiment failed"
    exit 1
fi
