"""
Test script for A*PEX-to-EMO integration.
Validates both GeoJSON and NPZ input paths.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required modules are available."""
    print("Checking dependencies...")
    
    missing = []
    try:
        import networkx as nx
        print("✓ networkx")
    except ImportError:
        missing.append("networkx")
        print("✗ networkx")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        missing.append("numpy")
        print("✗ numpy")
    
    try:
        from shapely.geometry import shape, LineString
        print("✓ shapely")
    except ImportError:
        missing.append("shapely")
        print("✗ shapely")
    
    try:
        from apex_core import ApexRouter
        print("✓ apex_core")
    except ImportError:
        missing.append("apex_core")
        print("✗ apex_core (custom module)")
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("\n✓ All dependencies available")
    return True


def check_input_files():
    """Check which input files are available."""
    print("\nChecking input files...")
    
    files = {
        "LI GeoJSON": Path("/workspace/data/output/LI/LI_roads.geojson"),
        "Austin NPZ": Path("./src/sample-test-set/austin_test_raster.npz"),
        "Seattle NPZ": Path("./src/sample-test-set/seattle_test_raster.npz"),
        "Portland NPZ": Path("./src/sample-test-set/portland_test_raster.npz"),
    }
    
    available = []
    for name, path in files.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if exists:
            available.append(name)
    
    if not available:
        print("\n⚠ No input files found!")
        return False
    
    print(f"\n✓ Found {len(available)} input file(s)")
    return True


def validate_npz_structure(npz_path: Path):
    """Validate NPZ file has required structure."""
    import numpy as np
    
    print(f"\nValidating {npz_path.name}...")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        required_keys = ['raster', 'width', 'height', 'bbox', 'goal_points']
        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            print(f"✗ Missing keys: {missing_keys}")
            return False
        
        print("✓ All required keys present")
        
        # Check structure
        raster = data['raster']
        width = int(data['width'])
        height = int(data['height'])
        goal_points = data['goal_points']
        
        print(f"  Raster shape: {raster.shape}")
        print(f"  Dimensions: {width} x {height}")
        print(f"  Goal points: {len(goal_points)}")
        
        if len(goal_points) < 2:
            print(f"✗ Need at least 2 goal points, found {len(goal_points)}")
            return False
        
        for i, gp in enumerate(goal_points):
            print(f"    {i+1}. {gp[2]}: ({gp[0]}, {gp[1]})")
        
        print("✓ NPZ structure valid")
        return True
        
    except Exception as e:
        print(f"✗ Error validating NPZ: {e}")
        return False


def test_graph_construction():
    """Test graph construction from NPZ."""
    import numpy as np
    import networkx as nx
    
    print("\nTesting graph construction from NPZ...")
    
    npz_path = Path("./src/sample-test-set/austin_test_raster.npz")
    if not npz_path.exists():
        print("✗ Austin test file not found, skipping")
        return False
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        raster = data['raster']
        width = int(data['width'])
        height = int(data['height'])
        
        print(f"Building graph from {width}x{height} raster...")
        
        # Build small sample graph (just corner)
        G = nx.Graph()
        sample_size = min(50, width, height)
        
        node_count = 0
        edge_count = 0
        
        for y in range(sample_size):
            for x in range(sample_size):
                node = (x, y)
                G.add_node(node, cost=float(raster[y, x]))
                node_count += 1
                
                # 4-connectivity for speed
                for dx, dy in [(1, 0), (0, 1)]:
                    nx_pos = x + dx
                    ny_pos = y + dy
                    
                    if nx_pos < sample_size and ny_pos < sample_size:
                        neighbor = (nx_pos, ny_pos)
                        G.add_edge(node, neighbor, length=10.0)
                        edge_count += 1
        
        print(f"✓ Sample graph: {node_count} nodes, {edge_count} edges")
        
        # Test basic connectivity
        source = (0, 0)
        target = (sample_size-1, sample_size-1)
        
        if nx.has_path(G, source, target):
            path = nx.shortest_path(G, source, target)
            print(f"✓ Path exists: {len(path)} nodes")
            return True
        else:
            print("✗ No path found in sample graph")
            return False
            
    except Exception as e:
        print(f"✗ Error in graph construction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_apex_router_mock():
    """Test ApexRouter with mock data."""
    print("\nTesting ApexRouter...")
    
    try:
        from apex_core import ApexRouter
        import networkx as nx
        
        # Create simple test graph
        G = nx.Graph()
        for i in range(5):
            for j in range(5):
                node = (i, j)
                G.add_node(node)
                if i < 4:
                    G.add_edge(node, (i+1, j), length=1.0, env_risk=0.1, tunnel=0.0)
                if j < 4:
                    G.add_edge(node, (i, j+1), length=1.0, env_risk=0.1, tunnel=0.0)
        
        router = ApexRouter(
            G,
            objective_keys=("length", "env_risk", "tunnel"),
            eps=(0.01, 0.01, 0.01)
        )
        
        source = (0, 0)
        target = (4, 4)
        weights = (1.0, 0.0, 0.0)
        
        path, objectives = router.route(source, target, weights)
        
        print(f"✓ ApexRouter returned path with {len(path)} nodes")
        print(f"  Objectives: {objectives}")
        return True
        
    except ImportError:
        print("⚠ apex_core not available, skipping")
        return None
    except Exception as e:
        print(f"✗ Error testing ApexRouter: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*60)
    print("A*PEX-EMO Integration Validation")
    print("="*60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Input Files": check_input_files(),
    }
    
    # Only run advanced tests if basics pass
    if results["Input Files"]:
        npz_path = Path("./src/sample-test-set/austin_test_raster.npz")
        if npz_path.exists():
            results["NPZ Structure"] = validate_npz_structure(npz_path)
            results["Graph Construction"] = test_graph_construction()
    
    if results["Dependencies"]:
        apex_result = test_apex_router_mock()
        if apex_result is not None:
            results["ApexRouter"] = apex_result
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for test, result in results.items():
        if result is True:
            print(f"✓ {test}")
        elif result is False:
            print(f"✗ {test}")
        else:
            print(f"⚠ {test} (skipped)")
    
    all_passed = all(r in [True, None] for r in results.values())
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        print("\nYou can now run:")
        print("  python apex_emo_seed_fixed.py austin")
        print("  python apex_emo_seed_fixed.py seattle")
        print("  python apex_emo_seed_fixed.py portland")
        return 0
    else:
        print("\n⚠ Some validation checks failed")
        print("Please fix the issues above before running the main script")
        return 1


if __name__ == "__main__":
    sys.exit(main())