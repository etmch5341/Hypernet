#!/usr/bin/env python3
"""
Run A* pathfinding on all test sets
Processes Austin, Seattle, and Portland test sets sequentially.
"""

import subprocess
import sys
import os

TEST_SETS = [
    {
        "name": "Austin",
        "input": "./src/sample-test-set/austin_test_raster.npz",
        "output": "./src/simple_astar_results/advanced/astar_output_austin"
    },
    {
        "name": "Seattle", 
        "input": "./src/sample-test-set/seattle_test_raster.npz",
        "output": "./src/simple_astar_results/advanced/astar_output_seattle"
    },
    {
        "name": "Portland",
        "input": "./src/sample-test-set/portland_test_raster.npz",
        "output": "./src/simple_astar_results/advanced/astar_output_portland"
    }
]


def run_test_set(test_config):
    """Run A* pathfinding on a single test set."""
    print("\n" + "=" * 80)
    print(f"Processing: {test_config['name']}")
    print("=" * 80)
    
    if not os.path.exists(test_config['input']):
        print(f"✗ Warning: Input file not found: {test_config['input']}")
        print("  Skipping this test set.")
        return False
    
    cmd = [
        sys.executable,
        "./src/simple_astar_results/advanced/hyperloop_astar_from_npz.py",
        "--input", test_config['input'],
        "--output", test_config['output']
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {test_config['name']} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error processing {test_config['name']}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user during {test_config['name']} processing.")
        raise


def main():
    print("=" * 80)
    print("Batch A* Pathfinding Runner")
    print("=" * 80)
    print(f"Processing {len(TEST_SETS)} test sets:")
    for test in TEST_SETS:
        print(f"  - {test['name']}")
    print("=" * 80)
    
    results = []
    for test_config in TEST_SETS:
        success = run_test_set(test_config)
        results.append((test_config['name'], success))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED/SKIPPED"
        print(f"{name:15s}: {status}")
    
    successful = sum(1 for _, s in results if s)
    print(f"\nCompleted: {successful}/{len(TEST_SETS)} test sets")
    print("=" * 80)
    
    return 0 if successful == len(TEST_SETS) else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user.")
        exit(1)