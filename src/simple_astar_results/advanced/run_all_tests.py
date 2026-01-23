#!/usr/bin/env python3
"""
Run A* and ARA* pathfinding on all test sets with comparison animations.
Processes Austin, Seattle, and Portland test sets sequentially.

For each test set:
1. Runs A* algorithm (blue visualization)
2. Runs ARA* algorithm (yellow visualization)
3. Creates comparison animation showing both algorithms
"""

import subprocess
import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

TEST_SETS = [
    {
        "name": "Austin",
        "input": "src/sample-test-set/austin_test_raster.npz",
    },
    {
        "name": "Seattle",
        "input": "src/sample-test-set/seattle_test_raster.npz",
    },
    {
        "name": "Portland",
        "input": "src/sample-test-set/portland_test_raster.npz",
    }
]


def run_astar(input_path, output_dir):
    """Run A* pathfinding."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "hyperloop_astar_from_npz.py"),
        "--input", input_path,
        "--output", output_dir,
        "--no-animation"  # We'll create comparison animation instead
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_ara(input_path, output_dir):
    """Run ARA* pathfinding."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "hyperloop_ara_from_npz.py"),
        "--input", input_path,
        "--output", output_dir,
        "--no-animation"  # We'll create comparison animation instead
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def create_comparison_animation(astar_dir, ara_dir, output_dir, name):
    """Create comparison animation for A* vs ARA*."""
    print(f"\nCreating comparison animation for {name}...")

    try:
        from astar_animator import create_dual_algorithm_animation

        output_path = create_dual_algorithm_animation(
            astar_dir=astar_dir,
            ara_dir=ara_dir,
            output_dir=output_dir,
            output_file=f"{name.lower()}_astar_ara_comparison.gif",
            fps=15,
            target_frames=150,
            figsize=(12, 10),
            dpi=100
        )
        print(f"  Animation saved: {output_path}")
        return True
    except Exception as e:
        print(f"  Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_test_set(test_config):
    """Run both A* and ARA* on a single test set and create comparison."""
    name = test_config['name']
    input_rel = test_config['input']
    input_path = os.path.join(PROJECT_ROOT, input_rel)

    # Output directories
    base_output = os.path.join(SCRIPT_DIR, f"comparison_output_{name.lower()}")
    astar_dir = os.path.join(base_output, "astar_data")
    ara_dir = os.path.join(base_output, "ara_data")

    print("\n" + "=" * 80)
    print(f"Processing: {name}")
    print("=" * 80)

    if not os.path.exists(input_path):
        print(f"  Warning: Input file not found: {input_path}")
        print("  Skipping this test set.")
        return False, False, False

    # Run A*
    print(f"\n--- Running A* on {name} ---")
    astar_success = run_astar(input_path, astar_dir)
    if not astar_success:
        print(f"  A* failed for {name}")
        return False, False, False
    print(f"  A* completed successfully!")

    # Run ARA*
    print(f"\n--- Running ARA* on {name} ---")
    ara_success = run_ara(input_path, ara_dir)
    if not ara_success:
        print(f"  ARA* failed for {name}")
        return True, False, False
    print(f"  ARA* completed successfully!")

    # Create comparison animation
    print(f"\n--- Creating Comparison Animation for {name} ---")
    anim_success = create_comparison_animation(astar_dir, ara_dir, base_output, name)

    return astar_success, ara_success, anim_success


def main():
    # Change to script directory for imports
    os.chdir(SCRIPT_DIR)

    print("=" * 80)
    print("A* vs ARA* Comparison Runner")
    print("=" * 80)
    print(f"Processing {len(TEST_SETS)} test sets:")
    for test in TEST_SETS:
        print(f"  - {test['name']}")
    print("\nFor each test set:")
    print("  1. Run A* algorithm (blue)")
    print("  2. Run ARA* algorithm (yellow)")
    print("  3. Create comparison animation")
    print("=" * 80)

    results = []
    for test_config in TEST_SETS:
        try:
            astar_ok, ara_ok, anim_ok = run_test_set(test_config)
            results.append((test_config['name'], astar_ok, ara_ok, anim_ok))
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during {test_config['name']} processing.")
            raise

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Test Set':<15} {'A*':<12} {'ARA*':<12} {'Animation':<12}")
    print("-" * 51)
    for name, astar_ok, ara_ok, anim_ok in results:
        a_status = "SUCCESS" if astar_ok else "FAILED"
        r_status = "SUCCESS" if ara_ok else "FAILED"
        n_status = "SUCCESS" if anim_ok else "FAILED"
        print(f"{name:<15} {a_status:<12} {r_status:<12} {n_status:<12}")

    total_success = sum(1 for _, a, r, n in results if a and r and n)
    print(f"\nFully completed: {total_success}/{len(TEST_SETS)} test sets")

    print("\nOutput locations:")
    for test in TEST_SETS:
        name = test['name'].lower()
        print(f"  {test['name']}: {SCRIPT_DIR}/comparison_output_{name}/")
    print("=" * 80)

    return 0 if total_success == len(TEST_SETS) else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user.")
        exit(1)