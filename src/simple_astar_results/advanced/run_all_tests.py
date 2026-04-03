#!/usr/bin/env python3
"""
Run all four pathfinding algorithms on all test sets with comparison animations.
Processes Austin, Seattle, and Portland test sets sequentially.

For each test set:
1. Runs A* algorithm (blue visualization)
2. Runs ARA* algorithm (yellow visualization)
3. Runs MHA* algorithm (purple visualization)
4. Runs NAMOA*-dr algorithm (orange visualization)
5. Creates 2-way comparison animation (A* vs ARA*)
6. Creates 4-way comparison animation (all algorithms)
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
        "--no-animation"
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
        "--no-animation"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_mha(input_path, output_dir):
    """Run MHA* pathfinding."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "hyperloop_mha_from_npz.py"),
        "--input", input_path,
        "--output", output_dir,
        "--no-animation",
        "--w2", "1.2"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_namoa(input_path, output_dir):
    """Run NAMOA*-dr pathfinding."""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "hyperloop_namoa_from_npz.py"),
        "--input", input_path,
        "--output", output_dir,
        "--no-animation",
        "--epsilon", "0.15"
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def create_comparison_animation(astar_dir, ara_dir, output_dir, name):
    """Create 2-way comparison animation for A* vs ARA*."""
    print(f"\nCreating A* vs ARA* comparison animation for {name}...")

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


def create_four_way_animation(astar_dir, ara_dir, mha_dir, namoa_dir, output_dir, name):
    """Create 4-way comparison animation for all algorithms."""
    print(f"\nCreating 4-algorithm comparison animation for {name}...")

    try:
        from astar_animator import create_four_algorithm_animation

        output_path = create_four_algorithm_animation(
            astar_dir=astar_dir,
            ara_dir=ara_dir,
            mha_dir=mha_dir,
            namoa_dir=namoa_dir,
            output_dir=output_dir,
            output_file=f"{name.lower()}_four_algorithm_comparison.gif",
            fps=12,
            target_frames=300,
            figsize=(14, 12),
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
    """Run all four algorithms on a single test set and create comparisons."""
    name = test_config['name']
    input_rel = test_config['input']
    input_path = os.path.join(PROJECT_ROOT, input_rel)

    # Output directories
    base_output = os.path.join(SCRIPT_DIR, f"comparison_output_{name.lower()}")
    astar_dir = os.path.join(base_output, "astar_data")
    ara_dir = os.path.join(base_output, "ara_data")
    mha_dir = os.path.join(base_output, "mha_data")
    namoa_dir = os.path.join(base_output, "namoa_data")

    print("\n" + "=" * 80)
    print(f"Processing: {name}")
    print("=" * 80)

    if not os.path.exists(input_path):
        print(f"  Warning: Input file not found: {input_path}")
        print("  Skipping this test set.")
        return False, False, False, False, False, False

    # Run A*
    print(f"\n--- Running A* on {name} ---")
    astar_success = run_astar(input_path, astar_dir)
    if not astar_success:
        print(f"  A* failed for {name}")
        return False, False, False, False, False, False
    print(f"  A* completed successfully!")

    # Run ARA*
    print(f"\n--- Running ARA* on {name} ---")
    ara_success = run_ara(input_path, ara_dir)
    if not ara_success:
        print(f"  ARA* failed for {name}")
    else:
        print(f"  ARA* completed successfully!")

    # Run MHA*
    print(f"\n--- Running MHA* on {name} ---")
    mha_success = run_mha(input_path, mha_dir)
    if not mha_success:
        print(f"  MHA* failed for {name}")
    else:
        print(f"  MHA* completed successfully!")

    # Run NAMOA*-dr
    print(f"\n--- Running NAMOA*-dr on {name} ---")
    namoa_success = run_namoa(input_path, namoa_dir)
    if not namoa_success:
        print(f"  NAMOA*-dr failed for {name}")
    else:
        print(f"  NAMOA*-dr completed successfully!")

    # Create 2-way A* vs ARA* comparison
    anim2_success = False
    if astar_success and ara_success:
        print(f"\n--- Creating A* vs ARA* Comparison for {name} ---")
        anim2_success = create_comparison_animation(astar_dir, ara_dir, base_output, name)

    # Create 4-way comparison
    anim4_success = False
    if astar_success and ara_success and mha_success and namoa_success:
        print(f"\n--- Creating 4-Algorithm Comparison for {name} ---")
        anim4_success = create_four_way_animation(
            astar_dir, ara_dir, mha_dir, namoa_dir, base_output, name)

    return astar_success, ara_success, mha_success, namoa_success, anim2_success, anim4_success


def main():
    # Change to script directory for imports
    os.chdir(SCRIPT_DIR)

    print("=" * 80)
    print("Four-Algorithm Comparison Runner")
    print("A* vs ARA* vs MHA* vs NAMOA*-dr")
    print("=" * 80)
    print(f"Processing {len(TEST_SETS)} test sets:")
    for test in TEST_SETS:
        print(f"  - {test['name']}")
    print("\nFor each test set:")
    print("  1. Run A* algorithm (blue)")
    print("  2. Run ARA* algorithm (yellow)")
    print("  3. Run MHA* algorithm (purple)")
    print("  4. Run NAMOA*-dr algorithm (orange)")
    print("  5. Create 2-way comparison animation (A* vs ARA*)")
    print("  6. Create 4-way comparison animation (all algorithms)")
    print("=" * 80)

    results = []
    for test_config in TEST_SETS:
        try:
            astar_ok, ara_ok, mha_ok, namoa_ok, anim2_ok, anim4_ok = run_test_set(test_config)
            results.append((test_config['name'], astar_ok, ara_ok, mha_ok, namoa_ok, anim2_ok, anim4_ok))
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during {test_config['name']} processing.")
            raise

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Test Set':<12} {'A*':<9} {'ARA*':<9} {'MHA*':<9} {'NAMOA*':<9} {'2-Way':<9} {'4-Way':<9}")
    print("-" * 66)
    for name, astar_ok, ara_ok, mha_ok, namoa_ok, anim2_ok, anim4_ok in results:
        row = f"{name:<12} "
        for ok in [astar_ok, ara_ok, mha_ok, namoa_ok, anim2_ok, anim4_ok]:
            row += f"{'OK':<9}" if ok else f"{'FAIL':<9}"
        print(row)

    total_full = sum(1 for _, a, r, m, n, a2, a4 in results if all([a, r, m, n, a2, a4]))
    print(f"\nFully completed: {total_full}/{len(TEST_SETS)} test sets")

    print("\nOutput locations:")
    for test in TEST_SETS:
        name = test['name'].lower()
        print(f"  {test['name']}: {SCRIPT_DIR}/comparison_output_{name}/")
    print("=" * 80)

    return 0 if total_full == len(TEST_SETS) else 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user.")
        exit(1)
