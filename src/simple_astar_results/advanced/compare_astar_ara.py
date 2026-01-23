#!/usr/bin/env python3
"""
A* vs ARA* Comparison Script

Runs both A* and ARA* on the same input NPZ file and creates a comparison
animation showing:
- Phase 1: A* search expansion in blue, with red final path
- Phase 2: ARA* search expansion in yellow, with green final path

Usage:
    python compare_astar_ara.py --input ../../sample-test-set/austin_test_raster.npz
    python compare_astar_ara.py --input test.npz --output ./my_comparison
"""

import argparse
import os
import sys
import subprocess


def run_astar(input_file, output_dir):
    """Run A* pathfinding on the input file."""
    print("\n" + "=" * 70)
    print("Running A* Algorithm...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "hyperloop_astar_from_npz.py",
        "--input", input_file,
        "--output", output_dir,
        "--no-animation"  # We'll create our own comparison animation
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_ara(input_file, output_dir, initial_epsilon=3.0, final_epsilon=1.0):
    """Run ARA* pathfinding on the input file."""
    print("\n" + "=" * 70)
    print("Running ARA* Algorithm...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "hyperloop_ara_from_npz.py",
        "--input", input_file,
        "--output", output_dir,
        "--no-animation",  # We'll create our own comparison animation
        "--initial-epsilon", str(initial_epsilon),
        "--final-epsilon", str(final_epsilon)
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare A* and ARA* algorithms with animated visualization"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to NPZ test set file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for comparison (default: ./comparison_output_<testname>)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Animation FPS (default: 15)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Total animation frames (default: 200)"
    )
    parser.add_argument(
        "--initial-epsilon",
        type=float,
        default=3.0,
        help="Initial epsilon for ARA* (default: 3.0)"
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=1.0,
        help="Final epsilon for ARA* (default: 1.0)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./comparison_output_{test_name}"

    # Create subdirectories for each algorithm's output
    astar_dir = os.path.join(args.output, "astar_data")
    ara_dir = os.path.join(args.output, "ara_data")

    print("=" * 70)
    print("A* vs ARA* Comparison")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"ARA* epsilon: {args.initial_epsilon} -> {args.final_epsilon}")
    print("=" * 70)

    # Run A*
    astar_success = run_astar(args.input, astar_dir)
    if not astar_success:
        print("\nError: A* failed to complete")
        return 1

    # Run ARA*
    ara_success = run_ara(args.input, ara_dir,
                          args.initial_epsilon, args.final_epsilon)
    if not ara_success:
        print("\nError: ARA* failed to complete")
        return 1

    # Create comparison animation
    print("\n" + "=" * 70)
    print("Creating Comparison Animation...")
    print("=" * 70)

    from astar_animator import create_dual_algorithm_animation

    try:
        output_path = create_dual_algorithm_animation(
            astar_dir=astar_dir,
            ara_dir=ara_dir,
            output_dir=args.output,
            output_file="astar_ara_comparison.gif",
            fps=args.fps,
            target_frames=args.frames,
            figsize=(12, 10),
            dpi=100
        )

        print("\n" + "=" * 70)
        print("Comparison Complete!")
        print("=" * 70)
        print(f"Animation: {output_path}")
        print(f"Static comparison: {os.path.join(args.output, 'astar_ara_comparison.png')}")
        print("=" * 70)

    except Exception as e:
        print(f"\nError creating animation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
