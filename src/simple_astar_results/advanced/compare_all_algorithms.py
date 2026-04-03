#!/usr/bin/env python3
"""
4-Algorithm Comparison: A* vs ARA* vs MHA* vs NAMOA*-dr

Runs all four pathfinding algorithms on the same input NPZ file and creates
a comparison animation showing their search patterns and final paths.

Usage:
    python compare_all_algorithms.py --input ../../sample-test-set/austin_test_raster.npz
    python compare_all_algorithms.py --input test.npz --output ./my_comparison
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
        "--no-animation"
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
        "--no-animation",
        "--initial-epsilon", str(initial_epsilon),
        "--final-epsilon", str(final_epsilon)
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_mha(input_file, output_dir, w1=2.0, w2=1.2):
    """Run MHA* pathfinding on the input file."""
    print("\n" + "=" * 70)
    print("Running MHA* Algorithm...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "hyperloop_mha_from_npz.py",
        "--input", input_file,
        "--output", output_dir,
        "--no-animation",
        "--w1", str(w1),
        "--w2", str(w2)
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_namoa(input_file, output_dir, roughness_window=5, epsilon=0.15):
    """Run NAMOA*-dr pathfinding on the input file."""
    print("\n" + "=" * 70)
    print("Running NAMOA*-dr Algorithm...")
    print("=" * 70)

    cmd = [
        sys.executable,
        "hyperloop_namoa_from_npz.py",
        "--input", input_file,
        "--output", output_dir,
        "--no-animation",
        "--roughness-window", str(roughness_window),
        "--epsilon", str(epsilon)
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare A*, ARA*, MHA*, and NAMOA*-dr algorithms with animated visualization"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to NPZ test set file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ./comparison_all_<testname>)"
    )
    parser.add_argument(
        "--fps", type=int, default=12,
        help="Animation FPS (default: 12)"
    )
    parser.add_argument(
        "--frames", type=int, default=300,
        help="Total animation frames (default: 300)"
    )
    parser.add_argument(
        "--initial-epsilon", type=float, default=3.0,
        help="Initial epsilon for ARA* (default: 3.0)"
    )
    parser.add_argument(
        "--final-epsilon", type=float, default=1.0,
        help="Final epsilon for ARA* (default: 1.0)"
    )
    parser.add_argument(
        "--w1", type=float, default=2.0,
        help="MHA* anchor suboptimality bound (default: 2.0)"
    )
    parser.add_argument(
        "--w2", type=float, default=1.2,
        help="MHA* inadmissible expansion bound (default: 1.2)"
    )
    parser.add_argument(
        "--roughness-window", type=int, default=5,
        help="NAMOA*-dr roughness window size (default: 5)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.15,
        help="NAMOA*-dr epsilon-dominance relaxation (default: 0.15)"
    )

    args = parser.parse_args()

    if args.output is None:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./comparison_all_{test_name}"

    # Subdirectories for each algorithm
    astar_dir = os.path.join(args.output, "astar_data")
    ara_dir = os.path.join(args.output, "ara_data")
    mha_dir = os.path.join(args.output, "mha_data")
    namoa_dir = os.path.join(args.output, "namoa_data")

    print("=" * 70)
    print("Four-Algorithm Comparison: A* vs ARA* vs MHA* vs NAMOA*-dr")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"ARA* epsilon: {args.initial_epsilon} -> {args.final_epsilon}")
    print(f"MHA* w1={args.w1}, w2={args.w2}")
    print(f"NAMOA*-dr roughness window: {args.roughness_window}")
    print("=" * 70)

    # Run all four algorithms
    results = {}
    results['astar'] = run_astar(args.input, astar_dir)
    results['ara'] = run_ara(args.input, ara_dir, args.initial_epsilon, args.final_epsilon)
    results['mha'] = run_mha(args.input, mha_dir, args.w1, args.w2)
    results['namoa'] = run_namoa(args.input, namoa_dir, args.roughness_window, args.epsilon)

    # Check results
    all_success = all(results.values())
    print("\n" + "=" * 70)
    print("Algorithm Results:")
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name.upper():<10} {status}")

    if not all_success:
        print("\nSome algorithms failed. Creating animation with available results...")
        # Only proceed if at least A* succeeded
        if not results['astar']:
            print("A* failed - cannot create comparison.")
            return 1

    # Create comparison animation
    print("\n" + "=" * 70)
    print("Creating Four-Algorithm Comparison Animation...")
    print("=" * 70)

    from astar_animator import create_four_algorithm_animation

    # Only create animation if all 4 algorithms succeeded
    if all_success:
        try:
            output_path = create_four_algorithm_animation(
                astar_dir=astar_dir,
                ara_dir=ara_dir,
                mha_dir=mha_dir,
                namoa_dir=namoa_dir,
                output_dir=args.output,
                output_file="four_algorithm_comparison.gif",
                fps=args.fps,
                target_frames=args.frames,
                figsize=(14, 12),
                dpi=100
            )

            print("\n" + "=" * 70)
            print("Comparison Complete!")
            print("=" * 70)
            print(f"Animation: {output_path}")
            print(f"Static: {os.path.join(args.output, 'four_algorithm_comparison.png')}")
            print("=" * 70)

        except Exception as e:
            print(f"\nError creating animation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Skipping 4-way animation due to algorithm failures.")

    # Also create the legacy 2-way A* vs ARA* comparison if both succeeded
    if results['astar'] and results['ara']:
        print("\nAlso creating A* vs ARA* comparison...")
        try:
            from astar_animator import create_dual_algorithm_animation

            test_name = os.path.splitext(os.path.basename(args.input))[0]
            create_dual_algorithm_animation(
                astar_dir=astar_dir,
                ara_dir=ara_dir,
                output_dir=args.output,
                output_file=f"{test_name}_astar_ara_comparison.gif",
                fps=args.fps,
                target_frames=200,
                figsize=(12, 10),
                dpi=100
            )
        except Exception as e:
            print(f"  Warning: Could not create 2-way comparison: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
