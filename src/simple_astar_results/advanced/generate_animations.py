#!/usr/bin/env python3
"""
Standalone Animation Generator
Run this script to create animations from existing A* output data.

Usage:
    python generate_animations.py [--output-dir ./astar_output] [--fps 15] [--frames 150]
"""

import argparse
from astar_animator import AStarAnimator


def main():
    parser = argparse.ArgumentParser(
        description="Generate animations from A* pathfinding output"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./astar_output",
        help="Directory containing A* output files (default: ./astar_output)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for animation (default: 15)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="Number of frames to interpolate (default: 150)"
    )
    parser.add_argument(
        "--animation-file",
        type=str,
        default="astar_animation.gif",
        help="Output filename for animation (default: astar_animation.gif)"
    )
    parser.add_argument(
        "--comparison-file",
        type=str,
        default="astar_comparison.png",
        help="Output filename for comparison image (default: astar_comparison.png)"
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Don't overlay the final path on the animation"
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip creating the animated GIF (only create comparison image)"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip creating the comparison image (only create animation)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("A* Animation Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Target frames: {args.frames}")
    print("=" * 60)
    
    try:
        # Create animator
        animator = AStarAnimator(output_dir=args.output_dir)
        
        # Create animation
        if not args.skip_animation:
            print("\nCreating animation...")
            animator.create_animation(
                output_file=args.animation_file,
                fps=args.fps,
                target_frames=args.frames,
                show_path=not args.no_path,
                figsize=(12, 10),
                dpi=100
            )
            print(f"✓ Animation saved: {args.output_dir}/{args.animation_file}")
        
        # Create comparison
        if not args.skip_comparison:
            print("\nCreating comparison image...")
            animator.create_static_comparison(
                output_file=args.comparison_file,
                figsize=(18, 6),
                dpi=150
            )
            print(f"✓ Comparison saved: {args.output_dir}/{args.comparison_file}")
        
        print("\n" + "=" * 60)
        print("Generation complete!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find required files in {args.output_dir}")
        print(f"  {e}")
        print("\nMake sure you've run the A* pathfinding script first!")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
