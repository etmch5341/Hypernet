#!/usr/bin/env python3
"""
Advanced Animation Examples
Demonstrates various customization options for A* animations.
"""

from astar_animator import AStarAnimator
import os


def example_1_basic():
    """Basic animation with default settings."""
    print("\n=== Example 1: Basic Animation ===")
    animator = AStarAnimator(output_dir="./astar_output")
    
    animator.create_animation(
        output_file="basic_animation.gif",
        fps=15,
        target_frames=150,
        show_path=True
    )
    print("Created: basic_animation.gif")


def example_2_fast_preview():
    """Fast, low-quality preview for quick debugging."""
    print("\n=== Example 2: Fast Preview ===")
    animator = AStarAnimator(output_dir="./astar_output")
    
    animator.create_animation(
        output_file="fast_preview.gif",
        fps=30,              # Higher FPS
        target_frames=50,    # Fewer frames
        show_path=True,
        figsize=(8, 7),      # Smaller figure
        dpi=75               # Lower resolution
    )
    print("Created: fast_preview.gif (quick & small)")


def example_3_high_quality():
    """High-quality, smooth animation for presentations."""
    print("\n=== Example 3: High Quality ===")
    animator = AStarAnimator(output_dir="./astar_output")
    
    animator.create_animation(
        output_file="presentation.gif",
        fps=15,
        target_frames=300,   # More frames = smoother
        show_path=True,
        figsize=(14, 12),    # Larger figure
        dpi=150              # Higher resolution
    )
    print("Created: presentation.gif (high quality)")


def example_4_search_only():
    """Show only the search process without the final path."""
    print("\n=== Example 4: Search Process Only ===")
    animator = AStarAnimator(output_dir="./astar_output")
    
    animator.create_animation(
        output_file="search_only.gif",
        fps=20,
        target_frames=150,
        show_path=False,     # Don't overlay final path
        figsize=(12, 10),
        dpi=100
    )
    print("Created: search_only.gif (no path overlay)")


def example_5_comparison_variants():
    """Create multiple comparison images with different settings."""
    print("\n=== Example 5: Multiple Comparisons ===")
    animator = AStarAnimator(output_dir="./astar_output")
    
    # Standard comparison
    animator.create_static_comparison(
        output_file="comparison_standard.png",
        figsize=(18, 6),
        dpi=150
    )
    
    # Large high-res comparison for poster
    animator.create_static_comparison(
        output_file="comparison_poster.png",
        figsize=(24, 8),
        dpi=300
    )
    
    print("Created: comparison_standard.png and comparison_poster.png")


def example_6_multiple_speeds():
    """Create animations at different speeds from same data."""
    print("\n=== Example 6: Multiple Speeds ===")
    animator = AStarAnimator(output_dir="./astar_output")
    
    # Slow motion
    animator.create_animation(
        output_file="slow_motion.gif",
        fps=10,              # Slower playback
        target_frames=200,
        show_path=True
    )
    
    # Normal speed
    animator.create_animation(
        output_file="normal_speed.gif",
        fps=15,
        target_frames=150,
        show_path=True
    )
    
    # Time lapse
    animator.create_animation(
        output_file="time_lapse.gif",
        fps=30,              # Faster playback
        target_frames=100,
        show_path=True
    )
    
    print("Created: slow_motion.gif, normal_speed.gif, time_lapse.gif")


def example_7_custom_colors():
    """
    Example showing how to customize colors.
    Note: This requires modifying the AStarAnimator class.
    See the comments for how to do it.
    """
    print("\n=== Example 7: Custom Colors ===")
    print("To customize colors, edit astar_animator.py:")
    print("In the create_animation() method, find this line:")
    print("  colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#4A90E2', '#FF0000']")
    print("")
    print("Color scheme:")
    print("  colors[0] = Empty terrain color (default: white #FFFFFF)")
    print("  colors[1] = Road color (default: gray #CCCCCC)")
    print("  colors[2] = Protected area color (default: pink #FFB6C1)")
    print("  colors[3] = Visited cells color (default: blue #4A90E2)")
    print("  colors[4] = Final path color (default: red #FF0000)")
    print("")
    print("Example alternative color schemes:")
    print("  Dark theme: ['#1a1a1a', '#444444', '#8b0000', '#00ff00', '#ffff00']")
    print("  Ocean theme: ['#e0f2f7', '#90caf9', '#ff6b6b', '#4a148c', '#ffd54f']")
    print("  Earth theme: ['#f5f5dc', '#8b7355', '#228b22', '#4169e1', '#ff4500']")


def main():
    """Run all examples or selected ones."""
    print("=" * 60)
    print("A* Animation Examples")
    print("=" * 60)
    print("\nChoose examples to run:")
    print("1. Basic animation (default settings)")
    print("2. Fast preview (quick debugging)")
    print("3. High quality (presentations)")
    print("4. Search only (no path overlay)")
    print("5. Comparison variants")
    print("6. Multiple speeds")
    print("7. Custom colors (info only)")
    print("8. Run all examples")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-8): ").strip()
    
    examples = {
        '1': example_1_basic,
        '2': example_2_fast_preview,
        '3': example_3_high_quality,
        '4': example_4_search_only,
        '5': example_5_comparison_variants,
        '6': example_6_multiple_speeds,
        '7': example_7_custom_colors,
    }
    
    if choice == '0':
        print("Exiting.")
        return
    elif choice == '8':
        # Run all
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 60)
    print("Done! Check ./astar_output/ for generated files.")
    print("=" * 60)


if __name__ == "__main__":
    # Check if output directory exists
    if not os.path.exists("./astar_output"):
        print("Error: ./astar_output directory not found!")
        print("Please run the A* pathfinding script first.")
        exit(1)
    
    main()
