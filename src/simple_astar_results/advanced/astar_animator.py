"""
A* Animation Module
Visualizes the pathfinding process using the sparse frames collected during execution.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import os

class AStarAnimator:
    """
    Creates animations from A* search frames.
    """
    
    def __init__(self, output_dir="./astar_output"):
        """
        Initialize the animator by loading data from the A* run.
        
        Args:
            output_dir: Directory containing A* output files
        """
        self.output_dir = output_dir
        self.load_data()
        
    def load_data(self):
        """Load all necessary data from the output directory."""
        # Load bitmaps
        road_data = np.load(os.path.join(self.output_dir, "road_bitmap.npz"))
        self.road_bitmap = road_data['road_bitmap']
        
        protected_data = np.load(os.path.join(self.output_dir, "protected_bitmap.npz"))
        self.protected_bitmap = protected_data['protected_bitmap']
        
        # Load metadata
        with open(os.path.join(self.output_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
            self.transform = meta['transform']
            self.bounds = meta['bounds']
        
        # Load sparse frames
        with open(os.path.join(self.output_dir, "astar_sparse_frames.pkl"), "rb") as f:
            self.sparse_frames = pickle.load(f)
        
        # Load final path
        self.final_path = np.load(os.path.join(self.output_dir, "astar_final_path.npy"))
        
        print(f"Loaded data: {len(self.sparse_frames)} sparse frames, path length: {len(self.final_path)}")
        
    def create_base_map(self):
        """
        Create the base visualization showing roads and protected areas.
        
        Returns:
            numpy array representing the base map
        """
        h, w = self.road_bitmap.shape
        base_map = np.zeros((h, w), dtype=np.uint8)
        
        # 0 = empty (white/light)
        # 1 = road (gray)
        # 2 = protected (light red/pink)
        base_map[self.road_bitmap == 1] = 1
        base_map[self.protected_bitmap == 1] = 2
        
        return base_map
    
    def interpolate_frames(self, target_frames=100):
        """
        Interpolate sparse frames to create smoother animation.
        
        Args:
            target_frames: Desired number of frames in the final animation
            
        Returns:
            List of frames with visited positions accumulated
        """
        if not self.sparse_frames:
            return []
        
        # Extract all visited positions in order
        all_visited = []
        for expansion_count, positions in self.sparse_frames:
            all_visited.extend(positions)
        
        total_visited = len(all_visited)
        if total_visited == 0:
            return []
        
        # Create evenly spaced frames
        interpolated_frames = []
        indices = np.linspace(0, total_visited - 1, min(target_frames, total_visited), dtype=int)
        
        for idx in indices:
            # Include all visited positions up to this point
            frame_visited = all_visited[:idx + 1]
            interpolated_frames.append(frame_visited)
        
        return interpolated_frames
    
    def create_animation(self, output_file="astar_animation.gif", 
                        fps=10, target_frames=100, show_path=True,
                        figsize=(12, 10), dpi=100):
        """
        Create an animated GIF showing the A* search progression.
        
        Args:
            output_file: Output filename for the animation
            fps: Frames per second
            target_frames: Number of frames to interpolate to
            show_path: Whether to overlay the final path
            figsize: Figure size in inches
            dpi: Resolution
        """
        print(f"Creating animation with {target_frames} frames at {fps} fps...")
        
        # Prepare frames
        frames = self.interpolate_frames(target_frames)
        if not frames:
            print("No frames to animate!")
            return
        
        # Create base map
        base_map = self.create_base_map()
        h, w = base_map.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define colormap
        # 0=white (empty), 1=gray (road), 2=pink (protected), 3=blue (visited), 4=red (path)
        colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#4A90E2', '#FF0000']
        cmap = ListedColormap(colors)
        
        # Initialize the image
        display_map = base_map.copy()
        im = ax.imshow(display_map, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        ax.set_title("A* Pathfinding - Frame 0", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"Expansions: 0", fontsize=12)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='black', label='Empty'),
            Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
            Patch(facecolor='#FFB6C1', edgecolor='black', label='Protected'),
            Patch(facecolor='#4A90E2', edgecolor='black', label='Visited'),
        ]
        if show_path and len(self.final_path) > 0:
            legend_elements.append(Patch(facecolor='#FF0000', edgecolor='black', label='Final Path'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        def update(frame_idx):
            """Update function for animation."""
            display_map = base_map.copy()
            
            # Mark visited cells
            visited_positions = frames[frame_idx]
            for pos in visited_positions:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3  # Blue for visited
            
            # Overlay final path if requested
            if show_path and len(self.final_path) > 0:
                for pos in self.final_path:
                    row, col = pos
                    if 0 <= row < h and 0 <= col < w:
                        display_map[row, col] = 4  # Red for path
            
            im.set_array(display_map)
            ax.set_title(f"A* Pathfinding - Frame {frame_idx + 1}/{len(frames)}", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f"Cells Visited: {len(visited_positions):,}", fontsize=12)
            
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(frames), 
                           interval=1000/fps, blit=True, repeat=True)
        
        # Save animation
        output_path = os.path.join(self.output_dir, output_file)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        
        plt.close(fig)
        print(f"Animation saved to: {output_path}")
        
    def create_static_comparison(self, output_file="astar_comparison.png", 
                                figsize=(18, 6), dpi=150):
        """
        Create a static image showing before/during/after states.
        
        Args:
            output_file: Output filename
            figsize: Figure size
            dpi: Resolution
        """
        print("Creating static comparison image...")
        
        base_map = self.create_base_map()
        h, w = base_map.shape
        
        # Get all visited cells
        all_visited = []
        for _, positions in self.sparse_frames:
            all_visited.extend(positions)
        
        # Create three maps
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        
        colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#4A90E2', '#FF0000']
        cmap = ListedColormap(colors)
        
        # Map 1: Base map only
        axes[0].imshow(base_map, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[0].set_title("Initial State\n(Roads & Protected Areas)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Map 2: With visited cells
        visited_map = base_map.copy()
        for pos in all_visited:
            row, col = pos
            if 0 <= row < h and 0 <= col < w:
                visited_map[row, col] = 3
        axes[1].imshow(visited_map, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[1].set_title(f"Search Process\n({len(all_visited):,} cells explored)", 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Map 3: Final path
        final_map = base_map.copy()
        for pos in all_visited:
            row, col = pos
            if 0 <= row < h and 0 <= col < w:
                final_map[row, col] = 3
        if len(self.final_path) > 0:
            for pos in self.final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    final_map[row, col] = 4
        axes[2].imshow(final_map, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[2].set_title(f"Final Solution\n(Path length: {len(self.final_path)} cells)", 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='black', label='Empty'),
            Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
            Patch(facecolor='#FFB6C1', edgecolor='black', label='Protected'),
            Patch(facecolor='#4A90E2', edgecolor='black', label='Visited'),
            Patch(facecolor='#FF0000', edgecolor='black', label='Final Path'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
                  fontsize=11, frameon=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        print(f"Comparison image saved to: {output_path}")
        

def create_dual_algorithm_animation(astar_dir, ara_dir, output_dir="./comparison_output",
                                     output_file="astar_ara_comparison.gif",
                                     fps=15, target_frames=200,
                                     figsize=(12, 10), dpi=100):
    """
    Create animation comparing A* (blue) and ARA* (yellow) on same input.

    Phase 1: Shows A* search expansion in blue, ending with red path
    Phase 2: Shows ARA* search expansion in yellow, ending with green path

    Args:
        astar_dir: Directory containing A* output files
        ara_dir: Directory containing ARA* output files
        output_dir: Directory to save the comparison animation
        output_file: Output filename for the animation
        fps: Frames per second
        target_frames: Total number of frames (split between both phases)
        figsize: Figure size in inches
        dpi: Resolution
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating dual algorithm comparison animation...")
    print(f"  A* data: {astar_dir}")
    print(f"  ARA* data: {ara_dir}")

    # Load A* data
    astar_road_data = np.load(os.path.join(astar_dir, "road_bitmap.npz"))
    astar_road_bitmap = astar_road_data['road_bitmap']

    with open(os.path.join(astar_dir, "astar_sparse_frames.pkl"), "rb") as f:
        astar_sparse_frames = pickle.load(f)

    astar_final_path = np.load(os.path.join(astar_dir, "astar_final_path.npy"))

    # Load ARA* data
    with open(os.path.join(ara_dir, "astar_sparse_frames.pkl"), "rb") as f:
        ara_sparse_frames = pickle.load(f)

    ara_final_path = np.load(os.path.join(ara_dir, "astar_final_path.npy"))

    print(f"  A* frames: {len(astar_sparse_frames)}, path length: {len(astar_final_path)}")
    print(f"  ARA* frames: {len(ara_sparse_frames)}, path length: {len(ara_final_path)}")

    # Extract all visited positions
    astar_all_visited = []
    for _, positions in astar_sparse_frames:
        astar_all_visited.extend(positions)

    ara_all_visited = []
    for _, positions in ara_sparse_frames:
        ara_all_visited.extend(positions)

    # Create interpolated frames for each algorithm
    frames_per_algo = target_frames // 2

    def interpolate_visits(all_visited, num_frames):
        if not all_visited:
            return []
        total = len(all_visited)
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        return [all_visited[:idx + 1] for idx in indices]

    astar_frames = interpolate_visits(astar_all_visited, frames_per_algo)
    ara_frames = interpolate_visits(ara_all_visited, frames_per_algo)

    # Create base map
    h, w = astar_road_bitmap.shape
    base_map = np.zeros((h, w), dtype=np.uint8)
    base_map[astar_road_bitmap == 1] = 1  # Roads = gray

    # Define extended colormap
    # 0=white (empty), 1=gray (road), 2=pink (protected),
    # 3=blue (A* visited), 4=yellow (ARA* visited),
    # 5=red (A* path), 6=green (ARA* path)
    colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#4A90E2', '#FFD700', '#FF0000', '#00FF00']
    cmap = ListedColormap(colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    display_map = base_map.copy()
    im = ax.imshow(display_map, cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    title = ax.set_title("A* vs ARA* Comparison", fontsize=14, fontweight='bold')
    xlabel = ax.set_xlabel("", fontsize=12)
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
        Patch(facecolor='#4A90E2', edgecolor='black', label='A* Visited'),
        Patch(facecolor='#FFD700', edgecolor='black', label='ARA* Visited'),
        Patch(facecolor='#FF0000', edgecolor='black', label='A* Path'),
        Patch(facecolor='#00FF00', edgecolor='black', label='ARA* Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    total_frames = len(astar_frames) + len(ara_frames) + 20  # +20 for pause frames

    def update(frame_idx):
        display_map = base_map.copy()

        # Phase 1: A* animation
        if frame_idx < len(astar_frames):
            # Show A* expansion
            visited = astar_frames[frame_idx]
            for pos in visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3  # Blue

            ax.set_title(f"Phase 1: A* Search - Frame {frame_idx + 1}/{len(astar_frames)}",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f"A* Cells Visited: {len(visited):,}", fontsize=12)

        # Pause frames showing A* result
        elif frame_idx < len(astar_frames) + 10:
            # Show all A* visited + final path
            for pos in astar_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3  # Blue

            for pos in astar_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 5  # Red path

            ax.set_title(f"A* Complete - Path Length: {len(astar_final_path)}",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f"A* Total Cells: {len(astar_all_visited):,}", fontsize=12)

        # Phase 2: ARA* animation
        elif frame_idx < len(astar_frames) + 10 + len(ara_frames):
            ara_idx = frame_idx - len(astar_frames) - 10

            # Keep A* visited visible (faded - using same color but will be overwritten)
            for pos in astar_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3  # Blue (A* base)

            # Show A* path
            for pos in astar_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 5  # Red path

            # Show ARA* expansion on top
            visited = ara_frames[ara_idx]
            for pos in visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    if display_map[row, col] != 5:  # Don't overwrite A* path
                        display_map[row, col] = 4  # Yellow

            ax.set_title(f"Phase 2: ARA* Search - Frame {ara_idx + 1}/{len(ara_frames)}",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f"ARA* Cells Visited: {len(visited):,}", fontsize=12)

        # Final frames showing both results
        else:
            # Show all A* visited
            for pos in astar_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3  # Blue

            # Show all ARA* visited
            for pos in ara_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    if display_map[row, col] != 3:
                        display_map[row, col] = 4  # Yellow

            # Show A* path
            for pos in astar_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 5  # Red

            # Show ARA* path
            for pos in ara_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 6  # Green

            ax.set_title("Comparison Complete: A* (Red) vs ARA* (Green)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f"A*: {len(astar_all_visited):,} cells | ARA*: {len(ara_all_visited):,} cells",
                         fontsize=12)

        im.set_array(display_map)
        return [im]

    # Create animation
    print(f"  Generating {total_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig, update, frames=total_frames,
                        interval=1000/fps, blit=True, repeat=True)

    # Save animation
    output_path = os.path.join(output_dir, output_file)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)

    plt.close(fig)
    print(f"  Animation saved to: {output_path}")

    # Also create static comparison
    create_static_dual_comparison(
        astar_road_bitmap, astar_all_visited, ara_all_visited,
        astar_final_path, ara_final_path, output_dir
    )

    return output_path


def create_static_dual_comparison(road_bitmap, astar_visited, ara_visited,
                                   astar_path, ara_path, output_dir,
                                   output_file="astar_ara_comparison.png",
                                   figsize=(20, 6), dpi=150):
    """Create a static side-by-side comparison image."""
    print("  Creating static comparison image...")

    h, w = road_bitmap.shape
    base_map = np.zeros((h, w), dtype=np.uint8)
    base_map[road_bitmap == 1] = 1

    colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#4A90E2', '#FFD700', '#FF0000', '#00FF00']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # Panel 1: A* only
    astar_map = base_map.copy()
    for pos in astar_visited:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            astar_map[row, col] = 3
    for pos in astar_path:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            astar_map[row, col] = 5
    axes[0].imshow(astar_map, cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    axes[0].set_title(f"A* Search\n({len(astar_visited):,} cells, path: {len(astar_path)})",
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: ARA* only
    ara_map = base_map.copy()
    for pos in ara_visited:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            ara_map[row, col] = 4
    for pos in ara_path:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            ara_map[row, col] = 6
    axes[1].imshow(ara_map, cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    axes[1].set_title(f"ARA* Search\n({len(ara_visited):,} cells, path: {len(ara_path)})",
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Both overlaid
    combined_map = base_map.copy()
    for pos in astar_visited:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            combined_map[row, col] = 3
    for pos in ara_visited:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            if combined_map[row, col] != 3:
                combined_map[row, col] = 4
    for pos in astar_path:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            combined_map[row, col] = 5
    for pos in ara_path:
        row, col = pos
        if 0 <= row < h and 0 <= col < w:
            combined_map[row, col] = 6
    axes[2].imshow(combined_map, cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    axes[2].set_title("Combined Comparison\n(A* blue/red, ARA* yellow/green)",
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
        Patch(facecolor='#4A90E2', edgecolor='black', label='A* Visited'),
        Patch(facecolor='#FFD700', edgecolor='black', label='ARA* Visited'),
        Patch(facecolor='#FF0000', edgecolor='black', label='A* Path'),
        Patch(facecolor='#00FF00', edgecolor='black', label='ARA* Path'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

    print(f"  Static comparison saved to: {output_path}")


def main():
    """Example usage of the animator."""
    # Create animator
    animator = AStarAnimator(output_dir="./astar_output")
    
    # Create animation (GIF)
    animator.create_animation(
        output_file="astar_animation.gif",
        fps=15,
        target_frames=150,
        show_path=True,
        figsize=(12, 10),
        dpi=100
    )
    
    # Create static comparison
    animator.create_static_comparison(
        output_file="astar_comparison.png",
        figsize=(18, 6),
        dpi=150
    )
    
    print("\nAnimation complete!")


if __name__ == "__main__":
    main()
