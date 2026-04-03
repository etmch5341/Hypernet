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


def create_iterative_ara_animation(astar_dir, ara_dir, output_dir="./comparison_output",
                                    output_file="astar_ara_iterative_comparison.gif",
                                    fps=12, frames_per_iteration=30,
                                    figsize=(12, 10), dpi=100):
    """
    Create animation showing A* then each ARA* iteration separately.

    This shows the progressive refinement of ARA* more clearly by
    animating each epsilon iteration one at a time.

    Args:
        astar_dir: Directory containing A* output files
        ara_dir: Directory containing ARA* output files
        output_dir: Directory to save the animation
        output_file: Output filename
        fps: Frames per second
        frames_per_iteration: Number of animation frames per ARA* iteration
        figsize: Figure size
        dpi: Resolution
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating iterative ARA* animation...")
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

    # Load iteration data
    iteration_data_path = os.path.join(ara_dir, "ara_iteration_data.pkl")
    if not os.path.exists(iteration_data_path):
        print(f"  Warning: No iteration data found. Run ARA* again to generate it.")
        return None

    with open(iteration_data_path, "rb") as f:
        iteration_data = pickle.load(f)

    print(f"  A* frames: {len(astar_sparse_frames)}, path length: {len(astar_final_path)}")
    print(f"  ARA* iterations: {len(iteration_data)}")
    for iter_info in iteration_data:
        print(f"    ε={iter_info['epsilon']:.2f}: frames {iter_info['start_frame_idx']}-{iter_info['end_frame_idx']}")

    # Extract all visited positions
    astar_all_visited = []
    for _, positions in astar_sparse_frames:
        astar_all_visited.extend(positions)

    # Extract visited positions per ARA* iteration
    ara_iterations_visited = []
    for iter_info in iteration_data:
        start_idx = iter_info['start_frame_idx']
        end_idx = iter_info['end_frame_idx']
        iter_visited = []
        for i in range(start_idx, end_idx):
            if i < len(ara_sparse_frames):
                _, positions = ara_sparse_frames[i]
                iter_visited.extend(positions)
        ara_iterations_visited.append({
            'epsilon': iter_info['epsilon'],
            'visited': iter_visited,
            'path': iter_info['path']
        })

    # Create base map
    h, w = astar_road_bitmap.shape
    base_map = np.zeros((h, w), dtype=np.uint8)
    base_map[astar_road_bitmap == 1] = 1

    # Color scheme
    colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#4A90E2', '#FFD700', '#FF0000', '#00FF00']
    cmap = ListedColormap(colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    display_map = base_map.copy()
    im = ax.imshow(display_map, cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    title = ax.set_title("A* vs ARA* Iterative Comparison", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
        Patch(facecolor='#4A90E2', edgecolor='black', label='A* Visited'),
        Patch(facecolor='#FFD700', edgecolor='black', label='ARA* Visited'),
        Patch(facecolor='#FF0000', edgecolor='black', label='A* Path'),
        Patch(facecolor='#00FF00', edgecolor='black', label='ARA* Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Calculate total frames
    astar_frames_count = frames_per_iteration
    pause_frames = 10
    ara_total_frames = len(iteration_data) * frames_per_iteration
    final_pause = 15
    total_frames = astar_frames_count + pause_frames + ara_total_frames + final_pause

    # Interpolate A* visits
    def interpolate_visits(all_visited, num_frames):
        if not all_visited:
            return [[]]
        total = len(all_visited)
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        return [all_visited[:idx + 1] for idx in indices]

    astar_anim_frames = interpolate_visits(astar_all_visited, astar_frames_count)

    # Interpolate each ARA* iteration
    ara_anim_frames = []
    for iter_data in ara_iterations_visited:
        iter_frames = interpolate_visits(iter_data['visited'], frames_per_iteration)
        ara_anim_frames.append({
            'epsilon': iter_data['epsilon'],
            'frames': iter_frames,
            'path': iter_data['path']
        })

    # Track cumulative ARA* visited (to show previous iterations)
    cumulative_ara_visited = []

    def update(frame_idx):
        nonlocal cumulative_ara_visited
        display_map = base_map.copy()

        # Phase 1: A* animation
        if frame_idx < astar_frames_count:
            visited = astar_anim_frames[min(frame_idx, len(astar_anim_frames) - 1)]
            for pos in visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3  # Blue

            ax.set_title(f"Phase 1: A* Search (ε=1.0)\nFrame {frame_idx + 1}/{astar_frames_count}",
                        fontsize=14, fontweight='bold')

        # Pause showing A* result
        elif frame_idx < astar_frames_count + pause_frames:
            for pos in astar_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3
            for pos in astar_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 5

            ax.set_title(f"A* Complete - Path Length: {len(astar_final_path)}",
                        fontsize=14, fontweight='bold')

        # Phase 2: ARA* iterations
        elif frame_idx < astar_frames_count + pause_frames + ara_total_frames:
            # Calculate which iteration and frame within iteration
            ara_frame = frame_idx - astar_frames_count - pause_frames
            iter_idx = ara_frame // frames_per_iteration
            frame_in_iter = ara_frame % frames_per_iteration

            # Show A* visited (faded)
            for pos in astar_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3

            # Show A* path
            for pos in astar_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 5

            # Show previous ARA* iterations (cumulative)
            if frame_in_iter == 0 and iter_idx > 0:
                # Add previous iteration's visited to cumulative
                prev_iter = ara_anim_frames[iter_idx - 1]
                if prev_iter['frames']:
                    cumulative_ara_visited.extend(prev_iter['frames'][-1])

            for pos in cumulative_ara_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    if display_map[row, col] != 5:
                        display_map[row, col] = 4

            # Show current iteration's exploration
            if iter_idx < len(ara_anim_frames):
                current_iter = ara_anim_frames[iter_idx]
                epsilon = current_iter['epsilon']
                if frame_in_iter < len(current_iter['frames']):
                    visited = current_iter['frames'][frame_in_iter]
                    for pos in visited:
                        row, col = pos
                        if 0 <= row < h and 0 <= col < w:
                            if display_map[row, col] != 5:
                                display_map[row, col] = 4

                    # Show this iteration's path if it exists
                    if current_iter['path'] and frame_in_iter == len(current_iter['frames']) - 1:
                        for pos in current_iter['path']:
                            row, col = pos
                            if 0 <= row < h and 0 <= col < w:
                                display_map[row, col] = 6

                ax.set_title(f"Phase 2: ARA* Iteration {iter_idx + 1}/{len(iteration_data)} (ε={epsilon:.2f})\n"
                            f"Frame {frame_in_iter + 1}/{frames_per_iteration}",
                            fontsize=14, fontweight='bold')

        # Final frames
        else:
            # Show all A* visited
            for pos in astar_all_visited:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 3

            # Show all ARA* visited
            for iter_data in ara_iterations_visited:
                for pos in iter_data['visited']:
                    row, col = pos
                    if 0 <= row < h and 0 <= col < w:
                        if display_map[row, col] == 0 or display_map[row, col] == 1:
                            display_map[row, col] = 4

            # Show A* path
            for pos in astar_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 5

            # Show ARA* final path
            for pos in ara_final_path:
                row, col = pos
                if 0 <= row < h and 0 <= col < w:
                    display_map[row, col] = 6

            ax.set_title(f"Complete: A* (Red) vs ARA* (Green)\n"
                        f"A* path: {len(astar_final_path)} | ARA* path: {len(ara_final_path)}",
                        fontsize=14, fontweight='bold')

        im.set_array(display_map)
        return [im]

    print(f"  Generating {total_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig, update, frames=total_frames,
                        interval=1000/fps, blit=True, repeat=True)

    output_path = os.path.join(output_dir, output_file)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)

    plt.close(fig)
    print(f"  Iterative animation saved to: {output_path}")

    return output_path


def create_four_algorithm_animation(astar_dir, ara_dir, mha_dir, namoa_dir,
                                     output_dir="./comparison_all_output",
                                     output_file="four_algorithm_comparison.gif",
                                     fps=12, target_frames=300,
                                     figsize=(14, 12), dpi=100):
    """
    Create a 2x2 grid animation comparing all four algorithms simultaneously.
    Each algorithm gets its own panel so they are easy to distinguish.

    Layout:
      [A* - blue/red]      [ARA* - yellow/green]
      [MHA* - purple/cyan] [NAMOA*-dr - orange/pink]

    All four panels animate in parallel, scaled to the same number of frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating four-algorithm comparison animation...")

    # Load data from all four algorithms
    algo_data = {}
    algo_names = ['astar', 'ara', 'mha', 'namoa']
    algo_dirs = [astar_dir, ara_dir, mha_dir, namoa_dir]
    algo_labels = ['A*', 'ARA*', 'MHA*', 'NAMOA*-dr']
    # (visited_color_hex, path_color_hex, visited_idx, path_idx)
    algo_styles = {
        'astar': ('#4A90E2', '#FF0000', 3, 5),
        'ara':   ('#FFD700', '#00FF00', 4, 6),
        'mha':   ('#9B59B6', '#00CED1', 7, 9),
        'namoa': ('#FF8C00', '#FF1493', 8, 10),
    }

    for name, dir_path, label in zip(algo_names, algo_dirs, algo_labels):
        print(f"  Loading {label} data from {dir_path}...")
        with open(os.path.join(dir_path, "astar_sparse_frames.pkl"), "rb") as f:
            sparse_frames = pickle.load(f)
        final_path = np.load(os.path.join(dir_path, "astar_final_path.npy"))

        all_visited = []
        for _, positions in sparse_frames:
            all_visited.extend(positions)

        algo_data[name] = {
            'sparse_frames': sparse_frames,
            'final_path': final_path,
            'all_visited': all_visited,
            'label': label
        }
        print(f"    {label}: {len(sparse_frames)} frames, "
              f"{len(all_visited)} visited cells, path length: {len(final_path)}")

    # Load road bitmap from A*
    road_data = np.load(os.path.join(astar_dir, "road_bitmap.npz"))
    road_bitmap = road_data['road_bitmap']

    h, w = road_bitmap.shape
    base_map = np.zeros((h, w), dtype=np.uint8)
    base_map[road_bitmap == 1] = 1

    # 11-color scheme
    colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1',
              '#4A90E2', '#FFD700',
              '#FF0000', '#00FF00',
              '#9B59B6', '#FF8C00',
              '#00CED1', '#FF1493']
    cmap = ListedColormap(colors)

    # Build per-algorithm frame snapshots (parallel progress, normalized)
    # Each frame i (0..target_frames-1) shows fraction i/target_frames of each algo's exploration.
    # Final `pause` frames show all paths drawn.
    pause = 20
    anim_frames = target_frames
    total_anim_frames = anim_frames + pause

    def snapshot_at(all_visited, frac):
        """Return visited cells up to fraction frac of total."""
        n = max(1, int(frac * len(all_visited)))
        return all_visited[:n]

    # Pre-build base maps for each panel (2x2 layout)
    # Panel positions: astar=top-left, ara=top-right, mha=bottom-left, namoa=bottom-right
    panel_layout = [
        ('astar', 0, 0),
        ('ara',   0, 1),
        ('mha',   1, 0),
        ('namoa', 1, 1),
    ]

    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    # Initialize each panel
    ims = {}
    for name, row, col in panel_layout:
        ax = axes[row][col]
        v_hex, p_hex, v_idx, p_idx = algo_styles[name]
        label = algo_data[name]['label']
        n_visited = len(algo_data[name]['all_visited'])
        n_path = len(algo_data[name]['final_path'])

        panel_map = base_map.copy()
        im = ax.imshow(panel_map, cmap=cmap, vmin=0, vmax=10, interpolation='nearest')
        ax.set_title(f"{label}\n0 cells explored", fontsize=11, fontweight='bold')
        ax.axis('off')

        # Small legend patch per panel
        legend_elements = [
            Patch(facecolor='#CCCCCC', label='Road'),
            Patch(facecolor=v_hex, label='Explored'),
            Patch(facecolor=p_hex, label='Path'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.8)
        ims[name] = im

    fig.suptitle("Four-Algorithm Pathfinding Comparison", fontsize=14, fontweight='bold', y=1.01)

    def update(frame_idx):
        artists = []
        if frame_idx < anim_frames:
            frac = (frame_idx + 1) / anim_frames
            show_path = False
        else:
            frac = 1.0
            show_path = True

        for name, row, col in panel_layout:
            ax = axes[row][col]
            v_hex, p_hex, v_idx, p_idx = algo_styles[name]
            label = algo_data[name]['label']
            all_visited = algo_data[name]['all_visited']
            final_path = algo_data[name]['final_path']

            panel_map = base_map.copy()
            visited_now = snapshot_at(all_visited, frac)

            for pos in visited_now:
                r, c = pos
                if 0 <= r < h and 0 <= c < w:
                    panel_map[r, c] = v_idx

            if show_path:
                for pos in final_path:
                    r, c = pos
                    if 0 <= r < h and 0 <= c < w:
                        panel_map[r, c] = p_idx
                ax.set_title(
                    f"{label} — Done\n{len(all_visited):,} cells | path: {len(final_path)}",
                    fontsize=11, fontweight='bold')
            else:
                pct = int(frac * 100)
                ax.set_title(
                    f"{label} — {pct}%\n{len(visited_now):,} / {len(all_visited):,} cells",
                    fontsize=11, fontweight='bold')

            ims[name].set_array(panel_map)
            artists.append(ims[name])

        return artists

    print(f"  Generating {total_anim_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig, update, frames=total_anim_frames,
                         interval=1000/fps, blit=True, repeat=True)

    output_path = os.path.join(output_dir, output_file)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)

    plt.close(fig)
    print(f"  Animation saved to: {output_path}")

    # Also create static comparison
    create_static_four_comparison(
        road_bitmap,
        {n: algo_data[n]['all_visited'] for n in algo_names},
        {n: algo_data[n]['final_path'] for n in algo_names},
        output_dir
    )

    return output_path


def create_static_four_comparison(road_bitmap, all_visited_dict, all_paths_dict,
                                   output_dir, output_file="four_algorithm_comparison.png",
                                   figsize=(24, 12), dpi=150):
    """
    Static 2x3 comparison of all four algorithms.

    Layout:
      [A*]        [ARA*]      [MHA*]
      [NAMOA*-dr] [Combined]  [Stats/Legend]
    """
    print("  Creating static four-algorithm comparison...")

    h, w = road_bitmap.shape
    base_map = np.zeros((h, w), dtype=np.uint8)
    base_map[road_bitmap == 1] = 1

    colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1',
              '#4A90E2', '#FFD700',
              '#FF0000', '#00FF00',
              '#9B59B6', '#FF8C00',
              '#00CED1', '#FF1493']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)

    # Panel configs: (row, col, algo_name, visited_color, path_color, title)
    panels = [
        (0, 0, 'astar', 3, 5, 'A*'),
        (0, 1, 'ara', 4, 6, 'ARA*'),
        (0, 2, 'mha', 7, 9, 'MHA*'),
        (1, 0, 'namoa', 8, 10, 'NAMOA*-dr'),
    ]

    for row, col, name, v_color, p_color, title in panels:
        panel_map = base_map.copy()
        visited = all_visited_dict[name]
        path = all_paths_dict[name]

        for pos in visited:
            r, c = pos
            if 0 <= r < h and 0 <= c < w:
                panel_map[r, c] = v_color

        for pos in path:
            r, c = pos
            if 0 <= r < h and 0 <= c < w:
                panel_map[r, c] = p_color

        axes[row][col].imshow(panel_map, cmap=cmap, vmin=0, vmax=10, interpolation='nearest')
        axes[row][col].set_title(
            f"{title}\n({len(visited):,} cells, path: {len(path)})",
            fontsize=12, fontweight='bold')
        axes[row][col].axis('off')

    # Panel 5: Combined overlay
    combined_map = base_map.copy()
    for name, v_color, _ in [('astar', 3, 5), ('ara', 4, 6), ('mha', 7, 9), ('namoa', 8, 10)]:
        for pos in all_visited_dict[name]:
            r, c = pos
            if 0 <= r < h and 0 <= c < w:
                if combined_map[r, c] <= 1:
                    combined_map[r, c] = v_color

    for name, _, p_color in [('astar', 3, 5), ('ara', 4, 6), ('mha', 7, 9), ('namoa', 8, 10)]:
        for pos in all_paths_dict[name]:
            r, c = pos
            if 0 <= r < h and 0 <= c < w:
                combined_map[r, c] = p_color

    axes[1][1].imshow(combined_map, cmap=cmap, vmin=0, vmax=10, interpolation='nearest')
    axes[1][1].set_title("Combined Overlay", fontsize=12, fontweight='bold')
    axes[1][1].axis('off')

    # Panel 6: Legend and stats
    axes[1][2].axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
        Patch(facecolor='#4A90E2', edgecolor='black', label='A* Visited'),
        Patch(facecolor='#FF0000', edgecolor='black', label='A* Path'),
        Patch(facecolor='#FFD700', edgecolor='black', label='ARA* Visited'),
        Patch(facecolor='#00FF00', edgecolor='black', label='ARA* Path'),
        Patch(facecolor='#9B59B6', edgecolor='black', label='MHA* Visited'),
        Patch(facecolor='#00CED1', edgecolor='black', label='MHA* Path'),
        Patch(facecolor='#FF8C00', edgecolor='black', label='NAMOA* Visited'),
        Patch(facecolor='#FF1493', edgecolor='black', label='NAMOA* Path'),
    ]
    axes[1][2].legend(handles=legend_elements, loc='center', fontsize=10,
                      frameon=True, title="Legend", title_fontsize=12)

    # Stats text
    stats_text = "Cells Explored:\n"
    for name, label in [('astar', 'A*'), ('ara', 'ARA*'), ('mha', 'MHA*'), ('namoa', 'NAMOA*-dr')]:
        stats_text += f"  {label}: {len(all_visited_dict[name]):,}\n"
    stats_text += "\nPath Length:\n"
    for name, label in [('astar', 'A*'), ('ara', 'ARA*'), ('mha', 'MHA*'), ('namoa', 'NAMOA*-dr')]:
        stats_text += f"  {label}: {len(all_paths_dict[name])}\n"

    axes[1][2].text(0.5, 0.05, stats_text, transform=axes[1][2].transAxes,
                    fontsize=9, verticalalignment='bottom', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

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
