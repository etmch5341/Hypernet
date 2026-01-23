import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import os

class HPAStarAnimator:
    """
    Animator for Hierarchical A* outputs. Expects the same output layout
    as the single-level animator (road_bitmap.npz, protected_bitmap.npz,
    astar_sparse_frames.pkl, astar_final_path.npy).
    """
    def __init__(self, output_dir="./hpa_output", cluster_size=40):
        self.output_dir = output_dir
        self.cluster_size = max(1, int(cluster_size))
        self.load_data()

    def load_data(self):
        road_data = np.load(os.path.join(self.output_dir, "road_bitmap.npz"))
        self.road_bitmap = road_data['road_bitmap']

        protected_data = np.load(os.path.join(self.output_dir, "protected_bitmap.npz"))
        self.protected_bitmap = protected_data['protected_bitmap']

        with open(os.path.join(self.output_dir, "astar_sparse_frames.pkl"), "rb") as f:
            self.sparse_frames = pickle.load(f)

        self.final_path = np.load(os.path.join(self.output_dir, "astar_final_path.npy"))

    def create_base_map(self, show_grid=True):
        """Creates the base map and overlays the hierarchical cluster grid."""
        h, w = self.road_bitmap.shape
        base_map = np.zeros((h, w), dtype=np.uint8)

        # 0=empty, 1=road, 2=protected
        base_map[self.road_bitmap == 1] = 1
        base_map[self.protected_bitmap == 1] = 2

        if show_grid and self.cluster_size > 0:
            # Draw cluster boundaries (value 5 for the grid); don't overwrite path of interest too strongly
            for r in range(0, h, self.cluster_size):
                base_map[r:r+1, :] = np.where(base_map[r:r+1, :] == 0, 5, base_map[r:r+1, :])
            for c in range(0, w, self.cluster_size):
                base_map[:, c:c+1] = np.where(base_map[:, c:c+1] == 0, 5, base_map[:, c:c+1])

        return base_map

    def interpolate_frames(self, target_frames=100):
        """Expand sparse frames into a list of accumulated visited positions per frame."""
        if not self.sparse_frames:
            return []

        all_visited = []
        for _, positions in self.sparse_frames:
            all_visited.extend(positions)

        total = len(all_visited)
        if total == 0:
            return []

        indices = np.linspace(0, total - 1, min(target_frames, total), dtype=int)
        return [all_visited[:idx + 1] for idx in indices]

    def create_animation(self, output_file="hpa_animation.gif", fps=15, target_frames=150, show_path=True,
                         figsize=(12, 10), dpi=100):
        frames = self.interpolate_frames(target_frames)
        if not frames:
            print("No frames to animate.")
            return

        base_map = self.create_base_map(show_grid=True)
        h, w = base_map.shape

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Colors: 0:White, 1:Gray(Road), 2:Pink(Protected), 3:Red(Visited), 4:DarkRed(Path), 5:LightGray(Grid)
        colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#FF0000', '#8B0000', '#E8F1F8']
        cmap = ListedColormap(colors)

        im = ax.imshow(base_map, cmap=cmap, vmin=0, vmax=len(colors)-1, interpolation='nearest')
        ax.set_title("Hierarchical A* Search Progression", fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add legend patches if desired
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='black', label='Empty'),
            Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
            Patch(facecolor='#FFB6C1', edgecolor='black', label='Protected'),
            Patch(facecolor='#FF0000', edgecolor='black', label='Visited (abstract/pixel)'),
            Patch(facecolor='#8B0000', edgecolor='black', label='Final Path'),
            Patch(facecolor='#E8F1F8', edgecolor='black', label='Cluster Grid'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        def update(frame_idx):
            display_map = base_map.copy()
            visited_positions = frames[frame_idx]
            # Mark visited nodes; for abstract gateway nodes we draw a small 3x3 block to be visible
            for pos in visited_positions:
                r, c = pos
                if 0 <= r < h and 0 <= c < w:
                    display_map[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = 3

            if show_path and self.final_path is not None and len(self.final_path) > 0:
                for pos in self.final_path:
                    r, c = pos
                    if 0 <= r < h and 0 <= c < w:
                        display_map[r, c] = 4

            im.set_array(display_map)
            ax.set_xlabel(f"Frame {frame_idx + 1}/{len(frames)}  —  Cells visited: {len(visited_positions):,}", fontsize=10)
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=True, repeat=True)
        out_path = os.path.join(self.output_dir, output_file)
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        plt.close(fig)
        print(f"Animation saved to: {out_path}")

    def create_static_comparison(self, output_file="hpa_comparison.png", figsize=(18, 6), dpi=150):
        base_map = self.create_base_map(show_grid=True)
        h, w = base_map.shape

        # Gather all visited cells
        all_visited = []
        for _, positions in self.sparse_frames:
            all_visited.extend(positions)

        # Build three-panel comparison
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        # Colors: 0:White, 1:Gray(Road), 2:Pink(Protected), 3:Red(Visited), 4:DarkRed(Path), 5:LightGray(Grid)
        colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#FF0000', '#8B0000', '#E8F1F8']
        cmap = ListedColormap(colors)

        # Initial
        axes[0].imshow(base_map, cmap=cmap, vmin=0, vmax=len(colors)-1, interpolation='nearest')
        axes[0].set_title("Initial State\n(Roads & Protected Areas)", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Visited
        visited_map = base_map.copy()
        for r, c in all_visited:
            if 0 <= r < h and 0 <= c < w:
                visited_map[r, c] = 3
        axes[1].imshow(visited_map, cmap=cmap, vmin=0, vmax=len(colors)-1, interpolation='nearest')
        axes[1].set_title(f"Search Process\n({len(all_visited):,} visited)", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Final path
        final_map = visited_map.copy()
        if self.final_path is not None and len(self.final_path) > 0:
            for r, c in self.final_path:
                if 0 <= r < h and 0 <= c < w:
                    final_map[r, c] = 4
        axes[2].imshow(final_map, cmap=cmap, vmin=0, vmax=len(colors)-1, interpolation='nearest')
        axes[2].set_title(f"Final Refined Path\n(length: {len(self.final_path)})", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='black', label='Empty'),
            Patch(facecolor='#CCCCCC', edgecolor='black', label='Road'),
            Patch(facecolor='#FFB6C1', edgecolor='black', label='Protected'),
            Patch(facecolor='#FF0000', edgecolor='black', label='Visited'),
            Patch(facecolor='#8B0000', edgecolor='black', label='Final Path'),
            Patch(facecolor='#E8F1F8', edgecolor='black', label='Cluster Grid'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10, frameon=True)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        out_path = os.path.join(self.output_dir, output_file)
        plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"Static comparison saved to: {out_path}")