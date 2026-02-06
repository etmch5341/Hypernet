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
        # Robust loading: if any file is missing/corrupt, fall back to sane defaults
        try:
            road_data = np.load(os.path.join(self.output_dir, "road_bitmap.npz"))
            self.road_bitmap = road_data['road_bitmap']
        except Exception as e:
            print(f"Warning: failed to load road_bitmap.npz: {e}")
            self.road_bitmap = np.zeros((1, 1), dtype=np.uint8)

        try:
            protected_data = np.load(os.path.join(self.output_dir, "protected_bitmap.npz"))
            self.protected_bitmap = protected_data['protected_bitmap']
        except Exception:
            # fallback to empty protected map with same shape as road_bitmap
            self.protected_bitmap = np.zeros_like(self.road_bitmap, dtype=np.uint8)

        try:
            with open(os.path.join(self.output_dir, "astar_sparse_frames.pkl"), "rb") as f:
                self.sparse_frames = pickle.load(f) or []
        except Exception:
            # ensure attribute exists and is a list of (expansion_count, positions)
            self.sparse_frames = []

        try:
            self.final_path = np.load(os.path.join(self.output_dir, "astar_final_path.npy"))
            # convert to list of tuples for easier handling
            if isinstance(self.final_path, np.ndarray):
                if self.final_path.ndim == 2 and self.final_path.shape[1] >= 2:
                    self.final_path = [tuple(p) for p in self.final_path.tolist()]
                else:
                    self.final_path = []
            else:
                self.final_path = list(self.final_path)
        except Exception:
            self.final_path = []

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
        """Build cumulative frames from sparse frames. Always return at least one frame (final state)."""
        # ensure sparse_frames exists
        sparse = getattr(self, "sparse_frames", []) or []

        # Build cumulative visited lists from the sparse snapshots (each snapshot is incremental)
        cumulative = []
        cum_set = []
        for _, positions in sparse:
            if positions:
                for p in positions:
                    cum_set.append(tuple(p))
            # append snapshot of cumulative state even if no new positions
            cumulative.append(list(cum_set))

        # If we have no sparse snapshots, fall back to final path (if any) or a single empty frame
        if not cumulative:
            if getattr(self, "final_path", None):
                # show final path as single frame
                return [list(self.final_path)]
            return [[]]

        # remove duplicate trailing identical frames to avoid useless frames
        dedup = []
        last = None
        for frame in cumulative:
            key = tuple(frame)
            if key != last:
                dedup.append(frame)
                last = key

        if len(dedup) == 0:
            if getattr(self, "final_path", None):
                return [list(self.final_path)]
            return [[]]

        # If we already have fewer frames than target, return them directly
        if len(dedup) <= target_frames:
            return dedup

        # Otherwise sample evenly to reach target_frames
        idxs = np.linspace(0, len(dedup) - 1, target_frames, dtype=int)
        return [dedup[i] for i in idxs]

    def create_animation(self, output_file="hpa_animation.gif", fps=15, target_frames=150, show_path=True,
                         figsize=(12, 10), dpi=100):
        """
        Create the animation. The final refined path is overlaid on every frame
        so it is visible for the entire animation.
        """
        print("HPAStarAnimator: starting animation creation...")
        frames = self.interpolate_frames(target_frames)

        # Ensure frames is non-empty (fallback to single empty/frame-with-path)
        if not frames:
            if getattr(self, "final_path", None):
                frames = [list(self.final_path)]
            else:
                frames = [[]]

        base_map = self.create_base_map(show_grid=True)
        h, w = base_map.shape

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Colors: 0:White, 1:Gray(Road), 2:Pink(Protected), 3:Red(Visited), 4:DarkRed(Path), 5:LightGray(Grid)
        colors = ['#FFFFFF', '#CCCCCC', '#FFB6C1', '#FF0000', '#8B0000', '#E8F1F8']
        cmap = ListedColormap(colors)

        im = ax.imshow(base_map, cmap=cmap, vmin=0, vmax=len(colors)-1, interpolation='nearest')
        ax.set_title("Hierarchical A* Search Progression", fontsize=14, fontweight='bold')
        ax.axis('off')

        # Precompute final path coordinates (ensure tuples and filtered to bounds)
        final_path_coords = []
        if getattr(self, "final_path", None):
            for p in self.final_path:
                try:
                    r, c = int(p[0]), int(p[1])
                except Exception:
                    continue
                if 0 <= r < h and 0 <= c < w:
                    final_path_coords.append((r, c))

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
            # Start from base map copy every frame
            display_map = base_map.copy()

            visited_positions = frames[frame_idx] or []
            # Normalize visited positions
            if isinstance(visited_positions, np.ndarray):
                if visited_positions.ndim == 2 and visited_positions.shape[1] >= 2:
                    visited_iter = (tuple(p) for p in visited_positions)
                else:
                    visited_iter = ()
            else:
                visited_iter = (tuple(p) for p in visited_positions)

            # Draw visited (red) as small blocks so gateways are visible
            for r, c in visited_iter:
                rr, cc = int(r), int(c)
                if 0 <= rr < h and 0 <= cc < w:
                    display_map[max(0, rr-1):min(h, rr+2), max(0, cc-1):min(w, cc+2)] = 3

            # ALWAYS overlay the final refined path last so it remains visible on top
            if final_path_coords and show_path:
                for r, c in final_path_coords:
                    display_map[r, c] = 4

            im.set_array(display_map)
            ax.set_xlabel(f"Frame {frame_idx + 1}/{len(frames)}", fontsize=10)
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=True, repeat=True)
        out_path = os.path.join(self.output_dir, output_file)
        try:
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
            print(f"Animation saved to: {out_path}")
        except Exception as e:
            # fallback: save a static PNG if GIF saving fails
            print(f"Animation save failed: {e}. Saving single-frame PNG instead.")
            single = update(0)[0].get_array()
            plt.imsave(out_path.replace(".gif", ".png"), single, cmap=cmap)
        finally:
            plt.close(fig)

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