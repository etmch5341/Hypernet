# animate_astar.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load pre-recorded animation frames and final path
frames = np.load("./data/astar_animation_frames.npy", allow_pickle=True)
path = np.load("./data/astar_final_path.npy", allow_pickle=True)

# Setup plot
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(frames[0], cmap="nipy_spectral", origin="upper")
plt.title("A* Pathfinding Animation")

def update(frame_index):
    im.set_array(frames[frame_index])
    return [im]

ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=100, blit=True
)

ani.save("astar_search_animation.mp4", writer="ffmpeg", fps=10)
plt.show()
