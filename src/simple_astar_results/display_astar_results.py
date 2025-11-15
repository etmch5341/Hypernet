import numpy as np
import matplotlib.pyplot as plt

road_bitmap = np.load("./astar_output/road_bitmap.npz")["road_bitmap"]
path = np.load("./astar_output/astar_final_path.npy", allow_pickle=True)

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(road_bitmap, cmap="gray", origin="upper")

# Draw path
for (x, y) in path:
    ax.scatter(y, x, s=2, c="red")

plt.title("Final A* Path")
plt.show()

# import pickle
# frames = pickle.load(open(".astar_sparse_frames.pkl","rb"))
# import matplotlib.pyplot as plt
# import numpy as np

# # road_bitmap must be in memory (load if saved)
# for expansion, pts in frames:
#     frame = np.zeros_like(road_bitmap)
#     for (x, y) in pts:
#         frame[x, y] = 1  # mark visited
    
#     plt.figure(figsize=(7,7))
#     plt.imshow(road_bitmap, cmap="gray", origin="upper")
#     plt.imshow(frame, cmap="hot", alpha=0.5, origin="upper")
#     plt.title(f"Visited up to expansion {expansion}")
#     plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import numpy as np
# import pickle

# frames = pickle.load(open("./astar_sparse_frames.pkl","rb"))
# road_bitmap = np.load("./road_bitmap.npz")

# fig, ax = plt.subplots(figsize=(8,8))
# img = ax.imshow(road_bitmap, cmap="gray", origin="upper")

# def update(i):
#     _, pts = frames[i]
#     overlay = np.zeros_like(road_bitmap)
#     for (x, y) in pts:
#         overlay[x, y] = 1
#     img.set_data(overlay)
#     ax.set_title(f"Expansion frame {i}")
#     return [img]

# ani = FuncAnimation(fig, update, frames=len(frames), interval=80)
# plt.show()
