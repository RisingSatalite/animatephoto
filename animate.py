import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os

# ------------------------------
# 1. Generate Depth Map with MiDaS
# ------------------------------
def generate_depth(image_path, depth_path="depth.jpg"):
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.eval()

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).unsqueeze(0)

    # Run model
    with torch.no_grad():
        prediction = midas(input_batch)
        depth = prediction.squeeze().cpu().numpy()

    # Normalize depth
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (255 * (depth - depth_min) / (depth_max - depth_min)).astype("uint8")

    cv2.imwrite(depth_path, depth_normalized)
    return img_rgb, depth_normalized

# ------------------------------
# 2. Create 3D Parallax Animation
# ------------------------------
def create_cinematic(img, depth, output="cinematic.mp4"):
    h, w = depth.shape

    # Normalize depth
    depth = depth.astype(np.float32) / 255.0
    depth = cv2.resize(depth, (w, h))

    # Meshgrid
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    Z = depth * 0.3  # scale depth exaggeration

    # Set up figure
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("off")

    # Animate function
    def update(frame):
        ax.cla()
        ax.axis("off")
        # Camera motion: pan + zoom effect
        ax.view_init(30, 30 + frame*0.5)
        ax.dist = 7 - (frame * 0.02)  # zoom-in effect
        ax.plot_surface(X, -Y, -Z, rstride=5, cstride=5,
                        facecolors=img/255, linewidth=0,
                        antialiased=False, shade=False)

    ani = animation.FuncAnimation(fig, update, frames=90, interval=50)
    ani.save(output, writer="ffmpeg", fps=20)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    image_path = "example.png"  # input image
    print("Looking for:", os.path.abspath(image_path))
    print("Exists?", os.path.exists(image_path))
    img, depth = generate_depth(image_path)
    create_cinematic(img, depth)
    print("✅ Cinematic video saved as cinematic.mp4")
