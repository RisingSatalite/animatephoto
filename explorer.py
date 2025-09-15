import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
from PIL import Image
import numpy as np
import timm
import matplotlib
import imageio_ffmpeg as ffmpeg
import plotly.graph_objects as go

# ------------------------------
# 1. Generate Depth Map with MiDaS
# ------------------------------
def generate_depth(image_path, depth_path="depth.jpg"):
    import torch, cv2, numpy as np

    # Pick model type
    model_type = "DPT_Large"  # "DPT_Hybrid" if slow, or "MiDaS_small" for lightweight
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transform
    input_data = transform(img_rgb)

    # Ensure batch dimension
    if input_data.ndim == 3:   # (C, H, W)
        input_batch = input_data.unsqueeze(0)
    else:  # Already batched
        input_batch = input_data

    # Run inference
    with torch.no_grad():
        prediction = midas(input_batch)
        depth = prediction.squeeze().cpu().numpy()

    # Normalize depth to 0–255
    depth_min, depth_max = depth.min(), depth.max()
    depth_normalized = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    # 🔑 Resize back to original image size
    depth_resized = cv2.resize(depth_normalized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Save depth map
    cv2.imwrite(depth_path, depth_resized)
    print(f"Depth map saved to {depth_path} with shape {depth_resized.shape}")

    return img_rgb, depth_resized

# ------------------------------
# 2. Create 3D Parallax Animation
# ------------------------------
def explore_depth(img, depth):
    h, w = depth.shape

    # Normalize depth
    depth = depth.astype(np.float32) / 255.0
    depth = cv2.resize(depth, (w, h))

    # Meshgrid
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    Z = depth * 0.03

    # Plot interactively
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.axis("off")

    ax.plot_surface(-X, Y, Z, rstride=5, cstride=5,
                    facecolors=img/255, linewidth=0,
                    antialiased=False, shade=False)

    plt.show()  # <-- interactive viewer

def explore_depth_plotly(img, depth):
    h, w = depth.shape
    depth = depth.astype(np.float32) / 255.0
    depth = cv2.resize(depth, (w, h))

    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    Z = depth * 0.03

    surface = go.Surface(
        x=-X, y=Y, z=Z,
        surfacecolor=(depth),  # optional coloring
        colorscale="Viridis"
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))
    fig.show()

import plotly.graph_objects as go

def explore_depth_plotly_rgb(img, depth):
    h, w = depth.shape
    depth = depth.astype(np.float32) / 255.0

    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    Z = depth * 0.03

    # Flatten
    x = -X.flatten()
    y = Y.flatten()
    z = Z.flatten()

    # Flatten colors (normalize to 0–1)
    colors = img.reshape(-1, 3) / 255.0
    colors = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
              for r, g, b in colors]

    # Build mesh (each grid cell = 2 triangles)
    I, J, K = [], [], []
    for i in range(h-1):
        for j in range(w-1):
            idx = i*w + j
            I += [idx, idx, idx+1]
            J += [idx+1, idx+w, idx+w]
            K += [idx+w, idx+w+1, idx+w+1]

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        vertexcolor=colors,  # 🔑 per-vertex pixel colors
        showscale=False
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))
    fig.show()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    image_path = "cat.jpg"  # input image
    print("Looking for:", os.path.abspath(image_path))
    print("Exists?", os.path.exists(image_path))
    img, depth = generate_depth(image_path)
    explore_depth_plotly_rgb(img, depth)
