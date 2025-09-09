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

    # Convert BGR → RGB and enforce dtype
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Make sure dtype is uint8
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    print("img_rgb dtype:", img_rgb.dtype, "shape:", img_rgb.shape, type(img_rgb))

    # Now safe to apply and call transform
    input_data = transform(img_rgb)
    print("Transform output shape:", input_data.shape)

    # Ensure batch dimension is correct
    if input_data.ndim == 3:   # (C, H, W)
        input_batch = input_data.unsqueeze(0)
    elif input_data.ndim == 4: # (B, C, H, W)
        input_batch = input_data
    else:
        raise ValueError(f"Unexpected transform output shape: {input_data.shape}")

    # Run inference
    with torch.no_grad():
        prediction = midas(input_batch)
        depth = prediction.squeeze().cpu().numpy()

    # Normalize depth to 0–255
    depth_min, depth_max = depth.min(), depth.max()
    depth_normalized = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    cv2.imwrite(depth_path, depth_normalized)
    print(f"Depth map saved to {depth_path}")

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

    # Ensure Matplotlib uses imageio-ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg.get_ffmpeg_exe()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='me'), bitrate=1800)

    ani = animation.FuncAnimation(fig, update, frames=90, interval=50)
    ani.save("output.mp4", writer=writer)

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
