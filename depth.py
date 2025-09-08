import torch
import cv2
import urllib.request
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Load your image
img = cv2.imread("example.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).unsqueeze(0)

# Run depth prediction
with torch.no_grad():
    prediction = midas(input_batch)
    depth = prediction.squeeze().cpu().numpy()

# Normalize to 0-255
depth_min = depth.min()
depth_max = depth.max()
depth_normalized = (255 * (depth - depth_min) / (depth_max - depth_min)).astype("uint8")

# Save depth map
cv2.imwrite("depth.png", depth_normalized)
