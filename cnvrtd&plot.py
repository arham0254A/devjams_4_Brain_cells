import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Step 1: Load the H5 file ===
with h5py.File(r"C:\Users\akash\OneDrive\Desktop\devjams\file_1.h5", "r") as f:
    print("Datasets in file:", list(f.keys()))
    road_matrix = f["road"][:]   # Load depth matrix

print("Road matrix shape:", road_matrix.shape)

# === Step 2: Create coordinate grid ===
width_px, length_px = road_matrix.shape
x = np.arange(width_px)    # X = road width direction
y = np.arange(length_px)   # Y = road length direction
X, Y = np.meshgrid(y, x)   # Mesh for plotting

# === Step 3: Plot as heatmap ===
plt.figure(figsize=(10, 6))
plt.imshow(road_matrix, cmap="viridis", origin="lower", aspect="auto")
plt.colorbar(label="Depth")
plt.xlabel("Road Length (Y axis)")
plt.ylabel("Road Width (X axis)")
plt.title("Road Depth Heatmap")
plt.show()
