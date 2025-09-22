import h5py
import numpy as np
import matplotlib.pyplot as plt
# === Step 1: Load the H5 file ===
with h5py.File(r"C:\Users\arham\OneDrive\Desktop\devjams\devjams_4_Brain_cells\sample_road_10.h5","r") as f:
    print("Datasets in file:", list(f.keys()))
    road_matrix = f["road_depth"][:]   # Load depth matrix
print("Road matrix shape:", road_matrix.shape)
# === Step 2: Create coordinate grid ===
width_px, length_px = road_matrix.shape
x = np.arange(width_px)    # X = road width direction
y = np.arange(length_px)   # Y = road length direction
X, Y = np.meshgrid(x, y)   # Mesh for plotting
# === Step 3: Plot as heatmap ===
plt.figure(figsize=(5,100))
plt.imshow(road_matrix, cmap="binary", origin="lower", aspect="auto")
plt.colorbar(label="Depth",)
plt.xlabel("")
plt.ylabel("")
plt.title("Potholes Detection on Roads")
plt.show()

