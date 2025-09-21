#generating h5py
import h5py
import numpy as np

# Road dimensions
length_m = 400   # meters
width_m = 13     # meters
divisions_per_meter = 10

# Calculate matrix size
length_px = length_m * divisions_per_meter
width_px = width_m * divisions_per_meter

# Create full matrix filled with random depths 0–9
road_matrix = np.random.randint(0, 10, size=(width_px, length_px))

# Save into an HDF5 file
with h5py.File(r"C:\Users\arham\OneDrive\Desktop\devjams\file2.h5", "w") as f:
    f.create_dataset("road", data=road_matrix)

print("File saved with dataset shape:", road_matrix.shape)
