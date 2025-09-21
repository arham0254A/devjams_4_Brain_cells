import h5py
import numpy as np
length_m = 400  
width_m = 13    
divisions_per_meter = 10
length_px = length_m * divisions_per_meter
width_px = width_m * divisions_per_meter
road_matrix = np.random.randint(0, 10, size=(width_px, length_px))
with h5py.File(r"C:\Users\akash\OneDrive\Desktop\devjams\file_1.h5", "w") as f:
    f.create_dataset("road", data=road_matrix)
print("File saved with dataset shape:", road_matrix.shape)