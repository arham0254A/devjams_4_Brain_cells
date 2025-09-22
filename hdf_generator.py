import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
# --- 1. Define Pothole Masking Function (Rough Edges) ---
def create_pothole_mask_rough(shape, center, radii, max_depth, rotation_deg=0):
    """
    Generates an elliptical mask with added Gaussian noise for irregular edges
    and depth variation, simulating a rough pothole.
    """
    rows, cols = shape
    r, c = np.mgrid[0:rows, 0:cols]
    
    r -= center[0]
    c -= center[1]

    rotation_rad = np.deg2rad(rotation_deg)
    
    r_rot = r * np.cos(rotation_rad) - c * np.sin(rotation_rad)
    c_rot = r * np.sin(rotation_rad) + c * np.cos(rotation_rad)
    
    ellipse_dist = (c_rot / radii[1])**2 + (r_rot / radii[0])**2
    
    base_mask = np.exp(-ellipse_dist / 1.5) 
    noise = np.random.normal(loc=1.0, scale=0.2, size=shape)
    
    rough_mask = base_mask * noise
    
    rough_mask[ellipse_dist > 3.0] = 0.0
    
    final_mask = np.clip(rough_mask * max_depth, 0, max_depth)
    
    return final_mask

# ----------------------------------------------------------------------
# --- 2. Define Road Parameters and Georeferencing Metadata ---

ROAD_WIDTH = 7.0       # 7 meters
ROAD_LENGTH = 100.0    # 100 meters
GRID_RESOLUTION = 0.1 # Meters per pixel <-- CHANGED

# Calculate matrix size
MATRIX_ROWS = int(ROAD_LENGTH / GRID_RESOLUTION)  # 7m / 0.1m = 70 rows
MATRIX_COLS = int(ROAD_WIDTH / GRID_RESOLUTION) # 100m / 0.1m = 1000 columns

# Georeferencing metadata: New Delhi, India
START_LAT = 28.6139   
START_LON = 77.2090
CELL_SIZE_DEGREE = GRID_RESOLUTION * 0.000009 
CHUNK_SIZE = (30, 60) 

# Initialize the road matrix
road_matrix = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)

# --- 3. Simulate Dispersed Potholes Across the Entire Road ---

# Total number of potholes scattered over the 100m length
# Using a higher number to ensure adequate scattering at higher resolution
total_potholes = random.randint(15,40)

for i in range(total_potholes):
    # Potholes are distributed randomly across the ENTIRE width and length
    r_center = np.random.randint(3, MATRIX_ROWS - 3) 
    c_center = np.random.randint(5, MATRIX_COLS - 5)
    
    # Randomly assign severity (deeper holes are larger)
    severity_roll = np.random.rand()
    if severity_roll < 0.2: # 20% chance for a deep/large pothole
        max_depth = np.random.uniform(5.0, 9.0)
        r_radius = np.random.uniform(5, 10) # 0.5m to 1.0m radius in pixels
    elif severity_roll < 0.6: # 40% chance for a medium pothole
        max_depth = np.random.uniform(2.5, 5.0)
        r_radius = np.random.uniform(3, 5) # 0.3m to 0.5m radius
    else: # 40% chance for a shallow/small pothole
        max_depth = np.random.uniform(0.5, 2.5)
        r_radius = np.random.uniform(1, 3) # 0.1m to 0.3m radius

    c_radius = np.random.uniform(r_radius * 0.8, r_radius * 1.2) # Elliptical variation
    rotation = np.random.uniform(0, 180)
    
    # Calculate local matrix boundaries
    r_min = max(0, int(r_center - r_radius * 2)); r_max = min(MATRIX_ROWS, int(r_center + r_radius * 2))
    c_min = max(0, int(c_center - c_radius * 2)); c_max = min(MATRIX_COLS, int(c_center + c_radius * 2))
    
    local_shape = (r_max - r_min, c_max - c_min)
    local_center = (r_center - r_min, c_center - c_min)
    
    mask = create_pothole_mask_rough(local_shape, local_center, (r_radius, c_radius), max_depth, rotation)
    road_matrix[r_min:r_max, c_min:c_max] = np.maximum(road_matrix[r_min:r_max, c_min:c_max], mask)

# 4. Save to Chunked H5 File and Add Metadata

OUTPUT_H5_PATH = "india_road_pothole_data_dispersed_0.1m.h5"

with h5py.File(OUTPUT_H5_PATH, 'w') as f:
    dset = f.create_dataset(
        'road_depth', 
        data=road_matrix, 
        chunks=CHUNK_SIZE, 
        compression='gzip' 
    )
    
    # Add crucial georeferencing attributes
    dset.attrs['ROAD_WIDTH_M'] = ROAD_WIDTH
    dset.attrs['ROAD_LENGTH_M'] = ROAD_LENGTH
    dset.attrs['GRID_RESOLUTION_M'] = GRID_RESOLUTION # 0.1m
    dset.attrs['START_LAT'] = START_LAT
    dset.attrs['START_LON'] = START_LON
    dset.attrs['CELL_SIZE_DEGREE'] = CELL_SIZE_DEGREE
    
print(f" Generated and saved dispersed road data (0.1m res) to {OUTPUT_H5_PATH}")
print(f"Matrix Dimensions: {MATRIX_ROWS} rows (width) x {MATRIX_COLS} columns (length)")