
import numpy as np
import h5py
import matplotlib.pyplot as plt 

# --- 1. Define Pothole Masking Function ---
def create_pothole_mask(shape, center, radii, rotation_deg=0):
    """
    Generates an elliptical or circular mask within a given shape, 
    simulating a realistic, non-square pothole.
    """
    rows, cols = shape
    r, c = np.mgrid[0:rows, 0:cols]
    
    r -= center[0]
    c -= center[1]
    
    rotation_rad = np.deg2rad(rotation_deg)
    
    # Apply rotation transformation
    r_rot = r * np.cos(rotation_rad) - c * np.sin(rotation_rad)
    c_rot = r * np.sin(rotation_rad) + c * np.cos(rotation_rad)
    
    # Calculate elliptical distance
    ellipse_dist = (c_rot / radii[1])**2 + (r_rot / radii[0])**2
    
    # Create a smooth, deep center with fading edges
    mask = np.exp(-ellipse_dist)
    
    mask[ellipse_dist > 4] = 0.0  # Cut off the very faint edges
    
    return mask

# --- 2. Define Road Parameters and Georeferencing Metadata ---
ROAD_WIDTH = 13.0      # 13 meters (Matrix Rows)
ROAD_LENGTH = 100.0    # 100 meters (Matrix Columns) <-- CHANGED
GRID_RESOLUTION = 0.25 # Meters per pixel

# Calculate matrix size
MATRIX_ROWS = int(ROAD_WIDTH / GRID_RESOLUTION)  # 13m / 0.25m = 52 rows
MATRIX_COLS = int(ROAD_LENGTH / GRID_RESOLUTION) # 100m / 0.25m = 400 columns <-- CHANGED

# Actual georeferencing metadata (London, UK)
START_LAT = 51.5074   
START_LON = 0.1278
CELL_SIZE_DEGREE = GRID_RESOLUTION * 0.000009 
CHUNK_SIZE = (50, 100) 

# Initialize the road matrix
road_matrix = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)

# --- 3. Simulate Realistic Pothole Clusters ---

# Pothole Cluster 1: High Severity (Dense, near the start of the 100m segment)
CLUSTER_AREA_C = (50, 150) # Columns 50 to 150 (12.5m to 37.5m along the road)
num_potholes_1 = 15 # Reduced count for the shorter segment

for i in range(num_potholes_1):
    r_center = np.random.randint(10, MATRIX_ROWS - 10)
    c_center = np.random.randint(CLUSTER_AREA_C[0], CLUSTER_AREA_C[1])
    
    r_radius = np.random.uniform(3, 7) # In pixels
    c_radius = np.random.uniform(r_radius * 0.8, r_radius * 1.2) 
    
    max_depth = np.random.uniform(5.0, 9.0)
    rotation = np.random.uniform(0, 180)
    
    # --- Local Mask Calculation ---
    r_min = max(0, int(r_center - r_radius * 2))
    r_max = min(MATRIX_ROWS, int(r_center + r_radius * 2))
    c_min = max(0, int(c_center - c_radius * 2))
    c_max = min(MATRIX_COLS, int(c_center + c_radius * 2))
    
    local_shape = (r_max - r_min, c_max - c_min)
    local_center = (r_center - r_min, c_center - c_min)
    
    mask = create_pothole_mask(local_shape, local_center, (r_radius, c_radius), rotation)
    
    # Add depth to the main matrix
    road_matrix[r_min:r_max, c_min:c_max] = np.maximum(
        road_matrix[r_min:r_max, c_min:c_max],
        mask * max_depth
    )

# Pothole Cluster 2: Low Severity/Sparse (Near the end of the 100m segment)
CLUSTER_AREA_C_2 = (300, 380) # Columns 300 to 380 (75m to 95m along the road)
num_potholes_2 = 8

for i in range(num_potholes_2):
    r_center = np.random.randint(15, MATRIX_ROWS - 15)
    c_center = np.random.randint(CLUSTER_AREA_C_2[0], CLUSTER_AREA_C_2[1])
    r_radius = np.random.uniform(2, 4)
    c_radius = np.random.uniform(r_radius * 0.9, r_radius * 1.1)
    max_depth = np.random.uniform(1.0, 3.0)
    rotation = np.random.uniform(0, 180)
    
    r_min = max(0, int(r_center - r_radius * 2))
    r_max = min(MATRIX_ROWS, int(r_center + r_radius * 2))
    c_min = max(0, int(c_center - c_radius * 2))
    c_max = min(MATRIX_COLS, int(c_center + c_radius * 2))
    
    local_shape = (r_max - r_min, c_max - c_min)
    local_center = (r_center - r_min, c_center - c_min)
    
    mask = create_pothole_mask(local_shape, local_center, (r_radius, c_radius), rotation)
    
    road_matrix[r_min:r_max, c_min:c_max] = np.maximum(
        road_matrix[r_min:r_max, c_min:c_max],
        mask * max_depth
    )
# 4. Save to Chunked H5 File and Add Metadata
OUTPUT_H5_PATH = "chunked_road_pothole_data_13m_100m.h5"

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
    dset.attrs['START_LAT'] = START_LAT
    dset.attrs['START_LON'] = START_LON
    dset.attrs['CELL_SIZE_DEGREE'] = CELL_SIZE_DEGREE
    dset.attrs['GRID_RESOLUTION_M'] = GRID_RESOLUTION
    
print(f"✅ Generated and saved road data (13m width, 100m length) to {OUTPUT_H5_PATH}")
print(f"Matrix Dimensions: {MATRIX_ROWS} rows (width) x {MATRIX_COLS} columns (length)")