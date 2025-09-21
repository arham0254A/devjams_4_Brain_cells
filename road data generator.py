import numpy as np
import h5py
import os
import random as rd

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

ROAD_WIDTH = 7.0       # meters
ROAD_LENGTH = 100.0    # meters
GRID_RESOLUTION = 0.1  # meters per pixel

# Matrix size
MATRIX_ROWS = int(ROAD_LENGTH / GRID_RESOLUTION)   # 100 / 0.1 = 1000 rows (length)
MATRIX_COLS = int(ROAD_WIDTH / GRID_RESOLUTION)    # 7 / 0.1 = 70 cols (width)

# Georeferencing metadata
START_LAT = 28.6139   
START_LON = 77.2090
CELL_SIZE_DEGREE = GRID_RESOLUTION * 0.000009 
CHUNK_SIZE = (30, 60) 

# --- 3. Directory to Save ---
OUTPUT_DIR = r"C:\Users\akash\OneDrive\Desktop\devjams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 4. Generate 10 Random Road Samples ---
for file_idx in range(1, 11):
    road_matrix = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)
    
    total_potholes = rd.randint(15, 40)  # random number of potholes
    
    for i in range(total_potholes):
        # Random pothole center
        r_center = np.random.randint(3, MATRIX_ROWS - 3) 
        c_center = np.random.randint(5, MATRIX_COLS - 5)
        
        # Severity logic
        severity_roll = np.random.rand()
        if severity_roll < 0.2:   # Deep/large
            max_depth = np.random.uniform(5.0, 9.0)
            r_radius = np.random.uniform(5, 10)
        elif severity_roll < 0.6: # Medium
            max_depth = np.random.uniform(2.5, 5.0)
            r_radius = np.random.uniform(3, 5)
        else:                     # Small/shallow
            max_depth = np.random.uniform(0.5, 2.5)
            r_radius = np.random.uniform(1, 3)

        c_radius = np.random.uniform(r_radius * 0.8, r_radius * 1.2)
        rotation = np.random.uniform(0, 180)
        
        # Local patch
        r_min = max(0, int(r_center - r_radius * 2)); r_max = min(MATRIX_ROWS, int(r_center + r_radius * 2))
        c_min = max(0, int(c_center - c_radius * 2)); c_max = min(MATRIX_COLS, int(c_center + c_radius * 2))
        
        local_shape = (r_max - r_min, c_max - c_min)
        local_center = (r_center - r_min, c_center - c_min)
        
        mask = create_pothole_mask_rough(local_shape, local_center, (r_radius, c_radius), max_depth, rotation)
        road_matrix[r_min:r_max, c_min:c_max] = np.maximum(road_matrix[r_min:r_max, c_min:c_max], mask)
    
    # --- Save File ---
    output_path = os.path.join(OUTPUT_DIR, f"sample_road_{file_idx}.h5")
    
    with h5py.File(output_path, 'w') as f:
        dset = f.create_dataset(
            'road_depth', 
            data=road_matrix, 
            chunks=CHUNK_SIZE, 
            compression='gzip'
        )
        
        # Add attributes
        dset.attrs['ROAD_WIDTH_M'] = ROAD_WIDTH
        dset.attrs['ROAD_LENGTH_M'] = ROAD_LENGTH
        dset.attrs['GRID_RESOLUTION_M'] = GRID_RESOLUTION
        dset.attrs['START_LAT'] = START_LAT
        dset.attrs['START_LON'] = START_LON
        dset.attrs['CELL_SIZE_DEGREE'] = CELL_SIZE_DEGREE
    
    print(f" Saved {output_path} (with {total_potholes} potholes)")
