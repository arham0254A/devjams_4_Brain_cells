import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pandas as pd
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
# 3. Simulate Dispersed Potholes Across the Entire Road 

# Total number of potholes scattered over the 100m length
# Using a higher number to ensure adequate scattering at higher resolution
total_potholes = random.randint(0,30)
np.random.seed(random.randint(1,100000))
for i in range(total_potholes):
    # Potholes are distfributed randomly across the ENTIRE width and length
    r_center = np.random.randint(3, MATRIX_ROWS - 3) 
    c_center = np.random.randint(5, MATRIX_COLS - 5)
    # Randomly assign severity (deeper holes are larger)
    severity_roll = np.random.rand()
    if severity_roll < 0.01: # 20% chance for a deep/large pothole
        max_depth = np.random.uniform(5.1, 7.5)
        r_radius = np.random.uniform(5, 10) # 0.5m to 1.0m radius in pixels
    elif severity_roll < 0.4: # 40% chance for a medium pothole
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
# 4. Save 

OUTPUT_H5_PATH = "nw2.h5"

with h5py.File(OUTPUT_H5_PATH, 'w') as f:
    dset = f.create_dataset(
        'road_depth', 
        data=road_matrix, 
        chunks=CHUNK_SIZE, 
        compression='gzip' 
    )
    
    
    dset.attrs['ROAD_WIDTH_M'] = ROAD_WIDTH
    dset.attrs['ROAD_LENGTH_M'] = ROAD_LENGTH
    dset.attrs['GRID_RESOLUTION_M'] = GRID_RESOLUTION # 0.1m
    dset.attrs['START_LAT'] = START_LAT
    dset.attrs['START_LON'] = START_LON
    dset.attrs['CELL_SIZE_DEGREE'] = CELL_SIZE_DEGREE
    
print(f" Generated and saved dispersed road data (0.1m res) to {OUTPUT_H5_PATH}")
print(f"Matrix Dimensions: {MATRIX_ROWS} rows (width) x {MATRIX_COLS} columns (length)")



import numpy as np

# --- 1. Define Hazard Thresholds (Based on Pothole Metrics) ---

# These values are examples for a 7m x 100m road section (700 m² total area).
# These should be adjusted based on real-world engineering standards.

# Threshold 1: Maximum Tolerable Pothole Depth (Severity)
MAX_TOLERABLE_DEPTH_CM = 10.0 # Any pothole deeper than 5cm is a major concern.

# Threshold 2: Pothole Density (Frequency)
# Total pixels in the matrix: 70 * 1000 = 70,000 pixels
MAX_TOLERABLE_DENSITY = 0.007 # Max 5 potholes per 10,000 pixels (or 1 every 20m²)

# Threshold 3: Overall Road Roughness (Mean Depth)
MAX_MEAN_ROAD_DEPTH_CM = 0.5 # If the average depth of ALL damage exceeds 0.5cm.

# --- NEW: Threshold for HIGH FREQUENCY (Used in the combined check) ---
HIGH_FREQUENCY_THRESHOLD = MAX_TOLERABLE_DENSITY * 1.5 # e.g., 50% above critical density




def generate_road_hazard_rating(road_matrix, grid_resolution_m=0.1):
    # ... (Lines 159-160)
    pothole_mask = road_matrix > 0.5
    total_damage_pixels = np.sum(pothole_mask)

    if total_damage_pixels == 0:
        return ("Good", "Road is perfectly smooth.", 0.0, 0.0, 0)
    
    # --- 2. Calculate Key Metrics (MOVED INSIDE THE IF BLOCK) ---
    # These calculations now run ONLY if damage is present (total_damage_pixels > 0)
    damaged_depths = road_matrix[pothole_mask]
    max_depth = np.max(damaged_depths)
    mean_pothole_depth = np.mean(damaged_depths)
    damage_density = total_damage_pixels / road_matrix.size 
    
    # ... (Continue with the rating logic) ...

    # --- 2. Calculate Key Metrics ---
    
    
    max_depth = np.max(damaged_depths)
    mean_pothole_depth = np.mean(damaged_depths)
    damage_density = (total_damage_pixels / road_matrix.size) # Ratio of damaged pixels to total


    rating = ""
    reason = ""


    # A. CHECK HAZARDOUS CONDITIONS (MODIFIED LOGIC)
    
    # **NEW LOGIC: Requires BOTH high severity AND high frequency for immediate hazard**
    if (max_depth >= MAX_TOLERABLE_DEPTH_CM) and (damage_density >= MAX_TOLERABLE_DENSITY):
        rating = "HAZARDOUS"
        reason = (f"Critical Severity & Frequency: Max depth ({max_depth:.1f}cm) AND "
                  f"damage density ({damage_density*10000:.1f} per 10000) are both critical.")    # B. CHECK WARNING CONDITIONS (Remaining critical checks and general roughness)
    # Check 1: Still flag extreme depth even if frequency is low (a singular, massive pothole)
    elif max_depth >= MAX_TOLERABLE_DEPTH_CM-2:
        rating = "DANGEROUS"
        reason = f"Extreme Severity: Single pothole max depth ({max_depth:.1f}cm) is a severe hazard."
    elif damage_density>=MAX_TOLERABLE_DENSITY:
    # Check 2: Flag high density even if maximum pothole isn't extremely deep (overall roughness)
        rating = "CAUTIOUS"
        reason = f"Critical Frequency: Damage density ({damage_density*10000:.1f} per 10000 pixels) is too high."
    # Check 3: Moderate Density/Roughness Warning
    elif mean_pothole_depth > MAX_MEAN_ROAD_DEPTH_CM:
        rating = "ROUGHNESS WARNING"
        reason = f"Roughness Warning: Average damage depth is high ({mean_pothole_depth:.1f}cm), repair soon."
    # Check 4: Proactive Density Warning
    elif damage_density >= (MAX_TOLERABLE_DENSITY * 0.75): # If half the max density is reached
         rating = "MAINTENANCE WARNING"
         reason = f"Maintenance Warning: Damage frequency is rising, approaching hazardous level."       
    else:
        # All conditions passed, but there is some damage
        rating = "Decent (Minor Damage)"
        reason = "Minor surface defects present but safe for regular use."
    return (rating, reason, max_depth, mean_pothole_depth, total_damage_pixels)
road_matrix_dummy = np.random.rand(70, 1000) *random.uniform(1,10)
road_matrix_dummy[road_matrix_dummy < 2] = 0 
RATING, REASON, MAX_D, MEAN_D, TOTAL_D = generate_road_hazard_rating(road_matrix_dummy)

print("\n--- ROAD HAZARD ASSESSMENT ---")
print(f"Road Quality Rating: {RATING}")
print(f"Reason: {REASON}")
print("-" * 50)
print(f"1. Maximum Pothole Depth: {MAX_D:.2f} cm")
print(f"2. Average Damage Depth: {MEAN_D:.2f} cm")
print(f"3. Total Damaged Pixels: {TOTAL_D}")


print("Road matrix shape:", road_matrix.shape)
width_px, length_px = road_matrix.shape
x = np.arange(width_px)
y = np.arange(length_px) 
X, Y = np.meshgrid(x, y) 
plt.figure(figsize=(5,100))
plt.imshow(road_matrix, cmap="binary", origin="lower", aspect="auto",vmin=0,vmax=10)
plt.colorbar(label="Depth", shrink=0.225,orientation='vertical', pad=0.2)
plt.xlabel(" ")
plt.ylabel(" ")
plt.title(RATING)
plt.show()