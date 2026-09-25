import os
import time
import cv2
from statistics import mean
from tqdm import tqdm
import gc
import torch
from pathlib import Path
import numpy as np 

import data_preprocessing
import perspective_projection
import facade_extraction

# Configure root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Measure total script runtime
run_start = time.perf_counter()

# 1. DEFINE INPUT DATA
# --------------------
# Select height estimation method
# height_estimation_method = "unet_method" # "cc_method", "depth_method", "unet_method"

# Define input directories and files
tiles_directory = os.path.join(PROJECT_ROOT, 'data/3dbag_tiles')
underpasses_directory = os.path.join(PROJECT_ROOT, 'data/underpass_polygons')
geojson_2d_path = os.path.join(tiles_directory, 'lod22_2d.geojson')
geojson_3d_path = os.path.join(tiles_directory, 'lod22_3d.geojson')

underpasses_path = os.path.join(underpasses_directory, 'underpasses_rotterdam.geojson')

oblique_images_dir = os.path.join(PROJECT_ROOT, 'data/oblique_images')
image_footprints_path = os.path.join(oblique_images_dir, 'rotterdam_footprints_renamed.geojson')
tile_footprint_path = os.path.join(oblique_images_dir, 'intersected_tile_footprint.geojson')
camera_parameters_path = os.path.join(PROJECT_ROOT, 'output/rotterdam_camera_parameters.csv')

rotterdam_sample_dir = os.path.join(PROJECT_ROOT, 'rotterdam_sample')

oblique_image_dirs = [
    os.path.join(rotterdam_sample_dir, 'oblique_right'),
    os.path.join(rotterdam_sample_dir, 'oblique_left'),
    os.path.join(rotterdam_sample_dir, 'oblique_forward'),
    os.path.join(rotterdam_sample_dir, 'oblique_back')
]

# Output directory for all facades
facades_output_dir = Path(os.path.expanduser("~/Desktop")) / "extracted_facades"
facades_output_dir.mkdir(parents=True, exist_ok=True)

'''
# Load model if needed
if height_estimation_method == "depth_method":
    depth_map_model = height_estimation.load_depth_map_model(depth_model_directory)
elif height_estimation_method == "unet_method":
    unet_device, unet_model = height_estimation.load_unet_model(unet_model_directory)

# Define output file according to selected method
if height_estimation_method == "cc_method":
    output_path = os.path.join(PROJECT_ROOT, 'output/underpass_heights_ccmethod.geojson')
    # Define output for groud truth visualization
    output_ground_truth_path = os.path.join(PROJECT_ROOT, 'output/visualizations_cc')
    os.makedirs(output_ground_truth_path, exist_ok=True)
                                   
elif height_estimation_method == "depth_method":
    output_path = os.path.join(PROJECT_ROOT, 'output/underpass_heights_depthmethod.geojson')
    # Define output for groud truth visualization
    output_ground_truth_path = os.path.join(PROJECT_ROOT, 'output/visualizations_depth')
    os.makedirs(output_ground_truth_path, exist_ok=True)

elif height_estimation_method == "unet_method":
    output_path = os.path.join(PROJECT_ROOT, 'output/underpass_heights_unetmethod.geojson')
    # Define output for groud truth visualization
    output_ground_truth_path = os.path.join(PROJECT_ROOT, 'output/visualizations_unet')
    os.makedirs(output_ground_truth_path, exist_ok=True)
'''

# 2. LOAD INPUT DATA IN GEOPANDAS DATAFRAMES
# ------------------------------------------
df_camera_parameters, gdf_underpass_polygons, gdf_building_2d, gdf_building_3d, gdf_image_footprints = (
    data_preprocessing.load_input_data(
        camera_parameters_path, underpasses_path, geojson_2d_path, geojson_3d_path, image_footprints_path, min_length=2))

if 'id' in df_camera_parameters.columns:
    df_camera_parameters['image_id'] = df_camera_parameters['id']

print("[INFO] Computing critical edges and walls...")
    # 4. FIND CRITICAL EDGES (INTERSECTION OF BUILDING FOOTPRINTS WITH UNDERPASSES) 
    # (ONLY IF underpass edges are not provided)
    # -----------------------------------------------------------------------------
gdf_underpass_intersected, gdf_critical_edges = data_preprocessing.find_critical_edges(
    gdf_underpass_polygons, gdf_building_2d, buf_tol=0.1, simpl_tol=0.2, min_length=2
)

    # 5. FIND CRITICAL WALLS (EXTRUDE CRITICAL EDGES TO 3D)
    # -----------------------------------------------------
gdf_critical_walls = data_preprocessing.find_critical_walls(
    gdf_critical_edges, None, gdf_building_3d, buf_tol=0.5, extend_length=0
)

    # 6. CONSTRUCT IMAGE - WALL VISIBILITY TABLE (INTERSECTION OF CRITICAL WALLS WITH IMAGE FOOTPRINTS)
    # -------------------------------------------------------------------------------------------------
gdf_image_visibility = data_preprocessing.infere_image_visibility(
    gdf_image_footprints, gdf_critical_walls, theta=60
)

saved_count = 0
missing_count = 0
no_wall_count = 0

# Track exact rejection between the stages
proj_failed_count = 0
facade_warp_failed_count = 0

def find_image_path(image_id, search_dirs):
    """Searches for an image file across multiple directories."""
    filename = str(image_id)
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
        filename += '.jpg'

    for directory in search_dirs:
        candidate_path = os.path.join(directory, filename)
        if os.path.exists(candidate_path):
            return candidate_path
            
    return None

# check only on 1 image to see if this is the issue ??

img1 = "217000682_0034_01_0169_P00_01"

gdf_image_visibility = gdf_image_visibility[gdf_image_visibility['image_id'].astype(str).str.contains(img1)]


    # 7. PERFORM PERSPECTIVE PROJECTION OF CRITICAL WALLS ONTO OBLIQUE IMAGES
    # -----------------------------------------------------------------------
for _, row in tqdm(
    gdf_image_visibility.iterrows(),
    total=len(gdf_image_visibility),
    desc="Processing visible images",
    unit="image"):
    wall_ids = row.get('visible_walls', [])
    if not wall_ids:
        no_wall_count += 1
        continue

    raw_id = str(row['image_id'])
    image_id = raw_id if raw_id.lower().endswith('.jpg') else f"{raw_id}.jpg"

    # 1. Search for image file across the 4 oblique directories
    image_path = find_image_path(image_id, oblique_image_dirs)

    if image_path is None:
        missing_count += 1
        continue

    # 2. Read image with OpenCV
    oblique_image = cv2.imread(image_path)
    if oblique_image is None:
        missing_count += 1
        continue

    # 3. Wall Projection Step
    rectangles_2d = perspective_projection.project_walls_on_image(
        image_id,
        wall_ids,
        df_camera_parameters,
        gdf_critical_walls
    )

    valid = [r for r in rectangles_2d if r is not None]
    if valid:
        dbg = oblique_image.copy()
        for r in valid:
            cv2.polylines(dbg, [r.reshape(-1, 1, 2)], True, (0, 0, 255), 20)
            for i, (x, y) in enumerate(r):
                cv2.putText(dbg, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 12)
        cv2.imwrite(os.path.expanduser('~/Desktop/debug_projection.jpg'),
                    cv2.resize(dbg, None, fx=0.2, fy=0.2))
            
    if not rectangles_2d:
        proj_failed_count += 1
        continue
          
    none_rect_count = 0 
    
    # 4. Extract and save facades
    for wall_id, rect_2d in zip(wall_ids, rectangles_2d):
        if rect_2d is None:
            none_rect_count += 1
            continue

        facade_image = facade_extraction.extract_facade(rect_2d, oblique_image)
        if facade_image is None:
            facade_warp_failed_count += 1
            continue

        out_filename = f"facade_{image_id}_{wall_id}"
        if not out_filename.endswith('.jpg'):
            out_filename += '.jpg'
            
        cv2.imwrite(os.path.join(facades_output_dir, out_filename), facade_image)
        saved_count += 1

        del facade_image

    del oblique_image
    gc.collect()

print(f" [INFO] Pipeline Breakdown:")
print(f" Saved: {saved_count}")
print(f" Projection returned None/empty: {proj_failed_count}")
print(f" Warping/extraction returned None: {facade_warp_failed_count}")
elapsed = time.perf_counter() - run_start
print(f" [INFO] Pipeline completed in {elapsed} seconds.")
print(f" [INFO] Saved: {saved_count} | Missing images: {missing_count} | No visible walls: {no_wall_count}")
