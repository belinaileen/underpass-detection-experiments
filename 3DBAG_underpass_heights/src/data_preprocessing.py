import geopandas as gpd
import pandas as pd
from shapely.ops import transform
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon
import pyvista as pv
import os

def load_input_data(camera_parameters_path, underpasses_path, image_footprints_path, min_length):
    """
    Load input data from specified paths and preprocess them into GeoDataFrames.

    Args:
        camera_parameters_path (str): Path to the camera parameters file.
        image_footprints_path (str): Path to the image footprints file.
        underpasses_path (str): Path to the underpass polygons file.
        underpass_edges_path (str): Path to the underpass edges file (optional, can be None).
        min_length (float): Minimum length of edges to be considered as critical.

    Returns:
        tuple: A tuple containing the following GeoDataFrames:
            - df_camera_parameters (GeoDataFrame): GeoDataFrame containing camera parameters.
            - gdf_image_footprints (GeoDataFrame): GeoDataFrame containing image footprints.
            - gdf_underpass_polygons (GeoDataFrame): GeoDataFrame containing underpass polygons.
            - gdf_underpass_edges (GeoDataFrame or None): GeoDataFrame containing underpass edges if provided, otherwise None.
    Changes from Juan's code:
        - takes XYZ present within the footprint file
        - removal of underpass edges, ground truth
        - does not add observed_heights column 
    """
    # Load camera parameters
    df_camera_parameters = pd.read_csv(camera_parameters_path, sep=',', dtype={'image_id': str})

    # Load image footprints. Add camera center per footprint
    gdf_image_footprints = gpd.read_file(image_footprints_path)
    gdf_image_footprints = gdf_image_footprints.set_crs("EPSG:7415", allow_override=True)

    # Load underpass polygons
    gdf_underpass_polygons = gpd.read_file(underpasses_path)
    gdf_underpass_polygons = gdf_underpass_polygons.set_crs("EPSG:7415", allow_override=True)
    gdf_underpass_polygons = gdf_underpass_polygons.rename(columns={'identificatie': 'building_id'})
    gdf_underpass_polygons['underpass_id'] = gdf_underpass_polygons.index + 1

    # merge for ading camera center per footprint
    # gdf_image_footprints = gdf_image_footprints.merge(df_camera_parameters, left_on='image_id', right_on='image_id', how='left')
    gdf_image_footprints = gdf_image_footprints[['image_id', 'geometry', 'X', 'Y', 'Z']].rename(columns={'X': 'camera_x', 'Y': 'camera_y', 'Z': 'camera_z'})

    return df_camera_parameters, gdf_underpass_polygons, gdf_image_footprints


def load_tile_data( geojson_2d_path, geojson_3d_path):

    """
    Load building footprints and 3D geometries from a given 3D BAG tile (GeoJSON) and preprocess them into GeoDataFrames.

    Args:
        tile_path (str): Path to the 3D BAG tile GeoJSON file.

    Returns:
        tuple: A tuple containing the following GeoDataFrames:
            - gdf_building_footprints (GeoDataFrame): GeoDataFrame containing building footprints with columns 'building_id' and 'geometry'.
            - gdf_building_3d (GeoDataFrame): GeoDataFrame containing building 3D geometries with columns 'building_id' and 'geometry'.

    """

    # Read 2D geometries into geodata frame
    gdf_building_2d = gpd.read_file(geojson_2d_path)
    gdf_building_2d = gdf_building_2d.rename(columns={'identificatie': 'building_id'})
    gdf_building_2d = gdf_building_2d[['building_id', 'geometry']]

    # Read 3D geometries into geodata frame
    gdf_building_3d = gpd.read_file(geojson_3d_path)
    gdf_building_3d = gdf_building_3d.rename(columns={'identificatie': 'building_id'})
    gdf_building_3d = gdf_building_3d[['building_id', 'geometry']]

    return gdf_building_2d, gdf_building_3d

def find_critical_edges(gdf_underpass_polygons, gdf_building_footprints, buf_tol, simpl_tol, min_length):
    """Find critical edges of underpass polygons that intersect with building footprints.

    Args:
        gdf_underpass_polygons (GeoDataFrame): GeoDataFrame containing underpass polygons with columns 'underpass_id', 'building_id', and 'geometry'.
        gdf_building_footprints (GeoDataFrame): GeoDataFrame containing building footprints with columns 'building_id' and 'geometry'.
        buf_tol (float): Buffer tolerance for the intersection of underpass polygons with building fottprints.
        simpl_tol (float): Simplification tolerance for underpass polygons to reduce the number of points in the extracted edges.
        min_length (float): Minimum length of edges to be considered as critical.

    Returns:
        GeoDataFrame: A GeoDataFrame containing critical edges with columns 'edge_id', 'geometry', 'underpass_id', and 'building_id'.
    """

    # Intersect buildings with underpass polygons
    gdf_building_2d = gdf_building_2d.to_crs(gdf_underpass_polygons.crs)
    gdf_underpass_intersected = gpd.sjoin(gdf_underpass_polygons, gdf_building_2d, how='inner', predicate='intersects')
    gdf_underpass_intersected = gdf_underpass_intersected[['underpass_id', 'building_id_left', 'geometry']].rename(columns={'building_id_left': 'building_id'})

    # Extract edges from intersected underpass polygons
    edge_records = []
    edge_id = 1
    for _, row in gdf_underpass_intersected.iterrows():
        underpass_id = row['underpass_id']
        building_id = row['building_id']

        # Simplify polygon to reduce the amount of points in the exracted edges
        poly = row['geometry'].simplify(tolerance=simpl_tol, preserve_topology=True)
        coords = list(poly.exterior.coords)

        building_geom = gdf_building_footprints.loc[gdf_building_2d.building_id == building_id, 'geometry'].iloc[0]
        building_boundary_buffered = building_geom.boundary.buffer(buf_tol)

        for i in range(len(coords) - 1):
            candidate_edge = shapely.geometry.LineString([coords[i], coords[i+1]])
            # Intersect edge with buidling ID, if True, label as critical edge
            if candidate_edge.length < min_length:
                continue
            if candidate_edge.intersects(building_boundary_buffered):
                edge_records.append({'edge_id': edge_id, 'geometry': candidate_edge, 'underpass_id': underpass_id, 'building_id': building_id})
                edge_id += 1

    gdf_critical_edges = gpd.GeoDataFrame(edge_records, crs=gdf_underpass_polygons.crs)

    return gdf_underpass_intersected, gdf_critical_edges

def visualize_critical_edges(gdf_underpass_intersected, gdf_critical_edges):

    fig, ax = plt.subplots(figsize=(10, 10))

    gdf_critical_edges.plot(
        ax=ax,
        color="red",
        linewidth=3
    )
    gdf_underpass_intersected.plot(
        ax=ax,
        color="gray",
        linewidth=1
    )

    ax.set_title("Critical Underpass Edges")
    ax.axis("off")
    ax.set_aspect("equal")
    plt.show()


def find_critical_walls(gdf_critical_edges, gdf_underpass_edges, gdf_building_3d, buf_tol, extend_length):

    """
    Find critical walls by extruding critical edges to 3D using the height of building 3D geometries. 
    If underpass edges are provided, use them to find the critical walls instead of critical edges.

    Args:
        gdf_critical_edges (GeoDataFrame): GeoDataFrame containing critical edges.
        gdf_underpass_edges (GeoDataFrame or None): GeoDataFrame containing underpass edges if provided, otherwise None.
        gdf_building_3d (GeoDataFrame): GeoDataFrame containing building 3D geometries with columns 'building_id' and 'geometry'.
        buf_tol (float): Buffer tolerance for the intersection of underpass edges with building 3D geometries.
        extend_length (float): Length to extend the walls on both sides of the edge.
    
    Returns:
        GeoDataFrame: A GeoDataFrame containing critical walls.

    """
    # Extract 3d wall polygons from buildings
    wall_polygon_records = []
    wall_polygon_id = 1
    for _, row in gdf_building_3d.iterrows():
        building_id = row['building_id']
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            wall_polygon_records.append({'wall_id': wall_polygon_id, 'geometry': geom, 'building_id': building_id})
            wall_polygon_id += 1
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                wall_polygon_records.append({'wall_id': wall_polygon_id, 'geometry': poly, 'building_id': building_id})
                wall_polygon_id += 1

    gdf_wall_polygons = gpd.GeoDataFrame(wall_polygon_records, crs=gdf_building_3d.crs)
    
    # Intersect 3d wall polygons with critical edges (buffered) or provided underpass edges
    if gdf_underpass_edges is not None:
        gdf_underpass_edges_buffered = gdf_underpass_edges.copy()
        gdf_underpass_edges_buffered['geometry'] = gdf_underpass_edges_buffered.geometry.buffer(buf_tol)

        gdf_underpass_edges_buffered = gdf_underpass_edges_buffered.to_crs(gdf_wall_polygons.crs)
        gdf_walls_intersected = gpd.sjoin(gdf_wall_polygons, gdf_underpass_edges_buffered, how='inner', predicate='intersects')
        # Make sure that the extracted 3D polygons belong to the same building as the underpass edge
        gdf_walls_intersected = gdf_walls_intersected[
            gdf_walls_intersected['building_id_left'] == gdf_walls_intersected['building_id_right']
        ]
        gdf_walls_intersected = gdf_walls_intersected[
            ['wall_id', 'geometry', 'edge_id', 'building_id_right', 'underpass_id']
        ].rename(columns={'building_id_right': 'building_id'})
    
    else:
        gdf_critical_edges_buffered = gdf_critical_edges.copy()
        gdf_critical_edges_buffered['geometry'] = gdf_critical_edges_buffered.geometry.buffer(buf_tol)

        gdf_critical_edges_buffered = gdf_critical_edges_buffered.to_crs(gdf_wall_polygons.crs)
        gdf_walls_intersected = gpd.sjoin(gdf_wall_polygons, gdf_critical_edges_buffered, how='inner', predicate='intersects')

        # Make sure that the extracted 3D polygons belong to the same building as the underpass edge
        gdf_walls_intersected = gdf_walls_intersected[
            gdf_walls_intersected['building_id_left'] == gdf_walls_intersected['building_id_right']
        ]
        gdf_walls_intersected = gdf_walls_intersected[
            ['wall_id', 'geometry', 'edge_id', 'building_id_right', 'underpass_id']
        ].rename(columns={'building_id_right': 'building_id'})

    # Create walls using the height of wall polygons and the edges
    wall_records = []
    wall_id = 1
    for edge_id, group in gdf_walls_intersected.groupby('edge_id'):
        # Find the maximum and minimum height of the wall polygons which belong to an edge
        max_z = -float('inf')
        min_z = float('inf')
        for geom in group['geometry']:
            for x, y, z in geom.exterior.coords:
                if z > max_z:
                    max_z = z
                elif z < min_z:
                    min_z = z
        # Find the geometry of the edge (depending on provided table)
        if gdf_underpass_edges is not None:
            edge_geom = gdf_underpass_edges[gdf_underpass_edges['edge_id'] == edge_id]['geometry'].iloc[0]
        else:
            edge_geom = gdf_critical_edges[gdf_critical_edges['edge_id'] == edge_id]['geometry'].iloc[0]
        # bottom2d = list(edge_geom.coords)
        p1 = np.array(edge_geom.coords[0])
        p2 = np.array(edge_geom.coords[1])
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        direction_unit = direction / length
        # Extend walls on both sides (create wider walls)
        new_p1 = p1 - direction_unit * extend_length
        new_p2 = p2 + direction_unit * extend_length
        bottom2d = [tuple(new_p1), tuple(new_p2)]

        bottom3d = [(x, y, min_z) for x, y, *_ in bottom2d]
        upper3d = [(x, y, max_z) for x, y, *_  in reversed(bottom2d)]

        wall_coords = bottom3d + upper3d
        
        # Build the wall geometry
        wall_geom = shapely.geometry.polygon.orient(Polygon(wall_coords), sign=1.0)
        # Build wall records
        underpass_id = group['underpass_id'].iloc[0]
        wall_records.append({
            'wall_id': wall_id,
            'geometry': wall_geom,
            'wall_height': max_z - min_z,
            'underpass_id': underpass_id,
            'edge_id': edge_id,
            'building_id': group['building_id'].iloc[0]
        })
        wall_id += 1

    gdf_critical_walls = gpd.GeoDataFrame(wall_records, crs=gdf_wall_polygons.crs)

    # If there are two overlapping walls or they are too close, keep the largest wall
    walls_to_remove = set()
    for i in range(len(gdf_critical_walls)):
        if i in walls_to_remove:
            continue
        geom_i = gdf_critical_walls.iloc[i]['geometry']
        area_i = geom_i.area
        for j in range(i + 1, len(gdf_critical_walls)):
            if j in walls_to_remove:
                continue
            geom_j = gdf_critical_walls.iloc[j]['geometry']
            # Check if walls overlap
            if  geom_i.within(geom_j) or geom_j.within(geom_i):
                area_j = geom_j.area
                # Keep the larger one, mark the smaller for removal
                if area_i > area_j:
                    walls_to_remove.add(j)
                else:
                    walls_to_remove.add(i)
                    break
    
    # Remove marked walls and reset indices
    gdf_critical_walls = gdf_critical_walls.drop(list(walls_to_remove)).reset_index(drop=True)

    return gdf_critical_walls


def visualize_critical_walls(gdf_critical_walls):

    """
    Visualize critical walls in 3D using PyVista.

    Args:
        gdf_critical_walls (GeoDataFrame): GeoDataFrame containing critical walls.

    Returns:
        None

    """

    plotter = pv.Plotter()
    for geom in gdf_critical_walls.geometry:
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords)
            poly = pv.PolyData(coords)
            poly.faces = np.hstack([[len(coords)], np.arange(len(coords))])
            plotter.add_mesh(poly, color="lightblue", show_edges=True)

    plotter.show()


def infere_image_visibility(gdf_image_footprints, gdf_critical_walls, theta):

    """
    Infer the visibility of critical walls in each image by intersecting image footprints with critical walls.
    The function checks for the visibility of the walls based on two criteria:
    1) The wall normal should face the camera (dot product test)
    2) The angle between the wall normal and the camera plane normal should be less than a threshold theta (remove too oblique views)

    Args:
        gdf_image_footprints (GeoDataFrame): GeoDataFrame containing image footprints with camera parameters.
        gdf_critical_walls (GeoDataFrame): GeoDataFrame containing critical walls with 3D geometries.
        theta (float): Angle threshold in degrees to filter out oblique views.

    Returns:
        gdf_image_visibility (GeoDataFrame): A GeoDataFrame containing image IDs, visible walls, and camera parameters
    
    """

    # Intersect image footprints with critical walls
    gdf_image_footprints = gdf_image_footprints.to_crs(gdf_critical_walls.crs)
    gdf_image_visibility = gpd.sjoin(gdf_image_footprints, gdf_critical_walls, how='inner', predicate='intersects')

    gdf_image_visibility = (
        gdf_image_visibility
        .groupby("image_id", as_index=False)
        .agg({
            "wall_id": list,
            "camera_x": "first",
            "camera_y": "first",
            "camera_z": "first",
        })
    )

    gdf_image_visibility = gdf_image_visibility[['image_id', 'wall_id', 'camera_x', 'camera_y', 'camera_z']].rename(columns={'wall_id': 'visible_walls'})

    # Check for the visibility of the walls, remove if not visible. 
    for idx, row in gdf_image_visibility.iterrows():

        wall_ids = row['visible_walls']
        camera_x = row['camera_x']
        camera_y = row['camera_y']
        camera_z = row['camera_z']

        filtered_walls = []
        for wall_id in wall_ids:
            # Get 3D geometry of the wall to calculate normal
            wall_geom_3d = gdf_critical_walls[gdf_critical_walls['wall_id'] == wall_id]['geometry'].iloc[0]
            coords = list(wall_geom_3d.exterior.coords)

            p0 = np.array(coords[0])
            p1 = np.array(coords[1])
            p3 = np.array(coords[3])
            # vertical edge (z direction)
            v_vert = p1 - p0
            # horizontal edge (xy direction)
            v_horiz = p3 - p0

            # Compute and normalize wall normal
            wall_normal = np.cross(v_horiz, v_vert)
            wall_normal_norm = np.linalg.norm(wall_normal)
            if wall_normal_norm != 0:
                wall_normal = wall_normal / wall_normal_norm

            # Calculate center of the wall
            centroid = wall_geom_3d.centroid
            wall_center_x = centroid.x
            wall_center_y = centroid.y
            wall_center_z = np.mean([c[2] for c in coords])

            # Compute vector wall center - camera
            v = np.array([camera_x - wall_center_x, camera_y - wall_center_y, camera_z - wall_center_z], dtype=float)
            v_norm = np.linalg.norm(v)
            if v_norm != 0:
                v = v / v_norm

            # Perform dot product test. Remove if wall faces away the camera (<= 0)
            dot_product = np.dot(wall_normal, v)
            angle_rad = np.arccos(np.clip(abs(dot_product), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            if dot_product < 0 and angle_deg < theta:
                filtered_walls.append(wall_id)

        gdf_image_visibility.at[idx, 'visible_walls'] = filtered_walls

    # Keep only rows where visible_walls list is not empty
    gdf_image_visibility = gdf_image_visibility[gdf_image_visibility['visible_walls'].apply(len) > 0]

    return gdf_image_visibility
