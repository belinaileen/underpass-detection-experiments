import numpy as np
import cv2


def project_walls_on_image(image_id, wall_ids, df_camera_parameters, gdf_critical_walls): 

    """
    Project the 3D coordinates of critical walls onto the corresponding image using the camera parameters.

    Args:
        - image_id: The ID of the image for which the walls are being projected.
        - wall_ids: A list of wall IDs that need to be projected onto the image.
        - df_camera_parameters: A DataFrame containing the camera parameters for each image.
        - gdf_critical_walls: A GeoDataFrame containing the geometries of the critical walls with their corresponding wall IDs.

    Returns:
        - rectangles_2d: A list of 2D rectangles (as numpy arrays) corresponding to the projected walls on the image.

    """

    image_width = df_camera_parameters[df_camera_parameters['image_id'] == image_id]['width'].iloc[0]
    image_height = df_camera_parameters[df_camera_parameters['image_id'] == image_id]['height'].iloc[0]

    # Compute camera matrix
    fx = float(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['fx'].iloc[0])
    fy = float(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['fy'].iloc[0])
    cx = float(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['cx'].iloc[0])
    cy = float(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['cy'].iloc[0])
        
    # Compute r matrix
    omega = np.deg2rad(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['omega'].iloc[0])
    phi = np.deg2rad(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['phi'].iloc[0])
    kappa = np.deg2rad(df_camera_parameters[df_camera_parameters['image_id'] == image_id]['kappa'].iloc[0])
    r_omega = np.array([[1, 0, 0],
                        [0, np.cos(omega), -np.sin(omega)],
                        [0, np.sin(omega),  np.cos(omega)]])
    r_phi   = np.array([[ np.cos(phi), 0, np.sin(phi)],
                        [0, 1, 0],
                        [-np.sin(phi), 0, np.cos(phi)]])
    r_kappa = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                        [np.sin(kappa),  np.cos(kappa), 0],
                        [0, 0, 1]])

    R_cam_to_world = r_omega @ r_phi @ r_kappa

    # Adjusting reflection across plane YZ 
    R = np.diag([1, -1, -1]) @ R_cam_to_world.T

    # Compute translation vector
    x = df_camera_parameters[df_camera_parameters['image_id'] == image_id]['X'].iloc[0]
    y = df_camera_parameters[df_camera_parameters['image_id'] == image_id]['Y'].iloc[0]
    z = df_camera_parameters[df_camera_parameters['image_id'] == image_id]['Z'].iloc[0]
    t = -R @ np.array([[x], [y], [z]]).astype(np.float32).reshape(3, 1)
    k = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]]).astype(np.float64)
    # Compute projection matrix
    m = k @ np.hstack((R, t))

    rectangles_2d = []
    for wall_id in wall_ids:
        # Extract wall 3d coordinates
        wall_geom = gdf_critical_walls[gdf_critical_walls['wall_id'] == wall_id]['geometry'].iloc[0]
        rect_3d = list(wall_geom.exterior.coords[:-1])

        rect_2d = []
        for vertex in rect_3d:
        # Express rectangle vertex in homogeneous coordinates
            vertex_3d_homogeneous = np.hstack((vertex, np.array([1])))
            # Compute projected point 
            vertex_2d_homogeneous = m @ vertex_3d_homogeneous.reshape(4, 1)
            # Convert projected point back to Cartesian coordinates
            vertex_2d = vertex_2d_homogeneous[:2] / vertex_2d_homogeneous[2]
            # Desestimate rectangles which are projected too far from image boundaries
            if vertex_2d[0][0] < 0 or vertex_2d[0][0] > (image_width + 100) or vertex_2d[1][0] < 0 or vertex_2d[1][0] > (image_height + 100):
                break
            # Append projected vertex to list of 2D vertices for the rectangle
            rect_2d.append([int(vertex_2d[0][0]), int(vertex_2d[1][0])])

        if len(rect_2d) != 4:
            # Append None for invalid projections
            rectangles_2d.append(None)
            continue

        # Keep one flat array per projected wall: shape (num_vertices, 2)
        rect_2d = np.array(rect_2d, dtype=np.int32)
        rectangles_2d.append(rect_2d)

    return rectangles_2d


def display_image(rectangles_2d, image, limit_width = 1920, limit_height = 1080, linewidth = 50):

    """
    Display the image with projected walls outlined.
    Args:
        - rectangles_2d: A list of 2D rectangles (as numpy arrays) corresponding to the projected walls on the image.
        - image: The original image (CV2 image) on which the walls were projected.
        - limit_width: The maximum width for displaying the image (default is 1920 pixels).
        - limit_height: The maximum height for displaying the image (default is 1080 pixels

    Returns:
        None
    
    """
    
    image_h, image_w = image.shape[:2]
    
    for rect_2d in rectangles_2d:
        contour = np.asarray(rect_2d, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [contour], True, (0, 0, 255), linewidth)

    if image_h > limit_height or image_w > limit_width:
        display_image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)

    else:
        display_image = image

    cv2.imshow("Projected facades", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
