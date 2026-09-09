import numpy as np


OPENCV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def intrinsic(width, height, fovy_deg):
    fovy = np.deg2rad(fovy_deg)
    fy = height / (2 * np.tan(fovy / 2))
    fx = fy 
    cx = (width -1 ) / 2
    cy = (height - 1) / 2
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])

    return K


def pixel_to_3d(u, v, depth, K):
    
    u = np.ravel(np.asarray(u, dtype=np.float64))
    v = np.ravel(np.asarray(v, dtype=np.float64))
    depth = np.ravel(np.asarray(depth, dtype=np.float64))

    pixels = np.stack([u, v, np.ones_like(u)])
    rays = OPENCV_TO_MUJOCO @ np.linalg.inv(K) @ pixels
    return (rays * depth).T


def camera_to_world(points_3d, camera_rotation, camera_position):
    points_3d = np.atleast_2d(points_3d)

    R = np.asarray(camera_rotation).reshape(3, 3)
    t = np.asarray(camera_position)

    world_points = points_3d @ R.T + t

    return world_points


def ground_points(depth_image, K, camera_rotation, camera_position,
                  stride=4, max_range=8.0, ground_z=0.10):
    height, width = depth_image.shape
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    rows = rows.ravel()
    cols = cols.ravel()

    depths = depth_image[rows, cols]
    usable = (depths > 0.1) & (depths < max_range)

    if not np.any(usable):
        return np.empty((0, 3))

    points_camera = pixel_to_3d(cols[usable], rows[usable], depths[usable], K)

    R = np.asarray(camera_rotation).reshape(3, 3)
    world = points_camera @ R.T + np.asarray(camera_position)

    return world[np.abs(world[:, 2]) < ground_z]
