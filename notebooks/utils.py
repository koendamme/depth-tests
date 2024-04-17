import numpy as np
from datetime import datetime


def pcd_to_depth_map(pcd):
    all_points = np.array(pcd.points)
    min_p, max_p = np.min(all_points, axis=0), np.max(all_points, axis=0)
    all_points -= np.array([min_p[0], 0, min_p[2]])

    width, height = 640, 480
    proj = np.zeros((height, width))

    new_x_max, new_z_max = np.max(all_points[:, 0], axis=0), np.max(all_points[:, 2], axis=0)

    x_scale = (width - 1) / new_x_max
    z_scale = (height - 1) / new_z_max

    for p in all_points:
        x, z = int(p[0] * x_scale), int(p[2] * z_scale)
        proj[z, x] = p[1]

    return proj[100:300, 250:500]


def pcd_to_volume(pcd):
    all_points = np.array(pcd.points)
    return np.sum(all_points[:, 1])/all_points.shape[0]





