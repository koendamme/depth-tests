import glob
import os
import open3d
import cv2
import numpy as np
import matplotlib as mpl
from depth_capture.intrinsics import Intrinsics
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import pickle
import pandas as pd
import copy


def select_plane_points(path, scale):
    # Read the image
    depth = np.asanyarray(open3d.io.read_image(path)) * scale
    depth = depth.astype(np.float32)
    # depth = open3d.geometry.Image(depth.astype(np.float32))
    scaled = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    cv2.imshow("Image", scaled)

    positions = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            positions.append((x, y))
            cv2.circle(scaled, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", scaled)

    # Set mouse callback
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for three clicks
    while len(positions) < 3:
        cv2.waitKey(1)

    # Close the window
    cv2.destroyAllWindows()
    return positions


def apply_bounding_box(pcd):
    # Test 1
    # p1 = p1 + np.array([1, 1, 10])
    # p2 = p2 + np.array([-1, -1, -10])

    # 320x240
    p1 = np.array([.3, 0, .4])
    p2 = np.array([1.3, .35, 2])

    points_to_filter = np.asarray(pcd.points)

    x_bound = (points_to_filter[:, 0] > p1[0]) & (points_to_filter[:, 0] < p2[0])
    z_bound = (points_to_filter[:, 2] > p1[2]) & (points_to_filter[:, 2] < p2[2])
    y_bound = (points_to_filter[:, 1] > p1[1]) & (points_to_filter[:, 1] < p2[1])

    return pcd.select_by_index(np.where(x_bound & z_bound & y_bound)[0])


def get_rotation_and_translation(depth_image, points, intrinsics):
    point_mask = np.zeros(depth_image.shape)

    for p in points:
        point_mask[p[1], p[0]] = depth_image[p[1], p[0]]

    markers_o3d = open3d.geometry.Image(point_mask.astype(np.float32))

    m_pcd = open3d.geometry.PointCloud.create_from_depth_image(markers_o3d, intrinsic=intrinsics)
    m_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    m_pcd.paint_uniform_color([1.0, 0, 0])

    points3d = np.asarray(m_pcd.points).copy()

    u_1 = points3d[1] - points3d[2]
    u_2 = points3d[0] - points3d[2]

    n = np.cross(u_1, u_2)
    R = Rotation.align_vectors([0, 1, 0], -n)[0].as_matrix()
    return R, -points3d[2]


def get_reference(path, rotation, translation, scale, intrinsics):
    depth_reference = np.asanyarray(open3d.io.read_image(path)) * scale
    depth_reference = open3d.geometry.Image(depth_reference.astype(np.float32))
    pcd_reference = open3d.geometry.PointCloud.create_from_depth_image(depth=depth_reference,
                                                                           intrinsic=intrinsics)

    pcd_reference.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_reference.translate(translation)
    pcd_reference.rotate(rotation, center=(0, 0, 0))

    return pcd_reference


def main():
    scale = 0.00025
    # paths = glob.glob(os.path.join("D:", os.sep,  "techmed_synchronisatie_1-5\Test1\depth\*.png"))
    paths = glob.glob(os.path.join("D:", os.sep, "MRI-28-5", "depth", "session1_no_temperal", "depth", "*.png"))
    paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "").split("_")[-1]))
    # points = select_plane_points(paths[0], scale)
    points = [(162, 15), (197, 97), (378, 9)]
    # 320x240
    # points = [(78, 8), (105, 69), (202, 7)]

    # 640x480
    # points = [(164, 11), (207, 125), (404, 10)]
    # print(points)

    npdepth = np.asanyarray(open3d.io.read_image(paths[0])) * scale

    intr = Intrinsics(width=640, height=480)
    R, translation = get_rotation_and_translation(npdepth, points, intr.get_intrinsics())

    pcd_reference = get_reference(paths[0], R, translation, scale, intr.get_intrinsics())
    pcd_reference = apply_bounding_box(pcd_reference)

    axes = open3d.geometry.TriangleMesh.create_coordinate_frame()
    open3d.visualization.draw_geometries([pcd_reference, axes])

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    geometry = open3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    vis.add_geometry(axes)

    diff = []
    timestamps = []
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        timestamp = float(path.split("_")[-1].replace(".png", ""))
        timestamps.append(timestamp)

        depth = np.asanyarray(open3d.io.read_image(path)) * scale
        depth = open3d.geometry.Image(depth.astype(np.float32))

        pcd = open3d.geometry.PointCloud.create_from_depth_image(depth=depth, intrinsic=intr.get_intrinsics())

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.translate(translation)
        pcd.rotate(R, center=(0, 0, 0))

        pcd = apply_bounding_box(pcd)
        dists = pcd.compute_point_cloud_distance(pcd_reference)
        m_dists = np.mean(np.asarray(dists))
        diff.append(m_dists)

        geometry.points = pcd.points

        if i == 0:
            vis.add_geometry(geometry)

        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    # with open("depth_wave_arend_experiment.pickle", 'wb') as f:
    #     pickle.dump({"distandes": diff, "timestamps": timestamps}, f)

    plt.plot(diff)
    plt.show()


if __name__ == '__main__':
    main()