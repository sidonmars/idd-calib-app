from __future__ import annotations

import cv2
import numpy as np


def project_lidar_points_to_image(
    points_L: np.ndarray,
    T_CL: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray = None,
    image_shape: tuple = None,
    min_depth_m: float = 0.05,
):
    points_L = np.asarray(points_L, dtype=np.float64)
    if points_L.ndim != 2 or points_L.shape[1] < 3:
        raise ValueError(f"Expected LiDAR points with shape (N, 3) or wider, got {points_L.shape}")

    points_L_xyz = points_L[:, :3]
    T_CL = np.asarray(T_CL, dtype=np.float64).reshape(4, 4)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    points_L_h = np.column_stack([points_L_xyz, np.ones(len(points_L_xyz))])
    points_C = (T_CL @ points_L_h.T).T[:, :3]
    finite_mask = np.isfinite(points_C).all(axis=1)
    depth_mask = points_C[:, 2] > min_depth_m
    valid_mask = finite_mask & depth_mask

    image_points = np.full((len(points_L_xyz), 2), np.nan, dtype=np.float64)
    if np.any(valid_mask):
        projected, _ = cv2.projectPoints(
            objectPoints=points_C[valid_mask],
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.zeros((3, 1), dtype=np.float64),
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
        image_points[valid_mask] = projected.reshape(-1, 2)

    if image_shape is not None:
        height, width = image_shape[:2]
        in_image_mask = (
            valid_mask
            & (image_points[:, 0] >= 0)
            & (image_points[:, 0] < width)
            & (image_points[:, 1] >= 0)
            & (image_points[:, 1] < height)
        )
    else:
        in_image_mask = valid_mask

    return {
        "image_points": image_points,
        "points_C": points_C,
        "depths_m": points_C[:, 2],
        "valid_mask": valid_mask,
        "in_image_mask": in_image_mask,
    }


def visualize_lidar_projection_on_image(
    image: np.ndarray,
    points_L: np.ndarray,
    T_CL: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray = None,
    reflectivity: np.ndarray = None,
    color_by: str = "depth",
    point_radius: int = 2,
    alpha: float = 0.85,
    max_points: int = 30000,
    min_depth_m: float = 0.05,
    output_path: str = None,
    show: bool = False,
):
    if image is None:
        raise ValueError("image is None.")

    output = image.copy()
    projection = project_lidar_points_to_image(
        points_L,
        T_CL,
        camera_matrix,
        dist_coeffs=dist_coeffs,
        image_shape=output.shape,
        min_depth_m=min_depth_m,
    )

    mask = projection["in_image_mask"]
    indices = np.flatnonzero(mask)
    if max_points is not None and len(indices) > max_points:
        stride = int(np.ceil(len(indices) / max_points))
        indices = indices[::stride]

    if len(indices) == 0:
        if output_path is not None:
            cv2.imwrite(output_path, output)
        return {
            "image": output,
            "projection": projection,
            "drawn_indices": indices,
        }

    if color_by == "reflectivity" and reflectivity is not None:
        values = np.asarray(reflectivity, dtype=np.float64).reshape(-1)[indices]
    elif color_by == "height":
        values = np.asarray(points_L, dtype=np.float64)[indices, 2]
    else:
        values = projection["depths_m"][indices]

    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        norm_values = np.zeros(len(values), dtype=np.float64)
    else:
        vmin = float(np.percentile(finite_values, 2))
        vmax = float(np.percentile(finite_values, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm_values = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    values_u8 = np.round(norm_values * 255.0).astype(np.uint8).reshape(-1, 1)
    colors_bgr = cv2.applyColorMap(values_u8, cv2.COLORMAP_TURBO).reshape(-1, 3)

    overlay = output.copy()
    image_points = projection["image_points"][indices]
    for point, color in zip(image_points, colors_bgr):
        u, v = np.round(point).astype(int)
        cv2.circle(
            overlay,
            (u, v),
            point_radius,
            color.tolist(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)

    if output_path is not None:
        cv2.imwrite(output_path, output)

    if show:
        import matplotlib.pyplot as plt

        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(output_rgb)
        plt.axis("off")
        plt.show()

    return {
        "image": output,
        "projection": projection,
        "drawn_indices": indices,
    }
