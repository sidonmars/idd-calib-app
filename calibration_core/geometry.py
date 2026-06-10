from __future__ import annotations

import numpy as np


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def estimate_rigid_transform_3d(src_points: np.ndarray, dst_points: np.ndarray):
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 3)

    if src.shape != dst.shape:
        raise ValueError(f"Point shape mismatch: {src.shape} vs {dst.shape}")
    if len(src) < 3:
        raise ValueError("Need at least three 3D correspondences.")

    v1_src, v2_src = src[1] - src[0], src[2] - src[0]
    n_src = np.cross(v1_src, v2_src)
    n_src /= np.linalg.norm(n_src) + 1e-12

    v1_dst, v2_dst = dst[1] - dst[0], dst[2] - dst[0]
    n_dst = np.cross(v1_dst, v2_dst)
    n_dst /= np.linalg.norm(n_dst) + 1e-12

    src_fit = np.vstack([src, src[0] + n_src])
    dst_fit = np.vstack([dst, dst[0] + n_dst])

    src_center = np.mean(src_fit, axis=0)
    dst_center = np.mean(dst_fit, axis=0)
    src_centered = src_fit - src_center
    dst_centered = dst_fit - dst_center

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = np.mean(dst, axis=0) - R @ np.mean(src, axis=0)
    residuals = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return R, t, rmse, residuals


def compute_convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    finite_mask = np.isfinite(points_2d).all(axis=1)
    points_2d = points_2d[finite_mask]
    if len(points_2d) == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = np.unique(points_2d, axis=0)
    if len(points) <= 2:
        return points

    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def make_lidar_plane_frame(board_points_L: np.ndarray, plane_model: np.ndarray) -> dict:
    board_points_L = np.asarray(board_points_L, dtype=np.float64)[:, :3]
    a, b, c, d = np.asarray(plane_model, dtype=np.float64)

    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12

    centroid = np.mean(board_points_L, axis=0)
    origin = centroid - normal * (np.dot(normal, centroid) + d)

    centered = board_points_L - origin
    centered_on_plane = centered - np.outer(centered @ normal, normal)
    _, _, vh = np.linalg.svd(centered_on_plane, full_matrices=False)

    x_axis = vh[0]
    x_axis = x_axis - normal * np.dot(normal, x_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    return {
        "origin_L": origin,
        "x_axis_L": x_axis,
        "y_axis_L": y_axis,
        "normal_L": normal,
        "plane_model": np.asarray(plane_model, dtype=np.float64),
    }


def project_lidar_points_to_plane_uv(points_L: np.ndarray, frame: dict):
    points_L = np.asarray(points_L, dtype=np.float64)[:, :3]
    rel = points_L - frame["origin_L"]
    u = rel @ frame["x_axis_L"]
    v = rel @ frame["y_axis_L"]
    signed_dist = rel @ frame["normal_L"]
    return np.column_stack([u, v]), signed_dist


def pca_align_uv_coordinates(uv: np.ndarray) -> dict:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(uv) < 2:
        raise ValueError("Need at least two UV points for PCA alignment.")

    centroid = np.mean(uv, axis=0)
    centered = uv - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    x_axis = vh[0].copy()
    dominant_component = np.argmax(np.abs(x_axis))
    if x_axis[dominant_component] < 0:
        x_axis *= -1.0

    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float64)
    pca_basis = np.vstack([x_axis, y_axis])
    uv_pca = centered @ pca_basis.T

    axis_ratio = np.inf
    if len(singular_values) > 1 and singular_values[1] > 1e-12:
        axis_ratio = float(singular_values[0] / singular_values[1])

    return {
        "uv": uv_pca,
        "centroid_uv": centroid,
        "basis_uv": pca_basis,
        "singular_values": singular_values,
        "axis_ratio": axis_ratio,
    }


def make_pca_aligned_lidar_plane_frame(frame: dict, pca_result: dict) -> dict:
    centroid_uv = np.asarray(pca_result["centroid_uv"], dtype=np.float64)
    basis_uv = np.asarray(pca_result["basis_uv"], dtype=np.float64)

    x_axis_L = np.asarray(frame["x_axis_L"], dtype=np.float64)
    y_axis_L = np.asarray(frame["y_axis_L"], dtype=np.float64)
    origin_L = np.asarray(frame["origin_L"], dtype=np.float64)

    pca_origin_L = origin_L + centroid_uv[0] * x_axis_L + centroid_uv[1] * y_axis_L
    pca_x_axis_L = basis_uv[0, 0] * x_axis_L + basis_uv[0, 1] * y_axis_L
    pca_y_axis_L = basis_uv[1, 0] * x_axis_L + basis_uv[1, 1] * y_axis_L

    pca_x_axis_L /= np.linalg.norm(pca_x_axis_L) + 1e-12
    pca_y_axis_L /= np.linalg.norm(pca_y_axis_L) + 1e-12

    return {
        "origin_L": pca_origin_L,
        "x_axis_L": pca_x_axis_L,
        "y_axis_L": pca_y_axis_L,
        "normal_L": np.asarray(frame["normal_L"], dtype=np.float64),
        "plane_model": np.asarray(frame["plane_model"], dtype=np.float64),
    }


def square_align_uv_coordinates(
    uv: np.ndarray,
    square_size_m: float = 1.0,
    percentile_bounds: tuple = (1.0, 99.0),
    angle_step_deg: float = 0.25,
    fit_source: str = "points",
    use_hull_edge_angles: bool = True,
    fit_mode: str = "target_size",
) -> dict:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(uv) < 4:
        raise ValueError("Need at least four UV points for square alignment.")

    hull = compute_convex_hull_2d(uv)
    if fit_source == "points":
        fit_points = uv
    elif fit_source == "hull":
        fit_points = hull
    else:
        raise ValueError("fit_source must be 'points' or 'hull'.")

    if len(fit_points) < 4:
        raise ValueError(
            f"Need at least four {fit_source} points for square alignment, "
            f"got {len(fit_points)}."
        )

    center_uv = np.mean(fit_points, axis=0)
    centered = uv - center_uv
    centered_fit_points = fit_points - center_uv
    lower_percentile, upper_percentile = percentile_bounds
    angles = list(np.deg2rad(np.arange(0.0, 90.0, angle_step_deg, dtype=np.float64)))

    if use_hull_edge_angles and len(hull) >= 3:
        edges = np.roll(hull, -1, axis=0) - hull
        edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
        edge_angles = np.mod(edge_angles, np.pi / 2.0)
        angles.extend(edge_angles.tolist())

    angles = np.unique(np.round(np.asarray(angles, dtype=np.float64), decimals=12))

    best = None
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        basis_uv = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float64)
        rotated = centered_fit_points @ basis_uv.T

        if fit_mode == "min_enclosing":
            low = np.min(rotated, axis=0)
            high = np.max(rotated, axis=0)
        elif fit_mode == "target_size":
            low = np.percentile(rotated, lower_percentile, axis=0)
            high = np.percentile(rotated, upper_percentile, axis=0)
        else:
            raise ValueError("fit_mode must be 'target_size' or 'min_enclosing'.")

        extents = high - low

        if fit_mode == "min_enclosing":
            side_length = float(np.max(extents))
            square_min = low - 0.5 * (side_length - extents)
            square_max = square_min + side_length
            score = side_length
        else:
            side_length = float(square_size_m)
            square_center = 0.5 * (low + high)
            square_min = square_center - 0.5 * side_length
            square_max = square_center + 0.5 * side_length
            size_error = np.sum((extents - square_size_m) ** 2)
            aspect_error = (extents[0] - extents[1]) ** 2
            score = float(size_error + aspect_error)

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "angle_rad": float(angle),
                "basis_uv": basis_uv,
                "bounds_min": low,
                "bounds_max": high,
                "square_min": square_min,
                "square_max": square_max,
                "side_length": side_length,
                "extents": extents,
            }

    uv_square = centered @ best["basis_uv"].T - best["square_min"]
    origin_uv = center_uv + best["square_min"] @ best["basis_uv"]

    return {
        "uv": uv_square,
        "origin_uv": origin_uv,
        "basis_uv": best["basis_uv"],
        "angle_rad": best["angle_rad"],
        "bounds_min": best["bounds_min"],
        "bounds_max": best["bounds_max"],
        "square_min": best["square_min"],
        "square_max": best["square_max"],
        "extents": best["extents"],
        "square_size_m": best["side_length"],
        "requested_square_size_m": square_size_m,
        "score": best["score"],
        "fit_source": fit_source,
        "fit_mode": fit_mode,
        "num_fit_points": len(fit_points),
    }


def make_uv_aligned_lidar_plane_frame(
    frame: dict,
    origin_uv: np.ndarray,
    basis_uv: np.ndarray,
) -> dict:
    origin_uv = np.asarray(origin_uv, dtype=np.float64)
    basis_uv = np.asarray(basis_uv, dtype=np.float64)

    x_axis_L = np.asarray(frame["x_axis_L"], dtype=np.float64)
    y_axis_L = np.asarray(frame["y_axis_L"], dtype=np.float64)
    origin_L = np.asarray(frame["origin_L"], dtype=np.float64)

    aligned_origin_L = origin_L + origin_uv[0] * x_axis_L + origin_uv[1] * y_axis_L
    aligned_x_axis_L = basis_uv[0, 0] * x_axis_L + basis_uv[0, 1] * y_axis_L
    aligned_y_axis_L = basis_uv[1, 0] * x_axis_L + basis_uv[1, 1] * y_axis_L

    aligned_x_axis_L /= np.linalg.norm(aligned_x_axis_L) + 1e-12
    aligned_y_axis_L /= np.linalg.norm(aligned_y_axis_L) + 1e-12

    return {
        "origin_L": aligned_origin_L,
        "x_axis_L": aligned_x_axis_L,
        "y_axis_L": aligned_y_axis_L,
        "normal_L": np.asarray(frame["normal_L"], dtype=np.float64),
        "plane_model": np.asarray(frame["plane_model"], dtype=np.float64),
    }


def plane_uv_to_lidar_points(uv: np.ndarray, frame: dict) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return (
        frame["origin_L"][None, :]
        + uv[:, 0:1] * frame["x_axis_L"][None, :]
        + uv[:, 1:2] * frame["y_axis_L"][None, :]
    )
