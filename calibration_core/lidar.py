from __future__ import annotations

from typing import List

import numpy as np
import open3d as o3d
from tqdm import tqdm

try:
    from .geometry import (
        compute_convex_hull_2d,
        make_lidar_plane_frame,
        make_pca_aligned_lidar_plane_frame,
        make_uv_aligned_lidar_plane_frame,
        pca_align_uv_coordinates,
        plane_uv_to_lidar_points,
        project_lidar_points_to_plane_uv,
        square_align_uv_coordinates,
    )
except ImportError:
    from geometry import (
        compute_convex_hull_2d,
        make_lidar_plane_frame,
        make_pca_aligned_lidar_plane_frame,
        make_uv_aligned_lidar_plane_frame,
        pca_align_uv_coordinates,
        plane_uv_to_lidar_points,
        project_lidar_points_to_plane_uv,
        square_align_uv_coordinates,
    )


HOLE_CENTERS_B_M = np.array(
    [
        [0.45, 0.15],
        [0.15, 0.45],
        [0.75, 0.45],
        [0.45, 0.75],
    ],
    dtype=np.float64,
)

HOLE_RADII_M = np.array([0.12, 0.12, 0.12, 0.12], dtype=np.float64)

RETRO_MARKER_CENTERS_B = np.array(
    [
        [0.45, 0.15, 0.0],
        [0.15, 0.45, 0.0],
        [0.75, 0.45, 0.0],
        [0.45, 0.75, 0.0],
    ],
    dtype=np.float64,
)


def filter_ptcld_by_range(ptcld, crop_range: List):
    x_min, x_max, y_min, y_max, z_min, z_max = crop_range
    mask = (
        (ptcld[:, 0] >= x_min)
        & (ptcld[:, 0] <= x_max)
        & (ptcld[:, 1] >= y_min)
        & (ptcld[:, 1] <= y_max)
        & (ptcld[:, 2] >= z_min)
        & (ptcld[:, 2] <= z_max)
    )
    return ptcld[mask], mask


def filter_list_of_ptclds_by_range(ptcld_list: List[dict], crop_range: List[int]):
    pointclouds = []
    for ptcld in ptcld_list:
        xyz, reflectivity = ptcld["xyz"], ptcld["reflectivity"]
        cropped_xyz, indices_to_keep = filter_ptcld_by_range(xyz, crop_range)
        cropped_reflectivity = reflectivity[indices_to_keep]
        ptcld_info = {"xyz": cropped_xyz, "reflectivity": cropped_reflectivity}
        for key, value in ptcld.items():
            if key not in ptcld_info:
                if (
                    isinstance(value, np.ndarray)
                    and value.ndim > 0
                    and len(value) == len(indices_to_keep)
                ):
                    ptcld_info[key] = value[indices_to_keep]
                else:
                    ptcld_info[key] = value
        pointclouds.append(ptcld_info)
    return pointclouds


def filter_list_of_ptclds_by_reflectivity(
    ptcld_list: List[dict],
    reflectivity_threshold: float = 75,
):
    pointclouds = []
    for ptcld in ptcld_list:
        if "reflectivity" not in ptcld:
            raise KeyError("Point cloud dict is missing required 'reflectivity' key.")

        reflectivity = np.asarray(ptcld["reflectivity"]).reshape(-1)
        mask = reflectivity >= reflectivity_threshold
        filtered_ptcld = {}

        for key, value in ptcld.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(reflectivity):
                filtered_ptcld[key] = value[mask]
            else:
                filtered_ptcld[key] = value

        pointclouds.append(filtered_ptcld)

    return pointclouds


def segment_plane_from_pointcloud(
    pointcloud: np.ndarray,
    distance_threshold: float = 0.025,
    ransac_n: int = 3,
    num_iterations: int = 1500,
    min_inliers: int = 50,
):
    points = np.asarray(pointcloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected pointcloud with shape (N, 3) or wider, got {points.shape}")

    points_xyz = points[:, :3]
    if len(points_xyz) < ransac_n:
        raise ValueError(
            f"Need at least {ransac_n} finite points to segment a plane, got {len(points_xyz)}"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    inlier_indices = np.asarray(inlier_indices, dtype=np.int64)

    if len(inlier_indices) < min_inliers:
        raise RuntimeError(
            f"Plane segmentation found only {len(inlier_indices)} inliers; "
            f"expected at least {min_inliers}"
        )

    outlier_mask = np.ones(len(points_xyz), dtype=bool)
    outlier_mask[inlier_indices] = False

    return {
        "plane_model": np.asarray(plane_model, dtype=np.float64),
        "inlier_points": points_xyz[inlier_indices],
        "outlier_points": points_xyz[outlier_mask],
        "inlier_indices": inlier_indices,
    }


def filter_list_of_ptclds_by_plane(
    ptcld_list: List[dict],
    distance_threshold: float = 0.015,
    ransac_n: int = 3,
    num_iterations: int = 100,
    min_inliers: int = 50,
):
    pointclouds = []
    for ptcld in tqdm(ptcld_list, desc="Segmenting LiDAR planes"):
        if "xyz" not in ptcld:
            raise KeyError("Point cloud dict is missing required 'xyz' key.")

        xyz = np.asarray(ptcld["xyz"])
        plane_result = segment_plane_from_pointcloud(
            xyz,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            min_inliers=min_inliers,
        )

        mask = np.zeros(len(xyz), dtype=bool)
        mask[plane_result["inlier_indices"]] = True
        filtered_ptcld = {}

        for key, value in ptcld.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(xyz):
                filtered_ptcld[key] = value[mask]
            else:
                filtered_ptcld[key] = value

        filtered_ptcld["plane_model"] = plane_result["plane_model"]
        filtered_ptcld["plane_inlier_indices"] = plane_result["inlier_indices"]
        pointclouds.append(filtered_ptcld)

    return pointclouds


def make_lidar_uv_planes_from_plane_filtered_ptclds(
    ptcld_list: List[dict],
    distance_threshold: float = 0.015,
    ransac_n: int = 3,
    num_iterations: int = 1500,
    min_inliers: int = 50,
    pca_align_uv: bool = False,
    square_align_uv: bool = True,
    square_size_m: float = 0.9,
    square_percentile_bounds: tuple = (1.0, 99.0),
    square_angle_step_deg: float = 0.25,
    square_fit_source: str = "points",
    square_use_hull_edge_angles: bool = True,
    square_fit_mode: str = "target_size",
):
    uv_planes = []
    for ptcld in ptcld_list:
        if "xyz" not in ptcld:
            raise KeyError("Point cloud dict is missing required 'xyz' key.")

        points_L = np.asarray(ptcld["xyz"], dtype=np.float64)[:, :3]
        if len(points_L) == 0:
            raise ValueError("Cannot build a UV plane from an empty point cloud.")

        if "plane_model" in ptcld:
            plane_model = np.asarray(ptcld["plane_model"], dtype=np.float64)
        else:
            plane_result = segment_plane_from_pointcloud(
                points_L,
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
                min_inliers=min_inliers,
            )
            plane_model = plane_result["plane_model"]

        frame = make_lidar_plane_frame(points_L, plane_model)
        uv, signed_dist = project_lidar_points_to_plane_uv(points_L, frame)
        pca_result = pca_align_uv_coordinates(uv) if pca_align_uv else None
        square_result = (
            square_align_uv_coordinates(
                uv,
                square_size_m=square_size_m,
                percentile_bounds=square_percentile_bounds,
                angle_step_deg=square_angle_step_deg,
                fit_source=square_fit_source,
                use_hull_edge_angles=square_use_hull_edge_angles,
                fit_mode=square_fit_mode,
            )
            if square_align_uv
            else None
        )

        uv_plane = {
            "xyz": points_L,
            "uv": uv,
            "uv_hull": compute_convex_hull_2d(uv),
            "signed_plane_dist": signed_dist,
            "plane_frame": frame,
            "plane_model": plane_model,
        }
        if pca_result is not None:
            uv_plane["uv_pca"] = pca_result["uv"]
            uv_plane["uv_pca_hull"] = compute_convex_hull_2d(pca_result["uv"])
            uv_plane["plane_frame_pca"] = make_pca_aligned_lidar_plane_frame(frame, pca_result)
            uv_plane["uv_pca_centroid"] = pca_result["centroid_uv"]
            uv_plane["uv_pca_basis"] = pca_result["basis_uv"]
            uv_plane["uv_pca_singular_values"] = pca_result["singular_values"]
            uv_plane["uv_pca_axis_ratio"] = pca_result["axis_ratio"]
        if square_result is not None:
            uv_plane["uv_square"] = square_result["uv"]
            uv_plane["uv_square_hull"] = compute_convex_hull_2d(square_result["uv"])
            uv_plane["plane_frame_square"] = make_uv_aligned_lidar_plane_frame(
                frame,
                square_result["origin_uv"],
                square_result["basis_uv"],
            )
            uv_plane["uv_square_origin"] = square_result["origin_uv"]
            uv_plane["uv_square_basis"] = square_result["basis_uv"]
            uv_plane["uv_square_angle_rad"] = square_result["angle_rad"]
            uv_plane["uv_square_extents"] = square_result["extents"]
            uv_plane["uv_square_bounds_min"] = square_result["bounds_min"]
            uv_plane["uv_square_bounds_max"] = square_result["bounds_max"]
            uv_plane["uv_square_min"] = square_result["square_min"]
            uv_plane["uv_square_max"] = square_result["square_max"]
            uv_plane["uv_square_size_m"] = square_result["square_size_m"]
            uv_plane["uv_square_requested_size_m"] = square_result["requested_square_size_m"]
            uv_plane["uv_square_score"] = square_result["score"]
            uv_plane["uv_square_fit_source"] = square_result["fit_source"]
            uv_plane["uv_square_fit_mode"] = square_result["fit_mode"]
            uv_plane["uv_square_num_fit_points"] = square_result["num_fit_points"]

        for key, value in ptcld.items():
            if key not in uv_plane:
                uv_plane[key] = value

        uv_planes.append(uv_plane)

    return uv_planes


def get_board_feature_centroids_from_square_frame(
    uv_plane: dict,
    centers_B_m: np.ndarray,
    frame_key: str = "plane_frame_square",
):
    if frame_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{frame_key}' key.")

    centers_B_m = np.asarray(centers_B_m, dtype=np.float64)
    if centers_B_m.ndim != 2 or centers_B_m.shape[1] < 2:
        raise ValueError(f"Expected board centers with shape (N, 2) or wider, got {centers_B_m.shape}")

    centers_uv = centers_B_m[:, :2]
    centers_L = plane_uv_to_lidar_points(centers_uv, uv_plane[frame_key])
    return {
        "centers_uv": centers_uv,
        "centers_L": centers_L,
        "frame_key": frame_key,
    }


def add_known_board_centroids_to_uv_plane(
    uv_plane: dict,
    hole_centers_B_m: np.ndarray = HOLE_CENTERS_B_M,
    retro_marker_centers_B: np.ndarray = RETRO_MARKER_CENTERS_B,
):
    holes = get_board_feature_centroids_from_square_frame(
        uv_plane,
        hole_centers_B_m,
        frame_key="plane_frame_square",
    )
    retro_markers = get_board_feature_centroids_from_square_frame(
        uv_plane,
        retro_marker_centers_B,
        frame_key="plane_frame_square",
    )

    uv_plane["hole_centers_uv"] = holes["centers_uv"]
    uv_plane["hole_centers_L"] = holes["centers_L"]
    uv_plane["retro_marker_centers_uv"] = retro_markers["centers_uv"]
    uv_plane["retro_marker_centers_L"] = retro_markers["centers_L"]
    return uv_plane


def make_planar_symmetry_uv_center_sets(centers_uv: np.ndarray):
    centers_uv = np.asarray(centers_uv, dtype=np.float64).reshape(-1, 2)
    board_center = np.mean(centers_uv, axis=0)
    symmetries = [
        ("identity", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)),
        ("rot90", np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)),
        ("rot180", np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)),
        ("rot270", np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)),
        ("mirror_x", np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)),
        ("mirror_y", np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)),
        ("mirror_diag", np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)),
        ("mirror_anti_diag", np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)),
    ]

    center_sets = []
    for name, matrix in symmetries:
        offset = board_center - matrix @ board_center
        center_sets.append({"name": name, "centers_uv": centers_uv @ matrix.T + offset})
    return center_sets


def refine_disk_center_near_uv(
    uv: np.ndarray,
    initial_center_uv: np.ndarray,
    radius_m: float,
    min_inliers: int,
    refinement_iterations: int = 8,
    weights: np.ndarray | None = None,
):
    center = np.asarray(initial_center_uv, dtype=np.float64).reshape(2).copy()
    inlier_mask = np.zeros(len(uv), dtype=bool)

    for _ in range(refinement_iterations):
        distances = np.linalg.norm(uv - center, axis=1)
        inlier_mask = distances <= radius_m
        if int(np.sum(inlier_mask)) < min_inliers:
            return None

        inlier_points = uv[inlier_mask]
        if weights is not None:
            local_weights = np.asarray(weights[inlier_mask], dtype=np.float64)
            local_weights = local_weights - np.min(local_weights) + 1e-6
            refined_center = np.average(inlier_points, axis=0, weights=local_weights)
        else:
            refined_center = np.mean(inlier_points, axis=0)

        if np.linalg.norm(refined_center - center) < 1e-5:
            center = refined_center
            break
        center = refined_center

    distances = np.linalg.norm(uv - center, axis=1)
    inlier_mask = distances <= radius_m
    if int(np.sum(inlier_mask)) < min_inliers:
        return None

    return {
        "center_uv": center,
        "inlier_mask": inlier_mask,
        "num_points": int(np.sum(inlier_mask)),
        "radial_mean_m": float(np.mean(distances[inlier_mask])),
        "radial_std_m": float(np.std(distances[inlier_mask])),
    }


def find_disk_centroids_near_expected_uv(
    uv_plane: dict,
    expected_centers_uv: np.ndarray,
    uv_key: str = "uv_square",
    frame_key: str = "plane_frame_square",
    radius_m: float = 0.12,
    search_radius_m: float | None = None,
    num_disks: int = 4,
    refinement_iterations: int = 8,
    min_inliers: int = 8,
    weight_key: str = "reflectivity",
):
    if uv_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{uv_key}' key.")
    if frame_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{frame_key}' key.")

    uv_all = np.asarray(uv_plane[uv_key], dtype=np.float64).reshape(-1, 2)
    finite_mask = np.isfinite(uv_all).all(axis=1)
    finite_indices = np.flatnonzero(finite_mask)
    uv = uv_all[finite_indices]
    if len(uv) == 0:
        return {
            "disks": [],
            "centers_uv": np.empty((0, 2), dtype=np.float64),
            "centers_L": np.empty((0, 3), dtype=np.float64),
            "method": "expected_prior",
            "symmetry_name": None,
        }

    weights = None
    if weight_key is not None and weight_key in uv_plane:
        candidate_weights = np.asarray(uv_plane[weight_key], dtype=np.float64).reshape(-1)
        if len(candidate_weights) == len(uv_all):
            weights = candidate_weights[finite_indices]

    if search_radius_m is None:
        search_radius_m = max(radius_m * 1.35, radius_m + 0.04)

    best = None
    for center_set in make_planar_symmetry_uv_center_sets(expected_centers_uv):
        disks = []
        for expected_center_uv in center_set["centers_uv"][:num_disks]:
            refined = refine_disk_center_near_uv(
                uv,
                expected_center_uv,
                radius_m=search_radius_m,
                min_inliers=min_inliers,
                refinement_iterations=refinement_iterations,
                weights=weights,
            )
            if refined is None:
                continue

            local_indices = np.flatnonzero(refined["inlier_mask"])
            original_indices = finite_indices[local_indices]
            center_L = plane_uv_to_lidar_points(refined["center_uv"][None, :], uv_plane[frame_key])[0]
            disks.append({
                "center_uv": refined["center_uv"],
                "center_L": center_L,
                "expected_center_uv": np.asarray(expected_center_uv, dtype=np.float64),
                "radius_m": radius_m,
                "search_radius_m": search_radius_m,
                "indices": original_indices,
                "points_uv": uv_all[original_indices],
                "num_points": refined["num_points"],
                "radial_mean_m": refined["radial_mean_m"],
                "radial_std_m": refined["radial_std_m"],
                "method": "expected_prior",
                "symmetry_name": center_set["name"],
            })

        total_points = sum(disk["num_points"] for disk in disks)
        mean_std = float(np.mean([disk["radial_std_m"] for disk in disks])) if disks else np.inf
        score = (len(disks), total_points, -mean_std)
        if best is None or score > best["score"]:
            best = {"score": score, "disks": disks, "symmetry_name": center_set["name"]}

    disks = best["disks"] if best is not None else []
    centers_uv = np.vstack([disk["center_uv"] for disk in disks]) if disks else np.empty((0, 2), dtype=np.float64)
    centers_L = np.vstack([disk["center_L"] for disk in disks]) if disks else np.empty((0, 3), dtype=np.float64)
    return {
        "disks": disks,
        "centers_uv": centers_uv,
        "centers_L": centers_L,
        "method": "expected_prior",
        "symmetry_name": best["symmetry_name"] if best is not None else None,
    }


def find_disk_centroids_in_uv_plane(
    uv_plane: dict,
    uv_key: str = "uv_square",
    frame_key: str = "plane_frame_square",
    radius_m: float = 0.12,
    num_disks: int = 4,
    num_candidates: int = 1500,
    refinement_iterations: int = 8,
    min_inliers: int = 8,
    min_center_separation_m: float = 0.12,
    weight_key: str = "reflectivity",
    random_seed: int = 0,
    expected_centers_uv: np.ndarray | None = HOLE_CENTERS_B_M,
    expected_search_radius_m: float | None = None,
):
    if uv_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{uv_key}' key.")
    if frame_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{frame_key}' key.")

    uv = np.asarray(uv_plane[uv_key], dtype=np.float64).reshape(-1, 2)
    finite_mask = np.isfinite(uv).all(axis=1)
    remaining_indices = np.flatnonzero(finite_mask)
    rng = np.random.default_rng(random_seed)

    weights = None
    if weight_key is not None and weight_key in uv_plane:
        candidate_weights = np.asarray(uv_plane[weight_key], dtype=np.float64).reshape(-1)
        if len(candidate_weights) == len(uv):
            weights = candidate_weights

    disks = []
    while len(disks) < num_disks and len(remaining_indices) >= min_inliers:
        candidate_uv = uv[remaining_indices]
        candidate_weights = weights[remaining_indices] if weights is not None else None

        candidate_count = min(num_candidates, len(candidate_uv))
        sampled_local_indices = rng.choice(len(candidate_uv), size=candidate_count, replace=False)

        best = None
        for local_index in sampled_local_indices:
            center = candidate_uv[local_index].copy()

            for _ in range(refinement_iterations):
                distances = np.linalg.norm(candidate_uv - center, axis=1)
                inlier_mask = distances <= radius_m
                if np.sum(inlier_mask) < min_inliers:
                    break

                inlier_points = candidate_uv[inlier_mask]
                if candidate_weights is not None:
                    local_weights = candidate_weights[inlier_mask]
                    local_weights = local_weights - np.min(local_weights) + 1e-6
                    refined_center = np.average(inlier_points, axis=0, weights=local_weights)
                else:
                    refined_center = np.mean(inlier_points, axis=0)

                if np.linalg.norm(refined_center - center) < 1e-5:
                    center = refined_center
                    break
                center = refined_center

            if any(np.linalg.norm(center - disk["center_uv"]) < min_center_separation_m for disk in disks):
                continue

            distances = np.linalg.norm(candidate_uv - center, axis=1)
            inlier_mask = distances <= radius_m
            inlier_count = int(np.sum(inlier_mask))
            if inlier_count < min_inliers:
                continue

            radial_mean = float(np.mean(distances[inlier_mask]))
            radial_std = float(np.std(distances[inlier_mask]))
            score = (inlier_count, -radial_std)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "center_uv": center,
                    "inlier_mask": inlier_mask,
                    "radial_mean_m": radial_mean,
                    "radial_std_m": radial_std,
                }

        if best is None:
            break

        inlier_indices = remaining_indices[best["inlier_mask"]]
        center_L = plane_uv_to_lidar_points(best["center_uv"][None, :], uv_plane[frame_key])[0]
        disks.append({
            "center_uv": best["center_uv"],
            "center_L": center_L,
            "radius_m": radius_m,
            "indices": inlier_indices,
            "points_uv": uv[inlier_indices],
            "num_points": len(inlier_indices),
            "radial_mean_m": best["radial_mean_m"],
            "radial_std_m": best["radial_std_m"],
        })

        keep_mask = np.ones(len(remaining_indices), dtype=bool)
        keep_mask[best["inlier_mask"]] = False
        remaining_indices = remaining_indices[keep_mask]

    disks.sort(key=lambda disk: disk["num_points"], reverse=True)
    centers_uv = np.vstack([disk["center_uv"] for disk in disks]) if disks else np.empty((0, 2), dtype=np.float64)
    centers_L = np.vstack([disk["center_L"] for disk in disks]) if disks else np.empty((0, 3), dtype=np.float64)

    blind_result = {
        "disks": disks,
        "centers_uv": centers_uv,
        "centers_L": centers_L,
        "radius_m": radius_m,
        "remaining_indices": remaining_indices,
        "method": "blind_disk_fit",
        "symmetry_name": None,
    }

    if expected_centers_uv is not None and len(disks) < num_disks:
        prior_result = find_disk_centroids_near_expected_uv(
            uv_plane,
            expected_centers_uv=expected_centers_uv,
            uv_key=uv_key,
            frame_key=frame_key,
            radius_m=radius_m,
            search_radius_m=expected_search_radius_m,
            num_disks=num_disks,
            refinement_iterations=refinement_iterations,
            min_inliers=min_inliers,
            weight_key=weight_key,
        )
        prior_result["radius_m"] = radius_m
        prior_result["remaining_indices"] = remaining_indices
        if len(prior_result["disks"]) > len(disks):
            return prior_result

    return blind_result


def add_disk_centroids_to_uv_plane(
    uv_plane: dict,
    uv_key: str = "uv_square",
    radius_m: float = 0.12,
    num_disks: int = 4,
    num_candidates: int = 1500,
    min_inliers: int = 8,
    expected_centers_uv: np.ndarray | None = HOLE_CENTERS_B_M,
    expected_search_radius_m: float | None = None,
):
    result = find_disk_centroids_in_uv_plane(
        uv_plane,
        uv_key=uv_key,
        frame_key="plane_frame_square",
        radius_m=radius_m,
        num_disks=num_disks,
        num_candidates=num_candidates,
        min_inliers=min_inliers,
        expected_centers_uv=expected_centers_uv,
        expected_search_radius_m=expected_search_radius_m,
    )
    uv_plane["disk_clusters"] = result["disks"]
    uv_plane["disk_centers_uv"] = result["centers_uv"]
    uv_plane["disk_centers_L"] = result["centers_L"]
    uv_plane["disk_radius_m"] = result["radius_m"]
    uv_plane["disk_center_method"] = result.get("method", "unknown")
    uv_plane["disk_center_symmetry_name"] = result.get("symmetry_name")
    return uv_plane
