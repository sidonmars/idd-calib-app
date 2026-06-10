from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation

try:
    from .geometry import estimate_rigid_transform_3d, invert_transform, make_transform
    from .lidar import HOLE_CENTERS_B_M
except ImportError:
    from geometry import estimate_rigid_transform_3d, invert_transform, make_transform
    from lidar import HOLE_CENTERS_B_M


def match_board_centers_to_detected_uv(
    board_centers_B_m: np.ndarray,
    detected_centers_uv: np.ndarray,
    max_distance_m: float = 0.15,
):
    board_uv = np.asarray(board_centers_B_m, dtype=np.float64)[:, :2]
    detected_uv = np.asarray(detected_centers_uv, dtype=np.float64).reshape(-1, 2)
    if len(detected_uv) < len(board_uv):
        raise RuntimeError(f"Need at least {len(board_uv)} detected centers, got {len(detected_uv)}.")

    distances = np.linalg.norm(board_uv[:, None, :] - detected_uv[None, :, :], axis=2)
    candidates = []
    for board_i in range(distances.shape[0]):
        for detected_i in range(distances.shape[1]):
            if distances[board_i, detected_i] <= max_distance_m:
                candidates.append((distances[board_i, detected_i], board_i, detected_i))

    candidates.sort(key=lambda item: item[0])
    used_board = set()
    used_detected = set()
    matches = []
    for distance, board_i, detected_i in candidates:
        if board_i in used_board or detected_i in used_detected:
            continue
        used_board.add(board_i)
        used_detected.add(detected_i)
        matches.append((board_i, detected_i, float(distance)))

    if len(matches) != len(board_uv):
        raise RuntimeError(
            f"Matched only {len(matches)} of {len(board_uv)} board centers. "
            "Increase max_distance_m or check the square-frame orientation."
        )

    matches.sort(key=lambda item: item[0])
    return matches


def make_board_planar_symmetry_transforms(board_centers_B_m: np.ndarray):
    centers = np.asarray(board_centers_B_m, dtype=np.float64)[:, :2]
    center = np.mean(centers, axis=0)
    symmetries = [
        ("identity", np.array([[1.0, 0.0], [0.0, 1.0]])),
        ("rot90", np.array([[0.0, -1.0], [1.0, 0.0]])),
        ("rot180", np.array([[-1.0, 0.0], [0.0, -1.0]])),
        ("rot270", np.array([[0.0, 1.0], [-1.0, 0.0]])),
        ("mirror_x", np.array([[-1.0, 0.0], [0.0, 1.0]])),
        ("mirror_y", np.array([[1.0, 0.0], [0.0, -1.0]])),
        ("mirror_diag", np.array([[0.0, 1.0], [1.0, 0.0]])),
        ("mirror_anti_diag", np.array([[0.0, -1.0], [-1.0, 0.0]])),
    ]

    transforms = []
    for name, A in symmetries:
        b = center - A @ center
        T_SB = np.eye(4, dtype=np.float64)
        T_SB[:2, :2] = A
        T_SB[:2, 3] = b
        T_SB[2, 2] = np.linalg.det(A)
        transforms.append({
            "name": name,
            "T_SB": T_SB,
            "A": A,
            "b": b,
            "center": center,
            "normal_sign": float(T_SB[2, 2]),
        })
    return transforms


def estimate_lidar_board_pose_from_disk_centers(
    uv_plane: dict,
    board_centers_B_m: np.ndarray = HOLE_CENTERS_B_M,
    detected_uv_key: str = "disk_centers_uv",
    detected_lidar_key: str = "disk_centers_L",
    max_match_distance_m: float = 0.15,
    try_planar_symmetries: bool = False,
):
    if detected_uv_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{detected_uv_key}' key.")
    if detected_lidar_key not in uv_plane:
        raise KeyError(f"UV plane dict is missing required '{detected_lidar_key}' key.")

    board_centers_B_m = np.asarray(board_centers_B_m, dtype=np.float64)
    board_points_B = np.zeros((len(board_centers_B_m), 3), dtype=np.float64)
    board_points_B[:, :2] = board_centers_B_m[:, :2]

    detected_uv = np.asarray(uv_plane[detected_uv_key], dtype=np.float64).reshape(-1, 2)
    detected_L = np.asarray(uv_plane[detected_lidar_key], dtype=np.float64).reshape(-1, 3)

    symmetries = (
        make_board_planar_symmetry_transforms(board_points_B)
        if try_planar_symmetries
        else [{"name": "identity", "T_SB": np.eye(4, dtype=np.float64)}]
    )

    candidates = []
    for symmetry in symmetries:
        board_points_h = np.column_stack([board_points_B, np.ones(len(board_points_B))])
        board_points_S = (symmetry["T_SB"] @ board_points_h.T).T[:, :3]

        try:
            matches = match_board_centers_to_detected_uv(
                board_points_S,
                detected_uv,
                max_distance_m=max_match_distance_m,
            )
        except RuntimeError:
            continue

        board_indices = [match[0] for match in matches]
        detected_indices = [match[1] for match in matches]
        matched_board_S = board_points_S[board_indices]
        matched_detected_L = detected_L[detected_indices]

        R_LS, t_LS, rmse_m, residuals_m = estimate_rigid_transform_3d(
            matched_board_S,
            matched_detected_L,
        )
        T_LS = make_transform(R_LS, t_LS)
        T_LB = T_LS @ symmetry["T_SB"]

        candidates.append({
            "symmetry_name": symmetry["name"],
            "normal_sign": symmetry.get("normal_sign", 1.0),
            "T_LB": T_LB,
            "T_BL": invert_transform(T_LB),
            "T_SB": symmetry["T_SB"],
            "matches": matches,
            "matched_board_points_B": board_points_B[board_indices],
            "matched_board_points_S": matched_board_S,
            "matched_lidar_points_L": matched_detected_L,
            "rmse_m": rmse_m,
            "residuals_m": residuals_m,
        })

    if len(candidates) == 0:
        raise RuntimeError("Could not match board centers to detected disk centers for any symmetry.")

    best = min(candidates, key=lambda candidate: candidate["rmse_m"])
    T_LB = best["T_LB"]

    return {
        "T_LB": T_LB,
        "T_BL": invert_transform(T_LB),
        "R_LB": T_LB[:3, :3],
        "t_LB": T_LB[:3, 3],
        "matches": best["matches"],
        "matched_board_points_B": best["matched_board_points_B"],
        "matched_lidar_points_L": best["matched_lidar_points_L"],
        "rmse_m": best["rmse_m"],
        "residuals_m": best["residuals_m"],
        "symmetry_name": best["symmetry_name"],
        "normal_sign": best["normal_sign"],
        "candidate_poses": candidates,
    }


def estimate_camera_lidar_extrinsic_from_board_poses(T_CB: np.ndarray, T_LB: np.ndarray):
    T_CB = np.asarray(T_CB, dtype=np.float64).reshape(4, 4)
    T_LB = np.asarray(T_LB, dtype=np.float64).reshape(4, 4)
    T_CL = T_CB @ invert_transform(T_LB)
    return {
        "T_CL": T_CL,
        "T_LC": invert_transform(T_CL),
    }


def rotation_angle_between(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    R_delta = R_a @ R_b.T
    cos_angle = 0.5 * (np.trace(R_delta) - 1.0)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extrinsic_distance(
    T_a: np.ndarray,
    T_b: np.ndarray,
    translation_weight: float = 2.0,
):
    T_a = np.asarray(T_a, dtype=np.float64).reshape(4, 4)
    T_b = np.asarray(T_b, dtype=np.float64).reshape(4, 4)
    rotation_error_rad = rotation_angle_between(T_a[:3, :3], T_b[:3, :3])
    translation_error_m = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    score = rotation_error_rad + translation_weight * translation_error_m
    return score, rotation_error_rad, translation_error_m


def make_candidate_extrinsic_records(
    extrinsic_result: dict,
    normal_sign_filter: float = None,
):
    records = []
    for candidate in extrinsic_result["lidar_board_pose"]["candidate_poses"]:
        normal_sign = float(np.sign(candidate.get("normal_sign", 1.0)))
        if normal_sign_filter is not None and normal_sign != float(np.sign(normal_sign_filter)):
            continue

        candidate_extrinsic = estimate_camera_lidar_extrinsic_from_board_poses(
            extrinsic_result["T_CB"],
            candidate["T_LB"],
        )
        records.append({
            "frame_index": extrinsic_result["index"],
            "image_path": extrinsic_result["image_path"],
            "lidar_path": extrinsic_result["lidar_path"],
            "symmetry_name": candidate["symmetry_name"],
            "normal_sign": normal_sign,
            "T_CL": candidate_extrinsic["T_CL"],
            "T_LC": candidate_extrinsic["T_LC"],
            "T_LB": candidate["T_LB"],
            "lidar_board_rmse_m": candidate["rmse_m"],
            "source_result": extrinsic_result,
            "candidate": candidate,
        })
    return records


def find_named_candidate_extrinsic_record(
    extrinsic_result: dict,
    symmetry_name: str,
    normal_sign: float = None,
):
    candidates = make_candidate_extrinsic_records(
        extrinsic_result,
        normal_sign_filter=normal_sign,
    )
    for candidate in candidates:
        if candidate["symmetry_name"] == symmetry_name:
            return candidate
    normal_label = "any" if normal_sign is None else f"{np.sign(normal_sign):.0f}"
    raise RuntimeError(
        f"Frame {extrinsic_result['index']} has no candidate "
        f"{symmetry_name} with normal sign {normal_label}."
    )


def select_consistent_candidate_extrinsics(
    extrinsic_results: list,
    preferred_symmetry_name: str = "mirror_diag",
    preferred_normal_sign: float = -1.0,
    reference_frame_index: int = None,
    translation_weight: float = 2.0,
):
    if len(extrinsic_results) == 0:
        raise RuntimeError("Need at least one extrinsic result.")

    if reference_frame_index is None:
        reference_result = min(extrinsic_results, key=lambda result: result["lidar_board_rmse_m"])
    else:
        matches = [result for result in extrinsic_results if result["index"] == reference_frame_index]
        if len(matches) == 0:
            raise RuntimeError(f"Reference frame {reference_frame_index} is not in extrinsic_results.")
        reference_result = matches[0]

    reference_record = find_named_candidate_extrinsic_record(
        reference_result,
        preferred_symmetry_name,
        normal_sign=preferred_normal_sign,
    )
    reference_T_CL = reference_record["T_CL"]

    selected_records = []
    for result in extrinsic_results:
        candidates = make_candidate_extrinsic_records(
            result,
            normal_sign_filter=preferred_normal_sign,
        )
        if len(candidates) == 0:
            continue

        for candidate in candidates:
            score, rotation_error_rad, translation_error_m = extrinsic_distance(
                candidate["T_CL"],
                reference_T_CL,
                translation_weight=translation_weight,
            )
            candidate["consistency_score"] = score
            candidate["rotation_error_deg"] = float(np.rad2deg(rotation_error_rad))
            candidate["translation_error_m"] = translation_error_m

        selected_records.append(min(candidates, key=lambda candidate: candidate["consistency_score"]))

    return reference_record, selected_records


def optimize_camera_lidar_extrinsic_robust(
    points_L: np.ndarray,
    points_C: np.ndarray,
    initial_T_CL: np.ndarray,
) -> tuple[np.ndarray, float]:
    points_L = np.asarray(points_L, dtype=np.float64).reshape(-1, 3)
    points_C = np.asarray(points_C, dtype=np.float64).reshape(-1, 3)

    if len(points_L) != len(points_C):
        raise ValueError("Point counts must match")

    initial_R = Rotation.from_matrix(initial_T_CL[:3, :3]).as_rotvec()
    initial_t = initial_T_CL[:3, 3]
    x0 = np.concatenate([initial_R, initial_t])

    def residuals(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        t = x[3:]
        transformed = points_L @ R.T + t
        return (transformed - points_C).ravel()

    res = scipy.optimize.least_squares(
        residuals,
        x0,
        loss='huber',
        f_scale=0.05,
        max_nfev=1500
    )

    opt_R = Rotation.from_rotvec(res.x[:3]).as_matrix()
    opt_t = res.x[3:]

    T_CL_opt = np.eye(4, dtype=np.float64)
    T_CL_opt[:3, :3] = opt_R
    T_CL_opt[:3, 3] = opt_t

    final_residuals = (points_L @ opt_R.T + opt_t) - points_C
    final_rmse = float(np.sqrt(np.mean(np.sum(final_residuals**2, axis=1))))

    return T_CL_opt, final_rmse
