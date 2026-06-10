from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CHARUCO_BOARD_UNIT_TO_M = 0.001


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


DEFAULT_CHARUCO_ID_MAP = {
    0: 0,
    2: 4,
    3: 5,
    6: 9,
    1: 3,
    4: 7,
    5: 8,
    7: 12,
    16: 15,
    17: 19,
    18: 20,
    19: 24,
    8: 27,
    10: 31,
    11: 32,
    14: 36,
    9: 30,
    12: 34,
    13: 35,
    15: 39,
}


def get_cv2_charuco_board_and_mapping(
    dictionary_name: str = "DICT_5X5_1000",
    board_size: tuple[int, int] = (9, 9),
    id_map: dict[int, int] | None = None,
):
    if dictionary_name not in ARUCO_DICTIONARIES:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")

    board = cv2.aruco.CharucoBoard(
        size=tuple(int(value) for value in board_size),
        squareLength=100,
        markerLength=70,
        dictionary=cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name]),
    )

    return board, dict(DEFAULT_CHARUCO_ID_MAP if id_map is None else id_map)


def detect_charuco_frame(
    image: np.ndarray,
    min_corners: int = 4,
    dictionary_name: str = "DICT_5X5_1000",
    board_size: tuple[int, int] = (9, 9),
    id_map: dict[int, int] | None = None,
) -> dict:
    if image is None:
        raise ValueError("image is None.")

    board, id_map = get_cv2_charuco_board_and_mapping(
        dictionary_name,
        board_size=board_size,
        id_map=id_map,
    )
    detector = cv2.aruco.ArucoDetector(board.getDictionary())
    corners, ids, rejected = detector.detectMarkers(image)

    result = {
        "success": False,
        "valid": False,
        "marker_corners": [],
        "marker_ids": np.empty((0, 1), dtype=np.int32),
        "mapped_marker_ids": np.empty((0, 1), dtype=np.int32),
        "charuco_corners": None,
        "charuco_ids": None,
        "object_points": None,
        "image_points": None,
        "rejected": rejected,
        "num_markers": 0,
        "num_corners": 0,
        "reason": "No ArUco markers detected",
    }

    if ids is None or len(ids) == 0:
        return result

    mapped_ids = np.copy(ids)
    valid_mask = []
    for j in range(len(ids)):
        physical_id = int(ids[j][0])
        if physical_id in id_map:
            mapped_ids[j][0] = id_map[physical_id]
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    marker_corners = [corners[k] for k, is_valid in enumerate(valid_mask) if is_valid]
    marker_ids = ids[valid_mask]
    mapped_ids = mapped_ids[valid_mask]

    result.update({
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "mapped_marker_ids": mapped_ids,
        "num_markers": len(mapped_ids),
    })

    if len(mapped_ids) == 0:
        result["reason"] = "No board markers matched the configured ID map"
        return result

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        mapped_ids,
        image,
        board,
    )
    num_corners = 0 if charuco_corners is None else len(charuco_corners)

    result.update({
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "num_corners": num_corners,
    })

    if charuco_corners is None or charuco_ids is None or num_corners < min_corners:
        result["reason"] = f"Detected {num_corners} ChArUco corners; need {min_corners}"
        return result

    object_points, image_points = cv2.aruco.getBoardObjectAndImagePoints(
        board,
        charuco_corners,
        charuco_ids,
    )
    result.update({
        "success": True,
        "valid": True,
        "object_points": object_points,
        "image_points": image_points,
        "reason": "",
    })
    return result


def estimate_charuco_pose_from_detection(
    charuco_result: dict,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    if charuco_result is None or not charuco_result.get("valid", False):
        raise ValueError("A valid ChArUco detection is required for solvePnP.")

    object_points = np.asarray(
        charuco_result.get("object_points"),
        dtype=np.float64,
    ).reshape(-1, 3)
    image_points = np.asarray(
        charuco_result.get("image_points"),
        dtype=np.float64,
    ).reshape(-1, 2)
    if len(object_points) != len(image_points):
        raise ValueError(
            f"Point count mismatch: {len(object_points)} object vs "
            f"{len(image_points)} image points."
        )
    if len(object_points) < 4:
        raise ValueError("Need at least four points for solvePnP.")

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=np.ascontiguousarray(object_points),
        imagePoints=np.ascontiguousarray(image_points),
        cameraMatrix=np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        distCoeffs=np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {
            "success": False,
            "reason": "solvePnP failed",
        }

    R, _ = cv2.Rodrigues(rvec)
    T_CB = np.eye(4, dtype=np.float64)
    T_CB[:3, :3] = R
    T_CB[:3, 3] = tvec.reshape(3)

    projected, _ = cv2.projectPoints(
        objectPoints=object_points,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        distCoeffs=np.asarray(dist_coeffs, dtype=np.float64),
    )
    projected = projected.reshape(-1, 2)
    residuals = np.linalg.norm(projected - image_points, axis=1)
    reprojection_rmse_px = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "success": True,
        "T_CB": T_CB,
        "rvec": rvec,
        "tvec": tvec,
        "reprojection_rmse_px": reprojection_rmse_px,
        "reprojection_residuals_px": residuals,
        "num_points": len(object_points),
    }


def calibrate_camera_intrinsics(image_paths):
    images = [cv2.imread(str(path)) for path in image_paths]
    readable = [(path, image) for path, image in zip(image_paths, images) if image is not None]
    if not readable:
        raise RuntimeError("No calibration images could be read.")

    board, id_map = get_cv2_charuco_board_and_mapping()
    detector = cv2.aruco.ArucoDetector(board.getDictionary())

    obj_points_3d = []
    image_points_2d = []
    used_images = []
    used_image_paths = []

    for path, img in tqdm(readable, desc="Extracting ChArUco corners"):
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue

        mapped_ids = np.copy(ids)
        valid_mask = []
        for j in range(len(ids)):
            physical_id = int(ids[j][0])
            if physical_id in id_map:
                mapped_ids[j][0] = id_map[physical_id]
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        mapped_ids = mapped_ids[valid_mask]
        corners = [corners[k] for k, is_valid in enumerate(valid_mask) if is_valid]
        if len(mapped_ids) == 0:
            continue

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            mapped_ids,
            img,
            board,
        )
        if charuco_corners is None or len(charuco_corners) < 4:
            continue

        obj_points, img_points = cv2.aruco.getBoardObjectAndImagePoints(
            board,
            charuco_corners,
            charuco_ids,
        )
        obj_points_3d.append(obj_points)
        image_points_2d.append(img_points)
        used_images.append(img)
        used_image_paths.append(str(path))

    if not used_images:
        raise RuntimeError("No images had enough ChArUco corners for calibration.")

    image_size = (used_images[0].shape[1], used_images[0].shape[0])
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points_3d,
        imagePoints=image_points_2d,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    return (
        rms,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        obj_points_3d,
        image_points_2d,
        used_images,
        used_image_paths,
    )


def estimate_charuco_pose_rel_camera(
    images,
    obj_pts_3d,
    img_pts_2d,
    camera_matrix,
    dist_coeffs,
    visualize=True,
    axis_length=100,
    axes_output_dir="output/coordinate_axes",
):
    """Estimate board-to-camera transforms for detected ChArUco frames."""
    poses = []
    axes_output_path = Path(axes_output_dir)
    if visualize:
        axes_output_path.mkdir(parents=True, exist_ok=True)

    for i, (image, object_points, image_points) in enumerate(
        zip(images, obj_pts_3d, img_pts_2d)
    ):
        output = image.copy()
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            poses.append({
                "success": False,
                "image_index": i,
                "reason": "solvePnP failed",
                "output": output,
            })
            continue

        R, _ = cv2.Rodrigues(rvec)
        T_cb = np.eye(4, dtype=np.float64)
        T_cb[:3, :3] = R
        T_cb[:3, 3] = tvec.flatten()

        if visualize:
            cv2.drawFrameAxes(
                output,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                axis_length,
                1,
            )
            cv2.imwrite(str(axes_output_path / f"coordinate_axes_{i:03d}.png"), output)

        poses.append({
            "success": True,
            "image_index": i,
            "T_cb": T_cb,
            "T_CB": T_cb,
            "rvec": rvec,
            "tvec": tvec,
            "output": output,
        })

    return poses
