from __future__ import annotations

import os
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, permutations
from pathlib import Path

qt_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
if "cv2" in qt_plugin_path:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt6.QtCore import QObject, QEvent, QThread, Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QAction, QColor, QImage, QPixmap, QFont, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QColorDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget
)

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main_app.calibration_core.charuco import (
    ARUCO_DICTIONARIES,
    CHARUCO_BOARD_UNIT_TO_M,
    detect_charuco_frame,
    estimate_charuco_pose_from_detection,
)
from main_app.calibration_core.io import (
    SUPPORTED_IMAGE_SUFFIXES,
    SUPPORTED_LIDAR_SUFFIXES,
    FrameRecord,
    load_dataset,
    read_lidar_file,
)
from main_app.calibration_core.lidar import (
    HOLE_CENTERS_B_M,
    add_disk_centroids_to_uv_plane,
    make_lidar_uv_planes_from_plane_filtered_ptclds,
    segment_plane_from_pointcloud,
)
from main_app.calibration_core.extrinsics import (
    estimate_camera_lidar_extrinsic_from_board_poses,
    estimate_lidar_board_pose_from_disk_centers,
    extrinsic_distance,
    make_board_planar_symmetry_transforms,
    make_candidate_extrinsic_records,
    optimize_camera_lidar_extrinsic_robust,
)
from main_app.calibration_core.geometry import (
    estimate_rigid_transform_3d,
    invert_transform,
    make_transform,
)
from main_app.calibration_core.projection_window import (
    project_lidar_points_to_image,
    visualize_lidar_projection_on_image,
)
from main_app.ui.point_cloud_viewer import PointCloudViewerWidget


RECENT_CALIBRATION_STATE_PATH = Path("output/recent_calibration_state.pkl")
CENTER_DISTANCE_ABS_TOLERANCE_M = 0.12
CENTER_DISTANCE_REL_TOLERANCE = 0.25
CPU_CORE_COUNT = max(1, os.cpu_count() or 1)
MAIN_WINDOW_MIN_WIDTH = 1720
MAIN_WINDOW_MIN_HEIGHT = 860
WORKFLOW_STEP_CARD_MIN_WIDTH = 180


class SliderValueControl(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        suffix: str = "",
        decimals: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.step = float(step)
        self.suffix = suffix
        self.decimals = int(decimals)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._index_for_value(self.maximum))
        self.slider.valueChanged.connect(self._handle_slider_changed)

        self.value_label = QLabel()
        self.value_label.setMinimumWidth(72)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.value_label)
        self.setLayout(layout)
        self.setValue(value)

    def value(self) -> float:
        value = self.minimum + self.slider.value() * self.step
        return float(np.clip(value, self.minimum, self.maximum))

    def setValue(self, value: float):
        self.slider.setValue(self._index_for_value(value))
        self._update_label()

    def _index_for_value(self, value: float) -> int:
        return int(round((float(value) - self.minimum) / self.step))

    def _format_value(self) -> str:
        value = self.value()
        if self.decimals == 0:
            text = f"{value:.0f}"
        else:
            text = f"{value:.{self.decimals}f}"
        return f"{text}{self.suffix}"

    def _update_label(self):
        self.value_label.setText(self._format_value())

    def _handle_slider_changed(self):
        self._update_label()
        if not self.signalsBlocked():
            self.valueChanged.emit(self.value())


def clamp_thread_count(value: int) -> int:
    return max(1, min(int(value), CPU_CORE_COUNT))


def calibration_object_points(result: dict, frame_index: int) -> np.ndarray:
    points = result.get("object_points")
    if points is None:
        raise RuntimeError(f"Frame {frame_index} is missing object points.")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise RuntimeError(f"Frame {frame_index} has non-finite object points.")
    return np.ascontiguousarray(points)


def calibration_image_points(result: dict, frame_index: int) -> np.ndarray:
    points = result.get("image_points")
    if points is None:
        raise RuntimeError(f"Frame {frame_index} is missing image points.")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise RuntimeError(f"Frame {frame_index} has non-finite image points.")
    return np.ascontiguousarray(points)


def lidar_range_mask(xyz: np.ndarray, settings: dict) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    range_xmin = float(settings.get("range_xmin", -np.inf))
    range_xmax = float(settings.get("range_xmax", np.inf))
    range_ymin = float(settings.get("range_ymin", -np.inf))
    range_ymax = float(settings.get("range_ymax", np.inf))
    range_zmax = float(settings.get("range_zmax", np.inf))
    x_low, x_high = sorted((range_xmin, range_xmax))
    y_low, y_high = sorted((range_ymin, range_ymax))
    return (
        (xyz[:, 0] >= x_low)
        & (xyz[:, 0] <= x_high)
        & (xyz[:, 1] >= y_low)
        & (xyz[:, 1] <= y_high)
        & (xyz[:, 2] <= range_zmax)
    )


def filter_lidar_dict_by_mask(lidar_dict: dict, mask: np.ndarray) -> dict:
    xyz = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
    filtered = {"xyz": xyz[mask, :3]}
    for key, value in lidar_dict.items():
        if key == "xyz":
            continue
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(mask):
            filtered[key] = value[mask]
        else:
            filtered[key] = value
    return filtered


def region_query(points: np.ndarray, index: int, eps: float) -> list[int]:
    distances = np.linalg.norm(points - points[index], axis=1)
    return np.flatnonzero(distances <= eps).astype(int).tolist()


def dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.full(len(points), -1, dtype=np.int32)
    if len(points) == 0:
        return labels

    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    for index in range(len(points)):
        if visited[index]:
            continue
        visited[index] = True
        neighbors = region_query(points, index, eps)
        if len(neighbors) < min_samples:
            continue

        labels[index] = cluster_id
        seeds = list(neighbors)
        cursor = 0
        while cursor < len(seeds):
            neighbor = seeds[cursor]
            if not visited[neighbor]:
                visited[neighbor] = True
                neighbor_neighbors = region_query(points, neighbor, eps)
                if len(neighbor_neighbors) >= min_samples:
                    for candidate in neighbor_neighbors:
                        if candidate not in seeds:
                            seeds.append(candidate)
            if labels[neighbor] < 0:
                labels[neighbor] = cluster_id
            cursor += 1
        cluster_id += 1
    return labels


def fit_circle_center_3d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centroid = np.mean(points, axis=0)
    if len(points) < 3:
        return centroid

    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    uv = centered @ basis.T
    x = uv[:, 0]
    y = uv[:, 1]
    A = np.column_stack([x, y, np.ones(len(uv))])
    b = -(x * x + y * y)
    try:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return centroid
    center_uv = -0.5 * coeffs[:2]
    return centroid + center_uv @ basis


def circle_centers_from_labels(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        cluster_points = points[labels == label]
        if len(cluster_points) > 0:
            centers.append(fit_circle_center_3d(cluster_points))
    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(centers)


def sorted_pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2:
        return np.empty(0, dtype=np.float64)

    distances = []
    for i, j in combinations(range(len(points)), 2):
        distances.append(float(np.linalg.norm(points[i] - points[j])))
    return np.sort(np.asarray(distances, dtype=np.float64))


def check_center_distance_consistency(
    centers_L: np.ndarray,
    template_centers_B_m: np.ndarray,
    abs_tolerance_m: float = CENTER_DISTANCE_ABS_TOLERANCE_M,
    rel_tolerance: float = CENTER_DISTANCE_REL_TOLERANCE,
    max_candidate_centers: int = 12,
) -> dict:
    centers = np.asarray(centers_L, dtype=np.float64).reshape(-1, 3)
    template = np.asarray(template_centers_B_m, dtype=np.float64)
    if template.ndim != 2 or template.shape[1] < 2:
        raise ValueError("Template centers must have shape (N, 2) or wider.")

    template_xy = template[:, :2]
    expected_count = len(template_xy)
    template_distances = sorted_pairwise_distances(template_xy)

    if len(centers) < expected_count:
        return {
            "passed": False,
            "reason": f"need {expected_count} centers, got {len(centers)}",
            "expected_count": expected_count,
            "detected_count": len(centers),
            "template_distances_m": template_distances,
            "matched_indices": [],
        }

    candidate_indices = np.arange(len(centers), dtype=np.int64)
    if len(candidate_indices) > max_candidate_centers:
        candidate_indices = candidate_indices[:max_candidate_centers]

    best = None
    for combo in combinations(candidate_indices.tolist(), expected_count):
        detected_distances = sorted_pairwise_distances(
            centers[np.asarray(combo, dtype=np.int64)]
        )
        abs_errors = np.abs(detected_distances - template_distances)
        rel_errors = abs_errors / np.maximum(template_distances, 1e-9)
        score = (float(np.max(rel_errors)), float(np.mean(abs_errors)))
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "matched_indices": list(combo),
                "detected_distances_m": detected_distances,
                "abs_errors_m": abs_errors,
                "rel_errors": rel_errors,
                "max_abs_error_m": float(np.max(abs_errors)),
                "mean_abs_error_m": float(np.mean(abs_errors)),
                "max_rel_error": float(np.max(rel_errors)),
                "mean_rel_error": float(np.mean(rel_errors)),
            }

    if best is None:
        return {
            "passed": False,
            "reason": "no center combinations available",
            "expected_count": expected_count,
            "detected_count": len(centers),
            "template_distances_m": template_distances,
            "matched_indices": [],
        }

    passed = (
        best["max_abs_error_m"] <= abs_tolerance_m and
        best["max_rel_error"] <= rel_tolerance
    )
    reason = (
        "ok"
        if passed else
        (
            f"max distance error {best['max_abs_error_m']:.3f} m "
            f"({best['max_rel_error']:.1%}) exceeds tolerance "
            f"{abs_tolerance_m:.3f} m/{rel_tolerance:.0%}"
        )
    )
    return {
        "passed": passed,
        "reason": reason,
        "expected_count": expected_count,
        "detected_count": len(centers),
        "template_distances_m": template_distances,
        "abs_tolerance_m": abs_tolerance_m,
        "rel_tolerance": rel_tolerance,
        **best,
    }


def detect_lidar_centers_from_lidar_dict(lidar_dict: dict, settings: dict) -> dict:
    xyz = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("LiDAR point cloud must contain xyz points.")
    xyz = xyz[:, :3]

    source = settings["source"]
    selected_indices = np.arange(len(xyz), dtype=np.int64)

    range_mask = lidar_range_mask(xyz, settings)
    selected_indices = selected_indices[range_mask]
    selected_points = xyz[range_mask]
    selected_dict = filter_lidar_dict_by_mask(lidar_dict, range_mask)

    if len(selected_points) == 0:
        raise RuntimeError("No LiDAR points remain after range thresholding.")

    if source in {"Plane points", "Plane + reflectivity"}:
        plane = segment_plane_from_pointcloud(
            selected_points,
            distance_threshold=float(settings["plane_distance_threshold"]),
            min_inliers=max(10, int(settings["min_samples"])),
        )
        plane_indices = plane["inlier_indices"]
        selected_indices = selected_indices[plane_indices]
        selected_points = selected_points[plane_indices]
        selected_dict = {
            "xyz": selected_points,
            "plane_model": plane["plane_model"],
        }
        for key, value in lidar_dict.items():
            if isinstance(value, np.ndarray) and len(value) == len(xyz):
                selected_dict[key] = value[selected_indices]

    if source in {"Reflective points", "Plane + reflectivity"}:
        if "reflectivity" not in lidar_dict and "reflectivity" not in selected_dict:
            raise KeyError("Point cloud does not contain reflectivity values.")
        reflectivity = np.asarray(
            selected_dict.get("reflectivity", lidar_dict["reflectivity"]),
            dtype=np.float64,
        ).reshape(-1)
        mask = reflectivity >= float(settings["reflectivity_threshold"])
        selected_points = selected_points[mask]
        selected_indices = selected_indices[mask]
        for key, value in list(selected_dict.items()):
            if isinstance(value, np.ndarray) and len(value) == len(mask):
                selected_dict[key] = value[mask]
        selected_dict["xyz"] = selected_points

    if len(selected_points) == 0:
        raise RuntimeError("No LiDAR points remain after source filtering.")

    fit_mode = settings["fit_mode"]
    labels = np.full(len(selected_points), -1, dtype=np.int32)
    centers_L = np.empty((0, 3), dtype=np.float64)
    uv_plane_result = None

    if fit_mode == "Circle fit from DBSCAN clusters":
        labels = dbscan_labels(
            selected_points,
            eps=float(settings["eps"]),
            min_samples=int(settings["min_samples"]),
        )
        centers_L = circle_centers_from_labels(selected_points, labels)
    elif fit_mode == "Disk fitting on plane":
        uv_plane = make_lidar_uv_planes_from_plane_filtered_ptclds(
            [selected_dict],
            distance_threshold=float(settings["plane_distance_threshold"]),
            square_align_uv=True,
        )[0]
        uv_plane = add_disk_centroids_to_uv_plane(
            uv_plane,
            radius_m=float(settings["disk_radius_m"]),
            min_inliers=int(settings["min_samples"]),
        )
        centers_L = uv_plane.get("disk_centers_L", np.empty((0, 3), dtype=np.float64))
        uv_plane_result = uv_plane
    else:
        raise ValueError(f"Unknown LiDAR center fit mode: {fit_mode}")

    center_distance_check = check_center_distance_consistency(
        centers_L,
        template_centers_B_m=HOLE_CENTERS_B_M,
        abs_tolerance_m=float(
            settings.get(
                "center_distance_abs_tolerance_m",
                CENTER_DISTANCE_ABS_TOLERANCE_M,
            )
        ),
        rel_tolerance=float(
            settings.get(
                "center_distance_rel_tolerance",
                CENTER_DISTANCE_REL_TOLERANCE,
            )
        ),
    )

    return {
        "source": source,
        "fit_mode": fit_mode,
        "selected_indices": selected_indices,
        "cluster_points": selected_points,
        "cluster_labels": labels,
        "centers_L": centers_L,
        "uv_plane": uv_plane_result,
        "center_distance_check": center_distance_check,
        "num_clusters": len(set(int(label) for label in labels if label >= 0)),
        "show_clusters": bool(settings["show_clusters"]),
        "show_centers": bool(settings["show_centers"]),
    }


class IntrinsicsCalibrationWorker(QObject):
    progress = pyqtSignal(int, int, str, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        frame_data: list[dict],
        calibration_flags: int,
        thread_count: int,
    ):
        super().__init__()
        self.frame_data = frame_data
        self.calibration_flags = calibration_flags
        self.thread_count = clamp_thread_count(thread_count)

    @staticmethod
    def prepare_frame(frame: dict) -> dict:
        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {frame['image_path']}")
        frame_image_size = (image.shape[1], image.shape[0])
        result = frame["charuco_result"]
        obj = calibration_object_points(result, frame["frame_index"])
        img = calibration_image_points(result, frame["frame_index"])
        if len(obj) != len(img):
            raise RuntimeError(
                f"Frame {frame['frame_index']} has mismatched calibration "
                f"points: {len(obj)} object vs {len(img)} image."
            )
        if len(obj) < 4:
            raise RuntimeError(
                f"Frame {frame['frame_index']} has only {len(obj)} corners; "
                "need at least 4."
            )
        return {
            "frame_index": int(frame["frame_index"]),
            "image_size": frame_image_size,
            "object_points": obj,
            "image_points": img,
        }

    def run(self):
        prepared_frames = []
        total_steps = len(self.frame_data) + 1

        try:
            max_workers = min(self.thread_count, max(1, len(self.frame_data)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.prepare_frame, frame)
                    for frame in self.frame_data
                ]
                for step, future in enumerate(as_completed(futures), start=1):
                    prepared_frames.append(future.result())
                    self.progress.emit(
                        step,
                        total_steps,
                        f"Prepared {step}/{len(self.frame_data)} frame(s) (%p%)",
                        f"Preparing calibration frame {step}/{len(self.frame_data)}",
                    )

            prepared_frames.sort(key=lambda frame: frame["frame_index"])
            image_size = prepared_frames[0]["image_size"]
            for frame in prepared_frames:
                if frame["image_size"] != image_size:
                    raise RuntimeError(
                        "All calibration images must have the same size. "
                        f"Frame {frame['frame_index']} has {frame['image_size']}, "
                        f"expected {image_size}."
                    )
            object_points = [frame["object_points"] for frame in prepared_frames]
            image_points = [frame["image_points"] for frame in prepared_frames]

            self.progress.emit(
                len(self.frame_data),
                total_steps,
                "Solving calibration (%p%)",
                "Solving camera calibration...",
            )
            rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                objectPoints=object_points,
                imagePoints=image_points,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None,
                flags=self.calibration_flags,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(
            {
                "rms": float(rms),
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "frame_indices": [
                    int(frame["frame_index"]) for frame in prepared_frames
                ],
                "total_steps": total_steps,
            }
        )


class SolvePnPWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)

    def __init__(
        self,
        frame_data: list[dict],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        thread_count: int,
    ):
        super().__init__()
        self.frame_data = frame_data
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).copy()
        self.thread_count = clamp_thread_count(thread_count)

    def solve_frame(self, frame: dict) -> tuple[int, dict, bool]:
        frame_index = int(frame["frame_index"])
        try:
            pose = estimate_charuco_pose_from_detection(
                frame["charuco_result"],
                self.camera_matrix,
                self.dist_coeffs,
            )
            if not pose.get("success", False):
                raise RuntimeError(pose.get("reason", "solvePnP failed."))
            return frame_index, pose, True
        except Exception as exc:
            return frame_index, {
                "success": False,
                "reason": str(exc),
            }, False

    def run(self):
        poses = []
        failed = []
        total = len(self.frame_data)

        max_workers = min(self.thread_count, max(1, total))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.solve_frame, frame)
                for frame in self.frame_data
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                frame_index, pose, success = future.result()
                if not success:
                    failed.append(frame_index)
                poses.append((frame_index, pose))
                self.progress.emit(
                    index,
                    total,
                    f"SolvePnP frame {frame_index + 1}",
                )

        self.finished.emit(
            {
                "poses": poses,
                "failed": failed,
                "total": total,
            }
        )


class LidarCenterDetectionWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)

    def __init__(self, frame_data: list[dict], settings: dict):
        super().__init__()
        self.frame_data = frame_data
        self.settings = dict(settings)

    def detect_frame(self, frame: dict) -> tuple[int, dict | None, str | None]:
        frame_index = int(frame["frame_index"])
        try:
            lidar_dict = read_lidar_file(frame["lidar_path"])
            result = detect_lidar_centers_from_lidar_dict(
                lidar_dict,
                self.settings,
            )
            return frame_index, result, None
        except Exception as exc:
            return frame_index, None, str(exc)

    def run(self):
        results = []
        failures = []
        completed = 0
        total_centers = 0
        total = len(self.frame_data)

        for index, frame in enumerate(self.frame_data, start=1):
            frame_index, result, error = self.detect_frame(frame)
            if error is not None:
                failures.append((frame_index, error))
                results.append((frame_index, None))
            else:
                results.append((frame_index, result))
                completed += 1
                total_centers += len(result.get("centers_L", []))

            self.progress.emit(
                index,
                total,
                f"LiDAR center detection frame {frame_index + 1}",
            )

        self.finished.emit(
            {
                "results": results,
                "failures": failures,
                "completed": completed,
                "total_centers": total_centers,
                "total": total,
            }
        )


class CharucoDetectionWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)

    def __init__(
        self,
        frame_data: list[dict],
        dictionary_name: str,
        board_size: tuple[int, int],
        id_map: dict[int, int] | None,
        thread_count: int,
    ):
        super().__init__()
        self.frame_data = frame_data
        self.dictionary_name = dictionary_name
        self.board_size = board_size
        self.id_map = id_map
        self.thread_count = clamp_thread_count(thread_count)

    def detect_frame(self, frame: dict) -> tuple[int, dict]:
        frame_index = int(frame["frame_index"])
        try:
            image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {frame['image_path']}")
            result = detect_charuco_frame(
                image,
                dictionary_name=self.dictionary_name,
                board_size=self.board_size,
                id_map=self.id_map,
            )
        except Exception as exc:
            result = {
                "success": False,
                "valid": False,
                "num_markers": 0,
                "num_corners": 0,
                "reason": str(exc),
            }
        return frame_index, result

    def run(self):
        results = []
        total = len(self.frame_data)

        max_workers = min(self.thread_count, max(1, total))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.detect_frame, frame)
                for frame in self.frame_data
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                frame_index, result = future.result()
                results.append((frame_index, result))
                self.progress.emit(
                    index,
                    total,
                    f"Detected frame {frame_index + 1}/{total}",
                )

        valid_count = sum(1 for _frame_index, result in results if result.get("valid", False))
        self.finished.emit(
            {
                "results": results,
                "total": total,
                "valid_count": valid_count,
                "dictionary_name": self.dictionary_name,
                "board_size": self.board_size,
            }
        )


# Apply this stylesheet to your QApplication or specific widgets
modern_stylesheet = """
    QFrame#Sidebar {
        background-color: #1e1e24;
        border-radius: 8px;
        padding: 15px;
    }
    QLabel {
        font-family: 'Segoe UI', 'Inter', sans-serif;
        color: #E0E0E0;
    }
    QPushButton {
        background-color: #2F80ED;
        color: white;
        border-radius: 6px;
        padding: 4px 6px;
        min-height: 20px;
        font-size: 14px;
        font-weight: 400;
    }
    QPushButton:hover {
        background-color: #1B66C9;
    }
    QPushButton:pressed {
        background-color: #154FA0;
    }
    QPushButton:disabled {
        background-color: #D0D5DD;
        color: #475467;
    }
    QLineEdit {
        border: 2px solid #E0E0E0;
        border-radius: 6px;
        padding: 6px;
        background-color: #FFFFFF;
    }
    QLineEdit:focus {
        border: 2px solid #2F80ED;
    }
"""


class StepConfigurationDialog(QDialog):
    def __init__(self, step_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{step_name} Configuration")
        self.resize(420, 260)

        title = QLabel(step_name)
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        config_group = QGroupBox("Configuration Settings")
        config_layout = QVBoxLayout()
        placeholder = QLabel("No configuration settings available yet.")
        placeholder.setStyleSheet("QLabel { color: #667085; }")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        config_layout.addWidget(placeholder, stretch=1)
        config_group.setLayout(config_layout)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        execute_button = QPushButton("Execute")
        execute_button.setEnabled(False)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(close_button)
        actions.addWidget(execute_button)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(config_group, stretch=1)
        layout.addLayout(actions)
        self.setLayout(layout)


class SettingsDialog(QDialog):
    def __init__(self, thread_count: int, max_thread_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(360, 160)

        title = QLabel("Settings")
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, max_thread_count)
        self.thread_count_spin.setValue(clamp_thread_count(thread_count))
        self.thread_count_spin.setToolTip(
            f"Number of worker-pool threads. Maximum: {max_thread_count} CPU core(s)."
        )

        form = QGridLayout()
        label = QLabel("Worker threads")
        label.setStyleSheet("QLabel { color: #111827; }")
        max_label = QLabel(f"Max: {max_thread_count} CPU core(s)")
        max_label.setStyleSheet("QLabel { color: #667085; }")
        form.addWidget(label, 0, 0)
        form.addWidget(self.thread_count_spin, 0, 1)
        form.addWidget(max_label, 1, 1)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(cancel_button)
        actions.addWidget(save_button)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.setLayout(layout)

    def thread_count(self) -> int:
        return clamp_thread_count(self.thread_count_spin.value())


class LoadDataDialog(QDialog):
    def __init__(self, load_callback, parent=None):
        super().__init__(parent)
        self.load_callback = load_callback
        self.image_dir = ""
        self.lidar_dir = ""
        self.setWindowTitle("Load Data")
        self.resize(620, 220)

        title = QLabel("Load Data")
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("Select image directory")
        image_button = QPushButton("Browse")
        image_button.clicked.connect(self.select_image_dir)

        self.lidar_path_edit = QLineEdit()
        self.lidar_path_edit.setReadOnly(True)
        self.lidar_path_edit.setPlaceholderText("Select LiDAR directory")
        lidar_button = QPushButton("Browse")
        lidar_button.clicked.connect(self.select_lidar_dir)

        images_label = QLabel("Images")
        images_label.setStyleSheet("QLabel { color: #111827; }")
        lidar_label = QLabel("LiDAR")
        lidar_label.setStyleSheet("QLabel { color: #111827; }")

        form = QGridLayout()
        form.addWidget(images_label, 0, 0)
        form.addWidget(self.image_path_edit, 0, 1)
        form.addWidget(image_button, 0, 2)
        form.addWidget(lidar_label, 1, 0)
        form.addWidget(self.lidar_path_edit, 1, 1)
        form.addWidget(lidar_button, 1, 2)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(close_button)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.setLayout(layout)

    def select_image_dir(self):
        path = self.select_directory("Select Image Directory")
        if not path:
            return
        self.image_dir = path
        self.image_path_edit.setText(path)
        self.try_load_dataset()

    def select_lidar_dir(self):
        path = self.select_directory("Select LiDAR Directory")
        if not path:
            return
        self.lidar_dir = path
        self.lidar_path_edit.setText(path)
        self.try_load_dataset()

    def try_load_dataset(self):
        if not self.image_dir or not self.lidar_dir:
            return
        if self.load_callback(self.image_dir, self.lidar_dir):
            self.accept()

    def select_directory(self, title: str) -> str:
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.ViewMode.List)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                return selected_paths[0]
        return ""


class CharucoConfigurationDialog(QDialog):
    def __init__(
        self,
        dictionary_name: str,
        board_size: tuple[int, int],
        id_map: dict[int, int] | None,
        id_map_path: str,
        execute_callback,
        parent=None,
    ):
        super().__init__(parent)
        self.execute_callback = execute_callback
        self.id_map = id_map
        self.id_map_path = id_map_path
        self.setWindowTitle("ChArUco Configuration")
        self.resize(520, 300)

        title = QLabel("ChArUco")
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        dictionary_label = QLabel("Dictionary")
        dictionary_label.setStyleSheet("QLabel { color: #111827; }")
        self.dictionary_combo = QComboBox()
        self.dictionary_combo.addItems(sorted(ARUCO_DICTIONARIES))
        current_index = self.dictionary_combo.findText(dictionary_name)
        if current_index >= 0:
            self.dictionary_combo.setCurrentIndex(current_index)

        board_columns_label = QLabel("Board columns")
        board_columns_label.setStyleSheet("QLabel { color: #111827; }")
        self.board_columns_spin = QSpinBox()
        self.board_columns_spin.setRange(2, 40)
        self.board_columns_spin.setValue(int(board_size[0]))

        board_rows_label = QLabel("Board rows")
        board_rows_label.setStyleSheet("QLabel { color: #111827; }")
        self.board_rows_spin = QSpinBox()
        self.board_rows_spin.setRange(2, 40)
        self.board_rows_spin.setValue(int(board_size[1]))

        id_map_label = QLabel("ID map")
        id_map_label.setStyleSheet("QLabel { color: #111827; }")
        self.id_map_path_edit = QLineEdit()
        self.id_map_path_edit.setReadOnly(True)
        self.id_map_path_edit.setPlaceholderText("Using default ID map")
        self.id_map_path_edit.setText(id_map_path)
        load_id_map_button = QPushButton("Load JSON")
        load_id_map_button.clicked.connect(self.load_id_map_json)

        form = QGridLayout()
        form.addWidget(dictionary_label, 0, 0)
        form.addWidget(self.dictionary_combo, 0, 1)
        form.addWidget(board_columns_label, 1, 0)
        form.addWidget(self.board_columns_spin, 1, 1)
        form.addWidget(board_rows_label, 2, 0)
        form.addWidget(self.board_rows_spin, 2, 1)
        form.addWidget(id_map_label, 3, 0)
        form.addWidget(self.id_map_path_edit, 3, 1)
        form.addWidget(load_id_map_button, 3, 2)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(self.execute)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(close_button)
        actions.addWidget(execute_button)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.setLayout(layout)

    def execute(self):
        dictionary_name = self.dictionary_combo.currentText()
        board_size = (self.board_columns_spin.value(), self.board_rows_spin.value())
        id_map = self.id_map
        id_map_path = self.id_map_path
        self.accept()
        self.execute_callback(dictionary_name, board_size, id_map, id_map_path)

    def load_id_map_json(self):
        path = self.select_json_file()
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            mapping = data.get("id_map", data)
            if not isinstance(mapping, dict):
                raise ValueError("Expected a JSON object or an object with an 'id_map' object.")
            self.id_map = {
                int(source_id): int(target_id)
                for source_id, target_id in mapping.items()
            }
        except Exception as exc:
            QMessageBox.critical(self, "ID Map Load Failed", str(exc))
            return
        self.id_map_path = path
        self.id_map_path_edit.setText(path)

    def select_json_file(self) -> str:
        dialog = QFileDialog(self, "Load ID Map")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilter("JSON files (*.json)")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                return selected_paths[0]
        return ""


INTRINSICS_DISTORTION_OPTIONS = {
    "OpenCV default": 0,
    "Zero tangent distortion": cv2.CALIB_ZERO_TANGENT_DIST,
    "Fix k3": cv2.CALIB_FIX_K3,
    "Rational model": cv2.CALIB_RATIONAL_MODEL,
    "Thin prism model": cv2.CALIB_THIN_PRISM_MODEL,
    "Tilted sensor model": cv2.CALIB_TILTED_MODEL,
    "Rational + thin prism": cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL,
}


class IntrinsicsConfigurationDialog(QDialog):
    def __init__(self, distortion_model_name: str, execute_callback, parent=None):
        super().__init__(parent)
        self.execute_callback = execute_callback
        self.setWindowTitle("Intrinsics Configuration")
        self.resize(460, 220)

        title = QLabel("Intrinsics")
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        distortion_label = QLabel("Distortion model")
        distortion_label.setStyleSheet("QLabel { color: #111827; }")
        self.distortion_combo = QComboBox()
        self.distortion_combo.addItems(list(INTRINSICS_DISTORTION_OPTIONS))
        current_index = self.distortion_combo.findText(distortion_model_name)
        if current_index >= 0:
            self.distortion_combo.setCurrentIndex(current_index)

        form = QGridLayout()
        form.addWidget(distortion_label, 0, 0)
        form.addWidget(self.distortion_combo, 0, 1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(self.execute)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(close_button)
        actions.addWidget(execute_button)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.setLayout(layout)

    def execute(self):
        distortion_model_name = self.distortion_combo.currentText()
        self.accept()
        self.execute_callback(distortion_model_name)


class LidarCenterConfigurationDialog(QDialog):
    RANGE_SLIDER_SCALE = 100

    def __init__(
        self,
        settings: dict,
        execute_callback,
        preview_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.execute_callback = execute_callback
        self.preview_callback = preview_callback
        self.setWindowTitle("LiDAR Center Configuration")
        self.resize(680, 520)

        title = QLabel("LiDAR Center")
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Reflective points", "Plane points", "Plane + reflectivity"])
        self.source_combo.setCurrentText(settings.get("source", "Reflective points"))

        self.fit_combo = QComboBox()
        self.fit_combo.addItems(["Circle fit from DBSCAN clusters", "Disk fitting on plane"])
        self.fit_combo.setCurrentText(settings.get("fit_mode", "Disk fitting on plane"))

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.001, 2.0)
        self.eps_spin.setDecimals(3)
        self.eps_spin.setSingleStep(0.01)
        self.eps_spin.setValue(float(settings.get("eps", 0.08)))

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 500)
        self.min_samples_spin.setValue(int(settings.get("min_samples", 8)))

        self.reflectivity_spin = QDoubleSpinBox()
        self.reflectivity_spin.setRange(0.0, 100000.0)
        self.reflectivity_spin.setDecimals(2)
        self.reflectivity_spin.setValue(float(settings.get("reflectivity_threshold", 75.0)))

        self.plane_threshold_spin = QDoubleSpinBox()
        self.plane_threshold_spin.setRange(0.001, 0.5)
        self.plane_threshold_spin.setDecimals(3)
        self.plane_threshold_spin.setSingleStep(0.005)
        self.plane_threshold_spin.setValue(float(settings.get("plane_distance_threshold", 0.015)))

        self.disk_radius_spin = QDoubleSpinBox()
        self.disk_radius_spin.setRange(0.01, 1.0)
        self.disk_radius_spin.setDecimals(3)
        self.disk_radius_spin.setSingleStep(0.01)
        self.disk_radius_spin.setValue(float(settings.get("disk_radius_m", 0.12)))

        self.xmin_slider, self.xmin_spin, xmin_widget = self.make_range_slider(
            settings,
            "range_xmin",
            default=-3.0,
            minimum=-20.0,
            maximum=20.0,
        )
        self.xmax_slider, self.xmax_spin, xmax_widget = self.make_range_slider(
            settings,
            "range_xmax",
            default=1.0,
            minimum=-20.0,
            maximum=20.0,
        )
        self.ymin_slider, self.ymin_spin, ymin_widget = self.make_range_slider(
            settings,
            "range_ymin",
            default=-6.0,
            minimum=-20.0,
            maximum=20.0,
        )
        self.ymax_slider, self.ymax_spin, ymax_widget = self.make_range_slider(
            settings,
            "range_ymax",
            default=1.0,
            minimum=-20.0,
            maximum=20.0,
        )
        self.zmax_slider, self.zmax_spin, zmax_widget = self.make_range_slider(
            settings,
            "range_zmax",
            default=4.0,
            minimum=-10.0,
            maximum=20.0,
        )

        self.show_clusters_checkbox = QCheckBox("Show clusters")
        self.show_clusters_checkbox.setChecked(bool(settings.get("show_clusters", True)))

        self.show_centers_checkbox = QCheckBox("Show circle centers")
        self.show_centers_checkbox.setChecked(bool(settings.get("show_centers", True)))

        form = QGridLayout()
        labels = [
            ("Source", self.source_combo),
            ("Fit", self.fit_combo),
            ("DBSCAN eps", self.eps_spin),
            ("DBSCAN min samples", self.min_samples_spin),
            ("Reflectivity threshold", self.reflectivity_spin),
            ("Plane threshold m", self.plane_threshold_spin),
            ("Disk radius m", self.disk_radius_spin),
            ("x min m", xmin_widget),
            ("x max m", xmax_widget),
            ("y min m", ymin_widget),
            ("y max m", ymax_widget),
            ("z max m", zmax_widget),
        ]
        for row, (text, widget) in enumerate(labels):
            label = QLabel(text)
            label.setStyleSheet("QLabel { color: #111827; }")
            form.addWidget(label, row, 0)
            form.addWidget(widget, row, 1)
        form.addWidget(self.show_clusters_checkbox, len(labels), 0, 1, 2)
        form.addWidget(self.show_centers_checkbox, len(labels) + 1, 0, 1, 2)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(self.execute)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(close_button)
        actions.addWidget(execute_button)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(actions)
        self.setLayout(layout)

    def make_range_slider(
        self,
        settings: dict,
        key: str,
        default: float,
        minimum: float,
        maximum: float,
    ):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(
            int(round(minimum * self.RANGE_SLIDER_SCALE)),
            int(round(maximum * self.RANGE_SLIDER_SCALE)),
        )
        slider.setSingleStep(5)
        slider.setPageStep(50)

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(float(settings.get(key, default)))
        slider.setValue(int(round(spin.value() * self.RANGE_SLIDER_SCALE)))

        slider.valueChanged.connect(
            lambda value, target=spin: target.setValue(
                value / self.RANGE_SLIDER_SCALE
            )
        )
        spin.valueChanged.connect(
            lambda value, target=slider: target.setValue(
                int(round(value * self.RANGE_SLIDER_SCALE))
            )
        )
        slider.valueChanged.connect(lambda _value: self.preview_range_thresholds())
        spin.valueChanged.connect(lambda _value: self.preview_range_thresholds())

        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)
        widget.setLayout(layout)
        return slider, spin, widget

    def current_settings(self):
        return {
            "source": self.source_combo.currentText(),
            "fit_mode": self.fit_combo.currentText(),
            "eps": self.eps_spin.value(),
            "min_samples": self.min_samples_spin.value(),
            "reflectivity_threshold": self.reflectivity_spin.value(),
            "plane_distance_threshold": self.plane_threshold_spin.value(),
            "disk_radius_m": self.disk_radius_spin.value(),
            "range_xmin": self.xmin_spin.value(),
            "range_xmax": self.xmax_spin.value(),
            "range_ymin": self.ymin_spin.value(),
            "range_ymax": self.ymax_spin.value(),
            "range_zmax": self.zmax_spin.value(),
            "show_clusters": self.show_clusters_checkbox.isChecked(),
            "show_centers": self.show_centers_checkbox.isChecked(),
        }

    def preview_range_thresholds(self):
        if self.preview_callback is not None:
            self.preview_callback(self.current_settings())

    def execute(self):
        settings = self.current_settings()
        self.accept()
        self.execute_callback(settings)


class ExtrinsicCandidateSelectionDialog(QDialog):
    def __init__(self, candidates: list[dict], parent=None):
        super().__init__(parent)
        self.candidates = candidates
        self.selected_index: int | None = None
        self.current_index = 0
        self.setWindowTitle("Select LiDAR-Camera Extrinsic Candidate")
        self.resize(1180, 860)

        title = QLabel("Select the projection that best aligns LiDAR with the image")
        title.setStyleSheet("QLabel { color: #111827; font-size: 16px; font-weight: 700; }")

        self.info_label = QLabel()
        self.info_label.setStyleSheet("QLabel { color: #111827; font-weight: 600; }")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(980, 620)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background: #111827; color: #f9fafb; }")

        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(self.previous_candidate)

        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_candidate)

        select_button = QPushButton("Use This Orientation")
        select_button.clicked.connect(lambda: self.select_candidate(self.current_index))

        close_button = QPushButton("Cancel")
        close_button.clicked.connect(self.reject)

        nav = QHBoxLayout()
        nav.addWidget(previous_button)
        nav.addWidget(next_button)
        nav.addStretch(1)
        nav.addWidget(select_button)
        nav.addWidget(close_button)

        actions = QHBoxLayout()
        actions.addLayout(nav)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.info_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(actions)
        self.setLayout(layout)
        self.update_candidate_view()

    def previous_candidate(self):
        self.current_index = (self.current_index - 1) % len(self.candidates)
        self.update_candidate_view()

    def next_candidate(self):
        self.current_index = (self.current_index + 1) % len(self.candidates)
        self.update_candidate_view()

    def update_candidate_view(self):
        if not self.candidates:
            self.info_label.setText("No candidate orientations available")
            self.image_label.clear()
            return

        candidate = self.candidates[self.current_index]
        normal_label = "zpos" if candidate["normal_sign"] > 0 else "zneg"
        self.info_label.setText(
            f"{self.current_index + 1}/{len(self.candidates)} | "
            f"{candidate['symmetry_name']} / {normal_label} | "
            f"RMSE {candidate['lidar_board_rmse_m']:.4f} m | "
            f"drawn {candidate.get('drawn_count', 0)} | "
            f"valid {candidate.get('valid_count', 0)}"
        )
        pixmap = candidate["pixmap"].scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_candidate_view()

    def select_candidate(self, index: int):
        self.selected_index = index
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Camera Calibration")
        self.setMinimumSize(MAIN_WINDOW_MIN_WIDTH, MAIN_WINDOW_MIN_HEIGHT)
        self.resize(MAIN_WINDOW_MIN_WIDTH, 900)

        self.frames: list[FrameRecord] = []
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.intrinsics_rms: float | None = None
        self.intrinsics_frame_indices: list[int] = []
        self.excluded_intrinsics_frame_indices: list[int] = []
        self.charuco_dictionary_name = "DICT_5X5_1000"
        self.charuco_board_size = (9, 9)
        self.charuco_id_map: dict[int, int] | None = None
        self.charuco_id_map_path = ""
        self.intrinsics_distortion_model_name = "OpenCV default"
        self.intrinsics_calibration_flags = INTRINSICS_DISTORTION_OPTIONS[
            self.intrinsics_distortion_model_name
        ]
        self.lidar_center_settings = {
            "source": "Reflective points",
            "fit_mode": "Disk fitting on plane",
            "eps": 0.08,
            "min_samples": 8,
            "reflectivity_threshold": 75.0,
            "plane_distance_threshold": 0.015,
            "disk_radius_m": 0.12,
            "range_xmin": -3.0,
            "range_xmax": 1.0,
            "range_ymin": -6.0,
            "range_ymax": 1.0,
            "range_zmax": 4.0,
            "show_clusters": True,
            "show_centers": True,
        }
        self.projection_overlay_settings = {
            "show": True,
            "source": "Full point cloud",
            "color_by": "depth",
            "point_radius": 2,
            "alpha": 0.85,
            "max_points": 30000,
            "min_depth_m": 0.05,
        }
        self.point_cloud_filter_settings = {
            "enabled": False,
            "xmin": -50.0,
            "xmax": 50.0,
            "ymin": -50.0,
            "ymax": 50.0,
            "zmin": -10.0,
            "zmax": 10.0,
            "distance_min": 0.0,
            "distance_max": 100.0,
        }
        self.cross_view_ray_settings = {
            "enabled": True,
            "color": "#ffff00",
            "width": 3.0,
            "endpoint_size": 20.0,
            "max_pixel_distance": 80.0,
        }
        self._updating_table = False
        self.image_zoom_factor = 1.0
        self.current_image_pixmap: QPixmap | None = None
        self.image_pan_offset = QPoint(0, 0)
        self.image_drag_last_pos: QPoint | None = None
        self.image_click_start_pos: QPoint | None = None
        self.image_display_origin = QPoint(0, 0)
        self.image_display_size = None
        self.displayed_frame_index: int | None = None
        self.selected_image_point_px: tuple[float, float] | None = None
        self.configuration_dialogs: list[QDialog] = []
        self.worker_threads: list[QThread] = []
        self.worker_objects: list[QObject] = []
        self.active_background_task: str | None = None
        self.max_thread_count = CPU_CORE_COUNT
        self.thread_count = self.max_thread_count
        self.solve_pnp_after_calibration = False
        self.run_intrinsics_after_charuco = False
        self.pending_pnp_current_frame_index: int | None = None
        self.saved_calibration_path: str | None = None
        self.optimized_camera_lidar_extrinsic: dict | None = None
        self.workflow_steps = [
            "Load Data",
            "ChArUco",
            "Intrinsics + PnP",
            "LiDAR Center",
            "LiDAR-Cam Extrinsics",
            "Save Calibration",
        ]
        self.step_labels: list[QLabel] = []
        self.step_cards: list[QWidget] = []
        self.step_icon_labels: list[QLabel] = []
        self.step_name_labels: list[QLabel] = []
        self.step_status_labels: list[QLabel] = []
        self.step_connector_labels: list[QLabel] = []
        self.step_run_buttons: list[QPushButton] = []
        self.active_step_index = 0

        # self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) # For rounded window corners

        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset_clicked)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)

        self.frame_counter_label = QLabel("Image 0 / 0")
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_counter_label.setMinimumWidth(120)
        self.frame_counter_label.setStyleSheet("QLabel { color: #111827; font-weight: 600; }")

        self.detect_current_button = QPushButton("Detect Current")
        self.detect_current_button.clicked.connect(self.detect_current_frame)
        self.detect_current_button.setEnabled(False)

        self.detect_all_button = QPushButton("Detect All")
        self.detect_all_button.clicked.connect(lambda _checked=False: self.detect_all_frames())
        self.detect_all_button.setEnabled(False)

        self.run_calibration_button = QPushButton("Run Calibration")
        self.run_calibration_button.clicked.connect(lambda _checked=False: self.run_calibration())
        self.run_calibration_button.setEnabled(False)

        self.solve_pnp_current_button = QPushButton("SolvePnP Current")
        self.solve_pnp_current_button.clicked.connect(self.solve_pnp_current_frame)
        self.solve_pnp_current_button.setEnabled(False)

        self.solve_pnp_all_button = QPushButton("SolvePnP All")
        self.solve_pnp_all_button.clicked.connect(self.solve_pnp_all_frames)
        self.solve_pnp_all_button.setEnabled(False)

        self.save_intrinsics_button = QPushButton("Save Intrinsics")
        self.save_intrinsics_button.clicked.connect(self.save_intrinsics)
        self.save_intrinsics_button.setEnabled(False)

        self.load_intrinsics_button = QPushButton("Load Intrinsics")
        self.load_intrinsics_button.clicked.connect(self.load_intrinsics)

        for button in (
            self.run_calibration_button,
            self.solve_pnp_current_button,
            self.solve_pnp_all_button,
            self.save_intrinsics_button,
            self.load_intrinsics_button,
        ):
            button.setMinimumWidth(135)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.show_frame)

        self.image_label = QLabel("No dataset loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(240, 180)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.image_label.setStyleSheet("QLabel { background: #202124; color: #f1f3f4; }")
        self.image_label.installEventFilter(self)
        self.image_label.setMouseTracking(True)

        self.draw_markers_checkbox = QCheckBox("Draw markers")
        self.draw_markers_checkbox.setChecked(True)
        self.draw_markers_checkbox.stateChanged.connect(self.refresh_current_frame)

        self.draw_corners_checkbox = QCheckBox("Draw corners")
        self.draw_corners_checkbox.setChecked(True)
        self.draw_corners_checkbox.stateChanged.connect(self.refresh_current_frame)

        self.draw_axes_checkbox = QCheckBox("Draw board axes")
        self.draw_axes_checkbox.setChecked(True)
        self.draw_axes_checkbox.stateChanged.connect(self.refresh_current_frame)

        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 12)
        self.marker_size_spin.setValue(2)
        self.marker_size_spin.valueChanged.connect(self.refresh_current_frame)

        self.corner_size_spin = QSpinBox()
        self.corner_size_spin.setRange(2, 20)
        self.corner_size_spin.setValue(5)
        self.corner_size_spin.valueChanged.connect(self.refresh_current_frame)

        self.axis_length_spin = QSpinBox()
        self.axis_length_spin.setRange(10, 1000)
        self.axis_length_spin.setValue(100)
        self.axis_length_spin.valueChanged.connect(self.refresh_current_frame)

        self.marker_type_combo = QComboBox()
        self.marker_type_combo.addItems(["Outline", "Centers", "Outline + Centers"])
        self.marker_type_combo.currentIndexChanged.connect(self.refresh_current_frame)

        self.corner_type_combo = QComboBox()
        self.corner_type_combo.addItems(["Circle", "Square", "Cross"])
        self.corner_type_combo.currentIndexChanged.connect(self.refresh_current_frame)

        self.show_lidar_projection_checkbox = QPushButton()
        self.show_lidar_projection_checkbox.setCheckable(True)
        self.show_lidar_projection_checkbox.setChecked(
            bool(self.projection_overlay_settings["show"])
        )
        self.update_lidar_projection_button_text()
        self.show_lidar_projection_checkbox.toggled.connect(
            self.update_lidar_projection_button_text
        )
        self.show_lidar_projection_checkbox.toggled.connect(self.refresh_current_frame)

        self.projection_source_combo = QComboBox()
        self.projection_source_combo.addItems(
            ["Full point cloud", "LiDAR-center filtered points", "Detected centers"]
        )
        self.projection_source_combo.setCurrentText(
            self.projection_overlay_settings["source"]
        )
        self.projection_source_combo.currentIndexChanged.connect(self.refresh_current_frame)

        self.projection_color_combo = QComboBox()
        self.projection_color_combo.addItems(["depth", "reflectivity", "height"])
        self.projection_color_combo.setCurrentText(
            self.projection_overlay_settings["color_by"]
        )
        self.projection_color_combo.currentIndexChanged.connect(self.refresh_current_frame)

        self.projection_point_size_spin = QSpinBox()
        self.projection_point_size_spin.setRange(1, 12)
        self.projection_point_size_spin.setValue(
            int(self.projection_overlay_settings["point_radius"])
        )
        self.projection_point_size_spin.valueChanged.connect(self.refresh_current_frame)

        self.projection_alpha_spin = QDoubleSpinBox()
        self.projection_alpha_spin.setRange(0.05, 1.0)
        self.projection_alpha_spin.setSingleStep(0.05)
        self.projection_alpha_spin.setValue(
            float(self.projection_overlay_settings["alpha"])
        )
        self.projection_alpha_spin.valueChanged.connect(self.refresh_current_frame)

        self.projection_max_points_spin = QSpinBox()
        self.projection_max_points_spin.setRange(100, 1000000)
        self.projection_max_points_spin.setSingleStep(1000)
        self.projection_max_points_spin.setValue(
            int(self.projection_overlay_settings["max_points"])
        )
        self.projection_max_points_spin.valueChanged.connect(self.refresh_current_frame)

        self.projection_min_depth_spin = QDoubleSpinBox()
        self.projection_min_depth_spin.setRange(0.01, 100.0)
        self.projection_min_depth_spin.setSingleStep(0.05)
        self.projection_min_depth_spin.setValue(
            float(self.projection_overlay_settings["min_depth_m"])
        )
        self.projection_min_depth_spin.valueChanged.connect(self.refresh_current_frame)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setPlainText("Ready")

        self.frame_table = QTableWidget(0, 8)
        self.frame_table.setHorizontalHeaderLabels(
            ["Use", "Frame", "Image", "LiDAR", "Markers", "Corners", "Valid", "PnP RMSE"]
        )
        self.frame_table.cellClicked.connect(self.table_cell_clicked)
        self.frame_table.itemChanged.connect(self.table_item_changed)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)


        self.point_cloud_viewer = PointCloudViewerWidget(self)
        self.point_cloud_filter_widget = self.build_point_cloud_filter_widget()

        self._build_menu()
        self._build_layout()
        self.update_workflow_stepper(active_step=0)
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        self.load_dataset_action = QAction("Load Dataset", self)
        self.load_dataset_action.triggered.connect(self.load_dataset_clicked)
        file_menu.addAction(self.load_dataset_action)
        self.load_recent_calibration_action = QAction("Load Recent Calibration", self)
        self.load_recent_calibration_action.triggered.connect(self.load_recent_calibration)
        self.load_recent_calibration_action.setEnabled(
            RECENT_CALIBRATION_STATE_PATH.exists()
        )
        file_menu.addAction(self.load_recent_calibration_action)
        file_menu.addSeparator()
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()
        self.load_intrinsics_action = QAction("Load Intrinsics", self)
        self.load_intrinsics_action.triggered.connect(self.load_intrinsics)
        file_menu.addAction(self.load_intrinsics_action)
        self.save_intrinsics_action = QAction("Save Intrinsics", self)
        self.save_intrinsics_action.triggered.connect(self.save_intrinsics)
        self.save_intrinsics_action.setEnabled(False)
        file_menu.addAction(self.save_intrinsics_action)
        self.save_calibration_action = QAction("Save Calibration", self)
        self.save_calibration_action.triggered.connect(self.save_calibration)
        self.save_calibration_action.setEnabled(False)
        file_menu.addAction(self.save_calibration_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        self.previous_frame_action = QAction("Previous Frame", self)
        self.previous_frame_action.triggered.connect(self.previous_frame)
        self.previous_frame_action.setEnabled(False)
        self.next_frame_action = QAction("Next Frame", self)
        self.next_frame_action.triggered.connect(self.next_frame)
        self.next_frame_action.setEnabled(False)

        self.detect_current_action = QAction("Detect Current Frame", self)
        self.detect_current_action.triggered.connect(self.detect_current_frame)
        self.detect_current_action.setEnabled(False)
        self.detect_all_action = QAction("Detect All Frames", self)
        self.detect_all_action.triggered.connect(lambda _checked=False: self.detect_all_frames())
        self.detect_all_action.setEnabled(False)

        self.run_calibration_action = QAction("Run Calibration", self)
        self.run_calibration_action.triggered.connect(lambda _checked=False: self.run_calibration())
        self.run_calibration_action.setEnabled(False)
        self.solve_pnp_current_action = QAction("SolvePnP Current Frame", self)
        self.solve_pnp_current_action.triggered.connect(self.solve_pnp_current_frame)
        self.solve_pnp_current_action.setEnabled(False)
        self.solve_pnp_all_action = QAction("SolvePnP All Frames", self)
        self.solve_pnp_all_action.triggered.connect(self.solve_pnp_all_frames)
        self.solve_pnp_all_action.setEnabled(False)

        self.view_menu = menu_bar.addMenu("View")

        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_settings_dialog(self):
        dialog = SettingsDialog(
            thread_count=self.thread_count,
            max_thread_count=self.max_thread_count,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.thread_count = dialog.thread_count()
        self.statusBar().showMessage(
            f"Worker thread count set to {self.thread_count}"
        )
        self.append_log(
            f"Worker thread count set to {self.thread_count}/{self.max_thread_count}."
        )
        self.autosave_recent_calibration()

    def show_about_dialog(self):
        QMessageBox.about(
            self,
            "About",
            "LiDAR Camera Calibration\n\n"
            "Tools for reviewing ChArUco detections and calibrating camera intrinsics.",
        )

    def append_log(self, message: str):
        self.log_text.append(message)

    def toggle_log_panel(self, visible: bool):
        self.log_panel.setVisible(visible)

    def build_point_cloud_filter_widget(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(
            "QWidget { color: #111827; }"
            "QGroupBox { color: #111827; font-weight: 700; }"
            "QLabel, QCheckBox { color: #111827; }"
            "QSlider { min-height: 22px; }"
            "QPushButton { color: #111827; }"
        )
        layout = QVBoxLayout()

        filter_group = QGroupBox("Point Cloud Filter")
        filter_layout = QFormLayout()
        self.point_cloud_filter_enabled_checkbox = QCheckBox("Enable filtering")
        self.point_cloud_filter_enabled_checkbox.setChecked(
            bool(self.point_cloud_filter_settings["enabled"])
        )
        self.point_cloud_filter_enabled_checkbox.stateChanged.connect(
            self.apply_point_cloud_filter_from_controls
        )
        filter_layout.addRow(self.point_cloud_filter_enabled_checkbox)

        def make_filter_slider(
            key: str,
            minimum: float,
            maximum: float,
            step: float = 0.1,
        ) -> SliderValueControl:
            slider = SliderValueControl(
                minimum,
                maximum,
                step,
                float(self.point_cloud_filter_settings[key]),
                suffix=" m",
                decimals=1,
            )
            slider.valueChanged.connect(self.apply_point_cloud_filter_from_controls)
            return slider

        self.filter_xmin_spin = make_filter_slider("xmin", -100.0, 100.0)
        self.filter_xmax_spin = make_filter_slider("xmax", -100.0, 100.0)
        self.filter_ymin_spin = make_filter_slider("ymin", -100.0, 100.0)
        self.filter_ymax_spin = make_filter_slider("ymax", -100.0, 100.0)
        self.filter_zmin_spin = make_filter_slider("zmin", -20.0, 20.0)
        self.filter_zmax_spin = make_filter_slider("zmax", -20.0, 20.0)
        self.filter_distance_min_spin = make_filter_slider("distance_min", 0.0, 200.0)
        self.filter_distance_max_spin = make_filter_slider("distance_max", 0.0, 200.0)

        filter_layout.addRow("X min", self.filter_xmin_spin)
        filter_layout.addRow("X max", self.filter_xmax_spin)
        filter_layout.addRow("Y min", self.filter_ymin_spin)
        filter_layout.addRow("Y max", self.filter_ymax_spin)
        filter_layout.addRow("Z min", self.filter_zmin_spin)
        filter_layout.addRow("Z max", self.filter_zmax_spin)
        filter_layout.addRow("Distance min", self.filter_distance_min_spin)
        filter_layout.addRow("Distance max", self.filter_distance_max_spin)

        reset_filter_button = QPushButton("Reset Filter")
        reset_filter_button.clicked.connect(self.reset_point_cloud_filter)
        filter_layout.addRow(reset_filter_button)
        filter_group.setLayout(filter_layout)

        ray_group = QGroupBox("Image Click Ray")
        ray_layout = QFormLayout()
        self.cross_view_ray_enabled_checkbox = QCheckBox("Enable after extrinsics")
        self.cross_view_ray_enabled_checkbox.setChecked(
            bool(self.cross_view_ray_settings["enabled"])
        )
        self.cross_view_ray_enabled_checkbox.stateChanged.connect(
            self.apply_cross_view_ray_settings_from_controls
        )
        ray_layout.addRow(self.cross_view_ray_enabled_checkbox)

        self.ray_color_button = QPushButton(
            self.cross_view_ray_settings["color"].upper()
        )
        self.ray_color_button.clicked.connect(self.choose_cross_view_ray_color)
        self.update_ray_color_button()
        ray_layout.addRow("Ray color", self.ray_color_button)

        self.ray_width_spin = SliderValueControl(
            1.0,
            20.0,
            1.0,
            float(self.cross_view_ray_settings["width"]),
            suffix=" px",
            decimals=0,
        )
        self.ray_width_spin.valueChanged.connect(
            self.apply_cross_view_ray_settings_from_controls
        )
        ray_layout.addRow("Ray width", self.ray_width_spin)

        self.ray_endpoint_size_spin = SliderValueControl(
            4.0,
            50.0,
            1.0,
            float(self.cross_view_ray_settings["endpoint_size"]),
            suffix=" px",
            decimals=0,
        )
        self.ray_endpoint_size_spin.valueChanged.connect(
            self.apply_cross_view_ray_settings_from_controls
        )
        ray_layout.addRow("Endpoint size", self.ray_endpoint_size_spin)

        self.ray_max_pixel_distance_spin = SliderValueControl(
            1.0,
            500.0,
            1.0,
            float(self.cross_view_ray_settings["max_pixel_distance"]),
            suffix=" px",
            decimals=0,
        )
        self.ray_max_pixel_distance_spin.valueChanged.connect(
            self.apply_cross_view_ray_settings_from_controls
        )
        ray_layout.addRow("Pick radius px", self.ray_max_pixel_distance_spin)

        clear_ray_button = QPushButton("Clear Ray")
        clear_ray_button.clicked.connect(self.point_cloud_viewer.clear_pick_ray)
        ray_layout.addRow(clear_ray_button)
        ray_group.setLayout(ray_layout)

        layout.addWidget(filter_group)
        layout.addWidget(ray_group)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def _build_layout(self):
        image_panel = QWidget()
        image_panel.setMinimumSize(320, 240)
        image_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.image_label, stretch=1)
        image_panel.setLayout(image_layout)

        overlay_group = QGroupBox("Overlay")
        overlay_layout = QGridLayout()
        overlay_layout.addWidget(self.draw_markers_checkbox, 0, 0, 1, 2)
        overlay_layout.addWidget(QLabel("Marker type"), 1, 0)
        overlay_layout.addWidget(self.marker_type_combo, 1, 1)
        overlay_layout.addWidget(QLabel("Marker size"), 2, 0)
        overlay_layout.addWidget(self.marker_size_spin, 2, 1)
        overlay_layout.addWidget(self.draw_corners_checkbox, 3, 0, 1, 2)
        overlay_layout.addWidget(QLabel("Corner type"), 4, 0)
        overlay_layout.addWidget(self.corner_type_combo, 4, 1)
        overlay_layout.addWidget(QLabel("Corner size"), 5, 0)
        overlay_layout.addWidget(self.corner_size_spin, 5, 1)
        overlay_layout.addWidget(self.draw_axes_checkbox, 6, 0, 1, 2)
        overlay_layout.addWidget(QLabel("Axis length"), 7, 0)
        overlay_layout.addWidget(self.axis_length_spin, 7, 1)
        overlay_layout.addWidget(self.show_lidar_projection_checkbox, 8, 0, 1, 2)
        overlay_layout.addWidget(QLabel("LiDAR source"), 9, 0)
        overlay_layout.addWidget(self.projection_source_combo, 9, 1)
        overlay_layout.addWidget(QLabel("Color by"), 10, 0)
        overlay_layout.addWidget(self.projection_color_combo, 10, 1)
        overlay_layout.addWidget(QLabel("Point size"), 11, 0)
        overlay_layout.addWidget(self.projection_point_size_spin, 11, 1)
        overlay_layout.addWidget(QLabel("Opacity"), 12, 0)
        overlay_layout.addWidget(self.projection_alpha_spin, 12, 1)
        overlay_layout.addWidget(QLabel("Max points"), 13, 0)
        overlay_layout.addWidget(self.projection_max_points_spin, 13, 1)
        overlay_layout.addWidget(QLabel("Min depth"), 14, 0)
        overlay_layout.addWidget(self.projection_min_depth_spin, 14, 1)
        overlay_group.setLayout(overlay_layout)
        overlay_group.setMinimumWidth(320)
        overlay_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )

        self.viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer_splitter.setHandleWidth(8)
        self.viewer_splitter.addWidget(image_panel)
        self.viewer_splitter.addWidget(self.point_cloud_viewer)
        self.viewer_splitter.setStretchFactor(0, 3)
        self.viewer_splitter.setStretchFactor(1, 3)
        self.viewer_splitter.setCollapsible(0, False)
        self.viewer_splitter.setCollapsible(1, False)
        self.viewer_splitter.setSizes([700, 700])

        frame_controls_panel = QWidget()
        frame_controls_layout = QHBoxLayout()
        frame_controls_layout.setContentsMargins(12, 6, 12, 6)
        frame_controls_layout.setSpacing(8)
        frame_controls_layout.addStretch(1)
        frame_controls_layout.addWidget(self.frame_counter_label)
        frame_controls_layout.addWidget(self.prev_button)
        self.frame_slider.setMinimumWidth(360)
        self.frame_slider.setMaximumWidth(900)
        frame_controls_layout.addWidget(self.frame_slider, stretch=3)
        frame_controls_layout.addWidget(self.next_button)
        frame_controls_layout.addStretch(1)
        frame_controls_panel.setLayout(frame_controls_layout)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout()
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)
        viewer_layout.addWidget(self.viewer_splitter, stretch=1)
        viewer_layout.addWidget(frame_controls_panel)
        viewer_panel.setLayout(viewer_layout)

        self.stepper_widget = self.build_stepper()

        self.log_panel = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Log"))
        log_layout.addWidget(self.log_text)
        self.log_panel.setLayout(log_layout)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(8)
        self.splitter.addWidget(viewer_panel)
        self.splitter.addWidget(self.stepper_widget)
        self.splitter.addWidget(self.log_panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)
        self.splitter.setSizes([620, 150, 140])

        central = QWidget()
        central_layout = QHBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.splitter, stretch=1)
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        point_cloud_control_group = QGroupBox("Point Cloud")
        point_cloud_control_layout = QVBoxLayout()
        point_cloud_control_layout.addWidget(self.point_cloud_viewer.controls_widget)
        point_cloud_control_group.setLayout(point_cloud_control_layout)
        point_cloud_control_group.setMinimumWidth(320)
        point_cloud_control_group.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )

        visualization_widget = QWidget()
        visualization_widget.setStyleSheet(
            "QWidget { color: #111827; }"
            "QGroupBox { color: #111827; font-weight: 700; }"
            "QLabel, QCheckBox { color: #111827; }"
            "QPushButton { color: #111827; }"
            "QComboBox { color: #111827; background: #ffffff; }"
            "QSpinBox, QDoubleSpinBox { color: #111827; background: #ffffff; }"
        )
        visualization_layout = QVBoxLayout()
        visualization_layout.addWidget(overlay_group)
        visualization_layout.addWidget(point_cloud_control_group)
        visualization_layout.addStretch(1)
        visualization_widget.setLayout(visualization_layout)

        self.visualization_dock = QDockWidget("Visualization", self)
        self.visualization_dock.setWidget(visualization_widget)
        self.visualization_dock.setAllowedAreas(Qt.DockWidgetArea.NoDockWidgetArea)
        self.visualization_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.visualization_dock)
        self.visualization_dock.setFloating(True)
        self.visualization_dock.adjustSize()
        self.visualization_dock.hide()
        visualization_action = self.visualization_dock.toggleViewAction()
        visualization_action.setText("Visualization")
        self.view_menu.addAction(visualization_action)

        self.point_cloud_filter_dock = QDockWidget("Point Cloud Filter", self)
        self.point_cloud_filter_dock.setWidget(self.point_cloud_filter_widget)
        self.point_cloud_filter_dock.setAllowedAreas(Qt.DockWidgetArea.NoDockWidgetArea)
        self.point_cloud_filter_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.point_cloud_filter_dock)
        self.point_cloud_filter_dock.setFloating(True)
        self.point_cloud_filter_dock.resize(360, 560)
        self.point_cloud_filter_dock.hide()
        point_filter_action = self.point_cloud_filter_dock.toggleViewAction()
        point_filter_action.setText("Point Cloud Filter")
        self.view_menu.addAction(point_filter_action)

        self.log_action = QAction("Log", self)
        self.log_action.setCheckable(True)
        self.log_action.setChecked(True)
        self.log_action.triggered.connect(self.toggle_log_panel)
        self.view_menu.addAction(self.log_action)

        self.frames_dock = QDockWidget("Frames", self)
        self.frames_dock.setWidget(self.frame_table)
        self.frames_dock.setAllowedAreas(Qt.DockWidgetArea.NoDockWidgetArea)
        self.frames_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.frames_dock)
        self.frames_dock.setFloating(True)
        self.frames_dock.resize(760, 420)
        self.frames_dock.hide()
        frames_action = self.frames_dock.toggleViewAction()
        frames_action.setText("Frames")
        self.view_menu.addAction(frames_action)

    def build_stepper(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("workflowStepper")
        widget.setStyleSheet(
            "#workflowStepper {"
            "border-top: 1px solid #cbd5e1;"
            "border-bottom: 1px solid #cbd5e1;"
            "background: #f1f5f9;"
            "}"
            "QWidget#workflowStepCard {"
            "background: #ffffff;"
            "border: 1px solid #d8dee8;"
            "border-radius: 8px;"
            "}"
            "QWidget#workflowStepCard[state='completed'] {"
            "background: #f0fdf4;"
            "border: 1px solid #86efac;"
            "}"
            "QWidget#workflowStepCard[state='active'] {"
            "background: #eff6ff;"
            "border: 2px solid #2563eb;"
            "}"
            "QWidget#workflowStepCard[state='pending'] {"
            "background: #ffffff;"
            "border: 1px solid #d8dee8;"
            "}"
            "QLabel#stepBadge {"
            "border-radius: 16px;"
            "font-size: 12px;"
            "font-weight: 800;"
            "min-width: 32px;"
            "max-width: 32px;"
            "min-height: 32px;"
            "max-height: 32px;"
            "}"
            "QLabel#stepBadge[state='completed'] {"
            "background: #16a34a;"
            "color: #ffffff;"
            "}"
            "QLabel#stepBadge[state='active'] {"
            "background: #2563eb;"
            "color: #ffffff;"
            "}"
            "QLabel#stepBadge[state='pending'] {"
            "background: #ffffff;"
            "border: 2px solid #cbd5e1;"
            "color: #64748b;"
            "}"
            "QLabel#stepName {"
            "color: #334155;"
            "font-size: 13px;"
            "font-weight: 700;"
            "}"
            "QLabel#stepName[state='completed'] { color: #166534; }"
            "QLabel#stepName[state='active'] { color: #1d4ed8; font-size: 14px; }"
            "QLabel#stepName[state='pending'] { color: #475569; }"
            "QLabel#stepStatus {"
            "color: #64748b;"
            "font-size: 11px;"
            "font-weight: 600;"
            "}"
            "QLabel#stepStatus[state='completed'] { color: #15803d; }"
            "QLabel#stepStatus[state='active'] { color: #1d4ed8; }"
            "QLabel#stepConnector {"
            "background: #cbd5e1;"
            "border-radius: 2px;"
            "min-height: 4px;"
            "max-height: 4px;"
            "}"
            "QLabel#stepConnector[state='completed'] { background: #16a34a; }"
            "QLabel#stepConnector[state='active'] { background: #60a5fa; }"
            "QPushButton#stepRunButton {"
            "border-radius: 6px;"
            "font-size: 12px;"
            "font-weight: 700;"
            "padding: 4px 10px;"
            "min-height: 24px;"
            "}"
            "QPushButton#stepRunButton[state='completed'] {"
            "background: #dcfce7;"
            "color: #166534;"
            "}"
            "QPushButton#stepRunButton[state='active'] {"
            "background: #2563eb;"
            "color: #ffffff;"
            "}"
            "QPushButton#stepRunButton[state='pending'] {"
            "background: #e2e8f0;"
            "color: #475569;"
            "}"
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(8)

        self.step_labels = []
        self.step_cards = []
        self.step_icon_labels = []
        self.step_name_labels = []
        self.step_status_labels = []
        self.step_connector_labels = []
        self.step_run_buttons = []
        step_badges = ["D", "G", "C", "L", "T", "S"]
        for index, step_name in enumerate(self.workflow_steps):
            step_widget = QWidget()
            step_widget.setObjectName("workflowStepCard")
            step_widget.setMinimumWidth(WORKFLOW_STEP_CARD_MIN_WIDTH)
            step_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

            step_layout = QVBoxLayout()
            step_layout.setContentsMargins(10, 8, 10, 8)
            step_layout.setSpacing(8)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(8)

            badge = QLabel(step_badges[index])
            badge.setObjectName("stepBadge")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(1)

            name_label = QLabel(step_name)
            name_label.setObjectName("stepName")
            name_label.setMinimumHeight(18)
            name_label.setMinimumWidth(name_label.sizeHint().width())

            status_label = QLabel("Pending")
            status_label.setObjectName("stepStatus")
            status_label.setMinimumHeight(15)

            text_column.addWidget(name_label)
            text_column.addWidget(status_label)

            top_row.addWidget(badge)
            top_row.addLayout(text_column, stretch=1)

            run_button = QPushButton("Configure")
            run_button.setObjectName("stepRunButton")
            run_button.setMinimumWidth(96)
            run_button.clicked.connect(
                lambda _checked=False, step_index=index: self.open_step_configuration(
                    step_index
                )
            )

            step_layout.addLayout(top_row)
            step_layout.addWidget(run_button)
            step_widget.setLayout(step_layout)

            self.step_cards.append(step_widget)
            self.step_labels.append(name_label)
            self.step_icon_labels.append(badge)
            self.step_name_labels.append(name_label)
            self.step_status_labels.append(status_label)
            self.step_run_buttons.append(run_button)

            steps_layout.addWidget(step_widget, stretch=1)

            if index < len(self.workflow_steps) - 1:
                connector = QLabel()
                connector.setObjectName("stepConnector")
                connector.setFixedWidth(30)
                connector.setSizePolicy(
                    QSizePolicy.Policy.Fixed,
                    QSizePolicy.Policy.Fixed,
                )
                steps_layout.addWidget(
                    connector,
                    alignment=Qt.AlignmentFlag.AlignVCenter,
                )
                self.step_connector_labels.append(connector)

        layout.addLayout(steps_layout)

        widget.setLayout(layout)
        return widget

    def show_configuration_dialog(self, dialog: QDialog):
        dialog.setWindowFlag(Qt.WindowType.Window, True)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.finished.connect(
            lambda _result, active_dialog=dialog: self.forget_configuration_dialog(
                active_dialog
            )
        )
        self.configuration_dialogs.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def forget_configuration_dialog(self, dialog: QDialog):
        if dialog in self.configuration_dialogs:
            self.configuration_dialogs.remove(dialog)

    def open_step_configuration(self, step_index: int):
        if step_index == 0:
            dialog = LoadDataDialog(self.load_dataset_from_paths)
            self.show_configuration_dialog(dialog)
            return
        if step_index == 1:
            dialog = CharucoConfigurationDialog(
                self.charuco_dictionary_name,
                self.charuco_board_size,
                self.charuco_id_map,
                self.charuco_id_map_path,
                self.execute_charuco_detection,
            )
            self.show_configuration_dialog(dialog)
            return
        if step_index == 2:
            dialog = IntrinsicsConfigurationDialog(
                self.intrinsics_distortion_model_name,
                self.execute_intrinsics_calibration,
            )
            self.show_configuration_dialog(dialog)
            return
        if step_index == 3:
            dialog = LidarCenterConfigurationDialog(
                self.lidar_center_settings,
                self.execute_lidar_center_detection,
                self.preview_lidar_center_range_thresholds,
            )
            self.show_configuration_dialog(dialog)
            return
        if step_index == 4:
            self.run_lidar_camera_extrinsics()
            return
        if step_index == 5:
            self.save_calibration()
            return
        dialog = StepConfigurationDialog(self.workflow_steps[step_index])
        self.show_configuration_dialog(dialog)

    def set_step_widget_state(self, widget: QWidget, state: str):
        widget.setProperty("state", state)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def update_workflow_stepper(self, active_step: int | None = None):
        frame_count = len(self.frames)
        completed_steps = set()
        if self.frames:
            completed_steps.add(0)

        valid_charuco_count = sum(
            1
            for frame in self.frames
            if frame.charuco_result is not None
            and frame.charuco_result.get("valid", False)
        )
        if valid_charuco_count:
            completed_steps.add(1)
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            completed_steps.add(2)
        lidar_center_count = sum(
            1
            for frame in self.frames
            if frame.lidar_processing_result
            and len(frame.lidar_processing_result.get("centers_L", [])) > 0
        )
        lidar_distance_pass_count = sum(
            1
            for frame in self.frames
            if frame.lidar_processing_result
            and frame.lidar_processing_result.get("center_distance_check", {}).get(
                "passed", False
            )
        )
        if lidar_center_count:
            completed_steps.add(3)
        extrinsic_count = sum(1 for frame in self.frames if frame.camera_lidar_extrinsic)
        optimized_count = sum(1 for frame in self.frames if frame.camera_lidar_extrinsic and frame.camera_lidar_extrinsic.get("optimized"))
        has_optimized_calibration = (
            optimized_count > 0
            or self.optimized_camera_lidar_extrinsic is not None
        )
        if has_optimized_calibration:
            completed_steps.add(4)
        if self.saved_calibration_path:
            completed_steps.add(5)

        step_statuses = [
            f"{frame_count} frame(s)" if frame_count else "Needs data",
            f"{valid_charuco_count}/{frame_count} valid" if frame_count else "Needs data",
        ]
        if self.intrinsics_rms is not None:
            step_statuses.append(f"RMS {self.intrinsics_rms:.3f}")
        elif valid_charuco_count:
            step_statuses.append("Ready")
        else:
            step_statuses.append("Needs ChArUco")
        if lidar_center_count:
            step_statuses.append(
                f"{lidar_distance_pass_count}/{lidar_center_count} passed"
            )
        elif frame_count:
            step_statuses.append("Ready")
        else:
            step_statuses.append("Needs data")
        if has_optimized_calibration:
            step_statuses.append("Optimized")
        elif extrinsic_count:
            step_statuses.append("Needs optimization")
        elif lidar_distance_pass_count:
            step_statuses.append("Ready")
        else:
            step_statuses.append("Needs centers")

        if self.saved_calibration_path:
            step_statuses.append(Path(self.saved_calibration_path).name)
        elif has_optimized_calibration:
            step_statuses.append("Ready")
        else:
            step_statuses.append("Needs extrinsics")

        if active_step is None:
            active_step = 0
            for index in range(len(self.workflow_steps)):
                if index not in completed_steps:
                    active_step = index
                    break
        self.active_step_index = active_step

        step_badges = ["D", "G", "C", "L", "T", "S"]
        base_step_enabled = [
            True,
            frame_count > 0,
            valid_charuco_count > 0,
            frame_count > 0,
            (
                self.camera_matrix is not None
                and self.dist_coeffs is not None
                and lidar_distance_pass_count > 0
            ),
            (
                self.camera_matrix is not None
                and self.dist_coeffs is not None
                and has_optimized_calibration
            ),
        ]
        step_enabled = [
            (enabled or index in completed_steps)
            and self.active_background_task is None
            for index, enabled in enumerate(base_step_enabled)
        ]

        for index, card in enumerate(self.step_cards):
            if index in completed_steps:
                state = "completed"
            elif index == active_step:
                state = "active"
            else:
                state = "pending"

            badge = self.step_icon_labels[index]
            badge.setText("OK" if state == "completed" else step_badges[index])
            self.step_status_labels[index].setText(step_statuses[index])

            button = self.step_run_buttons[index]
            if index == 5:
                button.setText("Save")
            elif index == 4:
                button.setText("Run")
            elif state == "completed":
                button.setText("Review")
            else:
                button.setText("Configure")
            button.setEnabled(step_enabled[index])

            self.set_step_widget_state(card, state)
            self.set_step_widget_state(badge, state)
            self.set_step_widget_state(self.step_name_labels[index], state)
            self.set_step_widget_state(self.step_status_labels[index], state)
            self.set_step_widget_state(button, state)

        for index, connector in enumerate(self.step_connector_labels):
            if index in completed_steps and index + 1 in completed_steps:
                connector_state = "completed"
            elif index in completed_steps and index + 1 == active_step:
                connector_state = "active"
            else:
                connector_state = "pending"
            self.set_step_widget_state(connector, connector_state)

    def start_worker_thread(self, worker: QObject, terminal_signals: list):
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        for signal in terminal_signals:
            signal.connect(worker.deleteLater)
            signal.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda worker_ref=worker, thread_ref=thread: self.forget_worker_thread(
                thread_ref,
                worker_ref,
            )
        )
        self.worker_threads.append(thread)
        self.worker_objects.append(worker)
        thread.start()

    def forget_worker_thread(self, thread: QThread, worker: QObject | None = None):
        self.worker_threads = [
            thread_item for thread_item in self.worker_threads
            if thread_item is not thread
        ]
        if worker is not None:
            self.worker_objects = [
                worker_item for worker_item in self.worker_objects
                if worker_item is not worker
            ]

    def set_background_task(self, task_name: str | None):
        self.active_background_task = task_name
        busy = task_name is not None
        has_frames = bool(self.frames)
        has_intrinsics = self.camera_matrix is not None and self.dist_coeffs is not None
        has_optimized_extrinsics = any(
            frame.camera_lidar_extrinsic
            and frame.camera_lidar_extrinsic.get("optimized")
            for frame in self.frames
        ) or self.optimized_camera_lidar_extrinsic is not None

        self.run_calibration_button.setEnabled(has_frames and not busy)
        self.run_calibration_action.setEnabled(has_frames and not busy)
        self.detect_current_button.setEnabled(has_frames and not busy)
        self.detect_all_button.setEnabled(has_frames and not busy)
        self.detect_current_action.setEnabled(has_frames and not busy)
        self.detect_all_action.setEnabled(has_frames and not busy)
        self.solve_pnp_current_button.setEnabled(has_frames and has_intrinsics and not busy)
        self.solve_pnp_all_button.setEnabled(has_frames and has_intrinsics and not busy)
        self.solve_pnp_current_action.setEnabled(has_frames and has_intrinsics and not busy)
        self.solve_pnp_all_action.setEnabled(has_frames and has_intrinsics and not busy)
        self.save_intrinsics_button.setEnabled(has_intrinsics and not busy)
        self.save_intrinsics_action.setEnabled(has_intrinsics and not busy)
        self.save_calibration_action.setEnabled(
            has_intrinsics and has_optimized_extrinsics and not busy
        )
        self.update_workflow_stepper(active_step=self.active_step_index)

    def finish_background_task(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        self.set_background_task(None)

    def recent_calibration_state(self) -> dict:
        return {
            "version": 1,
            "thread_count": self.thread_count,
            "charuco_dictionary_name": self.charuco_dictionary_name,
            "charuco_board_size": self.charuco_board_size,
            "charuco_id_map": self.charuco_id_map,
            "charuco_id_map_path": self.charuco_id_map_path,
            "intrinsics_distortion_model_name": self.intrinsics_distortion_model_name,
            "intrinsics_calibration_flags": self.intrinsics_calibration_flags,
            "lidar_center_settings": self.lidar_center_settings,
            "projection_overlay_settings": self.current_projection_overlay_settings(),
            "point_cloud_filter_settings": self.current_point_cloud_filter_settings(),
            "cross_view_ray_settings": self.current_cross_view_ray_settings(),
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "intrinsics_rms": self.intrinsics_rms,
            "intrinsics_frame_indices": self.intrinsics_frame_indices,
            "excluded_intrinsics_frame_indices": self.excluded_intrinsics_frame_indices,
            "saved_calibration_path": self.saved_calibration_path,
            "optimized_camera_lidar_extrinsic": self.optimized_camera_lidar_extrinsic,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "image_path": str(frame.image_path),
                    "lidar_path": str(frame.lidar_path),
                    "charuco_result": frame.charuco_result,
                    "camera_board_pose": frame.camera_board_pose,
                    "lidar_processing_result": frame.lidar_processing_result,
                    "lidar_board_pose": frame.lidar_board_pose,
                    "camera_lidar_extrinsic": frame.camera_lidar_extrinsic,
                    "projection_result": frame.projection_result,
                    "enabled": frame.enabled,
                }
                for frame in self.frames
            ],
        }

    def autosave_recent_calibration(self):
        if not self.frames:
            return
        try:
            RECENT_CALIBRATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with RECENT_CALIBRATION_STATE_PATH.open("wb") as file:
                pickle.dump(self.recent_calibration_state(), file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            self.append_log(f"Recent calibration autosave failed: {exc}")
            return

        if hasattr(self, "load_recent_calibration_action"):
            self.load_recent_calibration_action.setEnabled(True)

    def load_recent_calibration(self):
        if not RECENT_CALIBRATION_STATE_PATH.exists():
            QMessageBox.information(
                self,
                "No Recent Calibration",
                "No recent calibration state has been saved yet.",
            )
            return

        try:
            with RECENT_CALIBRATION_STATE_PATH.open("rb") as file:
                state = pickle.load(file)
            self.restore_recent_calibration_state(state)
        except Exception as exc:
            QMessageBox.critical(self, "Load Recent Calibration Failed", str(exc))
            self.append_log(f"Load recent calibration failed: {exc}")
            return

        self.statusBar().showMessage(
            f"Loaded recent calibration: {RECENT_CALIBRATION_STATE_PATH}"
        )
        self.append_log(f"Loaded recent calibration: {RECENT_CALIBRATION_STATE_PATH}")

    def restore_recent_calibration_state(self, state: dict):
        frame_states = state.get("frames", [])
        if not frame_states:
            raise RuntimeError("Recent calibration state does not contain any frames.")

        frames = []
        for frame_state in frame_states:
            image_path = Path(frame_state["image_path"])
            lidar_path = Path(frame_state["lidar_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            if not lidar_path.exists():
                raise FileNotFoundError(f"LiDAR file does not exist: {lidar_path}")

            frames.append(
                FrameRecord(
                    frame_index=int(frame_state["frame_index"]),
                    image_path=image_path,
                    lidar_path=lidar_path,
                    charuco_result=frame_state.get("charuco_result"),
                    camera_board_pose=frame_state.get("camera_board_pose"),
                    lidar_processing_result=frame_state.get("lidar_processing_result"),
                    lidar_board_pose=frame_state.get("lidar_board_pose"),
                    camera_lidar_extrinsic=frame_state.get("camera_lidar_extrinsic"),
                    projection_result=frame_state.get("projection_result"),
                    enabled=bool(frame_state.get("enabled", True)),
                )
            )

        self.frames = frames
        self.thread_count = clamp_thread_count(
            state.get("thread_count", self.thread_count)
        )
        self.charuco_dictionary_name = state.get(
            "charuco_dictionary_name",
            self.charuco_dictionary_name,
        )
        self.charuco_board_size = tuple(
            state.get("charuco_board_size", self.charuco_board_size)
        )
        self.charuco_id_map = state.get("charuco_id_map")
        self.charuco_id_map_path = state.get("charuco_id_map_path", "")
        self.intrinsics_distortion_model_name = state.get(
            "intrinsics_distortion_model_name",
            self.intrinsics_distortion_model_name,
        )
        self.intrinsics_calibration_flags = int(
            state.get("intrinsics_calibration_flags", self.intrinsics_calibration_flags)
        )
        self.lidar_center_settings = state.get(
            "lidar_center_settings",
            self.lidar_center_settings,
        )
        self.projection_overlay_settings = state.get(
            "projection_overlay_settings",
            self.projection_overlay_settings,
        )
        self.update_projection_overlay_controls()
        self.point_cloud_filter_settings = state.get(
            "point_cloud_filter_settings",
            self.point_cloud_filter_settings,
        )
        self.update_point_cloud_filter_controls()
        self.cross_view_ray_settings = state.get(
            "cross_view_ray_settings",
            self.cross_view_ray_settings,
        )
        self.update_cross_view_ray_controls()
        self.camera_matrix = state.get("camera_matrix")
        self.dist_coeffs = state.get("dist_coeffs")
        self.intrinsics_rms = state.get("intrinsics_rms")
        self.intrinsics_frame_indices = list(state.get("intrinsics_frame_indices", []))
        self.excluded_intrinsics_frame_indices = list(
            state.get("excluded_intrinsics_frame_indices", [])
        )
        saved_calibration_path = state.get("saved_calibration_path")
        self.saved_calibration_path = (
            saved_calibration_path
            if saved_calibration_path and Path(saved_calibration_path).exists()
            else None
        )
        self.optimized_camera_lidar_extrinsic = state.get(
            "optimized_camera_lidar_extrinsic"
        )
        if self.optimized_camera_lidar_extrinsic is None:
            for frame in frames:
                ext = frame.camera_lidar_extrinsic
                if ext and ext.get("optimized") and "T_CL" in ext:
                    self.optimized_camera_lidar_extrinsic = {
                        "T_CL": ext["T_CL"],
                        "T_LC": ext.get("T_LC", invert_transform(ext["T_CL"])),
                        "opt_rmse": ext.get("opt_rmse"),
                    }
                    break

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, len(self.frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(bool(self.frames))
        self.frame_slider.blockSignals(False)

        has_frames = bool(self.frames)
        has_intrinsics = self.camera_matrix is not None and self.dist_coeffs is not None
        has_optimized_extrinsics = any(
            frame.camera_lidar_extrinsic
            and frame.camera_lidar_extrinsic.get("optimized")
            for frame in self.frames
        ) or self.optimized_camera_lidar_extrinsic is not None
        self.prev_button.setEnabled(has_frames)
        self.next_button.setEnabled(has_frames)
        self.detect_current_button.setEnabled(has_frames)
        self.detect_all_button.setEnabled(has_frames)
        self.run_calibration_button.setEnabled(has_frames)
        self.previous_frame_action.setEnabled(has_frames)
        self.next_frame_action.setEnabled(has_frames)
        self.detect_current_action.setEnabled(has_frames)
        self.detect_all_action.setEnabled(has_frames)
        self.run_calibration_action.setEnabled(has_frames)
        self.save_intrinsics_button.setEnabled(has_intrinsics)
        self.save_intrinsics_action.setEnabled(has_intrinsics)
        self.save_calibration_action.setEnabled(
            has_intrinsics and has_optimized_extrinsics
        )
        self.solve_pnp_current_button.setEnabled(has_intrinsics and has_frames)
        self.solve_pnp_all_button.setEnabled(has_intrinsics and has_frames)
        self.solve_pnp_current_action.setEnabled(has_intrinsics and has_frames)
        self.solve_pnp_all_action.setEnabled(has_intrinsics and has_frames)

        self.log_text.setPlainText("Ready")
        self.update_frame_table()
        self.update_info()
        if has_intrinsics:
            self.update_intrinsics_display()
        self.update_workflow_stepper(active_step=None)
        self.show_frame(0)

    def load_dataset_clicked(self):
        image_dir = self.select_directory("Select Image Directory")
        if not image_dir:
            return

        lidar_dir = self.select_directory("Select LiDAR Directory")
        if not lidar_dir:
            return

        self.load_dataset_from_paths(image_dir, lidar_dir)

    def load_dataset_from_paths(self, image_dir: str, lidar_dir: str) -> bool:
        try:
            self.frames = load_dataset(
                image_dir=image_dir,
                lidar_dir=lidar_dir,
                image_suffix=SUPPORTED_IMAGE_SUFFIXES,
                lidar_suffix=SUPPORTED_LIDAR_SUFFIXES,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Dataset Load Failed", str(exc))
            self.statusBar().showMessage("Dataset load failed")
            self.append_log(f"Dataset load failed: {exc}")
            return False

        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics_rms = None
        self.intrinsics_frame_indices = []
        self.excluded_intrinsics_frame_indices = []
        self.saved_calibration_path = None
        self.optimized_camera_lidar_extrinsic = None
        self.image_zoom_factor = 1.0
        self.image_pan_offset = QPoint(0, 0)
        self.image_drag_last_pos = None
        self.image_click_start_pos = None
        self.displayed_frame_index = None
        self.selected_image_point_px = None
        self.log_text.setPlainText("Ready")
        self.append_log(f"Loaded dataset with {len(self.frames)} frame(s).")
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        self.save_intrinsics_button.setEnabled(False)
        self.save_intrinsics_action.setEnabled(False)
        self.save_calibration_action.setEnabled(False)
        self.solve_pnp_current_button.setEnabled(False)
        self.solve_pnp_all_button.setEnabled(False)
        self.solve_pnp_current_action.setEnabled(False)
        self.solve_pnp_all_action.setEnabled(False)

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, len(self.frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(bool(self.frames))
        self.frame_slider.blockSignals(False)

        has_frames = bool(self.frames)
        self.prev_button.setEnabled(has_frames)
        self.next_button.setEnabled(has_frames)
        self.detect_current_button.setEnabled(has_frames)
        self.detect_all_button.setEnabled(has_frames)
        self.run_calibration_button.setEnabled(has_frames)
        self.previous_frame_action.setEnabled(has_frames)
        self.next_frame_action.setEnabled(has_frames)
        self.detect_current_action.setEnabled(has_frames)
        self.detect_all_action.setEnabled(has_frames)
        self.run_calibration_action.setEnabled(has_frames)

        self.update_frame_table()
        self.update_info()
        self.update_workflow_stepper(active_step=1)
        self.statusBar().showMessage(f"Loaded {len(self.frames)} frame(s)")
        self.show_frame(0)
        self.autosave_recent_calibration()
        return True

    def select_directory(self, title: str) -> str:
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.ViewMode.List)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                return selected_paths[0]
        return ""

    def invalidate_lidar_camera_calibration(self):
        for frame in self.frames:
            frame.lidar_board_pose = None
            frame.camera_lidar_extrinsic = None
            frame.projection_result = None
        self.optimized_camera_lidar_extrinsic = None
        self.saved_calibration_path = None
        self.selected_image_point_px = None
        if hasattr(self, "save_calibration_action"):
            self.save_calibration_action.setEnabled(False)

    def previous_frame(self):
        self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))

    def next_frame(self):
        self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def update_frame_counter(self, frame_index: int | None = None):
        if not self.frames:
            self.frame_counter_label.setText("Image 0 / 0")
            return
        if frame_index is None:
            frame_index = self.frame_slider.value()
        self.frame_counter_label.setText(
            f"Image {frame_index + 1} / {len(self.frames)}"
        )

    def table_cell_clicked(self, row: int, _column: int):
        if self.frames and 0 <= row < len(self.frames):
            self.frame_slider.setValue(row)

    def table_item_changed(self, item: QTableWidgetItem):
        if self._updating_table or item.column() != 0:
            return
        row = item.row()
        if 0 <= row < len(self.frames):
            self.frames[row].enabled = item.checkState() == Qt.CheckState.Checked
            self.update_info()
            self.autosave_recent_calibration()

    def detect_current_frame(self):
        if not self.frames:
            return
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return
        frame = self.frames[self.frame_slider.value()]
        try:
            image = self.load_frame_image(frame)
            frame.charuco_result = detect_charuco_frame(
                image,
                dictionary_name=self.charuco_dictionary_name,
                board_size=self.charuco_board_size,
                id_map=self.charuco_id_map,
            )
            frame.camera_board_pose = None
            self.invalidate_lidar_camera_calibration()
        except Exception as exc:
            frame.charuco_result = {
                "success": False,
                "valid": False,
                "num_markers": 0,
                "num_corners": 0,
                "reason": str(exc),
            }
            frame.camera_board_pose = None
            self.invalidate_lidar_camera_calibration()
            self.append_log(f"Detect current failed for frame {frame.frame_index}: {exc}")
        self.update_frame_table()
        self.update_info()
        self.update_workflow_stepper(active_step=2)
        self.show_frame(frame.frame_index)
        self.autosave_recent_calibration()

    def execute_charuco_detection(
        self,
        dictionary_name: str,
        board_size: tuple[int, int],
        id_map: dict[int, int] | None,
        id_map_path: str,
    ):
        self.charuco_dictionary_name = dictionary_name
        self.charuco_board_size = board_size
        self.charuco_id_map = id_map
        self.charuco_id_map_path = id_map_path
        self.draw_markers_checkbox.setChecked(True)
        self.draw_corners_checkbox.setChecked(True)
        self.detect_all_frames(
            dictionary_name=dictionary_name,
            board_size=board_size,
            id_map=id_map,
        )

    def detect_all_frames(
        self,
        dictionary_name: str | None = None,
        board_size: tuple[int, int] | None = None,
        id_map: dict[int, int] | None = None,
    ):
        if not self.frames:
            return
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return
        if dictionary_name is None:
            dictionary_name = self.charuco_dictionary_name
        else:
            self.charuco_dictionary_name = dictionary_name
        if board_size is None:
            board_size = self.charuco_board_size
        else:
            self.charuco_board_size = board_size
        if id_map is None:
            id_map = self.charuco_id_map
        else:
            self.charuco_id_map = id_map

        frame_data = [
            {
                "frame_index": frame.frame_index,
                "image_path": frame.image_path,
            }
            for frame in self.frames
        ]
        self.progress_bar.setRange(0, len(self.frames))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("ChArUco detection (%p%)")
        self.progress_bar.setVisible(True)
        self.set_background_task("ChArUco detection")
        self.update_workflow_stepper(active_step=1)
        self.statusBar().showMessage(
            f"Detecting ChArUco on all frames with {dictionary_name}..."
        )

        worker = CharucoDetectionWorker(
            frame_data,
            dictionary_name=dictionary_name,
            board_size=board_size,
            id_map=id_map,
            thread_count=self.thread_count,
        )
        worker.progress.connect(self.handle_charuco_detection_progress)
        worker.finished.connect(self.handle_charuco_detection_finished)
        self.start_worker_thread(worker, [worker.finished])

    def handle_charuco_detection_progress(
        self,
        value: int,
        maximum: int,
        status_message: str,
    ):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat("ChArUco detection (%p%)")
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage(status_message)

    def handle_charuco_detection_finished(self, result_summary: dict):
        frame_by_index = {frame.frame_index: frame for frame in self.frames}
        for frame_index, result in result_summary["results"]:
            frame = frame_by_index.get(frame_index)
            if frame is None:
                continue
            frame.charuco_result = result
            frame.camera_board_pose = None
            self.invalidate_lidar_camera_calibration()

        self.finish_background_task()
        self.update_frame_table()
        self.update_info()
        self.update_workflow_stepper(active_step=2)
        self.show_frame(self.frame_slider.value())
        self.statusBar().showMessage("ChArUco detection complete")
        id_map_text = self.charuco_id_map_path or "default ID map"
        board_size = tuple(result_summary["board_size"])
        self.append_log(
            "ChArUco detection complete with "
            f"{result_summary['dictionary_name']}, board "
            f"{board_size[0]}x{board_size[1]}, {id_map_text}; "
            f"{result_summary['valid_count']}/{result_summary['total']} valid."
        )
        self.autosave_recent_calibration()
        if self.run_intrinsics_after_charuco:
            self.run_intrinsics_after_charuco = False
            self.run_calibration(
                calibration_flags=self.intrinsics_calibration_flags,
                solve_pnp_after=True,
            )

    def execute_intrinsics_calibration(self, distortion_model_name: str):
        self.intrinsics_distortion_model_name = distortion_model_name
        self.intrinsics_calibration_flags = INTRINSICS_DISTORTION_OPTIONS[
            distortion_model_name
        ]
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return
        if not self.frames:
            QMessageBox.warning(self, "No Dataset", "Load a dataset before calibrating intrinsics.")
            return
        if not any(frame.charuco_result is not None for frame in self.frames):
            self.run_intrinsics_after_charuco = True
            self.detect_all_frames()
            return
        else:
            self.append_log("Using existing ChArUco detections for intrinsics calibration.")
        self.run_calibration(
            calibration_flags=self.intrinsics_calibration_flags,
            solve_pnp_after=True,
        )

    def calibration_frames(self) -> list[FrameRecord]:
        return [
            frame for frame in self.frames
            if (
                frame.enabled
                and frame.charuco_result is not None
                and frame.charuco_result.get("valid", False)
            )
        ]

    def run_calibration(
        self,
        calibration_flags: int | None = None,
        solve_pnp_after: bool = False,
    ):
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return
        if calibration_flags is None:
            calibration_flags = self.intrinsics_calibration_flags
        selected_frames = self.calibration_frames()
        if len(selected_frames) < 3:
            QMessageBox.warning(
                self,
                "Not Enough Frames",
                "Need at least 3 enabled valid ChArUco frames to calibrate.",
            )
            return

        frame_data = [
            {
                "frame_index": frame.frame_index,
                "image_path": frame.image_path,
                "charuco_result": frame.charuco_result,
            }
            for frame in selected_frames
        ]
        total_steps = len(selected_frames) + 1

        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Preparing calibration (%p%)")
        self.progress_bar.setVisible(True)
        self.solve_pnp_after_calibration = solve_pnp_after
        self.set_background_task("Camera calibration")
        self.update_workflow_stepper(active_step=2)
        self.statusBar().showMessage("Running calibration...")

        worker = IntrinsicsCalibrationWorker(
            frame_data,
            calibration_flags,
            thread_count=self.thread_count,
        )
        worker.progress.connect(self.handle_calibration_progress)
        worker.finished.connect(self.handle_calibration_finished)
        worker.failed.connect(self.handle_calibration_failed)
        self.start_worker_thread(worker, [worker.finished, worker.failed])

    def handle_calibration_progress(
        self,
        value: int,
        maximum: int,
        progress_format: str,
        status_message: str,
    ):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(progress_format)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage(status_message)

    def handle_calibration_failed(self, message: str):
        QMessageBox.critical(self, "Calibration Failed", message)
        self.statusBar().showMessage("Calibration failed")
        self.append_log(f"Calibration failed: {message}")
        self.solve_pnp_after_calibration = False
        self.finish_background_task()

    def handle_calibration_finished(self, result: dict):
        self.progress_bar.setValue(int(result["total_steps"]))

        self.intrinsics_rms = float(result["rms"])
        self.camera_matrix = result["camera_matrix"]
        self.dist_coeffs = result["dist_coeffs"]
        self.saved_calibration_path = None
        self.optimized_camera_lidar_extrinsic = None
        self.intrinsics_frame_indices = list(result["frame_indices"])
        used_indices = set(self.intrinsics_frame_indices)
        self.excluded_intrinsics_frame_indices = [
            frame.frame_index for frame in self.frames
            if frame.frame_index not in used_indices
        ]
        for frame in self.frames:
            frame.camera_board_pose = None
            frame.lidar_board_pose = None
            frame.camera_lidar_extrinsic = None
        self.save_intrinsics_button.setEnabled(True)
        self.save_intrinsics_action.setEnabled(True)
        self.save_calibration_action.setEnabled(False)
        self.solve_pnp_current_button.setEnabled(bool(self.frames))
        self.solve_pnp_all_button.setEnabled(bool(self.frames))
        self.solve_pnp_current_action.setEnabled(bool(self.frames))
        self.solve_pnp_all_action.setEnabled(bool(self.frames))
        self.update_intrinsics_display()
        self.update_frame_table()
        self.update_workflow_stepper(active_step=3)
        self.statusBar().showMessage(
            f"Calibration complete | RMS {self.intrinsics_rms:.4f} px | "
            f"{len(self.intrinsics_frame_indices)} frame(s)"
        )
        self.append_log(
            f"Intrinsics calibration complete with {self.intrinsics_distortion_model_name}."
        )
        self.autosave_recent_calibration()
        solve_pnp_after = self.solve_pnp_after_calibration
        self.solve_pnp_after_calibration = False
        self.finish_background_task()
        if solve_pnp_after:
            self.solve_pnp_all_frames()

    @staticmethod
    def _calibration_object_points(result: dict, frame_index: int) -> np.ndarray:
        return calibration_object_points(result, frame_index)

    @staticmethod
    def _calibration_image_points(result: dict, frame_index: int) -> np.ndarray:
        return calibration_image_points(result, frame_index)

    def solve_pnp_current_frame(self):
        if not self.frames:
            return
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return
        frame = self.frames[self.frame_slider.value()]
        self.start_solve_pnp_worker([frame], current_frame_index=frame.frame_index)

    def solve_pnp_all_frames(self):
        if not self.frames:
            return
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            QMessageBox.warning(self, "No Intrinsics", "Run or load intrinsics first.")
            return

        candidates = [
            frame for frame in self.frames
            if frame.charuco_result is not None and frame.charuco_result.get("valid", False)
        ]
        if not candidates:
            QMessageBox.warning(
                self,
                "No Valid Frames",
                "Detect ChArUco corners before running solvePnP.",
            )
            return

        self.start_solve_pnp_worker(candidates)

    def start_solve_pnp_worker(
        self,
        candidates: list[FrameRecord],
        current_frame_index: int | None = None,
    ):
        if self.camera_matrix is None or self.dist_coeffs is None:
            QMessageBox.warning(self, "No Intrinsics", "Run or load intrinsics first.")
            return

        frame_data = [
            {
                "frame_index": frame.frame_index,
                "charuco_result": frame.charuco_result,
            }
            for frame in candidates
        ]
        if not frame_data:
            return

        self.progress_bar.setRange(0, len(candidates))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("SolvePnP (%p%)")
        self.progress_bar.setVisible(True)
        self.pending_pnp_current_frame_index = current_frame_index
        self.set_background_task("SolvePnP")
        self.statusBar().showMessage("Running SolvePnP...")

        worker = SolvePnPWorker(
            frame_data,
            self.camera_matrix,
            self.dist_coeffs,
            thread_count=self.thread_count,
        )
        worker.progress.connect(self.handle_solve_pnp_progress)
        worker.finished.connect(self.handle_solve_pnp_finished)
        self.start_worker_thread(worker, [worker.finished])

    def handle_solve_pnp_progress(
        self,
        value: int,
        maximum: int,
        status_message: str,
    ):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat("SolvePnP (%p%)")
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage(status_message)

    def handle_solve_pnp_finished(self, result: dict):
        frame_by_index = {frame.frame_index: frame for frame in self.frames}
        for frame_index, pose in result["poses"]:
            frame = frame_by_index.get(frame_index)
            if frame is not None:
                frame.camera_board_pose = pose
                self.invalidate_lidar_camera_calibration()

        failed = list(result["failed"])
        current_frame_index = self.pending_pnp_current_frame_index
        self.pending_pnp_current_frame_index = None

        self.finish_background_task()
        self.update_frame_table()
        self.update_info()
        self.show_frame(self.frame_slider.value())

        if current_frame_index is not None:
            frame = frame_by_index.get(current_frame_index)
            if frame is not None and frame.camera_board_pose:
                if frame.camera_board_pose.get("success", False):
                    self.update_pose_display(frame)
                    self.statusBar().showMessage(
                        f"SolvePnP frame {frame.frame_index}: "
                        f"{frame.camera_board_pose['reprojection_rmse_px']:.4f} px"
                    )
                else:
                    QMessageBox.critical(
                        self,
                        "SolvePnP Failed",
                        frame.camera_board_pose.get("reason", "solvePnP failed."),
                    )
                    self.statusBar().showMessage("SolvePnP failed")
        elif failed:
            self.statusBar().showMessage(f"SolvePnP complete with failures: {failed}")
            self.append_log(f"SolvePnP complete with failures: {failed}")
        else:
            self.statusBar().showMessage("SolvePnP complete")
            self.append_log(f"SolvePnP complete for {result['total']} frame(s).")
        self.autosave_recent_calibration()

    def solve_pnp_for_frame(self, frame: FrameRecord):
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Run or load intrinsics before solvePnP.")
        if frame.charuco_result is None or not frame.charuco_result.get("valid", False):
            raise RuntimeError(f"Frame {frame.frame_index} does not have a valid ChArUco detection.")

        frame.camera_board_pose = estimate_charuco_pose_from_detection(
            frame.charuco_result,
            self.camera_matrix,
            self.dist_coeffs,
        )
        if not frame.camera_board_pose.get("success", False):
            raise RuntimeError(
                frame.camera_board_pose.get("reason", "solvePnP failed.")
            )
        return frame.camera_board_pose

    def update_pose_display(self, frame: FrameRecord):
        pose = frame.camera_board_pose
        if not pose:
            return
        if not pose.get("success", False):
            self.append_log(f"SolvePnP failed: {pose.get('reason', 'Unknown error')}")
            return

        T_CB = pose["T_CB"]
        tvec = pose["tvec"].reshape(-1)
        t_text = np.array2string(tvec, precision=6, suppress_small=True)
        T_text = np.array2string(T_CB, precision=6, suppress_small=True)
        self.append_log(
            f"Frame: {frame.frame_index}\n"
            f"Reprojection RMSE: {pose['reprojection_rmse_px']:.6f} px\n"
            f"Points: {pose.get('num_points', 'N/A')}\n"
            f"t_CB: {t_text}\n\n"
            f"T_CB:\n{T_text}"
        )

    def update_intrinsics_display(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.append_log("No intrinsics loaded.")
            return

        k_text = np.array2string(
            self.camera_matrix,
            precision=6,
            suppress_small=True,
        )
        dist_text = np.array2string(
            self.dist_coeffs.reshape(-1),
            precision=6,
            suppress_small=True,
        )
        rms_text = "N/A" if self.intrinsics_rms is None else f"{self.intrinsics_rms:.6f} px"
        excluded_text = (
            ", ".join(str(index) for index in self.excluded_intrinsics_frame_indices)
            or "None"
        )
        self.append_log(
            f"RMS: {rms_text}\n"
            f"Excluded frames: {excluded_text}\n\n"
            f"K:\n{k_text}\n\n"
            f"Distortion:\n{dist_text}"
        )

    def save_intrinsics(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            QMessageBox.warning(self, "No Intrinsics", "Run or load intrinsics first.")
            return

        output_path = self.select_save_file(
            "Save Intrinsics",
            "NumPy archive (*.npz)",
            "camera_intrinsics.npz",
        )
        if not output_path:
            return

        output_path = str(output_path)
        if not output_path.lower().endswith(".npz"):
            output_path += ".npz"

        try:
            np.savez(
                output_path,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                rms=np.asarray(self.intrinsics_rms if self.intrinsics_rms is not None else np.nan),
                frame_indices=np.asarray(self.intrinsics_frame_indices, dtype=np.int32),
                excluded_frame_indices=np.asarray(
                    self.excluded_intrinsics_frame_indices,
                    dtype=np.int32,
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            self.append_log(f"Save intrinsics failed: {exc}")
            return

        self.statusBar().showMessage(f"Saved intrinsics: {output_path}")
        self.append_log(f"Saved intrinsics: {output_path}")
        self.autosave_recent_calibration()

    def optimized_extrinsic_frames(self) -> list[FrameRecord]:
        return [
            frame for frame in self.frames
            if frame.camera_lidar_extrinsic
            and frame.camera_lidar_extrinsic.get("optimized")
        ]

    @staticmethod
    def jsonable_array(value) -> list:
        return np.asarray(value, dtype=np.float64).tolist()

    @staticmethod
    def jsonable_float(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def np_float_or_nan(value) -> float:
        number = MainWindow.jsonable_float(value)
        return np.nan if number is None else number

    def calibration_export_payload(self) -> dict:
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Run or load intrinsics first.")

        extrinsic_frames = self.optimized_extrinsic_frames()
        if not extrinsic_frames:
            raise RuntimeError("Run nonlinear optimization before saving calibration.")

        final_extrinsic = extrinsic_frames[0].camera_lidar_extrinsic
        T_CL = np.asarray(final_extrinsic["T_CL"], dtype=np.float64).reshape(4, 4)
        T_LC = np.asarray(final_extrinsic["T_LC"], dtype=np.float64).reshape(4, 4)

        frames = []
        for frame in extrinsic_frames:
            ext = frame.camera_lidar_extrinsic
            per_frame_T_CL = np.asarray(
                ext.get("per_frame_T_CL", ext["T_CL"]),
                dtype=np.float64,
            ).reshape(4, 4)
            frames.append({
                "frame_index": int(frame.frame_index),
                "image_path": str(frame.image_path),
                "lidar_path": str(frame.lidar_path),
                "symmetry_name": ext.get("symmetry_name", ""),
                "normal_sign": self.jsonable_float(ext.get("normal_sign")),
                "consistency_score": self.jsonable_float(ext.get("consistency_score")),
                "rotation_error_deg": self.jsonable_float(ext.get("rotation_error_deg")),
                "translation_error_m": self.jsonable_float(ext.get("translation_error_m")),
                "lidar_board_rmse_m": self.jsonable_float(ext.get("lidar_board_rmse_m")),
                "opt_rmse_m": self.jsonable_float(ext.get("opt_rmse")),
                "is_reference": bool(ext.get("is_reference", False)),
                "T_CL": self.jsonable_array(ext["T_CL"]),
                "T_LC": self.jsonable_array(ext["T_LC"]),
                "per_frame_T_CL": self.jsonable_array(per_frame_T_CL),
                "per_frame_T_LC": self.jsonable_array(invert_transform(per_frame_T_CL)),
            })

        return {
            "version": 1,
            "format": "lidar_camera_calibration",
            "intrinsics": {
                "camera_matrix": self.jsonable_array(self.camera_matrix),
                "dist_coeffs": self.jsonable_array(self.dist_coeffs),
                "rms_px": self.jsonable_float(self.intrinsics_rms),
                "distortion_model": self.intrinsics_distortion_model_name,
                "calibration_flags": int(self.intrinsics_calibration_flags),
                "frame_indices": [int(index) for index in self.intrinsics_frame_indices],
                "excluded_frame_indices": [
                    int(index) for index in self.excluded_intrinsics_frame_indices
                ],
            },
            "extrinsics": {
                "convention": "T_CL maps LiDAR-frame points into the camera frame; T_LC is its inverse.",
                "T_CL": self.jsonable_array(T_CL),
                "T_LC": self.jsonable_array(T_LC),
                "optimized": True,
                "opt_rmse_m": self.jsonable_float(final_extrinsic.get("opt_rmse")),
                "frame_count": len(frames),
                "frames": frames,
            },
        }

    def save_calibration(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            QMessageBox.warning(self, "No Intrinsics", "Run or load intrinsics first.")
            return
        if not self.optimized_extrinsic_frames():
            QMessageBox.warning(
                self,
                "No Optimized Extrinsics",
                "Run nonlinear optimization before saving calibration.",
            )
            return

        output_path, selected_filter = self.select_save_file_with_filters(
            "Save Calibration",
            ["NumPy archive (*.npz)", "JSON (*.json)"],
            "lidar_camera_calibration.npz",
        )
        if not output_path:
            return

        output_path = str(output_path)
        suffix = Path(output_path).suffix.lower()
        if suffix not in (".npz", ".json"):
            output_path += ".json" if "JSON" in selected_filter else ".npz"
            suffix = Path(output_path).suffix.lower()

        try:
            payload = self.calibration_export_payload()
            if suffix == ".json":
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(payload, file, indent=2, allow_nan=False)
            else:
                extrinsic_frames = payload["extrinsics"]["frames"]
                np.savez(
                    output_path,
                    camera_matrix=np.asarray(self.camera_matrix, dtype=np.float64),
                    dist_coeffs=np.asarray(self.dist_coeffs, dtype=np.float64),
                    intrinsics_rms=np.asarray(
                        self.intrinsics_rms
                        if self.intrinsics_rms is not None
                        else np.nan,
                        dtype=np.float64,
                    ),
                    intrinsics_frame_indices=np.asarray(
                        self.intrinsics_frame_indices,
                        dtype=np.int32,
                    ),
                    excluded_intrinsics_frame_indices=np.asarray(
                        self.excluded_intrinsics_frame_indices,
                        dtype=np.int32,
                    ),
                    T_CL=np.asarray(payload["extrinsics"]["T_CL"], dtype=np.float64),
                    T_LC=np.asarray(payload["extrinsics"]["T_LC"], dtype=np.float64),
                    opt_rmse_m=np.asarray(
                        self.np_float_or_nan(payload["extrinsics"]["opt_rmse_m"]),
                        dtype=np.float64,
                    ),
                    frame_indices=np.asarray(
                        [frame["frame_index"] for frame in extrinsic_frames],
                        dtype=np.int32,
                    ),
                    per_frame_T_CL=np.asarray(
                        [frame["per_frame_T_CL"] for frame in extrinsic_frames],
                        dtype=np.float64,
                    ),
                    per_frame_T_LC=np.asarray(
                        [frame["per_frame_T_LC"] for frame in extrinsic_frames],
                        dtype=np.float64,
                    ),
                    symmetry_names=np.asarray(
                        [frame["symmetry_name"] for frame in extrinsic_frames],
                    ),
                    normal_signs=np.asarray(
                        [self.np_float_or_nan(frame["normal_sign"]) for frame in extrinsic_frames],
                        dtype=np.float64,
                    ),
                    consistency_scores=np.asarray(
                        [self.np_float_or_nan(frame["consistency_score"]) for frame in extrinsic_frames],
                        dtype=np.float64,
                    ),
                    lidar_board_rmse_m=np.asarray(
                        [self.np_float_or_nan(frame["lidar_board_rmse_m"]) for frame in extrinsic_frames],
                        dtype=np.float64,
                    ),
                    metadata_json=np.asarray(json.dumps(payload, allow_nan=False)),
                )
        except Exception as exc:
            QMessageBox.critical(self, "Save Calibration Failed", str(exc))
            self.append_log(f"Save calibration failed: {exc}")
            return

        self.saved_calibration_path = output_path
        self.save_calibration_action.setEnabled(True)
        self.update_workflow_stepper(active_step=5)
        self.statusBar().showMessage(f"Saved calibration: {output_path}")
        self.append_log(f"Saved calibration: {output_path}")
        self.autosave_recent_calibration()

    def load_intrinsics(self):
        input_path = self.select_open_file("Load Intrinsics", "NumPy archive (*.npz)")
        if not input_path:
            return

        try:
            with np.load(input_path) as data:
                self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
                self.dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
                self.intrinsics_rms = (
                    float(data["rms"])
                    if "rms" in data and np.isfinite(float(data["rms"]))
                    else None
                )
                self.intrinsics_frame_indices = (
                    [int(index) for index in data["frame_indices"]]
                    if "frame_indices" in data
                    else []
                )
                self.excluded_intrinsics_frame_indices = (
                    [int(index) for index in data["excluded_frame_indices"]]
                    if "excluded_frame_indices" in data
                    else []
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            self.append_log(f"Load intrinsics failed: {exc}")
            return

        self.save_intrinsics_button.setEnabled(True)
        self.save_intrinsics_action.setEnabled(True)
        self.solve_pnp_current_button.setEnabled(bool(self.frames))
        self.solve_pnp_all_button.setEnabled(bool(self.frames))
        self.solve_pnp_current_action.setEnabled(bool(self.frames))
        self.solve_pnp_all_action.setEnabled(bool(self.frames))
        for frame in self.frames:
            frame.camera_board_pose = None
            frame.lidar_board_pose = None
            frame.camera_lidar_extrinsic = None
        self.saved_calibration_path = None
        self.optimized_camera_lidar_extrinsic = None
        self.save_calibration_action.setEnabled(False)
        self.update_intrinsics_display()
        self.update_frame_table()
        self.update_workflow_stepper(active_step=3)
        self.statusBar().showMessage(f"Loaded intrinsics: {input_path}")
        self.autosave_recent_calibration()

    def select_save_file_with_filters(
        self,
        title: str,
        name_filters: list[str],
        default_name: str,
    ) -> tuple[str, str]:
        dialog = QFileDialog(self, title)
        dialog.setStyleSheet(
            "QFileDialog, QFileDialog * { color: #111827; }"
            "QLabel, QPushButton, QComboBox, QLineEdit { color: #111827; }"
            "QComboBox, QLineEdit, QListView, QTreeView { background: #ffffff; }"
            "QHeaderView::section { color: #111827; background: #f8fafc; }"
        )
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilters(name_filters)
        if name_filters:
            dialog.selectNameFilter(name_filters[0])
        dialog.selectFile(default_name)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                return selected_paths[0], dialog.selectedNameFilter()
        return "", ""

    def select_save_file(self, title: str, name_filter: str, default_name: str) -> str:
        dialog = QFileDialog(self, title)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilter(name_filter)
        dialog.selectFile(default_name)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                return selected_paths[0]
        return ""

    def select_open_file(self, title: str, name_filter: str) -> str:
        dialog = QFileDialog(self, title)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilter(name_filter)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                return selected_paths[0]
        return ""

    def show_frame(self, frame_index: int, update_point_cloud: bool = True):
        if not self.frames:
            self.image_label.setText("No dataset loaded")
            self.point_cloud_viewer.clear()
            self.update_frame_counter()
            return

        frame_index = max(0, min(frame_index, len(self.frames) - 1))
        if self.frame_slider.value() != frame_index:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_index)
            self.frame_slider.blockSignals(False)
        self.update_frame_counter(frame_index)
        frame = self.frames[frame_index]
        frame_changed = self.displayed_frame_index != frame_index
        if frame_changed:
            self.selected_image_point_px = None
            self.point_cloud_viewer.clear_pick_ray()

        try:
            image = self.render_frame_image(frame)
        except Exception as exc:
            self.image_label.setText(f"Could not read image:\n{frame.image_path}")
            self.statusBar().showMessage(f"Frame {frame_index}: {exc}")
            return

        self.current_image_pixmap = self.image_to_pixmap(image)
        self.image_drag_last_pos = None
        self.image_click_start_pos = None
        self.displayed_frame_index = frame_index
        self.update_image_display()
        self.frame_table.selectRow(frame_index)
        result = frame.charuco_result or {}
        valid_label = "valid" if result.get("valid", False) else "invalid"
        if frame.charuco_result is None:
            valid_label = "not detected"
        self.statusBar().showMessage(
            f"Frame {frame.frame_index + 1}/{len(self.frames)} | "
            f"{frame.image_path.name} | {valid_label}"
        )
        if update_point_cloud:
            self.update_point_cloud_viewer(frame)

    def update_image_display(self):
        if self.current_image_pixmap is None or self.current_image_pixmap.isNull():
            return

        fit_size = self.current_image_pixmap.size()
        fit_size.scale(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        scaled_size = fit_size * self.image_zoom_factor
        scaled_pixmap = self.current_image_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.clamp_image_pan(scaled_pixmap.size())

        viewport_size = self.image_label.size()
        viewport = QPixmap(viewport_size)
        viewport.fill(QColor("#202124"))

        x = (viewport_size.width() - scaled_pixmap.width()) // 2 + self.image_pan_offset.x()
        y = (viewport_size.height() - scaled_pixmap.height()) // 2 + self.image_pan_offset.y()
        self.image_display_origin = QPoint(x, y)
        self.image_display_size = scaled_pixmap.size()
        painter = QPainter(viewport)
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()
        self.image_label.setPixmap(viewport)

    def clamp_image_pan(self, scaled_size):
        viewport_size = self.image_label.size()
        overflow_x = max(0, scaled_size.width() - viewport_size.width())
        overflow_y = max(0, scaled_size.height() - viewport_size.height())
        min_x = -overflow_x // 2
        max_x = overflow_x - overflow_x // 2
        min_y = -overflow_y // 2
        max_y = overflow_y - overflow_y // 2
        clamped_x = min(max(self.image_pan_offset.x(), min_x), max_x)
        clamped_y = min(max(self.image_pan_offset.y(), min_y), max_y)
        self.image_pan_offset = QPoint(clamped_x, clamped_y)

    def zoom_image(self, wheel_delta: int):
        if self.current_image_pixmap is None:
            return
        zoom_step = 1.15
        if wheel_delta > 0:
            self.image_zoom_factor *= zoom_step
        else:
            self.image_zoom_factor /= zoom_step
        self.image_zoom_factor = max(1.0, min(self.image_zoom_factor, 12.0))
        self.update_image_display()

    def refresh_current_frame(self):
        if self.frames:
            self.show_frame(self.frame_slider.value())

    def current_projection_overlay_settings(self) -> dict:
        if not hasattr(self, "show_lidar_projection_checkbox"):
            return dict(self.projection_overlay_settings)
        return {
            "show": self.show_lidar_projection_checkbox.isChecked(),
            "source": self.projection_source_combo.currentText(),
            "color_by": self.projection_color_combo.currentText(),
            "point_radius": self.projection_point_size_spin.value(),
            "alpha": self.projection_alpha_spin.value(),
            "max_points": self.projection_max_points_spin.value(),
            "min_depth_m": self.projection_min_depth_spin.value(),
        }

    def update_lidar_projection_button_text(self, *_args):
        if not hasattr(self, "show_lidar_projection_checkbox"):
            return
        if self.show_lidar_projection_checkbox.isChecked():
            self.show_lidar_projection_checkbox.setText("Projected LiDAR: On")
        else:
            self.show_lidar_projection_checkbox.setText("Projected LiDAR: Off")

    def current_point_cloud_filter_settings(self) -> dict:
        if not hasattr(self, "point_cloud_filter_enabled_checkbox"):
            return dict(self.point_cloud_filter_settings)
        return {
            "enabled": self.point_cloud_filter_enabled_checkbox.isChecked(),
            "xmin": self.filter_xmin_spin.value(),
            "xmax": self.filter_xmax_spin.value(),
            "ymin": self.filter_ymin_spin.value(),
            "ymax": self.filter_ymax_spin.value(),
            "zmin": self.filter_zmin_spin.value(),
            "zmax": self.filter_zmax_spin.value(),
            "distance_min": self.filter_distance_min_spin.value(),
            "distance_max": self.filter_distance_max_spin.value(),
        }

    def apply_point_cloud_filter_from_controls(self, *_args):
        sender = self.sender()
        if (
            hasattr(self, "point_cloud_filter_enabled_checkbox")
            and sender is not self.point_cloud_filter_enabled_checkbox
            and not self.point_cloud_filter_enabled_checkbox.isChecked()
        ):
            self.point_cloud_filter_enabled_checkbox.blockSignals(True)
            self.point_cloud_filter_enabled_checkbox.setChecked(True)
            self.point_cloud_filter_enabled_checkbox.blockSignals(False)

        self.point_cloud_filter_settings = self.current_point_cloud_filter_settings()
        if self.frames:
            current_frame = self.frames[self.frame_slider.value()]
            self.update_point_cloud_viewer(current_frame)
            self.show_frame(self.frame_slider.value(), update_point_cloud=False)

    def reset_point_cloud_filter(self):
        defaults = {
            "enabled": False,
            "xmin": -50.0,
            "xmax": 50.0,
            "ymin": -50.0,
            "ymax": 50.0,
            "zmin": -10.0,
            "zmax": 10.0,
            "distance_min": 0.0,
            "distance_max": 100.0,
        }
        self.point_cloud_filter_settings = defaults
        controls = [
            self.point_cloud_filter_enabled_checkbox,
            self.filter_xmin_spin,
            self.filter_xmax_spin,
            self.filter_ymin_spin,
            self.filter_ymax_spin,
            self.filter_zmin_spin,
            self.filter_zmax_spin,
            self.filter_distance_min_spin,
            self.filter_distance_max_spin,
        ]
        for control in controls:
            control.blockSignals(True)
        self.point_cloud_filter_enabled_checkbox.setChecked(False)
        self.filter_xmin_spin.setValue(defaults["xmin"])
        self.filter_xmax_spin.setValue(defaults["xmax"])
        self.filter_ymin_spin.setValue(defaults["ymin"])
        self.filter_ymax_spin.setValue(defaults["ymax"])
        self.filter_zmin_spin.setValue(defaults["zmin"])
        self.filter_zmax_spin.setValue(defaults["zmax"])
        self.filter_distance_min_spin.setValue(defaults["distance_min"])
        self.filter_distance_max_spin.setValue(defaults["distance_max"])
        for control in controls:
            control.blockSignals(False)
        self.apply_point_cloud_filter_from_controls()

    def update_point_cloud_filter_controls(self):
        if not hasattr(self, "point_cloud_filter_enabled_checkbox"):
            return
        settings = dict(self.point_cloud_filter_settings)
        controls = [
            self.point_cloud_filter_enabled_checkbox,
            self.filter_xmin_spin,
            self.filter_xmax_spin,
            self.filter_ymin_spin,
            self.filter_ymax_spin,
            self.filter_zmin_spin,
            self.filter_zmax_spin,
            self.filter_distance_min_spin,
            self.filter_distance_max_spin,
        ]
        for control in controls:
            control.blockSignals(True)
        self.point_cloud_filter_enabled_checkbox.setChecked(bool(settings.get("enabled", False)))
        self.filter_xmin_spin.setValue(float(settings.get("xmin", -50.0)))
        self.filter_xmax_spin.setValue(float(settings.get("xmax", 50.0)))
        self.filter_ymin_spin.setValue(float(settings.get("ymin", -50.0)))
        self.filter_ymax_spin.setValue(float(settings.get("ymax", 50.0)))
        self.filter_zmin_spin.setValue(float(settings.get("zmin", -10.0)))
        self.filter_zmax_spin.setValue(float(settings.get("zmax", 10.0)))
        self.filter_distance_min_spin.setValue(float(settings.get("distance_min", 0.0)))
        self.filter_distance_max_spin.setValue(float(settings.get("distance_max", 100.0)))
        for control in controls:
            control.blockSignals(False)

    def current_cross_view_ray_settings(self) -> dict:
        if not hasattr(self, "cross_view_ray_enabled_checkbox"):
            return dict(self.cross_view_ray_settings)
        return {
            "enabled": self.cross_view_ray_enabled_checkbox.isChecked(),
            "color": self.cross_view_ray_settings.get("color", "#ffff00"),
            "width": self.ray_width_spin.value(),
            "endpoint_size": self.ray_endpoint_size_spin.value(),
            "max_pixel_distance": self.ray_max_pixel_distance_spin.value(),
        }

    def apply_cross_view_ray_settings_from_controls(self, *_args):
        self.cross_view_ray_settings = self.current_cross_view_ray_settings()

    def update_cross_view_ray_controls(self):
        if not hasattr(self, "cross_view_ray_enabled_checkbox"):
            return
        settings = dict(self.cross_view_ray_settings)
        controls = [
            self.cross_view_ray_enabled_checkbox,
            self.ray_width_spin,
            self.ray_endpoint_size_spin,
            self.ray_max_pixel_distance_spin,
        ]
        for control in controls:
            control.blockSignals(True)
        self.cross_view_ray_enabled_checkbox.setChecked(bool(settings.get("enabled", True)))
        self.ray_width_spin.setValue(float(settings.get("width", 3.0)))
        self.ray_endpoint_size_spin.setValue(float(settings.get("endpoint_size", 20.0)))
        self.ray_max_pixel_distance_spin.setValue(
            float(settings.get("max_pixel_distance", 80.0))
        )
        for control in controls:
            control.blockSignals(False)
        self.update_ray_color_button()

    def choose_cross_view_ray_color(self):
        color = QColorDialog.getColor(
            QColor(str(self.cross_view_ray_settings.get("color", "#ffff00"))),
            self,
            "Select Ray Color",
        )
        if not color.isValid():
            return
        self.cross_view_ray_settings["color"] = color.name()
        self.update_ray_color_button()
        self.apply_cross_view_ray_settings_from_controls()

    def update_ray_color_button(self):
        if not hasattr(self, "ray_color_button"):
            return
        color_name = str(self.cross_view_ray_settings.get("color", "#ffff00"))
        self.ray_color_button.setText(color_name.upper())
        self.ray_color_button.setStyleSheet(
            f"QPushButton {{ background: {color_name}; color: #111827; font-weight: 700; }}"
        )

    def ray_rgba(self) -> tuple[float, float, float, float]:
        color = QColor(str(self.cross_view_ray_settings.get("color", "#ffff00")))
        if not color.isValid():
            color = QColor("#ffff00")
        return (
            color.redF(),
            color.greenF(),
            color.blueF(),
            1.0,
        )

    def update_projection_overlay_controls(self):
        if not hasattr(self, "show_lidar_projection_checkbox"):
            return
        settings = dict(self.projection_overlay_settings)
        self.show_lidar_projection_checkbox.setChecked(bool(settings.get("show", True)))
        self.update_lidar_projection_button_text()
        self.projection_source_combo.setCurrentText(
            str(settings.get("source", "Full point cloud"))
        )
        self.projection_color_combo.setCurrentText(
            str(settings.get("color_by", "depth"))
        )
        self.projection_point_size_spin.setValue(
            int(settings.get("point_radius", 2))
        )
        self.projection_alpha_spin.setValue(float(settings.get("alpha", 0.85)))
        self.projection_max_points_spin.setValue(
            int(settings.get("max_points", 30000))
        )
        self.projection_min_depth_spin.setValue(
            float(settings.get("min_depth_m", 0.05))
        )

    def load_frame_image(self, frame: FrameRecord) -> np.ndarray:
        if frame.image is None:
            frame.image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if frame.image is None:
            raise RuntimeError(f"Could not read image: {frame.image_path}")
        return frame.image

    def load_frame_lidar(self, frame: FrameRecord) -> dict:
        if frame.lidar_dict is None:
            frame.lidar_dict = read_lidar_file(frame.lidar_path)
        return frame.lidar_dict

    @staticmethod
    def lidar_range_mask(xyz: np.ndarray, settings: dict) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        range_xmin = float(settings.get("range_xmin", -np.inf))
        range_xmax = float(settings.get("range_xmax", np.inf))
        range_ymin = float(settings.get("range_ymin", -np.inf))
        range_ymax = float(settings.get("range_ymax", np.inf))
        range_zmax = float(settings.get("range_zmax", np.inf))
        x_low, x_high = sorted((range_xmin, range_xmax))
        y_low, y_high = sorted((range_ymin, range_ymax))
        return (
            (xyz[:, 0] >= x_low)
            & (xyz[:, 0] <= x_high)
            & (xyz[:, 1] >= y_low)
            & (xyz[:, 1] <= y_high)
            & (xyz[:, 2] <= range_zmax)
        )

    @staticmethod
    def filter_lidar_dict_by_mask(lidar_dict: dict, mask: np.ndarray) -> dict:
        xyz = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
        filtered = {"xyz": xyz[mask, :3]}
        for key, value in lidar_dict.items():
            if key == "xyz":
                continue
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(mask):
                filtered[key] = value[mask]
            else:
                filtered[key] = value
        return filtered

    def filter_point_cloud_for_viewer(self, lidar_dict: dict) -> tuple[dict, int, int]:
        settings = dict(self.point_cloud_filter_settings)
        xyz = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            return lidar_dict, 0, 0

        total = len(xyz)
        if not settings.get("enabled", False):
            return lidar_dict, total, total

        points = xyz[:, :3]
        xmin, xmax = sorted((float(settings.get("xmin", -np.inf)), float(settings.get("xmax", np.inf))))
        ymin, ymax = sorted((float(settings.get("ymin", -np.inf)), float(settings.get("ymax", np.inf))))
        zmin, zmax = sorted((float(settings.get("zmin", -np.inf)), float(settings.get("zmax", np.inf))))
        dmin, dmax = sorted((
            float(settings.get("distance_min", 0.0)),
            float(settings.get("distance_max", np.inf)),
        ))
        distances = np.linalg.norm(points, axis=1)
        mask = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
            & (distances >= dmin)
            & (distances <= dmax)
        )
        return self.filter_lidar_dict_by_mask(lidar_dict, mask), int(np.sum(mask)), total

    def filter_projection_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not self.point_cloud_filter_settings.get("enabled", False):
            return points
        filtered_dict, _visible_count, _total_count = self.filter_point_cloud_for_viewer(
            {"xyz": points}
        )
        return np.asarray(filtered_dict.get("xyz", []), dtype=np.float64).reshape(-1, 3)

    def preview_lidar_center_range_thresholds(self, settings: dict):
        if not self.frames:
            return
        frame = self.frames[self.frame_slider.value()]
        try:
            lidar_dict = self.load_frame_lidar(frame)
            xyz = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
            if xyz.ndim != 2 or xyz.shape[1] < 3:
                return
            mask = self.lidar_range_mask(xyz[:, :3], settings)
            filtered_dict = self.filter_lidar_dict_by_mask(lidar_dict, mask)
        except Exception as exc:
            self.statusBar().showMessage(f"Range preview failed: {exc}")
            return

        self.point_cloud_viewer.set_point_cloud(
            filtered_dict,
            label=f"Frame {frame.frame_index}: range preview",
        )
        self.point_cloud_viewer.set_color_by(self.projection_color_combo.currentText())
        self.statusBar().showMessage(
            f"Range preview: {int(np.sum(mask))} / {len(mask)} LiDAR points"
        )

    def update_point_cloud_viewer(self, frame: FrameRecord):
        try:
            lidar_dict = self.load_frame_lidar(frame)
        except Exception as exc:
            self.point_cloud_viewer.clear()
            self.statusBar().showMessage(f"Point cloud load failed: {exc}")
            self.append_log(
                f"Point cloud load failed for frame {frame.frame_index}: {exc}"
            )
            return
        viewer_lidar_dict, visible_count, total_count = self.filter_point_cloud_for_viewer(lidar_dict)
        self.point_cloud_viewer.set_point_cloud(
            viewer_lidar_dict,
            label=f"Frame {frame.frame_index}: {frame.lidar_path.name}",
        )
        self.point_cloud_viewer.set_color_by(
            self.current_projection_overlay_settings().get("color_by", "depth")
        )
        if self.point_cloud_filter_settings.get("enabled", False):
            self.statusBar().showMessage(
                f"Point cloud filter: {visible_count}/{total_count} points visible"
            )
        result = frame.lidar_processing_result or {}
        if result:
            self.point_cloud_viewer.set_lidar_center_overlay(
                cluster_points=result.get("cluster_points"),
                cluster_labels=result.get("cluster_labels"),
                centers=result.get("centers_L"),
                show_clusters=bool(result.get("show_clusters", True)),
                show_centers=bool(result.get("show_centers", True)),
            )

    def execute_lidar_center_detection(self, settings: dict):
        self.lidar_center_settings = dict(settings)
        if not self.frames:
            QMessageBox.warning(self, "No Dataset", "Load a dataset before detecting LiDAR centers.")
            return
        if self.active_background_task is not None:
            self.statusBar().showMessage(
                f"{self.active_background_task} is already running."
            )
            return

        frame_data = [
            {
                "frame_index": frame.frame_index,
                "lidar_path": frame.lidar_path,
            }
            for frame in self.frames
        ]
        self.progress_bar.setRange(0, len(self.frames))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("LiDAR centers (%p%)")
        self.progress_bar.setVisible(True)
        self.set_background_task("LiDAR center detection")
        self.update_workflow_stepper(active_step=3)
        self.statusBar().showMessage("Running LiDAR center detection...")

        worker = LidarCenterDetectionWorker(
            frame_data,
            settings,
        )
        worker.progress.connect(self.handle_lidar_center_progress)
        worker.finished.connect(self.handle_lidar_center_finished)
        self.start_worker_thread(worker, [worker.finished])

    def handle_lidar_center_progress(
        self,
        value: int,
        maximum: int,
        status_message: str,
    ):
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat("LiDAR centers (%p%)")
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage(status_message)

    def handle_lidar_center_finished(self, result_summary: dict):
        frame_by_index = {frame.frame_index: frame for frame in self.frames}
        failures = list(result_summary["failures"])
        completed = int(result_summary["completed"])
        total_centers = int(result_summary["total_centers"])
        total = int(result_summary["total"])

        for frame_index, result in result_summary["results"]:
            frame = frame_by_index.get(frame_index)
            if frame is None:
                continue
            if result is None:
                frame.lidar_processing_result = None
                frame.enabled = False
                self.invalidate_lidar_camera_calibration()
                reason = next(
                    (
                        failure_reason
                        for failure_index, failure_reason in failures
                        if failure_index == frame_index
                    ),
                    "unknown error",
                )
                self.append_log(
                    f"LiDAR center detection failed for frame {frame.frame_index}: {reason}"
                )
            else:
                frame.lidar_processing_result = result
                self.invalidate_lidar_camera_calibration()
                distance_check = result.get("center_distance_check", {})
                frame.enabled = bool(distance_check.get("passed", False))
                distance_status = (
                    "distance ok"
                    if distance_check.get("passed", False)
                    else f"distance failed: {distance_check.get('reason', 'unknown')}"
                )
                self.append_log(
                    f"LiDAR centers frame {frame.frame_index}: "
                    f"{len(result.get('centers_L', []))} center(s), "
                    f"{result.get('num_clusters', 0)} cluster(s), "
                    f"{distance_status}."
                )

        self.finish_background_task()
        current_frame = self.frames[self.frame_slider.value()]
        self.update_point_cloud_viewer(current_frame)
        self.update_workflow_stepper(active_step=4)
        self.update_frame_table()

        if completed == 0:
            first_failure = failures[0][1] if failures else "No frames were processed."
            QMessageBox.critical(self, "LiDAR Center Failed", first_failure)
            return

        self.append_log(
            f"LiDAR center detection complete: {completed}/{total} frame(s), "
            f"{total_centers} center(s)."
        )
        self.statusBar().showMessage(
            f"LiDAR center detection complete: {completed}/{total} frame(s)"
        )
        self.autosave_recent_calibration()
        if failures:
            QMessageBox.warning(
                self,
                "LiDAR Center Partial Failure",
                f"Computed LiDAR centers for {completed}/{total} frame(s). "
                f"{len(failures)} frame(s) failed. See log for details.",
            )

    def detect_lidar_centers_for_frame(self, frame: FrameRecord, settings: dict) -> dict:
        lidar_dict = self.load_frame_lidar(frame)
        xyz = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError("LiDAR point cloud must contain xyz points.")
        xyz = xyz[:, :3]

        source = settings["source"]
        selected_indices = np.arange(len(xyz), dtype=np.int64)

        range_mask = self.lidar_range_mask(xyz, settings)
        selected_indices = selected_indices[range_mask]
        selected_points = xyz[range_mask]
        selected_dict = self.filter_lidar_dict_by_mask(lidar_dict, range_mask)

        if len(selected_points) == 0:
            raise RuntimeError("No LiDAR points remain after range thresholding.")

        if source in {"Plane points", "Plane + reflectivity"}:
            plane = segment_plane_from_pointcloud(
                selected_points,
                distance_threshold=float(settings["plane_distance_threshold"]),
                min_inliers=max(10, int(settings["min_samples"])),
            )
            plane_indices = plane["inlier_indices"]
            selected_indices = selected_indices[plane_indices]
            selected_points = selected_points[plane_indices]
            selected_dict = {
                "xyz": selected_points,
                "plane_model": plane["plane_model"],
            }
            for key, value in lidar_dict.items():
                if isinstance(value, np.ndarray) and len(value) == len(xyz):
                    selected_dict[key] = value[selected_indices]

        if source in {"Reflective points", "Plane + reflectivity"}:
            if "reflectivity" not in lidar_dict and "reflectivity" not in selected_dict:
                raise KeyError("Point cloud does not contain reflectivity values.")
            reflectivity = np.asarray(
                selected_dict.get("reflectivity", lidar_dict["reflectivity"]),
                dtype=np.float64,
            ).reshape(-1)
            mask = reflectivity >= float(settings["reflectivity_threshold"])
            selected_points = selected_points[mask]
            selected_indices = selected_indices[mask]
            for key, value in list(selected_dict.items()):
                if isinstance(value, np.ndarray) and len(value) == len(mask):
                    selected_dict[key] = value[mask]
            selected_dict["xyz"] = selected_points

        if len(selected_points) == 0:
            raise RuntimeError("No LiDAR points remain after source filtering.")

        fit_mode = settings["fit_mode"]
        labels = np.full(len(selected_points), -1, dtype=np.int32)
        centers_L = np.empty((0, 3), dtype=np.float64)
        uv_plane_result = None

        if fit_mode == "Circle fit from DBSCAN clusters":
            labels = self.dbscan_labels(
                selected_points,
                eps=float(settings["eps"]),
                min_samples=int(settings["min_samples"]),
            )
            centers_L = self.circle_centers_from_labels(selected_points, labels)
        elif fit_mode == "Disk fitting on plane":
            uv_plane = make_lidar_uv_planes_from_plane_filtered_ptclds(
                [selected_dict],
                distance_threshold=float(settings["plane_distance_threshold"]),
                square_align_uv=True,
            )[0]
            uv_plane = add_disk_centroids_to_uv_plane(
                uv_plane,
                radius_m=float(settings["disk_radius_m"]),
                min_inliers=int(settings["min_samples"]),
            )
            centers_L = uv_plane.get("disk_centers_L", np.empty((0, 3), dtype=np.float64))
            uv_plane_result = uv_plane
        else:
            raise ValueError(f"Unknown LiDAR center fit mode: {fit_mode}")

        center_distance_check = self.check_center_distance_consistency(
            centers_L,
            template_centers_B_m=HOLE_CENTERS_B_M,
            abs_tolerance_m=float(
                settings.get(
                    "center_distance_abs_tolerance_m",
                    CENTER_DISTANCE_ABS_TOLERANCE_M,
                )
            ),
            rel_tolerance=float(
                settings.get(
                    "center_distance_rel_tolerance",
                    CENTER_DISTANCE_REL_TOLERANCE,
                )
            ),
        )

        return {
            "source": source,
            "fit_mode": fit_mode,
            "selected_indices": selected_indices,
            "cluster_points": selected_points,
            "cluster_labels": labels,
            "centers_L": centers_L,
            "uv_plane": uv_plane_result,
            "center_distance_check": center_distance_check,
            "num_clusters": len(set(int(label) for label in labels if label >= 0)),
            "show_clusters": bool(settings["show_clusters"]),
            "show_centers": bool(settings["show_centers"]),
        }

    @staticmethod
    def sorted_pairwise_distances(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or len(points) < 2:
            return np.empty(0, dtype=np.float64)

        distances = []
        for i, j in combinations(range(len(points)), 2):
            distances.append(float(np.linalg.norm(points[i] - points[j])))
        return np.sort(np.asarray(distances, dtype=np.float64))

    @staticmethod
    def check_center_distance_consistency(
        centers_L: np.ndarray,
        template_centers_B_m: np.ndarray,
        abs_tolerance_m: float = CENTER_DISTANCE_ABS_TOLERANCE_M,
        rel_tolerance: float = CENTER_DISTANCE_REL_TOLERANCE,
        max_candidate_centers: int = 12,
    ) -> dict:
        centers = np.asarray(centers_L, dtype=np.float64).reshape(-1, 3)
        template = np.asarray(template_centers_B_m, dtype=np.float64)
        if template.ndim != 2 or template.shape[1] < 2:
            raise ValueError("Template centers must have shape (N, 2) or wider.")

        template_xy = template[:, :2]
        expected_count = len(template_xy)
        template_distances = MainWindow.sorted_pairwise_distances(template_xy)

        if len(centers) < expected_count:
            return {
                "passed": False,
                "reason": f"need {expected_count} centers, got {len(centers)}",
                "expected_count": expected_count,
                "detected_count": len(centers),
                "template_distances_m": template_distances,
                "matched_indices": [],
            }

        candidate_indices = np.arange(len(centers), dtype=np.int64)
        if len(candidate_indices) > max_candidate_centers:
            candidate_indices = candidate_indices[:max_candidate_centers]

        best = None
        for combo in combinations(candidate_indices.tolist(), expected_count):
            detected_distances = MainWindow.sorted_pairwise_distances(
                centers[np.asarray(combo, dtype=np.int64)]
            )
            abs_errors = np.abs(detected_distances - template_distances)
            rel_errors = abs_errors / np.maximum(template_distances, 1e-9)
            score = (float(np.max(rel_errors)), float(np.mean(abs_errors)))
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "matched_indices": list(combo),
                    "detected_distances_m": detected_distances,
                    "abs_errors_m": abs_errors,
                    "rel_errors": rel_errors,
                    "max_abs_error_m": float(np.max(abs_errors)),
                    "mean_abs_error_m": float(np.mean(abs_errors)),
                    "max_rel_error": float(np.max(rel_errors)),
                    "mean_rel_error": float(np.mean(rel_errors)),
                }

        if best is None:
            return {
                "passed": False,
                "reason": "no center combinations available",
                "expected_count": expected_count,
                "detected_count": len(centers),
                "template_distances_m": template_distances,
                "matched_indices": [],
            }

        passed = (
            best["max_abs_error_m"] <= abs_tolerance_m and
            best["max_rel_error"] <= rel_tolerance
        )
        reason = (
            "ok"
            if passed else
            (
                f"max distance error {best['max_abs_error_m']:.3f} m "
                f"({best['max_rel_error']:.1%}) exceeds tolerance "
                f"{abs_tolerance_m:.3f} m/{rel_tolerance:.0%}"
            )
        )
        return {
            "passed": passed,
            "reason": reason,
            "expected_count": expected_count,
            "detected_count": len(centers),
            "template_distances_m": template_distances,
            "abs_tolerance_m": abs_tolerance_m,
            "rel_tolerance": rel_tolerance,
            **best,
        }

    def extrinsic_ready_frames(self) -> list[FrameRecord]:
        return [
            frame for frame in self.frames
            if (
                frame.enabled
                and frame.camera_board_pose is not None
                and frame.camera_board_pose.get("success", False)
                and frame.lidar_processing_result is not None
                and frame.lidar_processing_result.get("center_distance_check", {}).get("passed", False)
            )
        ]

    def run_lidar_camera_extrinsics(self):
        if self.camera_matrix is None or self.dist_coeffs is None:
            QMessageBox.warning(self, "No Intrinsics", "Run intrinsics calibration first.")
            return

        ready_frames = self.extrinsic_ready_frames()
        if not ready_frames:
            QMessageBox.warning(
                self,
                "No Ready Frames",
                "Need enabled frames with SolvePnP poses and distance-checked LiDAR centers.",
            )
            return

        extrinsic_results = []
        failures = []
        for frame in ready_frames:
            try:
                extrinsic_results.append(self.make_frame_extrinsic_result(frame))
            except Exception as exc:
                failures.append((frame.frame_index, str(exc)))
                self.append_log(
                    f"LiDAR-camera candidate build failed for frame {frame.frame_index}: {exc}"
                )

        if not extrinsic_results:
            QMessageBox.critical(
                self,
                "LiDAR-Cam Extrinsics Failed",
                "Could not build any candidate extrinsics. See log for details.",
            )
            return

        reference_result = min(
            extrinsic_results,
            key=lambda result: min(
                candidate["lidar_board_rmse_m"]
                for candidate in result["candidate_records"]
            ),
        )
        reference_record = self.select_reference_extrinsic_candidate(reference_result)
        if reference_record is None:
            self.statusBar().showMessage("LiDAR-camera extrinsic selection cancelled")
            return
        self.append_log(
            f"Reference extrinsic frame {reference_record['frame_index']}: "
            f"{reference_record['symmetry_name']}/"
            f"{'zpos' if reference_record['normal_sign'] > 0 else 'zneg'}"
        )

        reference_T_CL = reference_record["T_CL"]
        selected_records = []

        for result in extrinsic_results:
            candidates = result["candidate_records"]
            for candidate in candidates:
                score, rotation_error_rad, translation_error_m = extrinsic_distance(
                    candidate["T_CL"],
                    reference_T_CL,
                    translation_weight=2.0,
                )
                candidate["consistency_score"] = score
                candidate["rotation_error_deg"] = float(np.rad2deg(rotation_error_rad))
                candidate["translation_error_m"] = translation_error_m

            selected = min(candidates, key=lambda candidate: candidate["consistency_score"])
            selected_records.append(selected)

        for result, selected in zip(extrinsic_results, selected_records):
            frame = self.frames[result["frame_list_index"]]
            frame.lidar_board_pose = selected["candidate"]
            frame.camera_lidar_extrinsic = {
                "T_CL": selected["T_CL"],
                "T_LC": invert_transform(selected["T_CL"]),
                "per_frame_T_CL": selected["T_CL"],
                "symmetry_name": selected["symmetry_name"],
                "normal_sign": selected["normal_sign"],
                "lidar_board_rmse_m": selected["lidar_board_rmse_m"],
                "consistency_score": selected["consistency_score"],
                "rotation_error_deg": selected["rotation_error_deg"],
                "translation_error_m": selected["translation_error_m"],
                "is_reference": result is reference_result,
                "optimized": False,
            }

        self.saved_calibration_path = None
        self.optimized_camera_lidar_extrinsic = None
        self.save_calibration_action.setEnabled(False)
        self.save_lidar_camera_extrinsics(reference_T_CL, reference_record, selected_records, failures)
        if self.run_nonlinear_optimization():
            self.statusBar().showMessage(
                f"LiDAR-camera extrinsics optimized: {len(selected_records)} frame(s)"
            )

    def run_nonlinear_optimization(self) -> bool:
        extrinsic_frames = [frame for frame in self.frames if frame.camera_lidar_extrinsic is not None]
        if not extrinsic_frames:
            QMessageBox.warning(self, "No Extrinsics", "Run LiDAR-Cam Extrinsics first.")
            return False

        all_points_L = []
        all_points_C = []
        reference_T_CL = None

        for frame in extrinsic_frames:
            ext = frame.camera_lidar_extrinsic
            if ext.get("is_reference"):
                reference_T_CL = ext["per_frame_T_CL"]

            candidate_info = frame.lidar_board_pose
            pts_L = candidate_info["matched_lidar_points_L"]
            pts_B = candidate_info["matched_board_points_B"]
            pts_B_h = np.column_stack([pts_B, np.ones(len(pts_B))])
            T_CB = self.camera_board_transform_m(frame)
            pts_C = (T_CB @ pts_B_h.T).T[:, :3]

            all_points_L.append(pts_L)
            all_points_C.append(pts_C)

        if not all_points_L:
            return False

        if reference_T_CL is None:
            reference_T_CL = extrinsic_frames[0].camera_lidar_extrinsic["per_frame_T_CL"]

        points_L_concat = np.vstack(all_points_L)
        points_C_concat = np.vstack(all_points_C)
        optimized_T_CL, opt_rmse = optimize_camera_lidar_extrinsic_robust(
            points_L_concat,
            points_C_concat,
            reference_T_CL,
        )
        optimized_T_LC = invert_transform(optimized_T_CL)

        self.append_log(f"Robust Optimization: RMSE {opt_rmse:.4f} m across {len(points_L_concat)} matched points.")
        self.optimized_camera_lidar_extrinsic = {
            "T_CL": optimized_T_CL,
            "T_LC": optimized_T_LC,
            "opt_rmse": opt_rmse,
        }

        selected_records = []
        reference_record = None
        failures = []

        for frame in extrinsic_frames:
            frame.camera_lidar_extrinsic["T_CL"] = optimized_T_CL
            frame.camera_lidar_extrinsic["T_LC"] = optimized_T_LC
            frame.camera_lidar_extrinsic["optimized"] = True
            frame.camera_lidar_extrinsic["opt_rmse"] = opt_rmse

            record = {
                "symmetry_name": frame.camera_lidar_extrinsic["symmetry_name"],
                "normal_sign": frame.camera_lidar_extrinsic["normal_sign"],
                "consistency_score": frame.camera_lidar_extrinsic["consistency_score"],
                "T_CL": frame.camera_lidar_extrinsic["per_frame_T_CL"],
                "T_LC": invert_transform(frame.camera_lidar_extrinsic["per_frame_T_CL"]),
                "lidar_board_rmse_m": frame.camera_lidar_extrinsic["lidar_board_rmse_m"],
                "frame_index": frame.frame_index,
            }
            selected_records.append(record)
            if frame.camera_lidar_extrinsic.get("is_reference"):
                reference_record = record

        if reference_record is None:
            reference_record = selected_records[0]

        self.save_lidar_camera_extrinsics(optimized_T_CL, reference_record, selected_records, failures)
        self.saved_calibration_path = None
        self.save_calibration_action.setEnabled(True)

        self.update_workflow_stepper(active_step=5)
        self.update_frame_table()
        self.update_info()
        self.show_frame(self.frame_slider.value())
        self.autosave_recent_calibration()
        self.statusBar().showMessage("Non-Linear Optimization complete")
        return True

    def make_frame_extrinsic_result(self, frame: FrameRecord) -> dict:
        lidar_result = frame.lidar_processing_result or {}
        uv_plane = lidar_result.get("uv_plane")
        T_CB_m = self.camera_board_transform_m(frame)
        if uv_plane is not None and "disk_centers_uv" in uv_plane:
            lidar_board_pose = estimate_lidar_board_pose_from_disk_centers(
                uv_plane,
                board_centers_B_m=HOLE_CENTERS_B_M,
                detected_uv_key="disk_centers_uv",
                detected_lidar_key="disk_centers_L",
                max_match_distance_m=0.20,
                try_planar_symmetries=True,
            )
            extrinsic_result = {
                "index": frame.frame_index,
                "frame_list_index": self.frames.index(frame),
                "image_path": str(frame.image_path),
                "lidar_path": str(frame.lidar_path),
                "T_CB": T_CB_m,
                "lidar_board_pose": lidar_board_pose,
                "lidar_board_rmse_m": lidar_board_pose["rmse_m"],
            }
            extrinsic_result["candidate_records"] = make_candidate_extrinsic_records(
                extrinsic_result
            )
            return extrinsic_result

        candidate_records = self.make_dbscan_center_candidate_records(frame)
        if not candidate_records:
            raise RuntimeError("No LiDAR-board pose candidates were generated.")
        return {
            "index": frame.frame_index,
            "frame_list_index": self.frames.index(frame),
            "image_path": str(frame.image_path),
            "lidar_path": str(frame.lidar_path),
            "T_CB": T_CB_m,
            "lidar_board_rmse_m": min(record["lidar_board_rmse_m"] for record in candidate_records),
            "candidate_records": candidate_records,
        }

    @staticmethod
    def camera_board_transform_m(frame: FrameRecord) -> np.ndarray:
        pose = frame.camera_board_pose or {}
        if "T_CB" not in pose:
            raise RuntimeError(f"Frame {frame.frame_index} is missing camera-board pose.")

        T_CB = np.asarray(pose["T_CB"], dtype=np.float64).reshape(4, 4).copy()
        translation_norm = float(np.linalg.norm(T_CB[:3, 3]))
        if translation_norm > 20.0:
            T_CB[:3, 3] *= CHARUCO_BOARD_UNIT_TO_M
        return T_CB

    def make_dbscan_center_candidate_records(self, frame: FrameRecord) -> list[dict]:
        lidar_result = frame.lidar_processing_result or {}
        centers_L = np.asarray(lidar_result.get("centers_L", []), dtype=np.float64).reshape(-1, 3)
        distance_check = lidar_result.get("center_distance_check", {})
        matched_indices = distance_check.get("matched_indices") or list(range(min(4, len(centers_L))))
        if len(matched_indices) < len(HOLE_CENTERS_B_M):
            raise RuntimeError("Not enough matched LiDAR centers for extrinsic candidates.")

        detected = centers_L[np.asarray(matched_indices[:len(HOLE_CENTERS_B_M)], dtype=np.int64)]
        board_points_B = np.zeros((len(HOLE_CENTERS_B_M), 3), dtype=np.float64)
        board_points_B[:, :2] = np.asarray(HOLE_CENTERS_B_M, dtype=np.float64)[:, :2]

        records = []
        for symmetry in make_board_planar_symmetry_transforms(board_points_B):
            board_points_h = np.column_stack([board_points_B, np.ones(len(board_points_B))])
            board_points_S = (symmetry["T_SB"] @ board_points_h.T).T[:, :3]

            best = None
            for order in permutations(range(len(detected)), len(board_points_S)):
                matched_detected = detected[np.asarray(order, dtype=np.int64)]
                R_LS, t_LS, rmse_m, residuals_m = estimate_rigid_transform_3d(
                    board_points_S,
                    matched_detected,
                )
                if best is None or rmse_m < best["rmse_m"]:
                    best = {
                        "matched_detected": matched_detected,
                        "rmse_m": rmse_m,
                        "residuals_m": residuals_m,
                        "T_LS": make_transform(R_LS, t_LS),
                        "order": order,
                    }

            if best is None:
                continue

            T_LB = best["T_LS"] @ symmetry["T_SB"]
            candidate_pose = {
                "symmetry_name": symmetry["name"],
                "normal_sign": symmetry.get("normal_sign", 1.0),
                "T_LB": T_LB,
                "T_BL": invert_transform(T_LB),
                "T_SB": symmetry["T_SB"],
                "matches": [(i, int(best["order"][i]), 0.0) for i in range(len(board_points_S))],
                "matched_board_points_B": board_points_B,
                "matched_board_points_S": board_points_S,
                "matched_lidar_points_L": best["matched_detected"],
                "rmse_m": best["rmse_m"],
                "residuals_m": best["residuals_m"],
            }
            candidate_extrinsic = estimate_camera_lidar_extrinsic_from_board_poses(
                self.camera_board_transform_m(frame),
                T_LB,
            )
            records.append({
                "frame_index": frame.frame_index,
                "image_path": str(frame.image_path),
                "lidar_path": str(frame.lidar_path),
                "symmetry_name": candidate_pose["symmetry_name"],
                "normal_sign": float(np.sign(candidate_pose["normal_sign"])),
                "T_CL": candidate_extrinsic["T_CL"],
                "T_LC": candidate_extrinsic["T_LC"],
                "T_LB": T_LB,
                "lidar_board_rmse_m": best["rmse_m"],
                "source_result": None,
                "candidate": candidate_pose,
            })

        return records

    def select_reference_extrinsic_candidate(self, reference_result: dict):
        candidates = sorted(
            reference_result["candidate_records"],
            key=lambda candidate: (candidate["normal_sign"], candidate["symmetry_name"]),
        )
        dialog_candidates = []
        image = self.load_frame_image(self.frames[reference_result["frame_list_index"]])
        lidar_dict = self.load_frame_lidar(self.frames[reference_result["frame_list_index"]])
        points_L = np.asarray(lidar_dict["xyz"], dtype=np.float64)[:, :3]
        reflectivity = lidar_dict.get("reflectivity")

        for candidate in candidates:
            projection = visualize_lidar_projection_on_image(
                image=image,
                points_L=points_L,
                T_CL=candidate["T_CL"],
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                reflectivity=reflectivity,
                color_by="depth",
                point_radius=2,
                alpha=0.85,
                max_points=12000,
                show=False,
            )
            drawn_count = len(projection["drawn_indices"])
            valid_count = int(np.sum(projection["projection"]["valid_mask"]))
            cv2.putText(
                projection["image"],
                f"{candidate['symmetry_name']} "
                f"{'zpos' if candidate['normal_sign'] > 0 else 'zneg'} | "
                f"drawn {drawn_count} valid {valid_count}",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255) if drawn_count > 0 else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            pixmap = self.image_to_pixmap(projection["image"])
            dialog_candidate = dict(candidate)
            dialog_candidate["pixmap"] = pixmap
            dialog_candidate["drawn_count"] = drawn_count
            dialog_candidate["valid_count"] = valid_count
            dialog_candidates.append(dialog_candidate)

        dialog = ExtrinsicCandidateSelectionDialog(dialog_candidates, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.selected_index is None:
            return None
        selected_dialog_candidate = dialog_candidates[dialog.selected_index]
        for candidate in candidates:
            if (
                candidate["symmetry_name"] == selected_dialog_candidate["symmetry_name"]
                and candidate["normal_sign"] == selected_dialog_candidate["normal_sign"]
            ):
                candidate["consistency_score"] = 0.0
                candidate["rotation_error_deg"] = 0.0
                candidate["translation_error_m"] = 0.0
                return candidate
        return None

    def save_lidar_camera_extrinsics(
        self,
        T_CL: np.ndarray,
        reference_record: dict,
        selected_records: list[dict],
        failures: list[tuple[int, str]],
    ):
        output_dir = Path("output/calibration")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "camera_lidar_extrinsics.npz"
        np.savez(
            output_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            T_CL=T_CL,
            T_LC=invert_transform(T_CL),
            selected_frame_index=np.asarray(reference_record["frame_index"], dtype=np.int32),
            selected_symmetry_names=np.asarray([record["symmetry_name"] for record in selected_records]),
            selected_normal_signs=np.asarray([record["normal_sign"] for record in selected_records], dtype=np.float64),
            selected_consistency_scores=np.asarray([record["consistency_score"] for record in selected_records], dtype=np.float64),
            selected_T_CL=np.asarray([record["T_CL"] for record in selected_records], dtype=np.float64),
            selected_T_LC=np.asarray([record["T_LC"] for record in selected_records], dtype=np.float64),
            lidar_board_rmse_m=np.asarray([record["lidar_board_rmse_m"] for record in selected_records], dtype=np.float64),
            frame_indices=np.asarray([record["frame_index"] for record in selected_records], dtype=np.int32),
            failed_frame_indices=np.asarray([index for index, _reason in failures], dtype=np.int32),
        )
        self.append_log(
            f"Saved LiDAR-camera extrinsics for {len(selected_records)} frame(s): {output_path}"
        )

    @staticmethod
    def dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        labels = np.full(len(points), -1, dtype=np.int32)
        if len(points) == 0:
            return labels
            return np.full(0, -1, dtype=np.int32)

        visited = np.zeros(len(points), dtype=bool)
        cluster_id = 0
        for index in range(len(points)):
            if visited[index]:
                continue
            visited[index] = True
            neighbors = MainWindow.region_query(points, index, eps)
            if len(neighbors) < min_samples:
                continue

            labels[index] = cluster_id
            seeds = list(neighbors)
            cursor = 0
            while cursor < len(seeds):
                neighbor = seeds[cursor]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    neighbor_neighbors = MainWindow.region_query(points, neighbor, eps)
                    if len(neighbor_neighbors) >= min_samples:
                        for candidate in neighbor_neighbors:
                            if candidate not in seeds:
                                seeds.append(candidate)
                if labels[neighbor] < 0:
                    labels[neighbor] = cluster_id
                cursor += 1
            cluster_id += 1
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(
            pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False),
            dtype=np.int32
        )
        return labels

    @staticmethod
    def region_query(points: np.ndarray, index: int, eps: float) -> list[int]:
        distances = np.linalg.norm(points - points[index], axis=1)
        return np.flatnonzero(distances <= eps).astype(int).tolist()

    @staticmethod
    def circle_centers_from_labels(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centers = []
        for label in sorted(set(int(value) for value in labels if value >= 0)):
            cluster_points = points[labels == label]
            if len(cluster_points) > 0:
                centers.append(MainWindow.fit_circle_center_3d(cluster_points))
        if not centers:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(centers)

    @staticmethod
    def fit_circle_center_3d(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        centroid = np.mean(points, axis=0)
        if len(points) < 3:
            return centroid

        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2]
        uv = centered @ basis.T
        x = uv[:, 0]
        y = uv[:, 1]
        A = np.column_stack([x, y, np.ones(len(uv))])
        b = -(x * x + y * y)
        try:
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return centroid
        center_uv = -0.5 * coeffs[:2]
        return centroid + center_uv @ basis

    def render_frame_image(self, frame: FrameRecord) -> np.ndarray:
        image = self.load_frame_image(frame).copy()
        result = frame.charuco_result
        if result is not None and self.draw_markers_checkbox.isChecked():
            self.draw_markers(image, result)
        if result is not None and self.draw_corners_checkbox.isChecked():
            self.draw_corners(image, result)
        if result is not None and self.draw_axes_checkbox.isChecked():
            self.draw_board_axes(image, frame)
        image = self.apply_lidar_projection_overlay(image, frame)
        self.draw_selected_image_point(image)
        return image

    def draw_selected_image_point(self, image: np.ndarray):
        if self.selected_image_point_px is None:
            return
        x, y = np.round(self.selected_image_point_px).astype(int)
        height, width = image.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        radius = 18
        color_outer = (0, 0, 0)
        color_inner = (0, 255, 255)
        cv2.circle(image, (x, y), radius + 5, color_outer, 4, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius, color_inner, 4, cv2.LINE_AA)
        cv2.line(image, (x - 30, y), (x + 30, y), color_outer, 7, cv2.LINE_AA)
        cv2.line(image, (x, y - 30), (x, y + 30), color_outer, 7, cv2.LINE_AA)
        cv2.line(image, (x - 30, y), (x + 30, y), color_inner, 3, cv2.LINE_AA)
        cv2.line(image, (x, y - 30), (x, y + 30), color_inner, 3, cv2.LINE_AA)

    def optimized_camera_lidar_transform(self) -> np.ndarray | None:
        if (
            self.optimized_camera_lidar_extrinsic is not None
            and "T_CL" in self.optimized_camera_lidar_extrinsic
        ):
            return np.asarray(
                self.optimized_camera_lidar_extrinsic["T_CL"],
                dtype=np.float64,
            ).reshape(4, 4)

        for frame in self.frames:
            ext = frame.camera_lidar_extrinsic
            if ext and ext.get("optimized") and "T_CL" in ext:
                return np.asarray(ext["T_CL"], dtype=np.float64).reshape(4, 4)
        return None

    def projection_camera_lidar_transform(self, frame: FrameRecord) -> np.ndarray | None:
        optimized_T_CL = self.optimized_camera_lidar_transform()
        if optimized_T_CL is not None:
            return optimized_T_CL

        extrinsic = frame.camera_lidar_extrinsic or {}
        if "T_CL" in extrinsic:
            return np.asarray(extrinsic["T_CL"], dtype=np.float64).reshape(4, 4)
        return None

    def image_label_pos_to_image_pixel(self, label_pos: QPoint) -> tuple[float, float] | None:
        if (
            self.current_image_pixmap is None
            or self.current_image_pixmap.isNull()
            or self.image_display_size is None
            or self.image_display_size.width() <= 0
            or self.image_display_size.height() <= 0
        ):
            return None

        local_x = label_pos.x() - self.image_display_origin.x()
        local_y = label_pos.y() - self.image_display_origin.y()
        if (
            local_x < 0
            or local_y < 0
            or local_x >= self.image_display_size.width()
            or local_y >= self.image_display_size.height()
        ):
            return None

        image_x = local_x * self.current_image_pixmap.width() / self.image_display_size.width()
        image_y = local_y * self.current_image_pixmap.height() / self.image_display_size.height()
        return float(image_x), float(image_y)

    def handle_image_pixel_clicked(self, label_pos: QPoint) -> bool:
        if not self.cross_view_ray_settings.get("enabled", True):
            return False
        T_CL = self.optimized_camera_lidar_transform()
        if T_CL is None:
            self.statusBar().showMessage(
                "Run LiDAR-Cam Extrinsics before using image-to-point ray picking."
            )
            return False
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False

        image_pixel = self.image_label_pos_to_image_pixel(label_pos)
        if image_pixel is None:
            return False

        points_L = self.point_cloud_viewer.points()
        if len(points_L) == 0:
            self.statusBar().showMessage("No visible point cloud points to pick.")
            return False

        projection = project_lidar_points_to_image(
            points_L,
            T_CL,
            self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_shape=(self.current_image_pixmap.height(), self.current_image_pixmap.width(), 3),
            min_depth_m=float(self.current_projection_overlay_settings().get("min_depth_m", 0.05)),
        )
        valid_mask = projection["in_image_mask"] & np.isfinite(projection["image_points"]).all(axis=1)
        if not np.any(valid_mask):
            self.statusBar().showMessage("No visible point cloud points project into this image.")
            return False

        indices = np.flatnonzero(valid_mask)
        clicked = np.asarray(image_pixel, dtype=np.float64)
        deltas = projection["image_points"][indices] - clicked
        distances_px = np.linalg.norm(deltas, axis=1)
        best_local = int(np.argmin(distances_px))
        best_distance_px = float(distances_px[best_local])
        max_distance_px = float(self.cross_view_ray_settings.get("max_pixel_distance", 80.0))
        if best_distance_px > max_distance_px:
            self.statusBar().showMessage(
                f"No projected LiDAR point within {max_distance_px:.0f} px of click."
            )
            return False

        best_index = int(indices[best_local])
        endpoint = np.asarray(points_L[best_index], dtype=np.float64).reshape(3)
        selected_projected_point = projection["image_points"][best_index]
        self.selected_image_point_px = (
            float(selected_projected_point[0]),
            float(selected_projected_point[1]),
        )
        self.point_cloud_viewer.set_pick_ray(
            endpoint,
            color=self.ray_rgba(),
            width=float(self.cross_view_ray_settings.get("width", 3.0)),
            endpoint_size=float(self.cross_view_ray_settings.get("endpoint_size", 20.0)),
        )
        if self.frames:
            self.show_frame(self.frame_slider.value(), update_point_cloud=False)
        self.statusBar().showMessage(
            "Selected LiDAR point "
            f"({endpoint[0]:.3f}, {endpoint[1]:.3f}, {endpoint[2]:.3f}) m | "
            f"{best_distance_px:.1f} px from click"
        )
        return True

    def apply_lidar_projection_overlay(
        self,
        image: np.ndarray,
        frame: FrameRecord,
    ) -> np.ndarray:
        settings = self.current_projection_overlay_settings()
        self.projection_overlay_settings = settings
        if not settings.get("show", True):
            return image
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        T_CL = self.projection_camera_lidar_transform(frame)
        if T_CL is None:
            return image

        try:
            points_L, reflectivity = self.projection_points_for_frame(frame, settings)
            if len(points_L) == 0:
                return image
            projection = visualize_lidar_projection_on_image(
                image,
                points_L,
                T_CL,
                self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                reflectivity=reflectivity,
                color_by=str(settings.get("color_by", "depth")),
                point_radius=max(
                    1,
                    int(round(float(settings.get("point_radius", 2)) * 1.5)),
                ),
                alpha=float(settings.get("alpha", 0.85)),
                max_points=int(settings.get("max_points", 30000)),
                min_depth_m=float(settings.get("min_depth_m", 0.05)),
            )
            frame.projection_result = {
                "drawn_count": len(projection["drawn_indices"]),
                "valid_count": int(np.sum(projection["projection"]["valid_mask"])),
                "in_image_count": int(np.sum(projection["projection"]["in_image_mask"])),
                "settings": settings,
            }
            return projection["image"]
        except Exception as exc:
            frame.projection_result = {
                "error": str(exc),
                "settings": settings,
            }
            return image

    def projection_points_for_frame(
        self,
        frame: FrameRecord,
        settings: dict,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        source = str(settings.get("source", "Full point cloud"))
        lidar_result = frame.lidar_processing_result or {}
        if source == "Detected centers":
            centers = np.asarray(
                lidar_result.get("centers_L", []),
                dtype=np.float64,
            ).reshape(-1, 3)
            return self.filter_projection_points(centers), None
        if source == "LiDAR-center filtered points":
            points = np.asarray(
                lidar_result.get("cluster_points", []),
                dtype=np.float64,
            ).reshape(-1, 3)
            return self.filter_projection_points(points), None

        lidar_dict = self.load_frame_lidar(frame)
        if self.point_cloud_filter_settings.get("enabled", False):
            lidar_dict, _visible_count, _total_count = self.filter_point_cloud_for_viewer(
                lidar_dict
            )
        points = np.asarray(lidar_dict.get("xyz"), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64), None
        reflectivity = None
        if "reflectivity" in lidar_dict:
            reflectivity = np.asarray(lidar_dict["reflectivity"], dtype=np.float64).reshape(-1)
            if len(reflectivity) != len(points):
                reflectivity = None
        elif "intensity" in lidar_dict:
            reflectivity = np.asarray(lidar_dict["intensity"], dtype=np.float64).reshape(-1)
            if len(reflectivity) != len(points):
                reflectivity = None
        return points[:, :3], reflectivity

    def draw_markers(self, image: np.ndarray, result: dict):
        corners = result.get("marker_corners") or []
        ids = result.get("marker_ids")
        if len(corners) == 0:
            return

        style = self.marker_type_combo.currentText()
        size = self.marker_size_spin.value()
        for index, corner in enumerate(corners):
            pts = np.round(corner.reshape(-1, 2)).astype(np.int32)
            center = tuple(np.round(np.mean(pts, axis=0)).astype(int))
            if "Outline" in style:
                cv2.polylines(image, [pts], True, (0, 220, 90), size, cv2.LINE_AA)
            if "Centers" in style:
                cv2.circle(image, center, max(2, size * 2), (255, 180, 0), -1, cv2.LINE_AA)
            if ids is not None and len(ids) > index:
                cv2.putText(
                    image,
                    str(int(ids[index][0])),
                    (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def draw_corners(self, image: np.ndarray, result: dict):
        corners = result.get("charuco_corners")
        if corners is None:
            return

        style = self.corner_type_combo.currentText()
        size = self.corner_size_spin.value()
        for point in corners.reshape(-1, 2):
            x, y = np.round(point).astype(int)
            if style == "Circle":
                cv2.circle(image, (x, y), size, (40, 40, 255), -1, cv2.LINE_AA)
            elif style == "Square":
                cv2.rectangle(
                    image,
                    (x - size, y - size),
                    (x + size, y + size),
                    (40, 40, 255),
                    -1,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(image, (x - size, y), (x + size, y), (40, 40, 255), 2, cv2.LINE_AA)
                cv2.line(image, (x, y - size), (x, y + size), (40, 40, 255), 2, cv2.LINE_AA)

    def draw_board_axes(self, image: np.ndarray, frame: FrameRecord):
        pose = frame.camera_board_pose
        if (
            pose is None
            or not pose.get("success", False)
            or self.camera_matrix is None
            or self.dist_coeffs is None
        ):
            return

        cv2.drawFrameAxes(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            pose["rvec"],
            pose["tvec"],
            float(self.axis_length_spin.value()),
            2,
        )

    @staticmethod
    def image_to_pixmap(image: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimage)

    def update_frame_table(self):
        self._updating_table = True
        self.frame_table.setRowCount(len(self.frames))
        for row, frame in enumerate(self.frames):
            result = frame.charuco_result or {}
            valid = bool(result.get("valid", False))
            detected = frame.charuco_result is not None
            pose = frame.camera_board_pose or {}
            pnp_rmse = ""
            if pose.get("success", False):
                pnp_rmse = f"{pose['reprojection_rmse_px']:.3f}"
            elif pose:
                pnp_rmse = "Failed"
            use_item = QTableWidgetItem()
            use_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            use_item.setCheckState(
                Qt.CheckState.Checked if frame.enabled else Qt.CheckState.Unchecked
            )
            self.frame_table.setItem(row, 0, use_item)

            values = [
                str(frame.frame_index),
                frame.image_path.name,
                frame.lidar_path.name,
                str(result.get("num_markers", "")),
                str(result.get("num_corners", "")),
                "Valid" if valid else ("Invalid" if detected else "Not detected"),
                pnp_rmse,
            ]
            for column_offset, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if detected:
                    color = QColor("#d9f2e3") if valid else QColor("#f7d7d7")
                    item.setBackground(color)
                    use_item.setBackground(color)
                if not frame.enabled:
                    item.setForeground(QColor("#777777"))
                    use_item.setForeground(QColor("#777777"))
                self.frame_table.setItem(row, column_offset, item)
        self.frame_table.resizeColumnsToContents()
        self._updating_table = False

    def update_info(self):
        total = len(self.frames)
        detected = [frame for frame in self.frames if frame.charuco_result is not None]
        valid = [
            frame for frame in detected
            if frame.charuco_result and frame.charuco_result.get("valid", False)
        ]
        valid_ids = {id(frame) for frame in valid}
        invalid = [frame for frame in detected if id(frame) not in valid_ids]
        calibration_ready = self.calibration_frames()
        posed = [
            frame for frame in self.frames
            if frame.camera_board_pose and frame.camera_board_pose.get("success", False)
        ]

        valid_indices = ", ".join(str(frame.frame_index) for frame in valid) or "None"
        invalid_indices = ", ".join(str(frame.frame_index) for frame in invalid) or "None"
        enabled_indices = ", ".join(str(frame.frame_index) for frame in calibration_ready) or "None"
        lines = [
            f"Frames: {total}",
            f"Detected: {len(detected)}",
            f"Valid: {len(valid)}",
            f"Invalid: {len(invalid)}",
            f"Enabled valid frames: {len(calibration_ready)}",
            f"SolvePnP poses: {len(posed)}",
            f"Valid frames: {valid_indices}",
            f"Invalid frames: {invalid_indices}",
            f"Calibration frames: {enabled_indices}",
        ]
        self.append_log("Detection summary:\n" + "\n".join(lines))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames:
            self.update_image_display()

    def eventFilter(self, obj, event):
        if obj is self.image_label:
            if event.type() == QEvent.Type.Resize:
                if self.frames:
                    self.update_image_display()
            elif event.type() == QEvent.Type.Wheel:
                self.zoom_image(event.angleDelta().y())
                return True
            elif event.type() == QEvent.Type.MouseButtonPress:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self.current_image_pixmap is not None
                ):
                    self.image_drag_last_pos = event.position().toPoint()
                    self.image_click_start_pos = self.image_drag_last_pos
                    self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            elif event.type() == QEvent.Type.MouseMove:
                if (
                    event.buttons() & Qt.MouseButton.LeftButton
                    and self.image_drag_last_pos is not None
                    and self.current_image_pixmap is not None
                ):
                    current_pos = event.position().toPoint()
                    delta = current_pos - self.image_drag_last_pos
                    self.image_drag_last_pos = current_pos
                    self.image_pan_offset = self.image_pan_offset + delta
                    self.update_image_display()
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    release_pos = event.position().toPoint()
                    click_start = self.image_click_start_pos
                    self.image_drag_last_pos = None
                    self.image_click_start_pos = None
                    self.image_label.unsetCursor()
                    if (
                        click_start is not None
                        and (release_pos - click_start).manhattanLength() <= 4
                    ):
                        self.handle_image_pixel_clicked(release_pos)
                    return True
        return super().eventFilter(obj, event)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(modern_stylesheet)
    app.setFont(QFont("Inter", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
