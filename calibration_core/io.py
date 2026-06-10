from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Any, Iterable, List, Optional
import os
from tqdm import tqdm


SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SUPPORTED_LIDAR_SUFFIXES = (".npz", ".npy", ".ply", ".pcd")


@dataclass
class FrameRecord:
    frame_index: int
    image_path: Path
    lidar_path: Path
    image: Optional[np.ndarray] = None
    lidar_dict: Optional[dict] = None
    charuco_result: Optional[Any] = None
    camera_board_pose: Optional[Any] = None
    lidar_processing_result: Optional[Any] = None
    lidar_board_pose: Optional[Any] = None
    camera_lidar_extrinsic: Optional[Any] = None
    projection_result: Optional[Any] = None
    enabled: bool = True


def _normalize_suffix(suffix: str) -> str:
    if not suffix:
        raise ValueError("File suffix cannot be empty.")
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return suffix.lower()


def _normalize_suffixes(suffixes: str | Iterable[str] | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if suffixes is None:
        return defaults
    if isinstance(suffixes, str):
        return (_normalize_suffix(suffixes),)
    normalized = tuple(_normalize_suffix(suffix) for suffix in suffixes)
    if not normalized:
        raise ValueError("At least one file suffix is required.")
    return normalized


def _files_by_stem(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    paths_by_stem = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in paths_by_stem:
            raise RuntimeError(
                f"Duplicate stem '{path.stem}' in {directory} for suffixes {suffixes}."
            )
        paths_by_stem[path.stem] = path
    return paths_by_stem


def create_frame_records(
    image_dir: str | Path,
    lidar_dir: str | Path,
    image_suffix: str | Iterable[str] | None = None,
    lidar_suffix: str | Iterable[str] | None = None,
) -> List[FrameRecord]:
    image_dir = Path(image_dir)
    lidar_dir = Path(lidar_dir)
    image_suffixes = _normalize_suffixes(image_suffix, SUPPORTED_IMAGE_SUFFIXES)
    lidar_suffixes = _normalize_suffixes(lidar_suffix, SUPPORTED_LIDAR_SUFFIXES)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"LiDAR directory does not exist: {lidar_dir}")

    image_paths_by_stem = _files_by_stem(image_dir, image_suffixes)
    lidar_paths_by_stem = _files_by_stem(lidar_dir, lidar_suffixes)

    if not image_paths_by_stem:
        raise FileNotFoundError(
            f"No image files matching {image_suffixes} in {image_dir}"
        )

    missing_lidar_stems = [
        stem for stem in image_paths_by_stem
        if stem not in lidar_paths_by_stem
    ]
    if missing_lidar_stems:
        preview = ", ".join(missing_lidar_stems[:10])
        suffix = "" if len(missing_lidar_stems) <= 10 else ", ..."
        raise FileNotFoundError(
            f"Missing LiDAR files for {len(missing_lidar_stems)} image frame(s): "
            f"{preview}{suffix}"
        )

    records = []
    for frame_index, stem in enumerate(sorted(image_paths_by_stem)):
        records.append(
            FrameRecord(
                frame_index=frame_index,
                image_path=image_paths_by_stem[stem],
                lidar_path=lidar_paths_by_stem[stem],
            )
        )
    return records


def load_dataset(
    image_dir: str | Path,
    lidar_dir: str | Path,
    image_suffix: str | Iterable[str] | None = None,
    lidar_suffix: str | Iterable[str] | None = None,
) -> List[FrameRecord]:
    return create_frame_records(
        image_dir=image_dir,
        lidar_dir=lidar_dir,
        image_suffix=image_suffix,
        lidar_suffix=lidar_suffix,
    )


def read_images(image_paths: List[str]) -> List:
    import cv2

    images = [cv2.imread(str(p)) for p in image_paths]
    return images


def _point_cloud_dict_from_npz(pt_cld) -> dict:
    x, y, z = pt_cld["x"], pt_cld["y"], pt_cld["z"]
    pts_3d = np.stack([x, y, z], axis=1)
    point_cloud = {"xyz": pts_3d}
    for key in ("intensity", "range", "reflectivity"):
        if key in pt_cld:
            point_cloud[key] = pt_cld[key]
    return point_cloud


def read_lidar_file(lidar_path: str | Path) -> dict:
    lidar_path = Path(lidar_path)
    suffix = lidar_path.suffix.lower()

    if suffix == ".npz":
        with np.load(lidar_path) as pt_cld:
            return _point_cloud_dict_from_npz(pt_cld)

    if suffix == ".npy":
        points = np.asarray(np.load(lidar_path))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"Expected {lidar_path} to contain an (N, 3+) array, "
                f"got shape {points.shape}."
            )
        point_cloud = {"xyz": points[:, :3]}
        if points.shape[1] > 3:
            point_cloud["reflectivity"] = points[:, 3]
        return point_cloud

    if suffix in {".ply", ".pcd"}:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(lidar_path))
        points = np.asarray(pcd.points, dtype=np.float64)
        if points.size == 0:
            raise ValueError(f"No points were read from {lidar_path}.")

        point_cloud = {"xyz": points}
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if len(colors) == len(points):
            point_cloud["colors"] = colors
        return point_cloud

    raise NotImplementedError(f"Unsupported LiDAR file format: {lidar_path.suffix}")


def read_corresponding_lidar_frames(image_paths: List[str], \
                                    lidar_dir: str, \
                                    lidar_suffix: str) -> List[str]:
    lidar_filepaths = []
    point_clouds = []
    for img_path in tqdm(image_paths):
        file_stem = Path(img_path).stem

        if lidar_suffix in SUPPORTED_LIDAR_SUFFIXES:

            lidar_file_path = os.path.join(lidar_dir, file_stem + lidar_suffix)
            pt_cld_dict = read_lidar_file(lidar_file_path)

            lidar_filepaths.append(lidar_file_path)
            point_clouds.append(pt_cld_dict)

        else:
            raise NotImplementedError(
                f"Currently only {SUPPORTED_LIDAR_SUFFIXES} files are supported"
            )

    return lidar_filepaths, point_clouds
