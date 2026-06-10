# LiDAR Camera Calibration App

This app is a PyQt6 desktop tool for calibrating a camera and LiDAR pair using synchronized image and point-cloud frames. It guides the user through loading paired data, detecting a ChArUco calibration target, estimating camera intrinsics and per-frame board poses, detecting LiDAR target centers, optimizing the LiDAR-to-camera extrinsics, visualizing projected LiDAR points on images, and saving the final calibration.

## Authors
Siddharth Tourani, Akash Kumbar, Gnana Prakash 

## Dependencies

Use Python 3.10 or newer. The app depends on:

- `PyQt6` for the desktop UI.
- `numpy` for numeric arrays.
- `opencv-contrib-python` for image loading, ChArUco/ArUco detection, camera calibration, PnP, and projection.
- `scipy` for nonlinear extrinsic optimization.
- `open3d` for PLY/PCD point-cloud loading and point-cloud processing.
- `pyqtgraph`, `PyOpenGL`, and `PyOpenGL_accelerate` for the 3D point-cloud viewer.
- `tqdm` for progress reporting in calibration utilities.
- `matplotlib` for optional projection debugging plots.

Install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install PyQt6 numpy scipy opencv-contrib-python open3d pyqtgraph PyOpenGL PyOpenGL_accelerate tqdm matplotlib
```

Run the app from the repository root:

```bash
python -m main_app.ui.main_window
```

## Data Format

Load data as two directories: one image directory and one LiDAR directory. Files are paired by matching file stem after sorting. For example, `000123.png` is paired with `000123.npz`, `000123.npy`, `000123.ply`, or `000123.pcd`.

Supported image formats:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

Supported LiDAR formats:

- `.npz`
- `.npy`
- `.ply`
- `.pcd`

For `.npz` LiDAR files, the archive must contain `x`, `y`, and `z` arrays with one value per point. Optional per-point arrays are `intensity`, `range`, and `reflectivity`.

For `.npy` LiDAR files, the array must have shape `(N, 3+)`. The first three columns are interpreted as `x`, `y`, and `z`. If a fourth column exists, it is used as `reflectivity`.

For `.ply` and `.pcd` LiDAR files, points are loaded with Open3D. Point colors are loaded when present and can be used as a reflectivity-like visualization value.

LiDAR coordinates should be in meters. Calibration target dimensions are also handled in metric units internally. The default ChArUco setup uses `DICT_5X5_1000` and a `9 x 9` board, with configurable detection settings in the app.

## Features

- Step-by-step workflow for loading data, ChArUco detection, camera intrinsics, LiDAR target detection, LiDAR-camera extrinsics, and calibration export.
- Image viewer with zoom, pan, ChArUco marker overlays, board axes, projected LiDAR overlays, and clicked-point highlighting.
- 3D point-cloud viewer with depth, height, and reflectivity coloring.
- Point-cloud range filtering by `x`, `y`, `z`, and distance, with filters used for display and image projection.
- LiDAR target center detection with plane filtering, clustering, and board-center matching.
- LiDAR-camera extrinsic candidate selection followed by nonlinear optimization.
- Cross-view interaction after extrinsics calibration: clicking an image point selects the nearest projected LiDAR point and draws a bright ray from the LiDAR-frame origin in the point-cloud viewer. Adjustable ray appearance and point-cloud point size.
- Calibration export to `.json` or `.npz`, including camera intrinsics, distortion coefficients, optimized `T_CL`, inverse `T_LC`, per-frame extrinsic metadata, and save-time metadata.
