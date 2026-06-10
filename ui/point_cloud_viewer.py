from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


class PointCloudViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = np.empty((0, 3), dtype=np.float32)
        self._reflectivity = None
        self._color_by = "depth"
        self._scatter = None
        self._cluster_scatter = None
        self._center_scatter = None
        self._ray_line = None
        self._ray_endpoint_scatter = None
        self._grid = None
        self._axis = None

        self.reset_view_button = QPushButton("Reset")
        self.reset_view_button.clicked.connect(self.reset_camera)

        self.top_view_button = QPushButton("Top")
        self.top_view_button.clicked.connect(self.set_top_view)

        self.front_view_button = QPushButton("Front")
        self.front_view_button.clicked.connect(self.set_front_view)

        self.side_view_button = QPushButton("Side")
        self.side_view_button.clicked.connect(self.set_side_view)

        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setRange(1, 24)
        self.point_size_slider.setValue(4)
        self.point_size_slider.valueChanged.connect(self._handle_point_size_changed)
        self.point_size_label = QLabel()
        self.point_size_label.setMinimumWidth(48)
        self.point_size_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_point_size_label()

        self.controls_widget = QWidget()
        controls = QHBoxLayout()
        controls.addWidget(self.reset_view_button)
        controls.addWidget(self.top_view_button)
        controls.addWidget(self.front_view_button)
        controls.addWidget(self.side_view_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Point size"))
        controls.addWidget(self.point_size_slider)
        controls.addWidget(self.point_size_label)
        controls.addStretch(1)
        self.controls_widget.setLayout(controls)
        self.controls_widget.setStyleSheet(
            "QWidget { color: #111827; }"
            "QLabel { color: #111827; }"
            "QSlider { min-width: 120px; }"
            "QPushButton { color: #111827; }"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if gl is None:
            self.view = None
            missing_label = QLabel(
                "pyqtgraph with OpenGL support is required for the point cloud viewer."
            )
            missing_label.setWordWrap(True)
            layout.addWidget(missing_label, stretch=1)
        else:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("#111827")
            self.view.setCameraPosition(distance=20.0, elevation=25.0, azimuth=-45.0)

            self._grid = gl.GLGridItem()
            self._grid.setSize(20, 20)
            self._grid.setSpacing(1, 1)
            self.view.addItem(self._grid)

            self._axis = gl.GLAxisItem()
            self._axis.setSize(2, 2, 2)
            self.view.addItem(self._axis)

            self._scatter = gl.GLScatterPlotItem(
                pos=self._points,
                color=np.empty((0, 4), dtype=np.float32),
                size=self.point_size_value(),
                pxMode=True,
            )
            self.view.addItem(self._scatter)

            self._cluster_scatter = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=4.0,
                pxMode=True,
            )
            self.view.addItem(self._cluster_scatter)

            self._center_scatter = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=12.0,
                pxMode=True,
            )
            self.view.addItem(self._center_scatter)

            self._ray_line = gl.GLLinePlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=(1.0, 1.0, 0.0, 1.0),
                width=3.0,
                antialias=True,
            )
            self.view.addItem(self._ray_line)

            self._ray_endpoint_scatter = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=18.0,
                pxMode=True,
            )
            self.view.addItem(self._ray_endpoint_scatter)
            layout.addWidget(self.view, stretch=1)

        self.setLayout(layout)

    def set_point_cloud(self, lidar_dict: dict, label: str = ""):
        points = np.asarray(lidar_dict.get("xyz", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            self._points = np.empty((0, 3), dtype=np.float32)
            self._reflectivity = None
        else:
            self._points = points[:, :3]
            reflectivity = lidar_dict.get("reflectivity")
            if reflectivity is None:
                reflectivity = lidar_dict.get("intensity")
            if reflectivity is None and "colors" in lidar_dict:
                colors = np.asarray(lidar_dict["colors"], dtype=np.float32)
                if colors.ndim == 2 and len(colors) == len(self._points):
                    reflectivity = np.mean(colors[:, :3], axis=1)
            if reflectivity is None:
                self._reflectivity = None
            else:
                reflectivity = np.asarray(reflectivity, dtype=np.float32).reshape(-1)
                self._reflectivity = (
                    reflectivity if len(reflectivity) == len(self._points) else None
                )

        self._refresh_scatter()
        self.clear_lidar_center_overlay()
        self.clear_pick_ray()

    def clear(self):
        self._points = np.empty((0, 3), dtype=np.float32)
        self._refresh_scatter()
        self.clear_lidar_center_overlay()
        self.clear_pick_ray()

    def points(self) -> np.ndarray:
        return np.asarray(self._points, dtype=np.float32)

    def set_color_by(self, color_by: str):
        color_by = str(color_by).lower()
        if color_by not in {"depth", "reflectivity", "height"}:
            color_by = "depth"
        self._color_by = color_by
        self._refresh_scatter()

    def set_lidar_center_overlay(
        self,
        cluster_points: np.ndarray | None = None,
        cluster_labels: np.ndarray | None = None,
        centers: np.ndarray | None = None,
        show_clusters: bool = True,
        show_centers: bool = True,
    ):
        if self._cluster_scatter is None or self._center_scatter is None:
            return

        if show_clusters and cluster_points is not None and len(cluster_points) > 0:
            cluster_points = np.asarray(cluster_points, dtype=np.float32).reshape(-1, 3)
            cluster_labels = np.asarray(cluster_labels, dtype=np.int32).reshape(-1)
            colors = self._cluster_colors(cluster_labels)
            self._cluster_scatter.setData(
                pos=cluster_points,
                color=colors,
                size=5.0,
                pxMode=True,
            )
        else:
            self._cluster_scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )

        if show_centers and centers is not None and len(centers) > 0:
            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
            center_colors = np.tile(
                np.asarray([[1.0, 0.05, 0.05, 1.0]], dtype=np.float32),
                (len(centers), 1),
            )
            self._center_scatter.setData(
                pos=centers,
                color=center_colors,
                size=14.0,
                pxMode=True,
            )
        else:
            self._center_scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )

    def clear_lidar_center_overlay(self):
        self.set_lidar_center_overlay(show_clusters=False, show_centers=False)

    def set_pick_ray(
        self,
        endpoint: np.ndarray,
        color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0),
        width: float = 3.0,
        endpoint_size: float = 18.0,
    ):
        if self._ray_line is None or self._ray_endpoint_scatter is None:
            return

        endpoint = np.asarray(endpoint, dtype=np.float32).reshape(3)
        line_points = np.vstack([
            np.zeros(3, dtype=np.float32),
            endpoint,
        ])
        self._ray_line.setData(
            pos=line_points,
            color=color,
            width=float(width),
            antialias=True,
        )
        self._ray_endpoint_scatter.setData(
            pos=endpoint.reshape(1, 3),
            color=np.asarray([color], dtype=np.float32),
            size=float(endpoint_size),
            pxMode=True,
        )

    def clear_pick_ray(self):
        if self._ray_line is not None:
            self._ray_line.setData(pos=np.empty((0, 3), dtype=np.float32))
        if self._ray_endpoint_scatter is not None:
            self._ray_endpoint_scatter.setData(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
            )

    def reset_camera(self):
        if self.view is None:
            return
        center = self._point_cloud_center()
        distance = self._camera_distance()
        self.view.opts["center"] = center
        self.view.setCameraPosition(distance=distance, elevation=25.0, azimuth=-45.0)

    def set_top_view(self):
        if self.view is None:
            return
        self.view.opts["center"] = self._point_cloud_center()
        self.view.setCameraPosition(
            distance=self._camera_distance(),
            elevation=90.0,
            azimuth=-90.0,
        )

    def set_front_view(self):
        if self.view is None:
            return
        self.view.opts["center"] = self._point_cloud_center()
        self.view.setCameraPosition(
            distance=self._camera_distance(),
            elevation=0.0,
            azimuth=-90.0,
        )

    def set_side_view(self):
        if self.view is None:
            return
        self.view.opts["center"] = self._point_cloud_center()
        self.view.setCameraPosition(
            distance=self._camera_distance(),
            elevation=0.0,
            azimuth=0.0,
        )

    def _refresh_scatter(self):
        if self._scatter is None:
            return
        colors = self._point_colors()
        point_size = self.point_size_value()
        self._scatter.setData(
            pos=self._points,
            color=colors,
            size=point_size,
            pxMode=True,
        )
        if self.view is not None:
            self.view.update()

    def _point_colors(self) -> np.ndarray:
        if self._color_by == "reflectivity" and self._reflectivity is not None:
            return self._value_colors(self._reflectivity)
        if self._color_by == "height" and len(self._points) > 0:
            return self._value_colors(self._points[:, 2])
        return self._range_colors(self._points)

    def point_size_value(self) -> float:
        return float(self.point_size_slider.value()) * 0.5

    def _update_point_size_label(self):
        self.point_size_label.setText(f"{self.point_size_value():.1f}px")

    def _handle_point_size_changed(self, *_args):
        self._update_point_size_label()
        self._refresh_scatter()

    def _point_cloud_center(self):
        if gl is None:
            return None
        import pyqtgraph as pg

        if len(self._points) == 0:
            return pg.Vector(0.0, 0.0, 0.0)
        finite = self._points[np.isfinite(self._points).all(axis=1)]
        if len(finite) == 0:
            return pg.Vector(0.0, 0.0, 0.0)
        center = np.median(finite, axis=0)
        return pg.Vector(float(center[0]), float(center[1]), float(center[2]))

    def _camera_distance(self) -> float:
        if len(self._points) == 0:
            return 20.0
        finite = self._points[np.isfinite(self._points).all(axis=1)]
        if len(finite) == 0:
            return 20.0
        spread = np.percentile(finite, 95, axis=0) - np.percentile(finite, 5, axis=0)
        return max(float(np.linalg.norm(spread)) * 1.4, 5.0)

    @staticmethod
    def _range_colors(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        ranges = np.linalg.norm(points[:, :3], axis=1)
        return PointCloudViewerWidget._value_colors(ranges)

    @staticmethod
    def _value_colors(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(values) == 0:
            return np.empty((0, 4), dtype=np.float32)

        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            norm = np.zeros(len(values), dtype=np.float32)
        else:
            low = float(np.percentile(finite, 2))
            high = float(np.percentile(finite, 98))
            if high <= low:
                high = low + 1e-6
            norm = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

        colors = np.empty((len(values), 4), dtype=np.float32)
        colors[:, 0] = norm
        colors[:, 1] = 1.0 - np.abs(norm - 0.5) * 1.6
        colors[:, 2] = 1.0 - norm
        colors[:, 3] = 0.95
        colors[:, :3] = np.clip(colors[:, :3], 0.05, 1.0)
        return colors

    @staticmethod
    def _cluster_colors(labels: np.ndarray) -> np.ndarray:
        if len(labels) == 0:
            return np.empty((0, 4), dtype=np.float32)
        palette = np.asarray(
            [
                [0.12, 0.55, 1.0, 1.0],
                [0.2, 0.85, 0.35, 1.0],
                [1.0, 0.7, 0.1, 1.0],
                [0.85, 0.25, 1.0, 1.0],
                [0.1, 0.85, 0.85, 1.0],
                [1.0, 0.35, 0.35, 1.0],
            ],
            dtype=np.float32,
        )
        colors = np.empty((len(labels), 4), dtype=np.float32)
        for index, label in enumerate(labels):
            if label < 0:
                colors[index] = [0.45, 0.45, 0.45, 0.45]
            else:
                colors[index] = palette[int(label) % len(palette)]
        return colors
