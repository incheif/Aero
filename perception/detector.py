"""
Image-based object detector for tabletop workspace.
Segments tabletop blocks directly from raw CameraObservation (RGB image).
Operates strictly as a sensor boundary without access to hidden physics state.
"""

import math
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

from .models import ObjectDetection
from simulation.sensor_interfaces import CameraObservation


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB [0..255] to HSV (H in [0..360), S in [0..1], V in [0..1])."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    c_max = max(rf, gf, bf)
    c_min = min(rf, gf, bf)
    delta = c_max - c_min

    # Hue
    if delta == 0.0:
        h = 0.0
    elif c_max == rf:
        h = (60.0 * ((gf - bf) / delta)) % 360.0
    elif c_max == gf:
        h = (60.0 * ((bf - rf) / delta) + 120.0) % 360.0
    else:
        h = (60.0 * ((rf - gf) / delta) + 240.0) % 360.0

    # Saturation
    s = 0.0 if c_max == 0.0 else delta / c_max

    # Value
    v = c_max
    return h, s, v


def classify_block_color(r: int, g: int, b: int) -> Optional[str]:
    """
    Classifies a single RGB pixel into one of the canonical block colors.
    Returns None if background or table.
    """
    h, s, v = rgb_to_hsv(r, g, b)

    # Discard table and background (low saturation or very low value)
    if s < 0.35 or v < 0.30:
        return None

    # Distinguish primary and secondary blocks
    if h >= 345.0 or h < 14.0:
        return "red"
    elif 14.0 <= h < 38.0:
        return "orange"
    elif 38.0 <= h < 70.0:
        return "yellow"
    elif 70.0 <= h < 165.0:
        return "green"
    elif 190.0 <= h < 255.0:
        return "blue"
    elif 260.0 <= h < 320.0:
        return "purple"

    return None


class BlockColorDetector:
    """
    Segmenter and detector for tabletop colored blocks.
    Takes only CameraObservation and returns detected objects with 2D bounds and centroids.
    """
    def __init__(self, min_cluster_pixels: int = 4):
        self.min_cluster_pixels = min_cluster_pixels

    def detect(self, observation: CameraObservation) -> List[ObjectDetection]:
        """
        Process camera observation to find all visible colored blocks.
        """
        rgb_img = observation.rgb
        width, height = rgb_img.size
        rgb_arr = np.array(rgb_img)

        # 1. Label each pixel with color category
        color_grid: List[List[Optional[str]]] = [[None for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                r, g, b = int(rgb_arr[y, x, 0]), int(rgb_arr[y, x, 1]), int(rgb_arr[y, x, 2])
                c_label = classify_block_color(r, g, b)
                if c_label:
                    color_grid[y][x] = c_label

        # 2. Extract connected components for each color
        visited = [[False for _ in range(width)] for _ in range(height)]
        detections: List[ObjectDetection] = []

        for y in range(height):
            for x in range(width):
                c_label = color_grid[y][x]
                if c_label is not None and not visited[y][x]:
                    # Connected component BFS
                    queue = [(x, y)]
                    visited[y][x] = True
                    cluster_pixels: List[Tuple[int, int]] = []

                    while queue:
                        cx, cy = queue.pop(0)
                        cluster_pixels.append((cx, cy))

                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if not visited[ny][nx] and color_grid[ny][nx] == c_label:
                                    visited[ny][nx] = True
                                    queue.append((nx, ny))

                    # Filter noise
                    if len(cluster_pixels) >= self.min_cluster_pixels:
                        xs = [p[0] for p in cluster_pixels]
                        ys = [p[1] for p in cluster_pixels]
                        u_min, u_max = min(xs), max(xs)
                        v_min, v_max = min(ys), max(ys)
                        centroid_u = sum(xs) / float(len(xs))
                        centroid_v = sum(ys) / float(len(ys))

                        # Average RGB
                        r_avg = int(np.mean([rgb_arr[p[1], p[0], 0] for p in cluster_pixels]))
                        g_avg = int(np.mean([rgb_arr[p[1], p[0], 1] for p in cluster_pixels]))
                        b_avg = int(np.mean([rgb_arr[p[1], p[0], 2] for p in cluster_pixels]))

                        conf = min(0.99, 0.90 + (len(cluster_pixels) / 100.0) * 0.08)

                        detection = ObjectDetection(
                            class_name="block",
                            color_name=c_label,
                            color_rgb=(r_avg, g_avg, b_avg),
                            bbox_xyxy=(u_min, v_min, u_max, v_max),
                            centroid_uv=(centroid_u, centroid_v),
                            pixel_area=len(cluster_pixels),
                            confidence=conf,
                            timestamp=observation.timestamp
                        )
                        detections.append(detection)

        return detections
