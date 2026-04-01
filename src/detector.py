from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class DetectionConfig:
    backend: str = "opencv"
    scale_factor: float = 1.05
    min_neighbors: int = 4
    min_size: tuple[int, int] = (60, 60)
    display_scale: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    use_profile_detector: bool = True
    overlap_threshold: float = 0.35


class FaceDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.backend = self.config.backend.lower()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )

        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        alt_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        self.classifiers = [
            cv2.CascadeClassifier(frontal_path),
            cv2.CascadeClassifier(alt_path),
        ]
        if self.config.use_profile_detector:
            self.classifiers.append(cv2.CascadeClassifier(profile_path))

        if any(classifier.empty() for classifier in self.classifiers):
            raise RuntimeError("Unable to load one or more OpenCV Haar cascade files.")

    def detect(self, frame):
        resized = self._resize_frame(frame)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        faces = []
        weights = []
        for classifier in self.classifiers:
            detections = classifier.detectMultiScale2(
                enhanced,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=self.config.min_size,
            )
            if len(detections) != 2:
                continue

            boxes, level_weights = detections
            for (x, y, w, h), weight in zip(boxes, level_weights):
                faces.append([int(x), int(y), int(w), int(h)])
                weights.append(float(weight))

        merged_faces = self._non_max_suppression(faces, weights)
        return resized, merged_faces

    def _resize_frame(self, frame):
        if self.config.display_scale == 1.0:
            return frame

        return cv2.resize(
            frame,
            None,
            fx=self.config.display_scale,
            fy=self.config.display_scale,
            interpolation=cv2.INTER_AREA,
        )

    def _non_max_suppression(self, boxes, weights):
        if not boxes:
            return []

        x1 = [box[0] for box in boxes]
        y1 = [box[1] for box in boxes]
        x2 = [box[0] + box[2] for box in boxes]
        y2 = [box[1] + box[3] for box in boxes]
        areas = [(x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1) for i in range(len(boxes))]
        order = sorted(range(len(boxes)), key=lambda idx: weights[idx], reverse=True)
        keep = []

        while order:
            current = order.pop(0)
            keep.append(tuple(boxes[current]))
            remaining = []
            for idx in order:
                xx1 = max(x1[current], x1[idx])
                yy1 = max(y1[current], y1[idx])
                xx2 = min(x2[current], x2[idx])
                yy2 = min(y2[current], y2[idx])

                width = max(0, xx2 - xx1 + 1)
                height = max(0, yy2 - yy1 + 1)
                intersection = width * height
                union = areas[current] + areas[idx] - intersection
                overlap = intersection / union if union else 0

                if overlap < self.config.overlap_threshold:
                    remaining.append(idx)

            order = remaining

        return keep
