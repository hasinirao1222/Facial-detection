from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from detector import DetectionConfig, FaceDetector


def _parse_source(source: str) -> int | str:
    try:
        return int(source)
    except ValueError:
        return source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect faces in a webcam or video stream.")
    parser.add_argument(
        "--source",
        default="0",
        help="webcam index or video file path (default: 0)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("snapshots"),
        help="directory for saved detection snapshots (default: snapshots)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    detector = FaceDetector(
        DetectionConfig(
            backend="opencv",
            scale_factor=1.05,
            min_neighbors=4,
            min_size=(60, 60),
            display_scale=1.0,
            use_profile_detector=True,
        )
    )

    capture = cv2.VideoCapture(_parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    prev_time = time.time()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video source ended or could not provide a frame.")
                break

            processed_frame, faces = detector.detect(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (40, 200, 90), 2)
                cv2.putText(
                    processed_frame,
                    "Face",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 200, 90),
                    2,
                )

            current_time = time.time()
            fps = 1 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            cv2.putText(
                processed_frame,
                f"Faces: {len(faces)} | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                processed_frame,
                "S: save snapshot | Q: quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
            )

            cv2.imshow("Real-Time Facial Detection", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                snapshot_path = args.snapshot_dir / f"face-detection-{timestamp}.jpg"
                if cv2.imwrite(str(snapshot_path), processed_frame):
                    print(f"Saved snapshot: {snapshot_path}")
                else:
                    print(f"Unable to save snapshot: {snapshot_path}")
            elif key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
