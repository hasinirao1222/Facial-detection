from __future__ import annotations

import time

import cv2

from detector import DetectionConfig, FaceDetector


def main() -> None:
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

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to access webcam. Check camera permissions and device usage.")

    prev_time = time.time()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Skipping frame because webcam read failed.")
                continue

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
                "Press Q to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
            )

            cv2.imshow("Real-Time Facial Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
