import os
import cv2
import time
import subprocess
import numpy as np
from ultralytics import YOLO
import shutil
from dotenv import load_dotenv

load_dotenv()


class VehicleDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Load model once at startup
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # ✅ Print classes at startup so you can verify
        print(f"[VehicleDetector] Model loaded: {model_path}")
        print(f"[VehicleDetector] Classes: {self.class_names}")

    # ─────────────────────────────────────────────
    # IMAGE INFERENCE
    # ─────────────────────────────────────────────
    def predict_image(self, image: np.ndarray, conf=0.15, iou=0.5):
        """
        conf=0.15 — foggy images have lower confidence scores.
        iou=0.5   — reduces duplicate suppression in fog.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy image array.")

        start_time = time.time()

        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False
        )

        inference_time = round(time.time() - start_time, 4)

        result = results[0]
        annotated = result.plot()

        counts = {}
        if result.boxes is not None and result.boxes.cls is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for cls_id in classes:
                name = self.class_names[cls_id]
                counts[name] = counts.get(name, 0) + 1

        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        return annotated, counts, inference_time

    # ─────────────────────────────────────────────
    # FFMPEG PATH RESOLVER — NO HARDCODING
    # ─────────────────────────────────────────────
    @staticmethod
    def _get_ffmpeg_path() -> str:
        """
        Resolve ffmpeg path with NO hardcoded paths.

        Priority:
          1. FFMPEG_PATH in .env file  ← set this once, works everywhere
          2. System PATH               ← works if ffmpeg added to PATH
          3. Clear error message       ← tells user exactly what to do

        To set up:
          Add to your .env file:
          FFMPEG_PATH=C:\\path\\to\\ffmpeg.exe
        """

        # 1. ✅ Read from .env / environment variable — no hardcoding needed
        env_path = os.environ.get("FFMPEG_PATH")
        if env_path:
            if os.path.isfile(env_path):
                print(f"[ffmpeg] Found via FFMPEG_PATH env: {env_path}")
                return env_path
            else:
                # Warn if env var is set but path is wrong
                raise RuntimeError(
                    f"FFMPEG_PATH is set in .env but file not found:\n"
                    f"  FFMPEG_PATH={env_path}\n"
                    f"Please check the path in your .env file."
                )

        # 2. ✅ Try system PATH
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            print(f"[ffmpeg] Found via system PATH: {ffmpeg}")
            return ffmpeg

        # 3. ❌ Nothing found — give clear instructions
        raise RuntimeError(
            "\n"
            "ffmpeg not found. Choose one of these fixes:\n\n"
            "  OPTION 1 (Recommended) — Add to your .env file:\n"
            "    FFMPEG_PATH=C:\\path\\to\\ffmpeg\\bin\\ffmpeg.exe\n\n"
            "  OPTION 2 — Add ffmpeg to System PATH:\n"
            "    Add C:\\path\\to\\ffmpeg\\bin to System Environment Variables\n"
            "    Then restart your terminal and uvicorn\n\n"
            "  OPTION 3 — Install ffmpeg fresh:\n"
            "    winget install --id Gyan.FFmpeg -e\n"
            "    (This auto-adds ffmpeg to PATH)\n"
        )

    # ─────────────────────────────────────────────
    # VIDEO FILE INFERENCE
    # ─────────────────────────────────────────────
    def predict_video(self, input_path: str, output_path: str, conf=0.15, iou=0.5):
        """
        conf=0.15 — foggy videos have lower confidence scores.
        iou=0.5   — reduces over-suppression of overlapping vehicles.

        Counting strategy:
        - Primary   : Track unique vehicle IDs (one count per unique vehicle)
        - Fallback  : When tracker loses IDs, count raw detections per frame
        - First-seen: Only record FIRST class seen per track_id to prevent
                      class flickering (e.g. bus -> car on same ID)
        """

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise RuntimeError("Could not open input video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            raise RuntimeError("Invalid video dimensions.")

        raw_output = output_path.replace(".mp4", "_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_output, fourcc, fps, (width, height))

        # ✅ Primary counter — unique tracked vehicles
        seen_track_ids: dict[int, str] = {}

        # ✅ Debug counters
        raw_detections_per_class: dict[str, int] = {}
        frames_with_no_ids = 0
        frame_count = 0

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = self.model.track(
                frame,
                conf=conf,
                iou=iou,
                persist=True,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()
            out.write(annotated)

            if result.boxes is None:
                continue

            # ✅ Always count raw detections for debug visibility
            if result.boxes.cls is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for cls_id in classes:
                    name = self.class_names[cls_id]
                    raw_detections_per_class[name] = (
                        raw_detections_per_class.get(name, 0) + 1
                    )

            if result.boxes.id is not None:
                # ✅ Normal path — tracking IDs available
                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for track_id, cls_id in zip(ids, classes):
                    class_name = self.class_names[cls_id]
                    # ✅ First-seen only — prevents bus/truck overwritten by car
                    if track_id not in seen_track_ids:
                        seen_track_ids[track_id] = class_name

            else:
                # ✅ Fallback — tracker dropped IDs (occlusion, fog, fast motion)
                frames_with_no_ids += 1
                classes = result.boxes.cls.cpu().numpy().astype(int)
                for i, cls_id in enumerate(classes):
                    class_name = self.class_names[cls_id]
                    # Negative fake ID avoids collision with real track IDs
                    fake_id = -(frame_count * 1000 + i)
                    seen_track_ids[fake_id] = class_name

        cap.release()
        out.release()

        # ✅ Build final unique vehicle counts
        final_counts: dict[str, int] = {}
        for class_name in seen_track_ids.values():
            final_counts[class_name] = final_counts.get(class_name, 0) + 1

        final_counts = dict(
            sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
        )

        # ✅ Debug report in uvicorn console
        print("\n" + "=" * 55)
        print("VIDEO DETECTION DEBUG REPORT")
        print("=" * 55)
        print(f"  Total frames processed      : {frame_count}")
        print(f"  Frames with no tracking IDs : {frames_with_no_ids}")
        print(f"  Total unique track IDs      : {len(seen_track_ids)}")
        print(f"\n  RAW detections per class (across all frames):")
        for cls, cnt in sorted(
            raw_detections_per_class.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {cls:<20}: {cnt:>6} detections")
        print(f"\n  FINAL unique vehicle counts:")
        for cls, cnt in final_counts.items():
            print(f"    {cls:<20}: {cnt:>6} vehicles")
        print("=" * 55 + "\n")

        # ─────────────────────────────────────────────
        # ffmpeg: re-encode to browser-compatible mp4
        # ─────────────────────────────────────────────
        ffmpeg_path = self._get_ffmpeg_path()

        subprocess.run([
            ffmpeg_path,
            "-y",
            "-i", raw_output,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True)

        if os.path.exists(raw_output):
            os.remove(raw_output)

        inference_time = round(time.time() - start_time, 4)
        return output_path, final_counts, inference_time